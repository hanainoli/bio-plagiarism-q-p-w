"""
bioRxiv S3 MECA Downloader
==========================
Downloads MECA files from bioRxiv's public S3 bucket.
Only downloads papers matching your category filter.

S3 Bucket Structure:
    s3://biorxiv-src-monthly/
    ├── Current_Content/
    │   ├── 2024-01/
    │   │   ├── 10.1101.2024.01.01.123456.meca
    │   │   └── ...
    │   ├── 2024-02/
    │   └── ...

Usage:
    # Download cancer biology papers
    python s3_downloader.py --category "cancer biology" --output ./data/cancer
    
    # Download from DOI list
    python s3_downloader.py --dois dois.json --output ./data/papers
    
    # Download with parallel workers
    python s3_downloader.py --category "cancer biology" --workers 4
"""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Generator, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging

# AWS S3
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

# Local imports
from .category_index import CategoryIndexBuilder, DEFAULT_INDEX_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# bioRxiv S3 bucket (public, no auth required)
BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
MEDRXIV_S3_BUCKET = "medrxiv-src-monthly"  # Same structure

# S3 path format
S3_PREFIX = "Current_Content"

# Download settings
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "meca_files"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DownloadResult:
    """Result of a single download attempt"""
    doi: str
    success: bool
    local_path: Optional[Path] = None
    s3_key: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "success": self.success,
            "local_path": str(self.local_path) if self.local_path else None,
            "s3_key": self.s3_key,
            "file_size": self.file_size,
            "error": self.error
        }


@dataclass
class DownloadStats:
    """Statistics for a download batch"""
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    total_bytes: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0
        return self.success / self.total * 100
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_bytes / 1024 / 1024, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "success_rate": round(self.success_rate, 2)
        }


# =============================================================================
# S3 DOWNLOADER
# =============================================================================

class BioRxivS3Downloader:
    """
    Downloads MECA files from bioRxiv's public S3 bucket.
    
    The bucket is publicly accessible (no AWS credentials needed).
    Files are organized by year-month based on DOI structure.
    """
    
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        skip_existing: bool = True
    ):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            skip_existing: Skip files that already exist locally
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        
        # S3 client with anonymous access
        self.s3 = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED),
            region_name='us-east-1'
        )
        
        logger.info(f"S3 Downloader initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def doi_to_s3_key(self, doi: str, server: str = "biorxiv") -> str:
        """
        Convert DOI to S3 object key.
        
        DOI format: 10.1101/2024.01.15.575685
        S3 key: Current_Content/2024-01/10.1101.2024.01.15.575685.meca
        
        Args:
            doi: Paper DOI
            server: "biorxiv" or "medrxiv"
            
        Returns:
            S3 object key
        """
        # Remove DOI prefix
        doi_suffix = doi.replace("10.1101/", "")
        
        # Parse date parts
        parts = doi_suffix.split(".")
        
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
            # New format: 2024.01.15.575685
            year = parts[0]
            month = parts[1].zfill(2)
            year_month = f"{year}-{month}"
        else:
            # Old/legacy format - try to handle gracefully
            year_month = "legacy"
        
        # Build S3 key
        filename = f"{doi.replace('/', '.')}.meca"
        s3_key = f"{S3_PREFIX}/{year_month}/{filename}"
        
        return s3_key
    
    def doi_to_local_path(self, doi: str, category: str = None) -> Path:
        """
        Convert DOI to local file path.
        
        Args:
            doi: Paper DOI
            category: Optional category for subdirectory
            
        Returns:
            Local file path
        """
        # Sanitize DOI for filename
        safe_doi = doi.replace("/", "_").replace(".", "_")
        filename = f"{safe_doi}.meca"
        
        if category:
            safe_category = category.replace(" ", "_").lower()
            return self.output_dir / safe_category / filename
        else:
            return self.output_dir / filename
    
    def check_s3_exists(self, s3_key: str, bucket: str = BIORXIV_S3_BUCKET) -> bool:
        """Check if object exists in S3"""
        try:
            self.s3.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def download_single(
        self,
        doi: str,
        server: str = "biorxiv",
        category: str = None,
        debug: bool = False
    ) -> DownloadResult:
        """
        Download a single MECA file.
        
        Args:
            doi: Paper DOI
            server: "biorxiv" or "medrxiv"
            category: Optional category for organization
            debug: Enable debug logging
            
        Returns:
            DownloadResult with status and path
        """
        # Determine paths
        s3_key = self.doi_to_s3_key(doi, server)
        local_path = self.doi_to_local_path(doi, category)
        bucket = BIORXIV_S3_BUCKET if server == "biorxiv" else MEDRXIV_S3_BUCKET
        
        if debug:
            logger.info(f"DEBUG: DOI={doi}, bucket={bucket}, key={s3_key}")
        
        # Skip if exists
        if self.skip_existing and local_path.exists():
            return DownloadResult(
                doi=doi,
                success=True,
                local_path=local_path,
                s3_key=s3_key,
                file_size=local_path.stat().st_size,
                error="skipped_existing"
            )
        
        # Create directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try multiple bucket/key combinations
        attempts_info = []
        
        # Try different S3 locations
        locations_to_try = [
            (bucket, s3_key),
            (BIORXIV_S3_BUCKET, s3_key),  # Always try biorxiv
            (MEDRXIV_S3_BUCKET, s3_key),  # Always try medrxiv
        ]
        
        for try_bucket, try_key in locations_to_try:
            try:
                response = self.s3.get_object(Bucket=try_bucket, Key=try_key)
                
                # Read and save
                meca_bytes = response['Body'].read()
                local_path.write_bytes(meca_bytes)
                
                if debug:
                    logger.info(f"SUCCESS: {try_bucket}/{try_key}")
                
                return DownloadResult(
                    doi=doi,
                    success=True,
                    local_path=local_path,
                    s3_key=try_key,
                    file_size=len(meca_bytes)
                )
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                attempts_info.append(f"{try_bucket}: {error_code}")
                
                if error_code != 'NoSuchKey':
                    # Some other error, wait and continue
                    time.sleep(RETRY_DELAY)
                    
            except Exception as e:
                attempts_info.append(f"{try_bucket}: {str(e)[:50]}")
        
        error_msg = "; ".join(attempts_info) if attempts_info else "Unknown error"
        
        return DownloadResult(
            doi=doi,
            success=False,
            s3_key=s3_key,
            error=error_msg
        )
    
    def download_batch(
        self,
        dois: List[str],
        server: str = "biorxiv",
        category: str = None,
        workers: int = 4,
        progress: bool = True
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """
        Download multiple MECA files in parallel.
        
        Args:
            dois: List of DOIs to download
            server: "biorxiv" or "medrxiv"
            category: Optional category for organization
            workers: Number of parallel download workers
            progress: Show progress bar
            
        Returns:
            Tuple of (results list, stats)
        """
        stats = DownloadStats(total=len(dois))
        stats.start_time = datetime.now()
        results = []
        
        logger.info(f"Starting download of {len(dois)} papers with {workers} workers")
        
        # Use thread pool for parallel downloads
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all download tasks
            future_to_doi = {
                executor.submit(self.download_single, doi, server, category): doi
                for doi in dois
            }
            
            # Process completed downloads
            iterator = as_completed(future_to_doi)
            if progress and HAS_TQDM:
                iterator = tqdm(iterator, total=len(dois), desc="Downloading")
            
            for future in iterator:
                result = future.result()
                results.append(result)
                
                if result.success:
                    if result.error == "skipped_existing":
                        stats.skipped += 1
                    else:
                        stats.success += 1
                    stats.total_bytes += result.file_size
                else:
                    stats.failed += 1
        
        stats.end_time = datetime.now()
        
        logger.info(f"Download complete: {stats.success} success, {stats.skipped} skipped, {stats.failed} failed")
        logger.info(f"Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
        logger.info(f"Duration: {stats.duration_seconds:.2f} seconds")
        
        return results, stats
    
    def download_category(
        self,
        categories: List[str],
        index_path: Path = DEFAULT_INDEX_PATH,
        workers: int = 4,
        max_papers: int = None,
        progress: bool = True
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """
        Download all papers in specified categories.
        
        Args:
            categories: List of category names (fuzzy matched)
            index_path: Path to category index
            workers: Number of parallel workers
            max_papers: Maximum papers to download (for testing)
            progress: Show progress bar
            
        Returns:
            Tuple of (results list, stats)
        """
        # Load index and get papers
        builder = CategoryIndexBuilder(index_path)
        papers = builder.get_papers_by_category(categories)
        
        if not papers:
            logger.warning(f"No papers found for categories: {categories}")
            return [], DownloadStats()
        
        logger.info(f"Found {len(papers)} papers in categories: {categories}")
        
        # Limit if requested
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        # Group by server
        biorxiv_dois = [p["doi"] for p in papers if p.get("server") == "biorxiv"]
        medrxiv_dois = [p["doi"] for p in papers if p.get("server") == "medrxiv"]
        
        # Use first category for folder name
        category_name = categories[0] if categories else "papers"
        
        all_results = []
        combined_stats = DownloadStats()
        combined_stats.start_time = datetime.now()
        
        # Download bioRxiv papers
        if biorxiv_dois:
            logger.info(f"Downloading {len(biorxiv_dois)} bioRxiv papers...")
            results, stats = self.download_batch(
                biorxiv_dois, "biorxiv", category_name, workers, progress
            )
            all_results.extend(results)
            combined_stats.success += stats.success
            combined_stats.failed += stats.failed
            combined_stats.skipped += stats.skipped
            combined_stats.total_bytes += stats.total_bytes
        
        # Download medRxiv papers
        if medrxiv_dois:
            logger.info(f"Downloading {len(medrxiv_dois)} medRxiv papers...")
            results, stats = self.download_batch(
                medrxiv_dois, "medrxiv", category_name, workers, progress
            )
            all_results.extend(results)
            combined_stats.success += stats.success
            combined_stats.failed += stats.failed
            combined_stats.skipped += stats.skipped
            combined_stats.total_bytes += stats.total_bytes
        
        combined_stats.total = len(papers)
        combined_stats.end_time = datetime.now()
        
        return all_results, combined_stats
    
    def download_from_file(
        self,
        dois_file: Path,
        server: str = "biorxiv",
        category: str = None,
        workers: int = 4,
        progress: bool = True
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """
        Download papers from a DOI list file.
        
        Args:
            dois_file: JSON file containing list of DOIs
            server: Default server for DOIs
            category: Optional category for organization
            workers: Number of parallel workers
            progress: Show progress bar
            
        Returns:
            Tuple of (results list, stats)
        """
        with open(dois_file) as f:
            dois = json.load(f)
        
        if isinstance(dois, dict):
            # Handle {server: [dois]} format
            all_results = []
            combined_stats = DownloadStats(total=sum(len(v) for v in dois.values()))
            combined_stats.start_time = datetime.now()
            
            for srv, doi_list in dois.items():
                results, stats = self.download_batch(
                    doi_list, srv, category, workers, progress
                )
                all_results.extend(results)
                combined_stats.success += stats.success
                combined_stats.failed += stats.failed
                combined_stats.skipped += stats.skipped
                combined_stats.total_bytes += stats.total_bytes
            
            combined_stats.end_time = datetime.now()
            return all_results, combined_stats
        else:
            # Simple list of DOIs
            return self.download_batch(dois, server, category, workers, progress)
    
    def list_downloaded(self, category: str = None) -> List[Path]:
        """List all downloaded MECA files"""
        if category:
            safe_category = category.replace(" ", "_").lower()
            search_dir = self.output_dir / safe_category
        else:
            search_dir = self.output_dir
        
        if not search_dir.exists():
            return []
        
        return list(search_dir.glob("**/*.meca"))
    
    def get_download_report(
        self,
        results: List[DownloadResult],
        stats: DownloadStats
    ) -> dict:
        """Generate a download report"""
        failed_dois = [r.doi for r in results if not r.success]
        
        return {
            "summary": stats.to_dict(),
            "failed_dois": failed_dois,
            "output_directory": str(self.output_dir),
            "generated_at": datetime.now().isoformat()
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def download_category(
    categories: List[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 4,
    max_papers: int = None
) -> Tuple[List[DownloadResult], DownloadStats]:
    """Download papers for given categories"""
    downloader = BioRxivS3Downloader(output_dir)
    return downloader.download_category(categories, workers=workers, max_papers=max_papers)


def download_dois(
    dois: List[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    server: str = "biorxiv",
    workers: int = 4
) -> Tuple[List[DownloadResult], DownloadStats]:
    """Download papers by DOI list"""
    downloader = BioRxivS3Downloader(output_dir)
    return downloader.download_batch(dois, server, workers=workers)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="bioRxiv S3 MECA Downloader")
    parser.add_argument("--category", type=str, nargs="+", help="Category/categories to download")
    parser.add_argument("--dois", type=str, help="JSON file with DOI list")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--max", type=int, help="Maximum papers to download")
    parser.add_argument("--index", type=str, default=str(DEFAULT_INDEX_PATH), help="Category index path")
    parser.add_argument("--list", action="store_true", help="List downloaded files")
    parser.add_argument("--report", type=str, help="Save download report to file")
    
    args = parser.parse_args()
    
    downloader = BioRxivS3Downloader(
        output_dir=Path(args.output),
        skip_existing=True
    )
    
    if args.list:
        files = downloader.list_downloaded(args.category[0] if args.category else None)
        print(f"\nDownloaded files: {len(files)}")
        for f in files[:20]:
            print(f"  {f}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
    
    elif args.category:
        results, stats = downloader.download_category(
            categories=args.category,
            index_path=Path(args.index),
            workers=args.workers,
            max_papers=args.max
        )
        
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {stats.success}")
        print(f"Skipped: {stats.skipped}")
        print(f"Failed: {stats.failed}")
        print(f"Total Size: {stats.total_bytes / 1024 / 1024:.2f} MB")
        print(f"Duration: {stats.duration_seconds:.2f} seconds")
        
        if args.report:
            report = downloader.get_download_report(results, stats)
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.report}")
    
    elif args.dois:
        results, stats = downloader.download_from_file(
            Path(args.dois),
            workers=args.workers
        )
        
        print(f"\nDownloaded {stats.success} papers")
    
    else:
        parser.print_help()
