"""
STEP 2 ALTERNATIVE: Download from bioRxiv S3 (MECA files)
=========================================================

This bypasses Cloudflare by downloading MECA files from bioRxiv's public S3 bucket.
MECA files contain PDF + XML + figures.

Workflow:
1. Download MECA from s3://biorxiv-src-monthly/
2. Extract PDF from MECA (it's a ZIP file)
3. Store PDF in Wasabi S3 (same as before)

Usage:
    python step2_meca_download.py --category "cancer biology" --max 100

"""

import os
import sys
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# bioRxiv public S3 bucket (no auth needed!)
BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
MEDRXIV_S3_BUCKET = "medrxiv-src-monthly"
S3_PREFIX = "Current_Content"

# DOI prefixes
OLD_DOI_PREFIX = "10.1101"
NEW_DOI_PREFIX = "10.64898"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DownloadResult:
    doi: str
    success: bool
    wasabi_key: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None


@dataclass
class DownloadStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    total_bytes: int = 0
    start_time: datetime = None
    end_time: datetime = None


# =============================================================================
# MECA DOWNLOADER
# =============================================================================

class MECAToWasabiDownloader:
    """
    Downloads MECA from bioRxiv S3 → Extracts PDF → Uploads to Wasabi.
    
    This bypasses Cloudflare completely!
    """
    
    def __init__(self, skip_existing: bool = True):
        self.skip_existing = skip_existing
        
        # bioRxiv S3 client (public, anonymous access)
        self.biorxiv_s3 = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED),
            region_name='us-east-1'
        )
        
        # Wasabi client (your storage)
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi()
        
        logger.info("MECA Downloader initialized")
        logger.info(f"  Source: s3://{BIORXIV_S3_BUCKET}/ (public)")
        if self.wasabi:
            logger.info(f"  Destination: Wasabi s3://{self.wasabi_bucket}/")
    
    def _init_wasabi(self):
        """Initialize Wasabi client"""
        try:
            from storage.wasabi_client import WasabiClient
            wasabi_client = WasabiClient()
            
            if wasabi_client.access_key and wasabi_client.secret_key:
                self.wasabi = wasabi_client.client
                self.wasabi_bucket = wasabi_client.bucket
                logger.info("✓ Wasabi S3 connected")
            else:
                logger.warning("⚠️  Wasabi credentials not found")
        except Exception as e:
            logger.warning(f"⚠️  Wasabi init failed: {e}")
    
    def doi_to_meca_key(self, doi: str) -> str:
        """
        Convert DOI to S3 MECA key.
        
        DOI: 10.1101/2024.01.15.575685
        Key: Current_Content/2024-01/10.1101.2024.01.15.575685.meca
        """
        # Handle both old and new DOI formats
        if doi.startswith(NEW_DOI_PREFIX):
            # New format: 10.64898/2025.12.09.693245
            doi_suffix = doi.replace(f"{NEW_DOI_PREFIX}/", "")
        else:
            # Old format: 10.1101/2024.01.15.575685
            doi_suffix = doi.replace(f"{OLD_DOI_PREFIX}/", "")
        
        # Parse date parts
        parts = doi_suffix.split(".")
        
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
            year = parts[0]
            month = parts[1].zfill(2)
            year_month = f"{year}-{month}"
        else:
            year_month = "legacy"
        
        # Build S3 key
        filename = f"{doi.replace('/', '.')}.meca"
        return f"{S3_PREFIX}/{year_month}/{filename}"
    
    def doi_to_wasabi_key(self, doi: str, server: str = "biorxiv", category: str = None) -> str:
        """Convert DOI to Wasabi PDF key"""
        safe_doi = doi.replace("/", "_").replace(".", "_")
        
        if category:
            safe_category = category.replace(" ", "_").lower()
            return f"pdfs/{server}/{safe_category}/{safe_doi}.pdf"
        else:
            return f"pdfs/{server}/{safe_doi}.pdf"
    
    def check_wasabi_exists(self, wasabi_key: str) -> bool:
        """Check if PDF already exists in Wasabi"""
        if not self.wasabi:
            return False
        try:
            self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=wasabi_key)
            return True
        except:
            return False
    
    def extract_pdf_from_meca(self, meca_bytes: bytes) -> Optional[bytes]:
        """Extract PDF from MECA (ZIP) file"""
        try:
            with zipfile.ZipFile(tempfile.SpooledTemporaryFile(max_size=50*1024*1024)) as zf:
                # Write MECA bytes to temp file
                import io
                zf_io = io.BytesIO(meca_bytes)
                
            with zipfile.ZipFile(io.BytesIO(meca_bytes), 'r') as zf:
                # Find PDF in archive
                pdf_names = [n for n in zf.namelist() if n.lower().endswith('.pdf')]
                
                if not pdf_names:
                    logger.warning("No PDF found in MECA")
                    return None
                
                # Get the main PDF (usually in content/ folder)
                main_pdf = None
                for name in pdf_names:
                    if 'content/' in name.lower() or 'manuscript' in name.lower():
                        main_pdf = name
                        break
                
                if not main_pdf:
                    main_pdf = pdf_names[0]
                
                # Extract PDF bytes
                pdf_bytes = zf.read(main_pdf)
                return pdf_bytes
                
        except Exception as e:
            logger.error(f"Error extracting PDF from MECA: {e}")
            return None
    
    def download_single(
        self,
        doi: str,
        server: str = "biorxiv",
        category: str = None
    ) -> DownloadResult:
        """
        Download single paper: MECA from bioRxiv S3 → PDF → Wasabi
        """
        wasabi_key = self.doi_to_wasabi_key(doi, server, category)
        
        # Skip if exists
        if self.skip_existing and self.check_wasabi_exists(wasabi_key):
            return DownloadResult(
                doi=doi,
                success=True,
                wasabi_key=wasabi_key,
                error="skipped_existing"
            )
        
        # Get MECA key
        meca_key = self.doi_to_meca_key(doi)
        bucket = BIORXIV_S3_BUCKET if server == "biorxiv" else MEDRXIV_S3_BUCKET
        
        try:
            # Download MECA from bioRxiv S3
            response = self.biorxiv_s3.get_object(Bucket=bucket, Key=meca_key)
            meca_bytes = response['Body'].read()
            
            # Extract PDF
            pdf_bytes = self.extract_pdf_from_meca(meca_bytes)
            
            if not pdf_bytes:
                return DownloadResult(
                    doi=doi,
                    success=False,
                    error="No PDF in MECA"
                )
            
            # Upload to Wasabi
            if self.wasabi:
                self.wasabi.put_object(
                    Bucket=self.wasabi_bucket,
                    Key=wasabi_key,
                    Body=pdf_bytes,
                    ContentType='application/pdf'
                )
                
                return DownloadResult(
                    doi=doi,
                    success=True,
                    wasabi_key=wasabi_key,
                    file_size=len(pdf_bytes)
                )
            else:
                # Save locally if no Wasabi
                local_path = Path("data/pdfs") / f"{doi.replace('/', '_')}.pdf"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(pdf_bytes)
                
                return DownloadResult(
                    doi=doi,
                    success=True,
                    file_size=len(pdf_bytes)
                )
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404' or error_code == 'NoSuchKey':
                return DownloadResult(
                    doi=doi,
                    success=False,
                    error=f"MECA not found in S3 (key: {meca_key})"
                )
            return DownloadResult(
                doi=doi,
                success=False,
                error=f"S3 error: {error_code}"
            )
        except Exception as e:
            return DownloadResult(
                doi=doi,
                success=False,
                error=str(e)[:100]
            )
    
    def download_batch(
        self,
        papers: List[dict],
        workers: int = 4
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """Download multiple papers"""
        stats = DownloadStats(total=len(papers))
        stats.start_time = datetime.now()
        results = []
        
        logger.info(f"Downloading {len(papers)} papers with {workers} workers")
        logger.info("Source: bioRxiv S3 (no Cloudflare!) → Wasabi")
        print()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            
            for paper in papers:
                future = executor.submit(
                    self.download_single,
                    doi=paper.get("doi"),
                    server=paper.get("server", "biorxiv"),
                    category=paper.get("category")
                )
                futures[future] = paper
            
            for i, future in enumerate(as_completed(futures), 1):
                paper = futures[future]
                result = future.result()
                results.append(result)
                
                doi = paper.get('doi', 'unknown')
                short_doi = doi.split('/')[-1] if '/' in doi else doi
                
                if result.success:
                    if result.error == "skipped_existing":
                        stats.skipped += 1
                        status = "⏭️  SKIPPED"
                    else:
                        stats.success += 1
                        stats.total_bytes += result.file_size
                        size_kb = result.file_size / 1024
                        status = f"✅ SUCCESS ({size_kb:.1f} KB)"
                else:
                    stats.failed += 1
                    status = f"❌ FAILED: {result.error[:40]}"
                
                print(f"  [{i:3d}/{len(papers)}] {short_doi[:30]:<30} {status}")
        
        stats.end_time = datetime.now()
        
        print()
        logger.info(f"Complete: {stats.success} success, {stats.skipped} skipped, {stats.failed} failed")
        logger.info(f"Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
        
        return results, stats
    
    def download_from_postgres(
        self,
        categories: List[str],
        max_papers: int = None,
        workers: int = 4
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """Download papers from PostgreSQL database"""
        papers = []
        
        try:
            from db.models import Paper, get_session
            from sqlalchemy import func
            
            session = get_session()
            
            for cat in categories:
                cat_papers = session.query(Paper).filter(
                    func.lower(Paper.category).like(f"%{cat.lower()}%")
                ).filter(
                    Paper.pdf_downloaded == False
                ).all()
                
                for p in cat_papers:
                    papers.append({
                        "doi": p.doi,
                        "title": p.title,
                        "category": p.category,
                        "server": p.server,
                    })
                
                logger.info(f"Found {len(cat_papers)} papers in '{cat}'")
            
            session.close()
            
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
            return [], DownloadStats()
        
        if not papers:
            logger.warning("No papers found")
            return [], DownloadStats()
        
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        return self.download_batch(papers, workers=workers)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def step2_download_meca(
    categories: List[str],
    max_papers: Optional[int] = None,
    workers: int = 4,
    force: bool = False
):
    """
    Download PDFs via MECA from bioRxiv S3 (bypasses Cloudflare!)
    
    This is the alternative to direct PDF download when blocked by Cloudflare.
    """
    print("\n" + "="*70)
    print("STEP 2: DOWNLOAD via S3 MECA (No Cloudflare!)")
    print("="*70)
    print(f"Categories: {categories}")
    print(f"Max papers: {max_papers or 'all'}")
    print(f"Workers: {workers}")
    print(f"Source: bioRxiv S3 (public bucket)")
    
    downloader = MECAToWasabiDownloader(skip_existing=not force)
    
    results, stats = downloader.download_from_postgres(
        categories=categories,
        max_papers=max_papers,
        workers=workers
    )
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"  Success: {stats.success}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
    
    # Update PostgreSQL for successful downloads
    if stats.success > 0:
        try:
            from db.progress import ProgressTracker
            tracker = ProgressTracker()
            
            for r in results:
                if r.success and r.error != "skipped_existing":
                    tracker.mark_downloaded(
                        doi=r.doi,
                        pdf_path=r.wasabi_key or "",
                        pdf_size=r.file_size
                    )
            
            tracker.close()
            logger.info("PostgreSQL updated")
        except Exception as e:
            logger.warning(f"Could not update PostgreSQL: {e}")
    
    return results, stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download via S3 MECA (bypasses Cloudflare)")
    parser.add_argument("--categories", "-c", nargs="+", default=["cancer biology"],
                        help="Categories to download")
    parser.add_argument("--max", "-m", type=int, help="Max papers to download")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    step2_download_meca(
        categories=args.categories,
        max_papers=args.max,
        workers=args.workers,
        force=args.force
    )
