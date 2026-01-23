"""
Download OLD papers from bioRxiv S3 (requester-pays)
=====================================================

Downloads papers with 10.1101 DOI prefix from bioRxiv's S3 bucket.
Uses the local biorxiv_index.json which has older papers.

Usage:
    python download_old_papers_s3.py --category "cancer biology" --max 100 --workers 4
"""

import os
import sys
import io
import json
import zipfile
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
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

OLD_DOI_PREFIX = "10.1101"
BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
MEDRXIV_S3_BUCKET = "medrxiv-src-monthly"
S3_PREFIX = "Current_Content"

INDEX_PATH = PROJECT_ROOT / "data" / "biorxiv_index.json"


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


# =============================================================================
# S3 DOWNLOADER
# =============================================================================

class OldPaperS3Downloader:
    """Downloads old papers (10.1101) from bioRxiv S3 to Wasabi"""
    
    def __init__(self, skip_existing: bool = True):
        self.skip_existing = skip_existing
        
        # AWS S3 client (uses ~/.aws/credentials)
        self.aws_s3 = boto3.client('s3', region_name='us-east-1')
        
        # Verify AWS credentials
        try:
            sts = boto3.client('sts', region_name='us-east-1')
            account = sts.get_caller_identity()['Account']
            logger.info(f"✓ AWS configured (Account: ...{account[-4:]})")
        except Exception as e:
            logger.error(f"AWS credentials not found: {e}")
            logger.error("Run 'aws configure' first")
            sys.exit(1)
        
        # Wasabi client
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi()
        
        logger.info(f"Source: s3://{BIORXIV_S3_BUCKET}/ (requester-pays)")
        if self.wasabi:
            logger.info(f"Destination: Wasabi s3://{self.wasabi_bucket}/")
    
    def _init_wasabi(self):
        """Initialize Wasabi client"""
        try:
            from storage.wasabi_client import WasabiClient
            wc = WasabiClient()
            if wc.access_key and wc.secret_key:
                self.wasabi = wc.client
                self.wasabi_bucket = wc.bucket
                logger.info("✓ Wasabi connected")
        except Exception as e:
            logger.warning(f"Wasabi not configured: {e}")
    
    def doi_to_meca_key(self, doi: str) -> str:
        """Convert DOI to S3 MECA key path"""
        # DOI: 10.1101/2024.01.15.575685
        # Key: Current_Content/2024-01/10.1101.2024.01.15.575685.meca
        
        doi_suffix = doi.replace(f"{OLD_DOI_PREFIX}/", "")
        parts = doi_suffix.split(".")
        
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
            year_month = f"{parts[0]}-{parts[1].zfill(2)}"
        else:
            year_month = "legacy"
        
        filename = f"{doi.replace('/', '.')}.meca"
        return f"{S3_PREFIX}/{year_month}/{filename}"
    
    def doi_to_wasabi_key(self, doi: str, server: str, category: str = None) -> str:
        """Convert DOI to Wasabi key"""
        safe_doi = doi.replace("/", "_").replace(".", "_")
        if category:
            safe_cat = category.replace(" ", "_").lower()
            return f"pdfs/{server}/{safe_cat}/{safe_doi}.pdf"
        return f"pdfs/{server}/{safe_doi}.pdf"
    
    def check_wasabi_exists(self, key: str) -> bool:
        """Check if file exists in Wasabi"""
        if not self.wasabi:
            return False
        try:
            self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=key)
            return True
        except:
            return False
    
    def extract_pdf_from_meca(self, meca_bytes: bytes) -> Optional[bytes]:
        """Extract PDF from MECA ZIP file"""
        try:
            with zipfile.ZipFile(io.BytesIO(meca_bytes), 'r') as zf:
                pdf_names = [n for n in zf.namelist() if n.lower().endswith('.pdf')]
                if not pdf_names:
                    return None
                
                # Find main PDF (prefer content/manuscript.pdf)
                main_pdf = None
                for name in pdf_names:
                    if 'content/' in name.lower() or 'manuscript' in name.lower():
                        main_pdf = name
                        break
                if not main_pdf:
                    main_pdf = pdf_names[0]
                
                return zf.read(main_pdf)
        except Exception as e:
            logger.debug(f"MECA extraction error: {e}")
            return None
    
    def download_single(self, doi: str, server: str, category: str) -> DownloadResult:
        """Download single paper from S3"""
        wasabi_key = self.doi_to_wasabi_key(doi, server, category)
        
        # Skip if exists
        if self.skip_existing and self.check_wasabi_exists(wasabi_key):
            return DownloadResult(doi=doi, success=True, wasabi_key=wasabi_key, 
                                  error="skipped_existing")
        
        meca_key = self.doi_to_meca_key(doi)
        bucket = BIORXIV_S3_BUCKET if server == "biorxiv" else MEDRXIV_S3_BUCKET
        
        try:
            # Download MECA with requester-pays
            response = self.aws_s3.get_object(
                Bucket=bucket,
                Key=meca_key,
                RequestPayer='requester'
            )
            meca_bytes = response['Body'].read()
            
            # Extract PDF
            pdf_bytes = self.extract_pdf_from_meca(meca_bytes)
            if not pdf_bytes:
                return DownloadResult(doi=doi, success=False, error="No PDF in MECA")
            
            # Upload to Wasabi
            if self.wasabi:
                self.wasabi.put_object(
                    Bucket=self.wasabi_bucket,
                    Key=wasabi_key,
                    Body=pdf_bytes,
                    ContentType='application/pdf'
                )
            
            return DownloadResult(doi=doi, success=True, wasabi_key=wasabi_key,
                                  file_size=len(pdf_bytes))
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            return DownloadResult(doi=doi, success=False, error=f"S3: {error_code}")
        except Exception as e:
            return DownloadResult(doi=doi, success=False, error=str(e)[:50])
    
    def load_papers_from_index(self, categories: List[str], max_papers: int = None) -> List[dict]:
        """Load old papers from biorxiv_index.json"""
        if not INDEX_PATH.exists():
            logger.error(f"Index not found: {INDEX_PATH}")
            return []
        
        with open(INDEX_PATH) as f:
            index_data = json.load(f)
        
        papers = []
        categories_lower = [c.lower() for c in categories]
        
        # Index structure: {"categories": {"cancer biology": [papers], ...}}
        all_categories = index_data.get('categories', {})
        
        for cat_name, cat_papers in all_categories.items():
            # Match category
            if not any(cat in cat_name.lower() for cat in categories_lower):
                continue
            
            for paper in cat_papers:
                doi = paper.get('doi', '')
                
                # Only old DOIs
                if not doi.startswith(OLD_DOI_PREFIX):
                    continue
                
                papers.append({
                    'doi': doi,
                    'server': paper.get('server', 'biorxiv'),
                    'category': cat_name,
                    'title': paper.get('title', '')
                })
        
        logger.info(f"Found {len(papers)} old papers (10.1101) in index for {categories}")
        
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        return papers
    
    def download_batch(
        self,
        papers: List[dict],
        workers: int = 4
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """Download multiple papers in parallel"""
        stats = DownloadStats(total=len(papers))
        results = []
        
        logger.info(f"Downloading {len(papers)} papers with {workers} workers")
        print()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.download_single,
                    p['doi'], p['server'], p['category']
                ): p for p in papers
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                paper = futures[future]
                result = future.result()
                results.append(result)
                
                short_doi = paper['doi'].split('/')[-1][:25]
                
                if result.success:
                    if result.error == "skipped_existing":
                        stats.skipped += 1
                        status = "⏭️  SKIP"
                    else:
                        stats.success += 1
                        stats.total_bytes += result.file_size
                        status = f"✅ OK ({result.file_size//1024}KB)"
                else:
                    stats.failed += 1
                    status = f"❌ {result.error[:35]}"
                
                print(f"  [{i:4d}/{len(papers)}] {short_doi:<25} {status}")
        
        return results, stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OLD papers from bioRxiv S3")
    parser.add_argument("--category", "-c", nargs="+", default=["cancer biology"],
                        help="Categories to download")
    parser.add_argument("--max", "-m", type=int, default=100, help="Max papers")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    
    args = parser.parse_args()
    
    print("="*60)
    print("DOWNLOAD OLD PAPERS FROM S3 (requester-pays)")
    print("="*60)
    
    downloader = OldPaperS3Downloader()
    
    # Load papers from local index
    papers = downloader.load_papers_from_index(args.category, args.max)
    
    if not papers:
        print("No papers found!")
        return
    
    # Download
    results, stats = downloader.download_batch(papers, args.workers)
    
    # Summary
    print()
    print("="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"  Success: {stats.success}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed:  {stats.failed}")
    print(f"  Size:    {stats.total_bytes / 1024 / 1024:.2f} MB")
    
    # Show failed DOIs
    failed = [r for r in results if not r.success]
    if failed and len(failed) <= 10:
        print(f"\nFailed DOIs:")
        for r in failed[:10]:
            print(f"  {r.doi}: {r.error}")


if __name__ == "__main__":
    main()
