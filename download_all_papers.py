"""
DOWNLOAD ALL PAPERS - Unified Downloader
=========================================

Downloads ALL papers regardless of DOI format:
- Old papers (10.1101) → bioRxiv S3 bucket (fast, no Cloudflare)
- New papers (10.64898) → Selenium browser (bypasses Cloudflare/CAPTCHA)

Usage:
    python download_all_papers.py --categories "cancer biology" --max 100
"""

import os
import sys
import io
import time
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

OLD_DOI_PREFIX = "10.1101"
NEW_DOI_PREFIX = "10.64898"

BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
MEDRXIV_S3_BUCKET = "medrxiv-src-monthly"
S3_PREFIX = "Current_Content"

BIORXIV_PDF_URL = "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
MEDRXIV_PDF_URL = "https://www.medrxiv.org/content/{doi}v{version}.full.pdf"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DownloadResult:
    doi: str
    success: bool
    method: str = ""  # "s3" or "selenium"
    wasabi_key: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None


@dataclass 
class DownloadStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    s3_downloads: int = 0
    selenium_downloads: int = 0
    total_bytes: int = 0
    start_time: datetime = None
    end_time: datetime = None


# =============================================================================
# UNIFIED DOWNLOADER
# =============================================================================

class UnifiedPaperDownloader:
    """
    Downloads ALL papers using the best method for each:
    - Old DOIs (10.1101) → S3 MECA download (requester-pays)
    - New DOIs (10.64898) → Selenium browser download
    """
    
    def __init__(self, skip_existing: bool = True):
        self.skip_existing = skip_existing
        
        # S3 client for bioRxiv (requester-pays - needs AWS credentials)
        self.biorxiv_s3 = None
        self.aws_configured = False
        self._init_aws_s3()
        
        # Wasabi client
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi()
        
        # Selenium driver (lazy init)
        self.driver = None
        self.selenium_ready = False
        
        logger.info("="*60)
        logger.info("UNIFIED PAPER DOWNLOADER")
        logger.info("="*60)
        if self.aws_configured:
            logger.info(f"Old papers (10.1101) → S3 MECA (requester-pays)")
        else:
            logger.info(f"Old papers (10.1101) → S3 NOT CONFIGURED (need AWS credentials)")
        logger.info(f"New papers (10.64898) → Selenium (browser)")
        if self.wasabi:
            logger.info(f"Destination: Wasabi s3://{self.wasabi_bucket}/")
        logger.info("="*60)
    
    def _init_aws_s3(self):
        """Initialize AWS S3 client with credentials for requester-pays"""
        
        # Try multiple sources for AWS credentials:
        # 1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        # 2. AWS CLI config (~/.aws/credentials) - boto3 picks this up automatically
        # 3. IAM role (if running on EC2)
        
        try:
            # Let boto3 find credentials automatically (env vars, ~/.aws/credentials, IAM role)
            self.biorxiv_s3 = boto3.client('s3', region_name='us-east-1')
            
            # Test if credentials work by checking caller identity
            sts = boto3.client('sts', region_name='us-east-1')
            identity = sts.get_caller_identity()
            account_id = identity.get('Account', 'unknown')
            
            self.aws_configured = True
            logger.info(f"✓ AWS configured (Account: ...{account_id[-4:]})")
            
        except Exception as e:
            logger.warning(f"⚠️  AWS credentials not found or invalid: {e}")
            logger.warning("   Run 'aws configure' or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
            logger.warning("   Old papers will fallback to Selenium")
            
            # Fallback to anonymous (won't work for requester-pays)
            self.biorxiv_s3 = boto3.client(
                's3',
                config=Config(signature_version=UNSIGNED),
                region_name='us-east-1'
            )
            self.aws_configured = False
    
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
            logger.warning(f"Wasabi init failed: {e}")
    
    def _init_selenium(self):
        """Initialize Selenium with undetected-chromedriver"""
        if self.selenium_ready:
            return True
        
        try:
            import undetected_chromedriver as uc
            
            logger.info("Starting browser for new papers...")
            
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920,1080')
            
            self.driver = uc.Chrome(options=options, use_subprocess=True)
            
            # Visit bioRxiv to establish session
            logger.info("  Opening bioRxiv...")
            self.driver.get("https://www.biorxiv.org")
            time.sleep(5)
            
            # Check for CAPTCHA
            page_source = self.driver.page_source.lower()
            if 'captcha' in page_source or 'challenge' in page_source:
                logger.warning("="*50)
                logger.warning("CAPTCHA DETECTED!")
                logger.warning("Please solve it in the browser window...")
                logger.warning("Waiting 60 seconds...")
                logger.warning("="*50)
                time.sleep(60)
            
            self.selenium_ready = True
            logger.info("✓ Browser ready")
            return True
            
        except ImportError:
            logger.error("undetected-chromedriver not installed!")
            logger.error("Run: pip install undetected-chromedriver")
            return False
        except Exception as e:
            logger.error(f"Selenium init failed: {e}")
            return False
    
    def _close_selenium(self):
        """Close browser"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
            self.selenium_ready = False
    
    # =========================================================================
    # S3 MECA DOWNLOAD (for old papers)
    # =========================================================================
    
    def _doi_to_meca_key(self, doi: str) -> str:
        """Convert DOI to S3 MECA key"""
        doi_suffix = doi.replace(f"{OLD_DOI_PREFIX}/", "")
        parts = doi_suffix.split(".")
        
        if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
            year_month = f"{parts[0]}-{parts[1].zfill(2)}"
        else:
            year_month = "legacy"
        
        filename = f"{doi.replace('/', '.')}.meca"
        return f"{S3_PREFIX}/{year_month}/{filename}"
    
    def _extract_pdf_from_meca(self, meca_bytes: bytes) -> Optional[bytes]:
        """Extract PDF from MECA ZIP"""
        try:
            with zipfile.ZipFile(io.BytesIO(meca_bytes), 'r') as zf:
                pdf_names = [n for n in zf.namelist() if n.lower().endswith('.pdf')]
                if not pdf_names:
                    return None
                
                # Find main PDF
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
    
    def _download_via_s3(self, doi: str, server: str, category: str) -> DownloadResult:
        """Download old paper via S3 MECA (requester-pays)"""
        
        # Check if AWS is configured
        if not self.aws_configured:
            # Fallback to Selenium for old papers too
            return self._download_via_selenium(doi, server, category)
        
        wasabi_key = self._doi_to_wasabi_key(doi, server, category)
        
        if self.skip_existing and self._check_wasabi_exists(wasabi_key):
            return DownloadResult(doi=doi, success=True, method="s3", 
                                  wasabi_key=wasabi_key, error="skipped_existing")
        
        meca_key = self._doi_to_meca_key(doi)
        bucket = BIORXIV_S3_BUCKET if server == "biorxiv" else MEDRXIV_S3_BUCKET
        
        try:
            # Download with RequestPayer='requester' for requester-pays bucket
            response = self.biorxiv_s3.get_object(
                Bucket=bucket, 
                Key=meca_key,
                RequestPayer='requester'  # IMPORTANT: requester-pays
            )
            meca_bytes = response['Body'].read()
            
            pdf_bytes = self._extract_pdf_from_meca(meca_bytes)
            if not pdf_bytes:
                return DownloadResult(doi=doi, success=False, method="s3", 
                                      error="No PDF in MECA")
            
            # Upload to Wasabi
            if self.wasabi:
                self.wasabi.put_object(
                    Bucket=self.wasabi_bucket,
                    Key=wasabi_key,
                    Body=pdf_bytes,
                    ContentType='application/pdf'
                )
            
            return DownloadResult(doi=doi, success=True, method="s3",
                                  wasabi_key=wasabi_key, file_size=len(pdf_bytes))
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            return DownloadResult(doi=doi, success=False, method="s3",
                                  error=f"S3: {error_code}")
        except Exception as e:
            return DownloadResult(doi=doi, success=False, method="s3",
                                  error=str(e)[:50])
    
    # =========================================================================
    # SELENIUM DOWNLOAD (for new papers)
    # =========================================================================
    
    def _download_via_selenium(self, doi: str, server: str, category: str) -> DownloadResult:
        """Download new paper via Selenium browser"""
        wasabi_key = self._doi_to_wasabi_key(doi, server, category)
        
        if self.skip_existing and self._check_wasabi_exists(wasabi_key):
            return DownloadResult(doi=doi, success=True, method="selenium",
                                  wasabi_key=wasabi_key, error="skipped_existing")
        
        if not self._init_selenium():
            return DownloadResult(doi=doi, success=False, method="selenium",
                                  error="Selenium not available")
        
        # Build PDF URL
        base_url = BIORXIV_PDF_URL if server == "biorxiv" else MEDRXIV_PDF_URL
        pdf_url = base_url.format(doi=doi, version=1)
        
        try:
            import base64
            
            # Navigate to PDF
            self.driver.get(pdf_url)
            time.sleep(3)
            
            # Check for Cloudflare challenge
            page_source = self.driver.page_source.lower()
            if 'challenge' in page_source or 'captcha' in page_source:
                logger.warning(f"Cloudflare challenge for {doi}, waiting...")
                time.sleep(10)
            
            # Try to fetch PDF via JavaScript
            pdf_script = """
            async function downloadPDF(url) {
                try {
                    const response = await fetch(url, {credentials: 'include'});
                    if (!response.ok) return {error: response.status};
                    const blob = await response.blob();
                    const reader = new FileReader();
                    return new Promise((resolve) => {
                        reader.onloadend = () => resolve({data: reader.result.split(',')[1]});
                        reader.readAsDataURL(blob);
                    });
                } catch (e) {
                    return {error: e.toString()};
                }
            }
            return await downloadPDF(arguments[0]);
            """
            
            result = self.driver.execute_script(pdf_script, pdf_url)
            
            if result and 'data' in result:
                pdf_bytes = base64.b64decode(result['data'])
                
                if pdf_bytes[:4] != b'%PDF':
                    return DownloadResult(doi=doi, success=False, method="selenium",
                                          error="Not a PDF")
                
                # Upload to Wasabi
                if self.wasabi:
                    self.wasabi.put_object(
                        Bucket=self.wasabi_bucket,
                        Key=wasabi_key,
                        Body=pdf_bytes,
                        ContentType='application/pdf'
                    )
                
                return DownloadResult(doi=doi, success=True, method="selenium",
                                      wasabi_key=wasabi_key, file_size=len(pdf_bytes))
            else:
                error = result.get('error', 'Unknown') if result else 'No response'
                return DownloadResult(doi=doi, success=False, method="selenium",
                                      error=f"Fetch failed: {error}")
                
        except Exception as e:
            return DownloadResult(doi=doi, success=False, method="selenium",
                                  error=str(e)[:50])
    
    # =========================================================================
    # COMMON METHODS
    # =========================================================================
    
    def _doi_to_wasabi_key(self, doi: str, server: str, category: str = None) -> str:
        """Convert DOI to Wasabi key"""
        safe_doi = doi.replace("/", "_").replace(".", "_")
        if category:
            safe_cat = category.replace(" ", "_").lower()
            return f"pdfs/{server}/{safe_cat}/{safe_doi}.pdf"
        return f"pdfs/{server}/{safe_doi}.pdf"
    
    def _check_wasabi_exists(self, key: str) -> bool:
        """Check if file exists in Wasabi"""
        if not self.wasabi:
            return False
        try:
            self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=key)
            return True
        except:
            return False
    
    def download_single(self, doi: str, server: str, category: str) -> DownloadResult:
        """Download single paper using appropriate method"""
        if doi.startswith(NEW_DOI_PREFIX):
            return self._download_via_selenium(doi, server, category)
        else:
            return self._download_via_s3(doi, server, category)
    
    def download_all(
        self,
        categories: List[str],
        max_papers: int = None,
        workers: int = 4
    ) -> Tuple[List[DownloadResult], DownloadStats]:
        """Download all papers from PostgreSQL"""
        
        # Get papers from PostgreSQL
        old_papers = []
        new_papers = []
        
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
                    paper_dict = {
                        "doi": p.doi,
                        "server": p.server or "biorxiv",
                        "category": p.category,
                    }
                    
                    if p.doi.startswith(NEW_DOI_PREFIX):
                        new_papers.append(paper_dict)
                    else:
                        old_papers.append(paper_dict)
            
            session.close()
            
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
            return [], DownloadStats()
        
        total_found = len(old_papers) + len(new_papers)
        logger.info(f"Found {total_found} papers to download:")
        logger.info(f"  - {len(old_papers)} old papers (10.1101) → S3")
        logger.info(f"  - {len(new_papers)} new papers (10.64898) → Selenium")
        
        # Apply max limit proportionally
        if max_papers and total_found > max_papers:
            ratio = max_papers / total_found
            old_limit = int(len(old_papers) * ratio)
            new_limit = max_papers - old_limit
            old_papers = old_papers[:old_limit]
            new_papers = new_papers[:new_limit]
            logger.info(f"Limited to {max_papers} papers ({len(old_papers)} old, {len(new_papers)} new)")
        
        stats = DownloadStats(total=len(old_papers) + len(new_papers))
        stats.start_time = datetime.now()
        results = []
        
        # =====================================================================
        # PHASE 1: Download OLD papers via S3 (parallel, fast)
        # =====================================================================
        if old_papers:
            print(f"\n{'='*60}")
            print(f"PHASE 1: Downloading {len(old_papers)} OLD papers via S3")
            print(f"{'='*60}\n")
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._download_via_s3,
                        p["doi"], p["server"], p["category"]
                    ): p for p in old_papers
                }
                
                for i, future in enumerate(as_completed(futures), 1):
                    paper = futures[future]
                    result = future.result()
                    results.append(result)
                    
                    short_doi = paper["doi"].split("/")[-1][:25]
                    
                    if result.success:
                        if result.error == "skipped_existing":
                            stats.skipped += 1
                            status = "⏭️  SKIP"
                        else:
                            stats.success += 1
                            stats.s3_downloads += 1
                            stats.total_bytes += result.file_size
                            status = f"✅ S3 ({result.file_size//1024}KB)"
                    else:
                        stats.failed += 1
                        status = f"❌ {result.error[:30]}"
                    
                    print(f"  [{i:4d}/{len(old_papers)}] {short_doi:<25} {status}")
        
        # =====================================================================
        # PHASE 2: Download NEW papers via Selenium (sequential, slower)
        # =====================================================================
        if new_papers:
            print(f"\n{'='*60}")
            print(f"PHASE 2: Downloading {len(new_papers)} NEW papers via Selenium")
            print(f"{'='*60}")
            print("(Browser window will open - solve CAPTCHA if it appears)\n")
            
            for i, paper in enumerate(new_papers, 1):
                result = self._download_via_selenium(
                    paper["doi"], paper["server"], paper["category"]
                )
                results.append(result)
                
                short_doi = paper["doi"].split("/")[-1][:25]
                
                if result.success:
                    if result.error == "skipped_existing":
                        stats.skipped += 1
                        status = "⏭️  SKIP"
                    else:
                        stats.success += 1
                        stats.selenium_downloads += 1
                        stats.total_bytes += result.file_size
                        status = f"✅ Browser ({result.file_size//1024}KB)"
                else:
                    stats.failed += 1
                    status = f"❌ {result.error[:30]}"
                
                print(f"  [{i:4d}/{len(new_papers)}] {short_doi:<25} {status}")
                
                # Small delay between Selenium downloads
                if result.success and result.error != "skipped_existing":
                    time.sleep(2)
        
        # Close browser
        self._close_selenium()
        
        stats.end_time = datetime.now()
        duration = (stats.end_time - stats.start_time).total_seconds()
        
        # Update PostgreSQL
        self._update_postgres(results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"  Total:    {stats.total}")
        print(f"  Success:  {stats.success} ({stats.s3_downloads} S3 + {stats.selenium_downloads} Selenium)")
        print(f"  Skipped:  {stats.skipped}")
        print(f"  Failed:   {stats.failed}")
        print(f"  Size:     {stats.total_bytes / 1024 / 1024:.2f} MB")
        print(f"  Duration: {duration:.1f} seconds")
        
        return results, stats
    
    def _update_postgres(self, results: List[DownloadResult]):
        """Update PostgreSQL with download results"""
        try:
            from db.progress import ProgressTracker
            tracker = ProgressTracker()
            
            for r in results:
                if r.success and r.error != "skipped_existing":
                    tracker.mark_downloaded(r.doi, r.wasabi_key or "", r.file_size)
            
            tracker.close()
        except Exception as e:
            logger.debug(f"PostgreSQL update failed: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download ALL papers (old via S3, new via Selenium)"
    )
    parser.add_argument("--categories", "-c", nargs="+", default=["cancer biology"])
    parser.add_argument("--max", "-m", type=int, help="Max papers to download")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Workers for S3 downloads")
    
    args = parser.parse_args()
    
    downloader = UnifiedPaperDownloader()
    downloader.download_all(
        categories=args.categories,
        max_papers=args.max,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
