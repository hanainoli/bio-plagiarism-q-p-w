"""
PDF Downloader for bioRxiv/medRxiv
==================================

Downloads PDFs directly from the bioRxiv/medRxiv website.
Stores PDFs in Wasabi S3 (or local if not configured).

This is FREE and doesn't require AWS credentials for downloading.
"""

import os
import time
import random
import requests
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
import logging

# Try to import cloudscraper for Cloudflare bypass
try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

# Try to import Selenium for stronger Cloudflare bypass
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# Try undetected-chromedriver (best for avoiding CAPTCHA)
try:
    import undetected_chromedriver as uc
    HAS_UNDETECTED = True
except ImportError:
    HAS_UNDETECTED = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

from .category_index import CategoryIndex, DEFAULT_INDEX_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Base URLs for PDF download
BIORXIV_PDF_URL = "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
MEDRXIV_PDF_URL = "https://www.medrxiv.org/content/{doi}v{version}.full.pdf"

# Alternative PDF URLs (without version, for fallback)
BIORXIV_PDF_ALT = "https://www.biorxiv.org/content/{doi}.full.pdf"
MEDRXIV_PDF_ALT = "https://www.medrxiv.org/content/{doi}.full.pdf"

# API for getting latest version (works for both old and new DOIs)
BIORXIV_API_URL = "https://api.biorxiv.org/details/biorxiv/{doi}"
MEDRXIV_API_URL = "https://api.biorxiv.org/details/medrxiv/{doi}"

# DOI prefixes
OLD_DOI_PREFIX = "10.1101"
NEW_DOI_PREFIX = "10.64898"  # Started December 1, 2025
OLD_DOI_PREFIX = "10.1101"

# Download settings
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "pdfs"
MAX_RETRIES = 4  # Increased retries
RETRY_DELAY = 5.0  # Longer delay for retries
RATE_LIMIT_DELAY = 2.5  # 2.5 seconds between requests to avoid 403

# Request headers - Use a realistic browser User-Agent to avoid 403 blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/pdf',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0'
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PDFDownloadResult:
    """Result of a single PDF download"""
    doi: str
    success: bool
    local_path: Optional[Path] = None
    s3_key: Optional[str] = None
    url: Optional[str] = None
    file_size: int = 0
    version: int = 1
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "success": self.success,
            "local_path": str(self.local_path) if self.local_path else None,
            "s3_key": self.s3_key,
            "url": self.url,
            "file_size": self.file_size,
            "version": self.version,
            "error": self.error
        }


@dataclass
class PDFDownloadStats:
    """Statistics for download batch"""
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
            "total_mb": round(self.total_bytes / 1024 / 1024, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "success_rate": round(self.success_rate, 2)
        }


# =============================================================================
# PDF DOWNLOADER
# =============================================================================

class BioRxivPDFDownloader:
    """
    Downloads PDFs directly from bioRxiv/medRxiv website.
    Stores in Wasabi S3 (or local if not configured).
    
    This is free and doesn't require AWS credentials for downloading.
    Rate-limited to be respectful to the servers.
    """
    
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        skip_existing: bool = True,
        use_wasabi: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        
        # Initialize Selenium browser for Cloudflare bypass
        self.driver = None
        self.use_selenium = False
        
        # Try undetected-chromedriver first (best for avoiding CAPTCHA)
        if HAS_UNDETECTED:
            try:
                logger.info("Setting up undetected Chrome browser...")
                
                options = uc.ChromeOptions()
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                # Don't use headless - it's more detectable
                # options.add_argument('--headless')
                
                self.driver = uc.Chrome(options=options, use_subprocess=True)
                self.use_selenium = True
                logger.info("✓ Using undetected-chromedriver (best CAPTCHA avoidance)")
                
                # Visit biorxiv to establish session
                logger.info("  Opening bioRxiv in browser...")
                self.driver.get("https://www.biorxiv.org")
                time.sleep(5)  # Wait for any challenges
                
                # Check if we passed the challenge
                if "biorxiv" in self.driver.title.lower() or "preprint" in self.driver.page_source.lower():
                    logger.info("  ✓ Browser session established successfully!")
                else:
                    logger.warning("  ⚠️  May need to solve CAPTCHA manually...")
                    logger.info("  Waiting 30 seconds for manual CAPTCHA solve if needed...")
                    time.sleep(30)
                
            except Exception as e:
                logger.warning(f"⚠️  undetected-chromedriver failed: {e}")
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
                self.use_selenium = False
        
        # Fallback to regular Selenium
        if not self.use_selenium and HAS_SELENIUM:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless')  # Run in background
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                
                # Auto-download and setup ChromeDriver
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                self.use_selenium = True
                logger.info("✓ Using Selenium for Cloudflare bypass")
                
                # Visit biorxiv once to establish session/cookies
                logger.info("  Initializing browser session with bioRxiv...")
                self.driver.get("https://www.biorxiv.org")
                time.sleep(3)  # Wait for Cloudflare challenge
                logger.info("  Browser session established")
                
            except Exception as e:
                logger.warning(f"⚠️  Selenium setup failed: {e}")
                logger.warning("   Falling back to cloudscraper/requests")
                self.use_selenium = False
                if self.driver:
                    try:
                        self.driver.quit()
                    except:
                        pass
                    self.driver = None
        
        # Fallback to cloudscraper or requests
        if not self.use_selenium:
            if HAS_CLOUDSCRAPER:
                self.session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'windows',
                        'desktop': True
                    }
                )
                logger.info("✓ Using cloudscraper for Cloudflare bypass")
            else:
                self.session = requests.Session()
                logger.warning("⚠️  No Cloudflare bypass available - may get blocked")
                logger.warning("   Install: pip install selenium webdriver-manager")
            
            self.session.headers.update(HEADERS)
        else:
            # Create a requests session for non-PDF requests (API calls)
            self.session = requests.Session()
            self.session.headers.update(HEADERS)
        
        # Initialize Wasabi client
        self.wasabi = None
        self.use_wasabi = use_wasabi
        
        if use_wasabi:
            try:
                # Try relative import first
                try:
                    from ..storage.wasabi_client import WasabiClient
                except ImportError:
                    from storage.wasabi_client import WasabiClient
                
                self.wasabi = WasabiClient()
                
                # Show connection info
                logger.info(f"Wasabi config: endpoint={self.wasabi.endpoint_url}, bucket={self.wasabi.bucket}")
                
                # Check if credentials are configured
                if self.wasabi.access_key and self.wasabi.secret_key:
                    # Test connection by trying to list objects
                    try:
                        self.wasabi.client.list_objects_v2(
                            Bucket=self.wasabi.bucket, 
                            MaxKeys=1,
                            Prefix='pdfs/'
                        )
                        logger.info(f"✓ Wasabi S3 storage enabled (bucket: {self.wasabi.bucket})")
                    except Exception as e:
                        error_code = str(e)
                        logger.debug(f"List objects result: {error_code}")
                        
                        # NoSuchBucket means bucket doesn't exist - try to create
                        if 'NoSuchBucket' in error_code or '404' in error_code:
                            try:
                                self.wasabi.client.create_bucket(Bucket=self.wasabi.bucket)
                                logger.info(f"✓ Created Wasabi bucket: {self.wasabi.bucket}")
                            except Exception as ce:
                                logger.warning(f"✗ Could not create Wasabi bucket: {ce}")
                                logger.info("  Falling back to local storage")
                                self.wasabi = None
                                self.use_wasabi = False
                        # Empty bucket returns success, so any error here is a real problem
                        elif 'AccessDenied' in error_code or '403' in error_code:
                            logger.warning(f"✗ Wasabi access denied - check credentials")
                            logger.info("  Falling back to local storage")
                            self.wasabi = None
                            self.use_wasabi = False
                        else:
                            # Try a simple put test as last resort
                            try:
                                test_key = ".connection_test"
                                self.wasabi.client.put_object(
                                    Bucket=self.wasabi.bucket,
                                    Key=test_key,
                                    Body=b"test"
                                )
                                self.wasabi.client.delete_object(
                                    Bucket=self.wasabi.bucket,
                                    Key=test_key
                                )
                                logger.info(f"✓ Wasabi S3 storage enabled (bucket: {self.wasabi.bucket})")
                            except Exception as pe:
                                logger.warning(f"✗ Wasabi access failed: {pe}")
                                logger.info("  Falling back to local storage")
                                self.wasabi = None
                                self.use_wasabi = False
                else:
                    logger.info("✗ Wasabi credentials not found in .env")
                    logger.info("  Set WASABI_ACCESS_KEY and WASABI_SECRET_KEY in .env")
                    logger.info("  Falling back to local storage")
                    self.wasabi = None
                    self.use_wasabi = False
                    
            except ImportError as e:
                logger.info(f"✗ Wasabi client import failed: {e}")
                logger.info("  Install boto3: pip install boto3")
                self.use_wasabi = False
            except Exception as e:
                logger.warning(f"✗ Wasabi initialization failed: {e}")
                self.use_wasabi = False
        
        logger.info(f"PDF Downloader initialized")
        if self.use_wasabi and self.wasabi:
            logger.info(f"Storage: Wasabi S3 → s3://{self.wasabi.bucket}/pdfs/")
        else:
            logger.info(f"Storage: Local → {self.output_dir}")
    
    def get_latest_version(self, doi: str, server: str = "biorxiv") -> int:
        """Get the latest version number for a paper"""
        try:
            # For new DOIs (10.64898), the API might need different handling
            # Try the standard API first
            api_url = BIORXIV_API_URL if server == "biorxiv" else MEDRXIV_API_URL
            url = api_url.format(doi=doi)
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("collection"):
                    # Get the highest version
                    versions = [int(p.get("version", 1)) for p in data["collection"]]
                    return max(versions) if versions else 1
            
            # If API fails for new DOIs, default to version 1
            # New papers (10.64898) are usually version 1
            if doi.startswith(NEW_DOI_PREFIX):
                logger.debug(f"New DOI format {doi}, defaulting to version 1")
                return 1
                
        except Exception as e:
            logger.debug(f"Could not get version for {doi}: {e}")
        
        return 1  # Default to version 1
    
    def doi_to_s3_key(self, doi: str, server: str = "biorxiv", category: str = None) -> str:
        """Convert DOI to S3 key path"""
        safe_doi = doi.replace("/", "_").replace(".", "_")
        
        if category:
            safe_category = category.replace(" ", "_").lower()
            return f"pdfs/{server}/{safe_category}/{safe_doi}.pdf"
        else:
            return f"pdfs/{server}/{safe_doi}.pdf"
    
    def doi_to_local_path(self, doi: str, category: str = None) -> Path:
        """Convert DOI to local file path"""
        safe_doi = doi.replace("/", "_").replace(".", "_")
        filename = f"{safe_doi}.pdf"
        
        if category:
            safe_category = category.replace(" ", "_").lower()
            return self.output_dir / safe_category / filename
        
        return self.output_dir / filename
    
    def check_exists_s3(self, s3_key: str) -> bool:
        """Check if file exists in S3"""
        if not self.wasabi:
            return False
        try:
            self.wasabi.client.head_object(Bucket=self.wasabi.bucket, Key=s3_key)
            return True
        except:
            return False
    
    def _visit_paper_page(self, doi: str, server: str, version: int) -> bool:
        """
        Visit the paper's HTML page first to establish a session and get cookies.
        This mimics real browser behavior.
        """
        try:
            # Visit the abstract page first
            paper_url = f"https://www.{server}.org/content/{doi}v{version}"
            response = self.session.get(paper_url, timeout=30)
            
            if response.status_code == 200:
                # Small delay to mimic human reading
                time.sleep(0.5)
                return True
            return False
        except:
            return False
    
    def _download_with_selenium(self, pdf_url: str, doi: str) -> Optional[bytes]:
        """
        Download PDF using Selenium browser.
        This bypasses Cloudflare by using a real browser.
        """
        if not self.driver:
            return None
        
        try:
            import base64
            
            # Navigate to the PDF URL
            self.driver.get(pdf_url)
            time.sleep(2)  # Wait for page/Cloudflare
            
            # Check if we got a PDF or a challenge page
            current_url = self.driver.current_url
            page_source = self.driver.page_source
            
            # If Cloudflare challenge, wait longer
            if 'challenge' in page_source.lower() or 'checking your browser' in page_source.lower():
                logger.debug(f"Cloudflare challenge detected, waiting...")
                time.sleep(5)
                page_source = self.driver.page_source
            
            # Try to get PDF via JavaScript fetch with browser's cookies
            pdf_script = """
            async function downloadPDF(url) {
                try {
                    const response = await fetch(url, {
                        credentials: 'include',
                        headers: {
                            'Accept': 'application/pdf'
                        }
                    });
                    if (!response.ok) {
                        return {error: response.status};
                    }
                    const blob = await response.blob();
                    const reader = new FileReader();
                    return new Promise((resolve) => {
                        reader.onloadend = () => {
                            resolve({data: reader.result.split(',')[1]});
                        };
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
                # Verify it's a PDF
                if pdf_bytes[:4] == b'%PDF':
                    return pdf_bytes
                else:
                    logger.debug(f"Downloaded content is not a PDF")
                    return None
            elif result and 'error' in result:
                logger.debug(f"Selenium fetch error: {result['error']}")
                return None
                
        except Exception as e:
            logger.debug(f"Selenium download failed for {doi}: {e}")
        
        return None
    
    def download_single(
        self,
        doi: str,
        server: str = "biorxiv",
        category: str = None,
        version: int = None
    ) -> PDFDownloadResult:
        """
        Download a single PDF and store in Wasabi S3 (or local).
        
        Args:
            doi: Paper DOI (e.g., "10.1101/2024.01.15.575685")
            server: "biorxiv" or "medrxiv"
            category: Optional category for organization
            version: Specific version (default: latest)
        """
        s3_key = self.doi_to_s3_key(doi, server, category)
        local_path = self.doi_to_local_path(doi, category)
        
        # Skip if exists (check S3 or local)
        if self.skip_existing:
            if self.use_wasabi and self.wasabi:
                if self.check_exists_s3(s3_key):
                    return PDFDownloadResult(
                        doi=doi,
                        success=True,
                        s3_key=s3_key,
                        error="skipped_existing"
                    )
            elif local_path.exists():
                return PDFDownloadResult(
                    doi=doi,
                    success=True,
                    local_path=local_path,
                    file_size=local_path.stat().st_size,
                    error="skipped_existing"
                )
        
        # Check if this is a new DOI (10.64898) or old DOI (10.1101)
        is_new_doi = doi.startswith(NEW_DOI_PREFIX)
        
        # Get version
        if version is None:
            version = self.get_latest_version(doi, server)
        
        # Build list of URLs to try (in order)
        urls_to_try = []
        
        # Primary URL with version
        base_url = BIORXIV_PDF_URL if server == "biorxiv" else MEDRXIV_PDF_URL
        urls_to_try.append(base_url.format(doi=doi, version=version))
        
        # Alternative URL without version
        alt_url = BIORXIV_PDF_ALT if server == "biorxiv" else MEDRXIV_PDF_ALT
        urls_to_try.append(alt_url.format(doi=doi))
        
        # For new DOIs, also try version 1 explicitly
        if is_new_doi and version != 1:
            urls_to_try.append(base_url.format(doi=doi, version=1))
        
        # Set referer to look like we're coming from the paper page
        paper_page_url = f"https://www.{server}.org/content/{doi}v{version}"
        
        pdf_url = urls_to_try[0]  # Track current URL for error reporting
        
        # TRY SELENIUM FIRST (best for Cloudflare bypass)
        if self.use_selenium and self.driver:
            for try_url in urls_to_try:
                pdf_url = try_url
                pdf_content = self._download_with_selenium(try_url, doi)
                
                if pdf_content:
                    # Store in Wasabi S3
                    if self.use_wasabi and self.wasabi:
                        try:
                            self.wasabi.client.put_object(
                                Bucket=self.wasabi.bucket,
                                Key=s3_key,
                                Body=pdf_content,
                                ContentType='application/pdf'
                            )
                            
                            return PDFDownloadResult(
                                doi=doi,
                                success=True,
                                s3_key=s3_key,
                                url=pdf_url,
                                file_size=len(pdf_content),
                                version=version
                            )
                        except Exception as e:
                            logger.warning(f"S3 upload failed, saving locally: {e}")
                    
                    # Store locally (fallback or if no Wasabi)
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(pdf_content)
                    
                    return PDFDownloadResult(
                        doi=doi,
                        success=True,
                        local_path=local_path,
                        url=pdf_url,
                        file_size=len(pdf_content),
                        version=version
                    )
                
                # Small delay between URL attempts
                time.sleep(1)
            
            # Selenium failed for all URLs
            return PDFDownloadResult(
                doi=doi,
                success=False,
                url=pdf_url,
                error="Selenium download failed - Cloudflare blocked"
            )
        
        # FALLBACK: Use requests/cloudscraper (if Selenium not available)
        # Download with retries
        last_error = None
        visited_page = False
        
        for attempt in range(MAX_RETRIES):
            # Try each URL pattern
            for url_idx, try_url in enumerate(urls_to_try):
                try:
                    pdf_url = try_url
                    
                    # On later attempts, visit the paper page first to get cookies
                    if attempt > 0 and not visited_page:
                        logger.debug(f"Visiting paper page for {doi} to get session cookies...")
                        self._visit_paper_page(doi, server, version)
                        visited_page = True
                        time.sleep(1.0)  # Extra delay after visiting page
                    
                    # Add referer header for this specific request
                    headers_with_referer = dict(self.session.headers)
                    headers_with_referer['Referer'] = paper_page_url
                    
                    response = self.session.get(pdf_url, timeout=60, stream=True, headers=headers_with_referer)
                    
                    if response.status_code == 200:
                        # Check if it's actually a PDF
                        content_type = response.headers.get('Content-Type', '')
                        if 'pdf' not in content_type.lower() and 'octet' not in content_type.lower():
                            last_error = f"Not a PDF: {content_type}"
                            continue  # Try next URL
                        
                        # Get PDF content
                        pdf_content = response.content
                        
                        # Store in Wasabi S3
                        if self.use_wasabi and self.wasabi:
                            try:
                                self.wasabi.client.put_object(
                                    Bucket=self.wasabi.bucket,
                                    Key=s3_key,
                                    Body=pdf_content,
                                    ContentType='application/pdf'
                                )
                                
                                return PDFDownloadResult(
                                    doi=doi,
                                    success=True,
                                    s3_key=s3_key,
                                    url=pdf_url,
                                    file_size=len(pdf_content),
                                    version=version
                                )
                            except Exception as e:
                                logger.warning(f"S3 upload failed, saving locally: {e}")
                                # Fall through to local storage
                        
                        # Store locally (fallback or if no Wasabi)
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(pdf_content)
                        
                        return PDFDownloadResult(
                            doi=doi,
                            success=True,
                            local_path=local_path,
                            url=pdf_url,
                            file_size=len(pdf_content),
                            version=version
                        )
                    
                    elif response.status_code == 403:
                        # Don't try other URLs on 403, it's a rate limit
                        raise Exception(f"HTTP 403")
                    
                    elif response.status_code == 404:
                        last_error = f"HTTP 404: URL {url_idx+1}/{len(urls_to_try)} not found"
                        continue  # Try next URL
                    
                    else:
                        last_error = f"HTTP {response.status_code}"
                        continue  # Try next URL
                        
                except Exception as e:
                    if "403" in str(e):
                        # Rate limited - wait and retry
                        wait_time = RETRY_DELAY * (2 ** attempt) + (attempt * 3)
                        logger.warning(f"HTTP 403 for {doi}, waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        last_error = f"HTTP 403: Access denied (rate limited?)"
                        break  # Exit URL loop, go to next attempt
                    else:
                        last_error = str(e)[:100]
                        continue  # Try next URL
        
        return PDFDownloadResult(
            doi=doi,
            success=False,
            url=pdf_url,
            error=last_error
        )
    
    def download_batch(
        self,
        papers: List[dict],
        workers: int = 2,  # Be nice to servers
        progress: bool = True
    ) -> Tuple[List[PDFDownloadResult], PDFDownloadStats]:
        """
        Download multiple PDFs.
        
        Args:
            papers: List of paper dicts with 'doi', 'server', 'category'
            workers: Number of parallel workers (keep low!)
            progress: Show progress bar
        """
        stats = PDFDownloadStats(total=len(papers))
        stats.start_time = datetime.now()
        
        results = []
        
        # Initialize PostgreSQL tracker for status updates
        tracker = None
        try:
            from db.progress import ProgressTracker
            tracker = ProgressTracker()
        except ImportError:
            logger.debug("PostgreSQL tracker not available")
        
        # Selenium must use single worker (one browser instance)
        if self.use_selenium:
            workers = 1
            logger.info("ℹ️  Using Selenium browser - single worker mode")
        elif workers > 3:
            logger.warning(f"⚠️  Reducing workers from {workers} to 3 to avoid rate limiting/403 errors")
            workers = 3
        
        storage_type = "Wasabi S3" if (self.use_wasabi and self.wasabi) else "Local"
        logger.info(f"Downloading {len(papers)} PDFs with {workers} workers")
        logger.info(f"Storage: {storage_type}")
        if self.use_selenium:
            logger.info("Note: Using Selenium for Cloudflare bypass (slower but reliable)")
        else:
            logger.info("Note: Rate-limited with delays to avoid 403 blocks")
        print()  # Blank line before progress
        
        # Use ThreadPoolExecutor for parallel downloads
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
                
                # Random delay between submissions to look more human-like
                jitter = random.uniform(0.5, 1.5)
                time.sleep(RATE_LIMIT_DELAY * jitter)
            
            # Collect results with detailed logging
            total = len(futures)
            for i, future in enumerate(as_completed(futures), 1):
                paper = futures[future]
                result = future.result()
                results.append(result)
                
                doi = paper.get('doi', 'unknown')
                short_doi = doi.split('/')[-1] if '/' in doi else doi
                
                if result.success:
                    if result.error == "skipped_existing":
                        stats.skipped += 1
                        status = "⏭️  SKIPPED (exists)"
                    else:
                        stats.success += 1
                        stats.total_bytes += result.file_size
                        size_kb = result.file_size / 1024 if result.file_size else 0
                        status = f"✅ SUCCESS ({size_kb:.1f} KB)"
                        
                        # UPDATE POSTGRESQL: Mark as downloaded
                        if tracker:
                            try:
                                tracker.mark_downloaded(
                                    doi=doi,
                                    pdf_path=result.s3_path or str(result.local_path),
                                    pdf_size=result.file_size
                                )
                            except Exception as e:
                                logger.debug(f"Failed to update DB for {doi}: {e}")
                else:
                    stats.failed += 1
                    status = f"❌ FAILED: {result.error[:50]}"
                    
                    # Record error in PostgreSQL
                    if tracker:
                        try:
                            tracker.mark_error(doi, result.error)
                        except:
                            pass
                
                # Print progress for each paper
                print(f"  [{i:3d}/{total}] {short_doi[:30]:<30} {status}")
        
        stats.end_time = datetime.now()
        
        # Close tracker
        if tracker:
            tracker.close()
        
        print()  # Blank line after progress
        logger.info(f"Download complete: {stats.success} success, {stats.skipped} skipped, {stats.failed} failed")
        logger.info(f"Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
        logger.info(f"Duration: {stats.duration_seconds:.2f} seconds")
        
        return results, stats
    
    def download_category(
        self,
        categories: List[str],
        index_path: Path = DEFAULT_INDEX_PATH,
        workers: int = 2,
        max_papers: int = None,
        use_postgres: bool = True
    ) -> Tuple[List[PDFDownloadResult], PDFDownloadStats]:
        """
        Download PDFs for papers in specified categories.
        
        Args:
            categories: List of category names
            index_path: Path to category index (legacy)
            workers: Number of parallel workers
            max_papers: Maximum papers to download
            use_postgres: Use PostgreSQL database instead of JSON index
        """
        papers = []
        
        if use_postgres:
            # Use PostgreSQL database
            try:
                from db.models import Paper, get_session
                from sqlalchemy import func
                
                session = get_session()
                
                for cat in categories:
                    # Find papers matching category (case-insensitive)
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
                            "date": str(p.date) if p.date else "",
                            "jatsxml": p.jatsxml or ""
                        })
                    
                    logger.info(f"Found {len(cat_papers)} papers in '{cat}'")
                
                session.close()
                
            except ImportError:
                logger.warning("PostgreSQL not available, falling back to JSON index")
                use_postgres = False
        
        if not use_postgres:
            # Legacy JSON index
            if not index_path.exists():
                raise FileNotFoundError(f"Index not found: {index_path}. Run --build-index first or use PostgreSQL.")
            
            index = CategoryIndex.load(index_path)
            logger.info(f"Loaded index with {index.total_papers} papers")
            
            for cat in categories:
                cat_papers = index.get_papers(cat, fuzzy=True)
                for p in cat_papers:
                    p["category"] = cat
                papers.extend(cat_papers)
                logger.info(f"Found {len(cat_papers)} papers in '{cat}'")
        
        if not papers:
            logger.warning(f"No papers found for categories: {categories}")
            return [], PDFDownloadStats()
        
        # Limit
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            logger.info(f"Limited to {max_papers} papers")
        
        # Download
        return self.download_batch(papers, workers=workers)
    
    def close(self):
        """Clean up resources (close Selenium browser)"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Browser closed")
            except:
                pass
            self.driver = None
    
    def __del__(self):
        """Destructor - ensure browser is closed"""
        self.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def download_pdfs(
    categories: List[str],
    max_papers: int = None,
    workers: int = 2,
    output_dir: Path = None,
    use_wasabi: bool = True
) -> PDFDownloadStats:
    """
    Download PDFs for specified categories.
    
    Args:
        categories: List of category names
        max_papers: Maximum papers to download
        workers: Number of parallel workers
        output_dir: Output directory (for local storage)
        use_wasabi: Whether to use Wasabi S3
    """
    downloader = BioRxivPDFDownloader(
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        use_wasabi=use_wasabi
    )
    
    _, stats = downloader.download_category(
        categories=categories,
        workers=workers,
        max_papers=max_papers
    )
    
    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PDFs from bioRxiv/medRxiv")
    parser.add_argument("--category", "-c", type=str, nargs="+", required=True,
                        help="Categories to download")
    parser.add_argument("--max", "-m", type=int, default=None,
                        help="Maximum papers to download")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Parallel download workers (keep low!)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (for local storage)")
    parser.add_argument("--local", action="store_true",
                        help="Use local storage instead of Wasabi")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    
    stats = download_pdfs(
        categories=args.category,
        max_papers=args.max,
        workers=args.workers,
        output_dir=output_dir,
        use_wasabi=not args.local
    )
    
    print(f"\nDownload Statistics:")
    print(f"  Success: {stats.success}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total MB: {stats.total_bytes / 1024 / 1024:.2f}")
