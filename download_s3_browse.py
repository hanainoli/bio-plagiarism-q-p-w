"""
Download papers by browsing bioRxiv S3 bucket
=============================================

The S3 bucket uses UUIDs, not DOIs. This script:
1. Lists available MECA files from a specific month folder
2. Downloads and extracts PDFs
3. Uploads to Wasabi

Usage:
    python download_s3_browse.py --month "2024-01" --max 100 --workers 4
    python download_s3_browse.py --list-months
"""

import os
import sys
import io
import json
import zipfile
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
S3_PREFIX = "Current_Content"


@dataclass
class DownloadResult:
    key: str
    doi: str
    success: bool
    wasabi_key: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None


class S3BucketBrowser:
    """Browse and download from bioRxiv S3 bucket"""
    
    def __init__(self):
        # AWS S3
        self.s3 = boto3.client('s3', region_name='us-east-1')
        
        # Verify credentials
        try:
            sts = boto3.client('sts')
            account = sts.get_caller_identity()['Account']
            logger.info(f"✓ AWS configured (Account: ...{account[-4:]})")
        except Exception as e:
            logger.error(f"AWS not configured: {e}")
            sys.exit(1)
        
        # Wasabi
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi()
    
    def _init_wasabi(self):
        try:
            from storage.wasabi_client import WasabiClient
            wc = WasabiClient()
            if wc.access_key and wc.secret_key:
                self.wasabi = wc.client
                self.wasabi_bucket = wc.bucket
                logger.info(f"✓ Wasabi connected ({wc.bucket})")
        except Exception as e:
            logger.warning(f"Wasabi not configured: {e}")
    
    def list_months(self) -> List[str]:
        """List available month folders in the S3 bucket"""
        months = []
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=BIORXIV_S3_BUCKET,
                Prefix=f"{S3_PREFIX}/",
                Delimiter='/',
                RequestPayer='requester'
            ):
                for prefix in page.get('CommonPrefixes', []):
                    folder = prefix['Prefix'].replace(f"{S3_PREFIX}/", "").rstrip('/')
                    months.append(folder)
            
            return sorted(months)
            
        except Exception as e:
            logger.error(f"Error listing months: {e}")
            return []
    
    def list_files_in_month(self, month: str, max_files: int = None) -> List[dict]:
        """List MECA files in a specific month folder"""
        files = []
        
        prefix = f"{S3_PREFIX}/{month}/"
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(
                Bucket=BIORXIV_S3_BUCKET,
                Prefix=prefix,
                RequestPayer='requester'
            ):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.meca'):
                        files.append({
                            'key': key,
                            'size': obj['Size'],
                            'filename': key.split('/')[-1]
                        })
                        
                        if max_files and len(files) >= max_files:
                            return files
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def extract_doi_from_meca(self, meca_bytes: bytes) -> Tuple[Optional[str], Optional[bytes]]:
        """Extract DOI and PDF from MECA file"""
        try:
            with zipfile.ZipFile(io.BytesIO(meca_bytes), 'r') as zf:
                # Find PDF
                pdf_names = [n for n in zf.namelist() if n.lower().endswith('.pdf')]
                if not pdf_names:
                    return None, None
                
                main_pdf = None
                for name in pdf_names:
                    if 'content/' in name.lower() or 'manuscript' in name.lower():
                        main_pdf = name
                        break
                if not main_pdf:
                    main_pdf = pdf_names[0]
                
                pdf_bytes = zf.read(main_pdf)
                
                # Try to find DOI from manifest.xml or transfer.xml
                doi = None
                for xml_file in ['manifest.xml', 'transfer.xml']:
                    if xml_file in zf.namelist():
                        try:
                            xml_content = zf.read(xml_file).decode('utf-8')
                            # Look for DOI pattern
                            match = re.search(r'10\.1101/[\d\.]+', xml_content)
                            if match:
                                doi = match.group(0)
                                break
                        except:
                            pass
                
                return doi, pdf_bytes
                
        except Exception as e:
            logger.debug(f"MECA extraction error: {e}")
            return None, None
    
    def download_single(self, file_info: dict, category: str = "biorxiv") -> DownloadResult:
        """Download and process a single MECA file"""
        key = file_info['key']
        
        try:
            # Download MECA
            response = self.s3.get_object(
                Bucket=BIORXIV_S3_BUCKET,
                Key=key,
                RequestPayer='requester'
            )
            meca_bytes = response['Body'].read()
            
            # Extract DOI and PDF
            doi, pdf_bytes = self.extract_doi_from_meca(meca_bytes)
            
            if not pdf_bytes:
                return DownloadResult(key=key, doi="", success=False, error="No PDF in MECA")
            
            # Generate wasabi key
            if doi:
                safe_doi = doi.replace("/", "_").replace(".", "_")
            else:
                # Use UUID from filename
                uuid = file_info['filename'].replace('.meca', '')
                safe_doi = f"uuid_{uuid}"
            
            wasabi_key = f"pdfs/biorxiv/{category}/{safe_doi}.pdf"
            
            # Check if exists
            if self.wasabi:
                try:
                    self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=wasabi_key)
                    return DownloadResult(key=key, doi=doi or "", success=True,
                                          wasabi_key=wasabi_key, error="skipped_existing")
                except:
                    pass
            
            # Upload to Wasabi
            if self.wasabi:
                self.wasabi.put_object(
                    Bucket=self.wasabi_bucket,
                    Key=wasabi_key,
                    Body=pdf_bytes,
                    ContentType='application/pdf'
                )
            
            return DownloadResult(key=key, doi=doi or "", success=True,
                                  wasabi_key=wasabi_key, file_size=len(pdf_bytes))
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            return DownloadResult(key=key, doi="", success=False, error=f"S3: {error_code}")
        except Exception as e:
            return DownloadResult(key=key, doi="", success=False, error=str(e)[:50])
    
    def download_month(self, month: str, max_files: int = 100, workers: int = 4, category: str = "all"):
        """Download papers from a specific month"""
        
        print(f"\n{'='*60}")
        print(f"DOWNLOADING FROM: {month}")
        print(f"{'='*60}")
        
        # List files
        logger.info(f"Listing files in {month}...")
        files = self.list_files_in_month(month, max_files)
        
        if not files:
            logger.warning("No files found!")
            return
        
        logger.info(f"Found {len(files)} MECA files")
        logger.info(f"Downloading with {workers} workers...")
        print()
        
        success = 0
        failed = 0
        skipped = 0
        total_bytes = 0
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.download_single, f, category): f 
                for f in files
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                
                short_name = result.key.split('/')[-1][:20]
                doi_short = result.doi.split('/')[-1][:15] if result.doi else "no-doi"
                
                if result.success:
                    if result.error == "skipped_existing":
                        skipped += 1
                        status = "⏭️  SKIP"
                    else:
                        success += 1
                        total_bytes += result.file_size
                        status = f"✅ OK ({result.file_size//1024}KB)"
                else:
                    failed += 1
                    status = f"❌ {result.error[:25]}"
                
                print(f"  [{i:4d}/{len(files)}] {doi_short:<15} {status}")
        
        print()
        print(f"{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"  Success: {success}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed:  {failed}")
        print(f"  Size:    {total_bytes / 1024 / 1024:.2f} MB")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Browse and download from bioRxiv S3")
    parser.add_argument("--list-months", action="store_true", help="List available months")
    parser.add_argument("--month", "-m", type=str, help="Month folder to download (e.g., 2024-01)")
    parser.add_argument("--max", type=int, default=100, help="Max files to download")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--category", "-c", type=str, default="all", help="Category folder name")
    
    args = parser.parse_args()
    
    browser = S3BucketBrowser()
    
    if args.list_months:
        print("\nAvailable months in S3 bucket:")
        print("-" * 40)
        months = browser.list_months()
        for m in months:
            print(f"  {m}")
        print(f"\nTotal: {len(months)} months")
        print("\nUsage: python download_s3_browse.py --month 2024-01 --max 100")
        
    elif args.month:
        browser.download_month(args.month, args.max, args.workers, args.category)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
