"""
Download ALL papers from bioRxiv S3 bucket
==========================================

Downloads papers from ALL available months in the S3 bucket.

Usage:
    python download_all_s3.py --max-per-month 1000 --workers 4
    python download_all_s3.py --start-month January_2024 --max-per-month 500
"""

import os
import sys
import io
import zipfile
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
except ImportError:
    pass

import boto3
from botocore.exceptions import ClientError

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BIORXIV_S3_BUCKET = "biorxiv-src-monthly"
MEDRXIV_S3_BUCKET = "medrxiv-src-monthly"
S3_PREFIX = "Current_Content"


@dataclass
class DownloadResult:
    key: str
    doi: str
    success: bool
    wasabi_key: Optional[str] = None
    file_size: int = 0
    error: Optional[str] = None


@dataclass
class MonthStats:
    month: str
    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    bytes: int = 0


class BulkS3Downloader:
    """Download ALL papers from bioRxiv S3"""
    
    def __init__(self, max_workers: int = 30):
        # Configure connection pool size to match workers
        from botocore.config import Config
        
        config = Config(
            max_pool_connections=max_workers + 10,  # Extra buffer
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        # Get AWS credentials from environment (.env file)
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = os.environ.get('AWS_REGION', 'us-east-1')
        
        if aws_access_key and aws_secret_key:
            # Use credentials from .env
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                config=config
            )
            logger.info(f"✓ AWS configured from .env")
        else:
            # Fall back to default credential chain (~/.aws/credentials)
            self.s3 = boto3.client('s3', region_name='us-east-1', config=config)
        
        try:
            sts = boto3.client('sts',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            ) if aws_access_key else boto3.client('sts')
            account = sts.get_caller_identity()['Account']
            logger.info(f"✓ AWS configured (Account: ...{account[-4:]})")
        except Exception as e:
            logger.error(f"AWS not configured: {e}")
            sys.exit(1)
        
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi(max_workers)
    
    def _init_wasabi(self, max_workers: int = 30):
        try:
            from storage.wasabi_client import WasabiClient
            from botocore.config import Config
            
            wc = WasabiClient()
            if wc.access_key and wc.secret_key:
                # Create wasabi client with larger connection pool
                config = Config(
                    max_pool_connections=max_workers + 10,
                    retries={'max_attempts': 3, 'mode': 'adaptive'}
                )
                
                # Get endpoint from WasabiClient or use default
                endpoint = getattr(wc, 'endpoint', None) or getattr(wc, 'endpoint_url', None) or f"https://s3.{wc.region}.wasabisys.com"
                region = getattr(wc, 'region', 'ap-southeast-1')
                
                self.wasabi = boto3.client(
                    's3',
                    endpoint_url=endpoint,
                    aws_access_key_id=wc.access_key,
                    aws_secret_access_key=wc.secret_key,
                    region_name=region,
                    config=config
                )
                self.wasabi_bucket = wc.bucket
                logger.info(f"✓ Wasabi connected ({wc.bucket})")
        except Exception as e:
            logger.warning(f"Wasabi not configured: {e}")
            # Try using the existing client from WasabiClient
            try:
                from storage.wasabi_client import WasabiClient
                wc = WasabiClient()
                if hasattr(wc, 'client') and wc.client:
                    self.wasabi = wc.client
                    self.wasabi_bucket = wc.bucket
                    logger.info(f"✓ Wasabi connected via existing client ({wc.bucket})")
            except:
                pass
    
    def list_months(self, include_back_content: bool = True) -> List[tuple]:
        """List all available months from both bioRxiv and medRxiv"""
        all_folders = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        # Process both buckets
        buckets = [
            (BIORXIV_S3_BUCKET, 'biorxiv'),
            (MEDRXIV_S3_BUCKET, 'medrxiv')
        ]
        
        for bucket, server in buckets:
            # Current_Content (2018/2019 onwards)
            try:
                for page in paginator.paginate(
                    Bucket=bucket,
                    Prefix=f"{S3_PREFIX}/",
                    Delimiter='/',
                    RequestPayer='requester'
                ):
                    for prefix in page.get('CommonPrefixes', []):
                        folder = prefix['Prefix'].replace(f"{S3_PREFIX}/", "").rstrip('/')
                        all_folders.append((bucket, server, 'Current_Content', folder))
            except Exception as e:
                logger.warning(f"Error listing {bucket} Current_Content: {e}")
            
            # Back_Content (pre-2018/2019 papers)
            if include_back_content:
                try:
                    for page in paginator.paginate(
                        Bucket=bucket,
                        Prefix="Back_Content/",
                        Delimiter='/',
                        RequestPayer='requester'
                    ):
                        for prefix in page.get('CommonPrefixes', []):
                            folder = prefix['Prefix'].replace("Back_Content/", "").rstrip('/')
                            all_folders.append((bucket, server, 'Back_Content', folder))
                except Exception as e:
                    logger.warning(f"Error listing {bucket} Back_Content: {e}")
        
        return all_folders
    
    def _month_sort_key(self, month: str) -> tuple:
        """Sort months chronologically"""
        month_order = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        parts = month.split('_')
        if len(parts) == 2:
            m, y = parts[0], parts[1]
            return (int(y), month_order.get(m, 0))
        return (0, 0)
    
    def list_files_in_month(self, bucket: str, content_type: str, month: str, max_files: int = None) -> List[dict]:
        """List MECA files in a month
        
        Args:
            bucket: S3 bucket name (biorxiv-src-monthly or medrxiv-src-monthly)
            content_type: 'Current_Content' or 'Back_Content'
            month: folder name (e.g., 'January_2024' or 'Batch_001')
        """
        files = []
        prefix = f"{content_type}/{month}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        
        print(f" listing...", end="", flush=True)
        
        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
            RequestPayer='requester'
        ):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.meca'):
                    files.append({
                        'key': key,
                        'size': obj['Size'],
                        'filename': key.split('/')[-1],
                        'bucket': bucket
                    })
                    if max_files and len(files) >= max_files:
                        print(f" {len(files)} files", end="", flush=True)
                        return files
        
        print(f" {len(files)} files", end="", flush=True)
        return files
    
    def extract_doi_and_pdf(self, meca_bytes: bytes) -> Tuple[Optional[str], Optional[bytes]]:
        """Extract DOI and PDF from MECA"""
        try:
            with zipfile.ZipFile(io.BytesIO(meca_bytes), 'r') as zf:
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
                
                doi = None
                for xml_file in ['manifest.xml', 'transfer.xml']:
                    if xml_file in zf.namelist():
                        try:
                            xml_content = zf.read(xml_file).decode('utf-8')
                            match = re.search(r'10\.1101/[\d\.]+', xml_content)
                            if match:
                                doi = match.group(0)
                                break
                        except:
                            pass
                
                return doi, pdf_bytes
        except:
            return None, None
    
    def download_single(self, file_info: dict, server: str, month: str, max_retries: int = 3) -> DownloadResult:
        """Download single MECA file with retry logic"""
        key = file_info['key']
        bucket = file_info.get('bucket', BIORXIV_S3_BUCKET)
        
        # Generate wasabi key using UUID FIRST (before downloading)
        uuid = file_info['filename'].replace('.meca', '')
        wasabi_key_uuid = f"pdfs/{server}/{month}/uuid_{uuid}.pdf"
        
        # STEP 1: Check if already exists in Wasabi BEFORE downloading MECA
        # This saves bandwidth by not downloading files we already have
        if self.wasabi:
            try:
                self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=wasabi_key_uuid)
                # File EXISTS in Wasabi → SKIP (don't download MECA at all!)
                return DownloadResult(key=key, doi="", success=True,
                                      wasabi_key=wasabi_key_uuid, error="skipped")
            except:
                pass  # File doesn't exist, continue to download
        
        for attempt in range(max_retries):
            try:
                # STEP 2: Download MECA from S3 (bioRxiv or medRxiv)
                response = self.s3.get_object(
                    Bucket=bucket,
                    Key=key,
                    RequestPayer='requester'
                )
                meca_bytes = response['Body'].read()
                
                # STEP 3: Extract PDF from MECA
                doi, pdf_bytes = self.extract_doi_and_pdf(meca_bytes)
                
                if not pdf_bytes:
                    return DownloadResult(key=key, doi="", success=False, error="No PDF in MECA")
                
                # Generate final wasabi key (prefer DOI-based if available)
                if doi:
                    safe_doi = doi.replace("/", "_").replace(".", "_")
                    wasabi_key = f"pdfs/{server}/{month}/{safe_doi}.pdf"
                    
                    # Also check if DOI-based key exists
                    if self.wasabi:
                        try:
                            self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=wasabi_key)
                            return DownloadResult(key=key, doi=doi, success=True,
                                                  wasabi_key=wasabi_key, error="skipped")
                        except:
                            pass
                else:
                    wasabi_key = wasabi_key_uuid
                
                # STEP 4: Upload to Wasabi (with retry)
                if self.wasabi:
                    for upload_attempt in range(max_retries):
                        try:
                            self.wasabi.put_object(
                                Bucket=self.wasabi_bucket,
                                Key=wasabi_key,
                                Body=pdf_bytes,
                                ContentType='application/pdf'
                            )
                            break  # Upload successful
                        except Exception as upload_err:
                            if upload_attempt < max_retries - 1:
                                import time
                                time.sleep(1 * (upload_attempt + 1))  # Backoff: 1s, 2s, 3s
                                continue
                            else:
                                return DownloadResult(key=key, doi=doi or "", success=False,
                                                      error=f"Upload failed: {str(upload_err)[:20]}")
                
                return DownloadResult(key=key, doi=doi or "", success=True,
                                      wasabi_key=wasabi_key, file_size=len(pdf_bytes))
                
            except ClientError as e:
                code = e.response.get('Error', {}).get('Code', 'Unknown')
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (attempt + 1))  # Backoff: 1s, 2s, 3s
                    continue
                return DownloadResult(key=key, doi="", success=False, error=f"S3:{code}")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1 * (attempt + 1))
                    continue
                return DownloadResult(key=key, doi="", success=False, error=str(e)[:30])
        
        return DownloadResult(key=key, doi="", success=False, error="Max retries reached")
    
    def download_month(self, bucket: str, server: str, content_type: str, month: str, max_files: int, workers: int) -> MonthStats:
        """Download all papers from a month/batch"""
        stats = MonthStats(month=f"{server}/{content_type}/{month}")
        
        files = self.list_files_in_month(bucket, content_type, month, max_files)
        stats.total = len(files)
        
        if not files:
            print(" → empty")
            return stats
        
        # Use month as folder name in Wasabi
        folder_name = month.replace(" ", "_")
        
        print(f" → downloading {len(files)} files...")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.download_single, f, server, folder_name): f 
                for f in files
            }
            
            # Progress tracking
            completed = 0
            total = len(futures)
            
            if HAS_TQDM:
                # Use tqdm progress bar with live stats
                pbar = tqdm(total=total, desc=f"    {folder_name[:20]}", 
                           unit="file", ncols=100)
                
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result.success:
                        if result.error == "skipped":
                            stats.skipped += 1
                        else:
                            stats.success += 1
                            stats.bytes += result.file_size
                    else:
                        stats.failed += 1
                    
                    # Update progress bar with stats
                    pbar.set_postfix({
                        '✓': stats.success, 
                        '⏭': stats.skipped, 
                        '✗': stats.failed,
                        'MB': f"{stats.bytes/1024/1024:.1f}"
                    }, refresh=True)
                    pbar.update(1)
                
                pbar.close()
            else:
                # Fallback without tqdm
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    
                    if result.success:
                        if result.error == "skipped":
                            stats.skipped += 1
                        else:
                            stats.success += 1
                            stats.bytes += result.file_size
                    else:
                        stats.failed += 1
                    
                    # Print progress every 5% or every 50 files
                    if completed % max(1, min(50, total // 20)) == 0 or completed == total:
                        pct = completed * 100 // total
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        print(f"\r    [{bar}] {pct:3d}% ({completed}/{total}) | ✓{stats.success} ⏭{stats.skipped} ✗{stats.failed}", end="", flush=True)
                
                print()  # New line after progress
        
        return stats
    
    def download_all(self, max_per_month: int = 10000, workers: int = 4, 
                     start_from: str = None, include_back_content: bool = True):
        """Download from ALL months including Back_Content from both bioRxiv and medRxiv"""
        
        print("="*70)
        print("DOWNLOAD ALL PAPERS FROM BIORXIV + MEDRXIV S3")
        print("="*70)
        print(f"Max per folder: {max_per_month}")
        print(f"Workers: {workers}")
        print(f"Include Back_Content (pre-2018/2019): {include_back_content}")
        print(f"Destination: Wasabi s3://{self.wasabi_bucket}/")
        print("="*70)
        print()
        
        all_folders = self.list_months(include_back_content)
        
        # Count by server
        biorxiv_count = len([f for f in all_folders if f[1] == 'biorxiv'])
        medrxiv_count = len([f for f in all_folders if f[1] == 'medrxiv'])
        
        logger.info(f"Found {len(all_folders)} folders to process:")
        logger.info(f"  - bioRxiv: {biorxiv_count} folders")
        logger.info(f"  - medRxiv: {medrxiv_count} folders")
        print()
        
        total_stats = {
            'folders': 0,
            'papers': 0,
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'bytes': 0
        }
        
        start_time = datetime.now()
        
        for i, (bucket, server, content_type, folder) in enumerate(all_folders, 1):
            display_name = f"{server}/{content_type}/{folder}"
            print(f"[{i:3d}/{len(all_folders)}] {display_name:<50}", end="", flush=True)
            
            stats = self.download_month(bucket, server, content_type, folder, max_per_month, workers)
            
            total_stats['folders'] += 1
            total_stats['papers'] += stats.total
            total_stats['success'] += stats.success
            total_stats['skipped'] += stats.skipped
            total_stats['failed'] += stats.failed
            total_stats['bytes'] += stats.bytes
            
            # Clear line and print final stats
            print(f"\r[{i:3d}/{len(all_folders)}] {display_name:<50} ✓ {stats.success:4d} new | {stats.skipped:4d} skip | {stats.failed:3d} fail | {stats.bytes/1024/1024:6.1f} MB")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print()
        print("="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"  Folders processed: {total_stats['folders']}")
        print(f"  Total papers:      {total_stats['papers']}")
        print(f"  New downloads:     {total_stats['success']}")
        print(f"  Already existed:   {total_stats['skipped']}")
        print(f"  Failed:            {total_stats['failed']}")
        print(f"  Total size:        {total_stats['bytes']/1024/1024/1024:.2f} GB")
        print(f"  Duration:          {duration/60:.1f} minutes")
        print(f"  Cost estimate:     ${total_stats['bytes']/1024/1024/1024 * 0.09:.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download ALL papers from bioRxiv S3")
    parser.add_argument("--max-per-month", type=int, default=10000,
                        help="Max papers per month (default: 10000 = all)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Parallel workers")
    parser.add_argument("--no-back-content", action="store_true",
                        help="Skip Back_Content (pre-2018 papers)")
    
    args = parser.parse_args()
    
    downloader = BulkS3Downloader(max_workers=args.workers)
    downloader.download_all(
        max_per_month=args.max_per_month,
        workers=args.workers,
        include_back_content=not args.no_back_content
    )


if __name__ == "__main__":
    main()
