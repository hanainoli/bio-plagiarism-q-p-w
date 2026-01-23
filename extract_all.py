"""
Extract content from ALL PDFs in Wasabi
=======================================

Works with the new month-based structure:
  pdfs/biorxiv/April_2019/xxx.pdf
  pdfs/medrxiv/June_2019/yyy.pdf

Features:
- Checks if already extracted (skip existing)
- Parallel processing
- Progress bar
- Saves extracted text/images to Wasabi

Usage:
    python extract_all.py --workers 20
    python extract_all.py --workers 20 --max 1000
    python extract_all.py --server biorxiv --month April_2019
"""

import os
import sys
import io
import json
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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
from botocore.config import Config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    pdf_key: str
    success: bool
    doi: Optional[str] = None
    word_count: int = 0
    figure_count: int = 0
    table_count: int = 0
    error: Optional[str] = None


class BulkExtractor:
    """Extract content from all PDFs in Wasabi"""
    
    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
        
        # Wasabi client with large connection pool
        config = Config(
            max_pool_connections=max_workers + 10,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        self.wasabi = None
        self.wasabi_bucket = None
        self._init_wasabi(config)
        
        if not self.wasabi:
            logger.error("Wasabi not configured!")
            sys.exit(1)
    
    def _init_wasabi(self, config):
        """Initialize Wasabi client"""
        try:
            from storage.wasabi_client import WasabiClient
            wc = WasabiClient()
            
            if wc.access_key and wc.secret_key:
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
            logger.error(f"Wasabi init failed: {e}")
    
    def list_pdf_folders(self) -> List[Tuple[str, str]]:
        """List all server/month folders containing PDFs"""
        folders = []
        paginator = self.wasabi.get_paginator('list_objects_v2')
        
        for server in ['biorxiv', 'medrxiv']:
            prefix = f"pdfs/{server}/"
            
            try:
                for page in paginator.paginate(
                    Bucket=self.wasabi_bucket,
                    Prefix=prefix,
                    Delimiter='/'
                ):
                    for p in page.get('CommonPrefixes', []):
                        folder = p['Prefix'].replace(prefix, '').rstrip('/')
                        folders.append((server, folder))
            except Exception as e:
                logger.warning(f"Error listing {server}: {e}")
        
        return folders
    
    def list_pdfs_in_folder(self, server: str, month: str, max_files: int = None) -> List[dict]:
        """List PDFs in a specific folder"""
        files = []
        prefix = f"pdfs/{server}/{month}/"
        paginator = self.wasabi.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.wasabi_bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pdf'):
                    files.append({
                        'key': key,
                        'server': server,
                        'month': month,
                        'filename': key.split('/')[-1]
                    })
                    if max_files and len(files) >= max_files:
                        return files
        
        return files
    
    def check_already_extracted(self, pdf_key: str) -> bool:
        """Check if PDF has already been extracted (JSON exists)"""
        # PDF: pdfs/biorxiv/April_2019/xxx.pdf
        # JSON: extracted/biorxiv/April_2019/xxx.json
        json_key = pdf_key.replace('pdfs/', 'extracted/').replace('.pdf', '.json')
        
        try:
            self.wasabi.head_object(Bucket=self.wasabi_bucket, Key=json_key)
            return True  # JSON exists, already extracted
        except:
            return False  # Not extracted yet
    
    def extract_single(self, pdf_info: dict) -> ExtractResult:
        """Extract content from a single PDF"""
        pdf_key = pdf_info['key']
        server = pdf_info['server']
        month = pdf_info['month']
        filename = pdf_info['filename']
        
        # Check if already extracted FIRST
        if self.check_already_extracted(pdf_key):
            return ExtractResult(pdf_key=pdf_key, success=True, error="skipped")
        
        temp_pdf_path = None
        
        try:
            # Download PDF to temp file
            response = self.wasabi.get_object(Bucket=self.wasabi_bucket, Key=pdf_key)
            pdf_bytes = response['Body'].read()
            
            temp_pdf_path = tempfile.mktemp(suffix='.pdf')
            with open(temp_pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # Extract content
            from extraction.pdf_extractor import PDFExtractor
            extractor = PDFExtractor()
            result = extractor.extract(temp_pdf_path)
            
            if not result:
                return ExtractResult(pdf_key=pdf_key, success=False, error="Extract failed")
            
            # Get DOI from filename
            doi = self._filename_to_doi(filename)
            
            # Prepare extraction data
            extraction_data = {
                'doi': doi,
                'server': server,
                'month': month,
                'filename': filename,
                'text': result.text or '',
                'word_count': len((result.text or '').split()),
                'figure_count': len(result.figures) if result.figures else 0,
                'table_count': len(result.tables) if result.tables else 0,
                'extracted_at': datetime.now().isoformat()
            }
            
            # Upload figures to Wasabi
            figure_keys = []
            if result.figures:
                for i, fig in enumerate(result.figures):
                    if hasattr(fig, 'image_bytes') and fig.image_bytes:
                        fig_key = f"figures/{server}/{month}/{filename.replace('.pdf', '')}_fig{i+1}.png"
                        try:
                            self.wasabi.put_object(
                                Bucket=self.wasabi_bucket,
                                Key=fig_key,
                                Body=fig.image_bytes,
                                ContentType='image/png'
                            )
                            figure_keys.append(fig_key)
                        except:
                            pass
            
            extraction_data['figure_keys'] = figure_keys
            
            # Upload extraction JSON
            json_key = pdf_key.replace('pdfs/', 'extracted/').replace('.pdf', '.json')
            self.wasabi.put_object(
                Bucket=self.wasabi_bucket,
                Key=json_key,
                Body=json.dumps(extraction_data).encode('utf-8'),
                ContentType='application/json'
            )
            
            return ExtractResult(
                pdf_key=pdf_key,
                success=True,
                doi=doi,
                word_count=extraction_data['word_count'],
                figure_count=extraction_data['figure_count'],
                table_count=extraction_data['table_count']
            )
            
        except Exception as e:
            return ExtractResult(pdf_key=pdf_key, success=False, error=str(e)[:50])
        
        finally:
            # Cleanup temp file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except:
                    pass
    
    def _filename_to_doi(self, filename: str) -> Optional[str]:
        """Convert filename to DOI"""
        name = filename.replace('.pdf', '')
        
        # uuid_xxx format
        if name.startswith('uuid_'):
            return None
        
        # 10_1101_xxx format → 10.1101/xxx
        if name.startswith('10_'):
            parts = name.split('_')
            if len(parts) >= 3:
                return f"{parts[0]}.{parts[1]}/{'.'.join(parts[2:])}"
        
        return None
    
    def extract_folder(self, server: str, month: str, max_files: int = None) -> dict:
        """Extract all PDFs in a folder"""
        stats = {'total': 0, 'success': 0, 'skipped': 0, 'failed': 0}
        
        files = self.list_pdfs_in_folder(server, month, max_files)
        stats['total'] = len(files)
        
        if not files:
            return stats
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_single, f): f for f in files}
            
            if HAS_TQDM:
                pbar = tqdm(total=len(futures), desc=f"    {month[:15]}", unit="pdf", ncols=100)
            
            for future in as_completed(futures):
                result = future.result()
                
                if result.success:
                    if result.error == "skipped":
                        stats['skipped'] += 1
                    else:
                        stats['success'] += 1
                else:
                    stats['failed'] += 1
                
                if HAS_TQDM:
                    pbar.set_postfix({
                        '✓': stats['success'],
                        '⏭': stats['skipped'],
                        '✗': stats['failed']
                    })
                    pbar.update(1)
            
            if HAS_TQDM:
                pbar.close()
        
        return stats
    
    def extract_all(self, max_per_folder: int = None, server_filter: str = None, month_filter: str = None):
        """Extract from all folders"""
        
        print("="*70)
        print("EXTRACT CONTENT FROM ALL PDFs")
        print("="*70)
        print(f"Workers: {self.max_workers}")
        print(f"Source: Wasabi s3://{self.wasabi_bucket}/pdfs/")
        print(f"Output: Wasabi s3://{self.wasabi_bucket}/extracted/")
        print("="*70)
        print()
        
        # Get all folders
        folders = self.list_pdf_folders()
        
        # Apply filters
        if server_filter:
            folders = [(s, m) for s, m in folders if s == server_filter]
        if month_filter:
            folders = [(s, m) for s, m in folders if m == month_filter]
        
        logger.info(f"Found {len(folders)} folders to process")
        print()
        
        total_stats = {'folders': 0, 'total': 0, 'success': 0, 'skipped': 0, 'failed': 0}
        start_time = datetime.now()
        
        for i, (server, month) in enumerate(folders, 1):
            display = f"{server}/{month}"
            print(f"[{i:3d}/{len(folders)}] {display:<45}", end="", flush=True)
            
            stats = self.extract_folder(server, month, max_per_folder)
            
            total_stats['folders'] += 1
            total_stats['total'] += stats['total']
            total_stats['success'] += stats['success']
            total_stats['skipped'] += stats['skipped']
            total_stats['failed'] += stats['failed']
            
            print(f"\r[{i:3d}/{len(folders)}] {display:<45} ✓ {stats['success']:4d} new | {stats['skipped']:4d} skip | {stats['failed']:3d} fail")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print()
        print("="*70)
        print("EXTRACTION COMPLETE")
        print("="*70)
        print(f"  Folders processed: {total_stats['folders']}")
        print(f"  Total PDFs:        {total_stats['total']}")
        print(f"  Newly extracted:   {total_stats['success']}")
        print(f"  Already existed:   {total_stats['skipped']}")
        print(f"  Failed:            {total_stats['failed']}")
        print(f"  Duration:          {duration/60:.1f} minutes")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract content from all PDFs in Wasabi")
    parser.add_argument("--workers", "-w", type=int, default=20, help="Parallel workers")
    parser.add_argument("--max-per-folder", type=int, help="Max PDFs per folder")
    parser.add_argument("--server", type=str, choices=['biorxiv', 'medrxiv'], help="Filter by server")
    parser.add_argument("--month", type=str, help="Filter by month (e.g., April_2019)")
    
    args = parser.parse_args()
    
    extractor = BulkExtractor(max_workers=args.workers)
    extractor.extract_all(
        max_per_folder=args.max_per_folder,
        server_filter=args.server,
        month_filter=args.month
    )


if __name__ == "__main__":
    main()
