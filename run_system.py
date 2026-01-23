#!/usr/bin/env python3
"""
=============================================================================
COMPLETE END-TO-END PLAGIARISM DETECTION SYSTEM
=============================================================================

This script provides the complete workflow from:
1. Building category index
2. Downloading papers from S3
3. Extracting content from MECA files
4. Building embeddings in Qdrant
5. User submission → Plagiarism report

Run with: python run_system.py --help
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    # Load from current directory or parent
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default locations
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# STEP 1: BUILD CATEGORY INDEX
# =============================================================================

def step1_build_index(update_only: bool = False, days: int = 7):
    """
    Build or update the category index.
    
    This fetches metadata from bioRxiv API and organizes papers by category.
    One-time operation takes 2-3 hours. Updates take 5-10 minutes.
    """
    print("\n" + "="*70)
    print("STEP 1: BUILD CATEGORY INDEX")
    print("="*70)
    
    from ingestion.category_index import CategoryIndexBuilder
    
    builder = CategoryIndexBuilder()
    
    if update_only:
        print(f"Updating index with papers from last {days} days...")
        new_count = builder.update_index(days=days)
        print(f"Added {new_count} new papers")
    else:
        print("Building full index (this takes 2-3 hours)...")
        print("You can Ctrl+C and resume later - progress is saved.")
        builder.build_full_index()
    
    # Print stats
    builder.print_stats()
    
    return builder.index


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _extract_doi_from_filename(filename: str) -> Optional[str]:
    """
    Extract DOI from filename.
    Converts: 10_1101_19000174 -> 10.1101/19000174
    """
    if not filename:
        return None
    
    # Remove extension
    name = filename.replace('.pdf', '').replace('.json', '')
    
    # Try to convert safe_doi to doi format
    if name.startswith('10_'):
        # Replace first _ with . and second _ with /
        parts = name.split('_', 2)
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}/{parts[2]}"
    
    return None


# =============================================================================
# STEP 2: DOWNLOAD PAPERS (PDF to Wasabi S3)
# =============================================================================

def step2_download_papers(
    categories: List[str],
    max_papers: Optional[int] = None,
    workers: int = 2,  # Keep low to be nice to servers
    force: bool = False  # Force re-download even if exists
):
    """
    Download PDFs directly from bioRxiv/medRxiv website.
    Stores in Wasabi S3 (or local if not configured).
    
    This is FREE for downloading - no AWS credentials needed!
    Rate-limited to be respectful to servers.
    """
    print("\n" + "="*70)
    print("STEP 2: DOWNLOAD PDFs")
    print("="*70)
    print(f"Categories: {categories}")
    print(f"Max papers: {max_papers or 'all'}")
    print(f"Workers: {workers} (kept low to be nice to servers)")
    if force:
        print(f"Force: Re-downloading even if exists")
    
    from ingestion.pdf_downloader import BioRxivPDFDownloader
    
    downloader = BioRxivPDFDownloader(
        output_dir=PROJECT_ROOT / "data" / "pdfs",
        skip_existing=not force,  # If force=True, don't skip
        use_wasabi=True  # Will use Wasabi if configured, else local
    )
    
    results, stats = downloader.download_category(
        categories=categories,
        workers=workers,
        max_papers=max_papers,
        use_postgres=True  # Use PostgreSQL database
    )
    
    print(f"\nDownload complete!")
    print(f"  Success: {stats.success}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total size: {stats.total_bytes / 1024 / 1024:.2f} MB")
    
    # Show storage location
    if downloader.use_wasabi and downloader.wasabi:
        print(f"\nStorage: Wasabi S3 (bucket: {downloader.wasabi.bucket})")
    else:
        print(f"\nStorage: Local ({PROJECT_ROOT / 'data' / 'pdfs'})")
    
    # Show some failed DOIs for debugging
    failed = [r for r in results if not r.success and r.error != "skipped_existing"]
    if failed and len(failed) <= 10:
        print(f"\nFailed downloads:")
        for r in failed[:10]:
            print(f"  - {r.doi}: {r.error}")
    
    return results, stats


# =============================================================================
# STREAMING EXTRACTION FROM WASABI (Parallel Processing)
# =============================================================================

def _process_single_pdf(args):
    """
    Process a single PDF: download, extract, upload results.
    Designed for parallel execution.
    SKIPS if extraction JSON already exists in Wasabi.
    
    Saves to: extractions/{server}/{month}/{filename}.json
    """
    from io import BytesIO
    import tempfile
    import os
    import json
    
    pdf_info, wasabi_config, extract_images, extract_tables = args
    key = pdf_info['key']
    category = pdf_info['category']  # This is the month (e.g., April_2019)
    server = pdf_info.get('server', 'biorxiv')  # Get server (biorxiv/medrxiv)
    filename = key.split('/')[-1]
    
    # Create new Wasabi client for this thread
    import boto3
    wasabi_client = boto3.client(
        's3',
        endpoint_url=wasabi_config['endpoint'],
        aws_access_key_id=wasabi_config['access_key'],
        aws_secret_access_key=wasabi_config['secret_key'],
        region_name=wasabi_config['region']
    )
    bucket = wasabi_config['bucket']
    
    # Get DOI/safe_doi from filename for checking
    if filename.startswith('10_1101_'):
        doi = filename.replace('.pdf', '').replace('_', '.', 1).replace('_', '/', 1).replace('_', '.')
    else:
        doi = filename.replace('.pdf', '')
    safe_doi = doi.replace('/', '_').replace('.', '_')
    
    # Build extraction path with server/month structure
    # extractions/biorxiv/April_2019/uuid_xxx.json
    extraction_key = f"extractions/{server}/{category}/{safe_doi}.json"
    
    # CHECK IF ALREADY EXTRACTED (skip if JSON exists)
    try:
        wasabi_client.head_object(Bucket=bucket, Key=extraction_key)
        # JSON exists - SKIP
        return {
            'success': True,
            'filename': filename,
            'word_count': 0,
            'fig_count': 0,
            'table_count': 0,
            'error': 'skipped',
            'skipped': True
        }
    except:
        pass  # JSON doesn't exist, continue with extraction
    
    temp_pdf_path = None
    result_info = {
        'success': False,
        'filename': filename,
        'word_count': 0,
        'fig_count': 0,
        'table_count': 0,
        'error': None,
        'skipped': False
    }
    
    try:
        # 1. Download PDF to temp file
        pdf_obj = wasabi_client.get_object(Bucket=bucket, Key=key)
        pdf_bytes = pdf_obj['Body'].read()
        
        temp_pdf_path = tempfile.mktemp(suffix='.pdf')
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        
        # 2. Extract content
        from extraction.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        result = extractor.extract(temp_pdf_path)
        
        if result:
            fig_count = 0
            table_count = 0
            figure_paths = []
            table_paths = []
            
            # 3. Upload figures to Wasabi
            if extract_images and result.figures:
                for i, fig in enumerate(result.figures):
                    try:
                        if hasattr(fig, 'image') and fig.image is not None:
                            img = fig.image
                            if img.mode in ('RGBA', 'P'):
                                img = img.convert('RGB')
                            
                            buffer = BytesIO()
                            img.save(buffer, format='PNG')
                            buffer.seek(0)
                            
                            # Save to figures/{server}/{month}/{doi}/fig_N.png
                            s3_key = f"figures/{server}/{category}/{safe_doi}/fig_{i+1}.png"
                            wasabi_client.put_object(
                                Bucket=bucket,
                                Key=s3_key,
                                Body=buffer.getvalue(),
                                ContentType='image/png'
                            )
                            figure_paths.append(s3_key)
                            fig_count += 1
                    except Exception as e:
                        pass
            
            # 4. Upload tables to Wasabi
            if extract_tables and result.tables:
                for i, table in enumerate(result.tables):
                    try:
                        if hasattr(table, 'image') and table.image is not None:
                            img = table.image
                            if img.mode in ('RGBA', 'P'):
                                img = img.convert('RGB')
                            
                            buffer = BytesIO()
                            img.save(buffer, format='PNG')
                            buffer.seek(0)
                            
                            # Save to tables/{server}/{month}/{doi}/table_N.png
                            s3_key = f"tables/{server}/{category}/{safe_doi}/table_{i+1}.png"
                            wasabi_client.put_object(
                                Bucket=bucket,
                                Key=s3_key,
                                Body=buffer.getvalue(),
                                ContentType='image/png'
                            )
                            table_paths.append(s3_key)
                            table_count += 1
                    except Exception as e:
                        pass
            
            # 5. Save extraction JSON to Wasabi
            word_count = len(result.full_text.split()) if result.full_text else 0
            extraction_data = {
                'doi': doi,
                'paper_id': doi,
                'safe_id': safe_doi,
                'category': category,
                'server': server,
                'title': getattr(result, 'title', ''),
                'abstract': getattr(result, 'abstract', ''),
                'full_text': getattr(result, 'full_text', '') or getattr(result, 'text', ''),
                'figure_count': fig_count,
                'table_count': table_count,
                'figure_paths': figure_paths,
                'table_paths': table_paths,
                'word_count': word_count
            }
            
            # Save to extractions/{server}/{month}/{filename}.json
            json_key = f"extractions/{server}/{category}/{safe_doi}.json"
            wasabi_client.put_object(
                Bucket=bucket,
                Key=json_key,
                Body=json.dumps(extraction_data, ensure_ascii=False),
                ContentType='application/json'
            )
            
            result_info = {
                'success': True,
                'filename': filename,
                'word_count': word_count,
                'fig_count': fig_count,
                'table_count': table_count,
                'error': None
            }
        else:
            result_info['error'] = "No content extracted"
            
    except Exception as e:
        result_info['error'] = str(e)[:50]
    
    finally:
        # 6. Delete temp file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except:
                pass
    
    return result_info


def _stream_extract_from_wasabi(
    wasabi,
    pdf_keys: List[dict],
    extractor,
    extract_images: bool = True,
    extract_tables: bool = True,
    workers: int = 4
):
    """
    PARALLEL stream-process PDFs from Wasabi.
    Downloads, extracts, uploads results in parallel.
    
    Args:
        workers: Number of parallel workers (default: 4)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    total_files = len(pdf_keys)
    
    print(f"\n📥 Parallel Streaming Extraction: {total_files} PDFs")
    print(f"   Workers: {workers} parallel threads")
    print(f"   (Download → Extract → Upload JSON + Images → Delete)")
    print(f"   Extraction JSONs saved to: s3://{wasabi.bucket}/extractions/\n")
    
    # Prepare wasabi config for workers
    wasabi_config = {
        'endpoint': wasabi.endpoint_url,  # Use endpoint_url
        'access_key': wasabi.access_key,
        'secret_key': wasabi.secret_key,
        'region': wasabi.region,
        'bucket': wasabi.bucket
    }
    
    # Prepare arguments for parallel processing
    work_items = [
        (pdf_info, wasabi_config, extract_images, extract_tables)
        for pdf_info in pdf_keys
    ]
    
    results = []
    total_figures = 0
    total_tables = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_pdf, item): i for i, item in enumerate(work_items)}
        
        with tqdm(total=total_files, desc="Processing", unit="pdf") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result.get('skipped'):
                    skipped_count += 1
                    status = "⏭️ skipped"
                elif result['success']:
                    success_count += 1
                    total_figures += result['fig_count']
                    total_tables += result['table_count']
                    status = f"✅ {result['word_count']:,} words, {result['fig_count']} figs, {result['table_count']} tables"
                else:
                    fail_count += 1
                    status = f"❌ {result['error']}"
                
                # Update progress bar description with latest file
                short_name = result['filename'].replace('.pdf', '')[:25]
                pbar.set_postfix_str(f"{short_name}: {status[:40]}")
                pbar.update(1)
                
                results.append(result)
    
    print(f"\n{'='*60}")
    print(f"PARALLEL EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Success: {success_count} papers")
    print(f"  ⏭️  Skipped: {skipped_count} papers (already extracted)")
    print(f"  ❌ Failed: {fail_count} papers")
    print(f"  📊 Figures saved: {total_figures}")
    print(f"  📊 Tables saved: {total_tables}")
    print(f"\n  Storage: Wasabi S3 (s3://{wasabi.bucket}/)")
    print(f"    - extractions/  → Text data (for embed step)")
    print(f"    - figures/      → Figure images")
    print(f"    - tables/       → Table images")
    
    return results


# =============================================================================
# STEP 3: EXTRACT CONTENT FROM PDFs
# =============================================================================

def step3_extract_content(
    categories: List[str],
    extract_images: bool = True,
    extract_tables: bool = True,
    from_wasabi: bool = False,
    workers: int = 4,
    max_papers: Optional[int] = None,
    month: Optional[str] = None,
    server: Optional[str] = None
):
    """
    Extract text, images, and tables from downloaded PDFs.
    Fetches from Wasabi S3 if PDFs not found locally or if from_wasabi=True.
    
    Args:
        categories: List of categories to process
        extract_images: Whether to extract figures
        extract_tables: Whether to extract tables
        from_wasabi: If True, stream from Wasabi S3 (skip local files)
        workers: Number of parallel workers for Wasabi streaming
        max_papers: Maximum number of papers to process (None = all)
        month: Filter by specific month (e.g., 'April_2019')
        server: Filter by server ('biorxiv' or 'medrxiv')
    """
    print("\n" + "="*70)
    print("STEP 3: EXTRACT CONTENT FROM PDFs")
    print("="*70)
    if month:
        print(f"Month filter: {month}")
    if server:
        print(f"Server filter: {server}")
    
    from extraction.pdf_extractor import PDFExtractor
    from tqdm import tqdm
    
    extractor = PDFExtractor()
    
    # If --from-wasabi flag, skip local and stream directly
    if from_wasabi:
        print(f"Mode: PARALLEL STREAMING from Wasabi S3 (--from-wasabi flag)")
        print(f"Workers: {workers}")
        if max_papers:
            print(f"Max papers: {max_papers}")
        all_pdf_files = []  # Force Wasabi fetch
    else:
        # Find PDF files locally first
        pdf_dir = PROJECT_ROOT / "data" / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        all_pdf_files = []
        
        for category in categories:
            cat_dir = pdf_dir / category.replace(" ", "_").lower()
            if cat_dir.exists():
                all_pdf_files.extend(cat_dir.glob("*.pdf"))
        
        if not all_pdf_files:
            # Try root pdfs directory
            all_pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # If no local PDFs (or from_wasabi), fetch from Wasabi (STREAMING - one at a time)
    if not all_pdf_files:
        if not from_wasabi:
            print(f"No local PDFs found. Checking Wasabi S3...")
        
        try:
            from storage.wasabi_client import WasabiClient
            wasabi = WasabiClient()
            
            if wasabi.access_key and wasabi.secret_key:
                print(f"Fetching PDFs from Wasabi bucket: {wasabi.bucket}")
                print(f"Mode: STREAMING (process one PDF at a time, minimal disk usage)")
                
                # Collect list of PDF keys to process (don't download yet)
                pdf_keys_to_process = []
                
                # Determine which servers to process
                servers_to_process = [server] if server else ['biorxiv', 'medrxiv']
                
                # Try month-based structure first (new download_all_s3.py)
                # Structure: pdfs/biorxiv/April_2019/xxx.pdf
                for srv in servers_to_process:
                    if month:
                        # Specific month
                        prefix = f"pdfs/{srv}/{month}/"
                        try:
                            paginator = wasabi.client.get_paginator('list_objects_v2')
                            for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                                if 'Contents' in page:
                                    for obj in page['Contents']:
                                        key = obj['Key']
                                        if key.endswith('.pdf'):
                                            pdf_keys_to_process.append({
                                                'key': key,
                                                'category': month,
                                                'server': srv
                                            })
                        except Exception as e:
                            logger.debug(f"Error listing {prefix}: {e}")
                    else:
                        # All months
                        prefix = f"pdfs/{srv}/"
                        try:
                            paginator = wasabi.client.get_paginator('list_objects_v2')
                            for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                                if 'Contents' in page:
                                    for obj in page['Contents']:
                                        key = obj['Key']
                                        if key.endswith('.pdf'):
                                            parts = key.split('/')
                                            m = parts[2] if len(parts) > 2 else 'unknown'
                                            pdf_keys_to_process.append({
                                                'key': key,
                                                'category': m,
                                                'server': srv
                                            })
                        except Exception as e:
                            logger.debug(f"Error listing {prefix}: {e}")
                
                # Fallback to category-based if no month-based found
                if not pdf_keys_to_process:
                    for category in categories:
                        safe_cat = category.replace(" ", "_").lower()
                        prefixes = [
                            f"pdfs/biorxiv/{safe_cat}/",
                            f"pdfs/medrxiv/{safe_cat}/"
                        ]
                        
                        for prefix in prefixes:
                            try:
                                paginator = wasabi.client.get_paginator('list_objects_v2')
                                for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                                    if 'Contents' in page:
                                        for obj in page['Contents']:
                                            key = obj['Key']
                                            if key.endswith('.pdf'):
                                                pdf_keys_to_process.append({
                                                    'key': key,
                                                    'category': safe_cat
                                                })
                            except Exception as e:
                                logger.debug(f"Error listing {prefix}: {e}")
                
                if pdf_keys_to_process:
                    # Apply max limit if specified
                    if max_papers and len(pdf_keys_to_process) > max_papers:
                        pdf_keys_to_process = pdf_keys_to_process[:max_papers]
                        print(f"\nLimited to {max_papers} PDFs (--max flag)")
                    
                    print(f"\nFound {len(pdf_keys_to_process)} PDFs to process (parallel streaming)")
                    
                    # Process in parallel streaming mode
                    return _stream_extract_from_wasabi(
                        wasabi=wasabi,
                        pdf_keys=pdf_keys_to_process,
                        extractor=extractor,
                        extract_images=extract_images,
                        extract_tables=extract_tables,
                        workers=workers
                    )
                
        except ImportError:
            print("Wasabi client not available")
        except Exception as e:
            print(f"Could not fetch from Wasabi: {e}")
    
    if not all_pdf_files:
        print(f"No PDF files found locally or in Wasabi")
        print("Run --download first to get papers.")
        return []
    
    # Apply max limit if specified
    if max_papers and len(all_pdf_files) > max_papers:
        all_pdf_files = all_pdf_files[:max_papers]
        print(f"Limited to {max_papers} PDFs (--max flag)")
    
    print(f"\nFound {len(all_pdf_files)} PDF files to process")
    
    # Initialize Wasabi for uploading figures/tables
    wasabi = None
    use_wasabi = True
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        if wasabi.access_key and wasabi.secret_key:
            print(f"Figures/Tables will be uploaded to Wasabi S3: {wasabi.bucket}")
        else:
            print("Wasabi not configured - saving locally")
            wasabi = None
            use_wasabi = False
    except:
        print("Wasabi not available - saving locally")
        use_wasabi = False
    
    results = []
    
    # Initialize PostgreSQL tracker for skip checking
    tracker = None
    try:
        from db.progress import ProgressTracker
        tracker = ProgressTracker()
    except ImportError:
        logger.debug("PostgreSQL tracker not available")
    
    # Local directories (for temporary or fallback storage)
    figures_dir = PROJECT_ROOT / "data" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = PROJECT_ROOT / "data" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    total_figures_saved = 0
    total_tables_saved = 0
    total_files = len(all_pdf_files)
    skipped_count = 0
    
    print(f"\nProcessing {total_files} PDF files...\n")
    
    for idx, pdf_path in enumerate(all_pdf_files, 1):
        filename = pdf_path.stem
        
        # CHECK POSTGRESQL: Skip if already extracted
        doi = _extract_doi_from_filename(filename)
        if tracker and doi:
            try:
                if tracker.is_extracted(doi):
                    short_name = filename[:38] if len(filename) > 38 else filename
                    print(f"  [{idx:3d}/{total_files}] {short_name:<38} ⏭️  SKIPPED (already extracted)")
                    skipped_count += 1
                    continue
            except:
                pass  # Continue with extraction if check fails
        short_name = filename[:35] + "..." if len(filename) > 35 else filename
        
        try:
            result = extractor.extract(str(pdf_path))
            
            if result:
                # Get DOI from filename (format: 10_1101_2024_01_15_575685.pdf)
                # Convert back to DOI format: 10.1101/2024.01.15.575685
                filename = pdf_path.stem
                if filename.startswith('10_1101_'):
                    # Convert underscore format to DOI format
                    doi = filename.replace('_', '.', 1).replace('_', '/', 1).replace('_', '.')
                else:
                    doi = filename
                
                # Attach DOI and paper_id to result for later use
                result.doi = doi
                result.paper_id = doi
                result.source_file = str(pdf_path)
                
                # Save figures → Wasabi S3
                if extract_images and result.figures:
                    for i, fig in enumerate(result.figures):
                        try:
                            if hasattr(fig, 'image') and fig.image is not None:
                                # Convert to RGB if needed
                                if fig.image.mode in ('RGBA', 'P'):
                                    img = fig.image.convert('RGB')
                                else:
                                    img = fig.image
                                
                                # Use DOI-based path (with safe filename)
                                safe_doi = doi.replace('/', '_').replace('.', '_')
                                
                                # Upload to Wasabi
                                if use_wasabi and wasabi:
                                    from io import BytesIO
                                    buffer = BytesIO()
                                    img.save(buffer, format='PNG')
                                    buffer.seek(0)
                                    
                                    s3_key = f"figures/{safe_doi}/fig_{i+1}.png"
                                    wasabi.client.put_object(
                                        Bucket=wasabi.bucket,
                                        Key=s3_key,
                                        Body=buffer.getvalue(),
                                        ContentType='image/png'
                                    )
                                    total_figures_saved += 1
                                else:
                                    # Fallback to local
                                    paper_fig_dir = figures_dir / safe_doi
                                    paper_fig_dir.mkdir(parents=True, exist_ok=True)
                                    fig_path = paper_fig_dir / f"fig_{i+1}.png"
                                    img.save(str(fig_path), 'PNG')
                                    total_figures_saved += 1
                                    
                        except Exception as e:
                            logger.debug(f"Failed to save figure {i}: {e}")
                
                # Save tables → Wasabi S3
                if extract_tables and result.tables:
                    for i, table in enumerate(result.tables):
                        try:
                            if hasattr(table, 'image') and table.image is not None:
                                # Convert to RGB if needed
                                if table.image.mode in ('RGBA', 'P'):
                                    img = table.image.convert('RGB')
                                else:
                                    img = table.image
                                
                                # Use DOI-based path (with safe filename)
                                safe_doi = doi.replace('/', '_').replace('.', '_')
                                
                                # Upload to Wasabi
                                if use_wasabi and wasabi:
                                    from io import BytesIO
                                    buffer = BytesIO()
                                    img.save(buffer, format='PNG')
                                    buffer.seek(0)
                                    
                                    s3_key = f"tables/{safe_doi}/table_{i+1}.png"
                                    wasabi.client.put_object(
                                        Bucket=wasabi.bucket,
                                        Key=s3_key,
                                        Body=buffer.getvalue(),
                                        ContentType='image/png'
                                    )
                                    total_tables_saved += 1
                                else:
                                    # Fallback to local
                                    paper_table_dir = tables_dir / safe_doi
                                    paper_table_dir.mkdir(parents=True, exist_ok=True)
                                    table_img_path = paper_table_dir / f"table_{i+1}.png"
                                    img.save(str(table_img_path), 'PNG')
                                    total_tables_saved += 1
                            
                        except Exception as e:
                            logger.debug(f"Failed to save table {i}: {e}")
                
                # Save extraction JSON to Wasabi (for embed step in Colab)
                if use_wasabi and wasabi:
                    try:
                        import json as json_module
                        safe_doi = doi.replace('/', '_').replace('.', '_')
                        
                        # Get category from path
                        category = "unknown"
                        for cat in categories:
                            if cat.replace(" ", "_").lower() in str(pdf_path).lower():
                                category = cat.replace(" ", "_").lower()
                                break
                        
                        fig_count = len(result.figures) if result.figures else 0
                        tbl_count = len(result.tables) if result.tables else 0
                        
                        extraction_data = {
                            'doi': doi,
                            'paper_id': doi,
                            'safe_id': safe_doi,
                            'category': category,
                            'title': getattr(result, 'title', ''),
                            'abstract': getattr(result, 'abstract', ''),
                            'full_text': getattr(result, 'full_text', '') or getattr(result, 'text', ''),
                            'figure_count': fig_count,
                            'table_count': tbl_count,
                            'figure_paths': [f"figures/{safe_doi}/fig_{i+1}.png" for i in range(fig_count)],
                            'table_paths': [f"tables/{safe_doi}/table_{i+1}.png" for i in range(tbl_count)],
                            'word_count': len(result.full_text.split()) if result.full_text else 0
                        }
                        
                        json_key = f"extractions/{category}/{safe_doi}.json"
                        wasabi.client.put_object(
                            Bucket=wasabi.bucket,
                            Key=json_key,
                            Body=json_module.dumps(extraction_data, ensure_ascii=False),
                            ContentType='application/json'
                        )
                    except Exception as e:
                        logger.debug(f"Failed to save extraction JSON: {e}")
                
                # UPDATE POSTGRESQL: Mark as extracted
                try:
                    from db.progress import ProgressTracker
                    tracker = ProgressTracker()
                    fig_count = len(result.figures) if result.figures else 0
                    tbl_count = len(result.tables) if result.tables else 0
                    text_chunks = len(result.full_text.split()) // 200 if result.full_text else 0  # Approx chunks
                    tracker.mark_extracted(
                        doi=doi,
                        text_chunks=text_chunks,
                        figures_count=fig_count,
                        tables_count=tbl_count
                    )
                    tracker.close()
                except Exception as e:
                    logger.debug(f"Failed to update extraction status: {e}")
                
                # Print progress for this file
                fig_count = len(result.figures) if result.figures else 0
                tbl_count = len(result.tables) if result.tables else 0
                word_count = len(result.full_text.split()) if result.full_text else 0
                print(f"  [{idx:3d}/{total_files}] {short_name:<38} ✅ {word_count:,} words, {fig_count} figs, {tbl_count} tables")
                
                results.append(result)
                
        except Exception as e:
            print(f"  [{idx:3d}/{total_files}] {short_name:<38} ❌ {str(e)[:30]}")
            logger.debug(f"Failed to extract {pdf_path.name}: {e}")
    
    # Count tables with actual images
    tables_with_images = sum(
        sum(1 for t in r.tables if hasattr(t, 'image') and t.image is not None) 
        if r.tables else 0 
        for r in results
    )
    
    print(f"\nExtraction complete!")
    print(f"  Processed: {len(results)} papers")
    print(f"  Skipped:   {skipped_count} papers (already extracted)")
    print(f"  Figures detected: {sum(len(r.figures) if r.figures else 0 for r in results)}")
    print(f"  Figures saved: {total_figures_saved}")
    print(f"  Tables with structure: {tables_with_images}")
    print(f"  Tables saved as images: {total_tables_saved}")
    
    # Close tracker
    if tracker:
        tracker.close()
    
    if use_wasabi and wasabi:
        print(f"\nStorage: Wasabi S3")
        print(f"  Figures: s3://{wasabi.bucket}/figures/")
        print(f"  Tables: s3://{wasabi.bucket}/tables/")
    else:
        print(f"\nStorage: Local")
        print(f"  Figures: {figures_dir}")
        print(f"  Tables: {tables_dir}")
    
    if tables_with_images == 0:
        print(f"\nNote: No structured tables found. PDFs may use:")
        print(f"  - Text-based tables (no grid lines)")
        print(f"  - Tables as images (already in figures)")
        print(f"  - Supplementary tables in separate files")
    
    return results


# =============================================================================
# STEP 4: BUILD EMBEDDINGS AND STORE IN QDRANT
# =============================================================================

def _list_extraction_keys_from_wasabi(
    categories: List[str] = None, 
    max_papers: Optional[int] = None,
    month: Optional[str] = None,
    server: Optional[str] = None
) -> List[dict]:
    """
    List extraction JSON keys from Wasabi WITHOUT downloading them.
    Returns list of {key, doi, category, server} dicts.
    
    Structure in Wasabi:
    - extractions/{server}/{month}/xxx.json  (e.g., extractions/biorxiv/April_2019/uuid_xxx.json)
    - extractions/{category}/xxx.json        (legacy category-based)
    
    This is FAST and uses minimal memory!
    """
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        
        if not wasabi.access_key or not wasabi.secret_key:
            return []
        
        keys = []
        
        # If month and/or server specified, use server/month structure
        if month or server:
            servers_to_check = [server] if server else ['biorxiv', 'medrxiv']
            
            for srv in servers_to_check:
                if month:
                    # Specific month: extractions/{server}/{month}/
                    prefix = f"extractions/{srv}/{month}/"
                    print(f"  Looking in: {prefix}")
                    try:
                        paginator = wasabi.client.get_paginator('list_objects_v2')
                        for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                            if 'Contents' in page:
                                for obj in page['Contents']:
                                    key = obj['Key']
                                    if key.endswith('.json'):
                                        filename = key.split('/')[-1].replace('.json', '')
                                        
                                        if filename.startswith('uuid_'):
                                            doi = filename
                                        else:
                                            parts = filename.split('_')
                                            if len(parts) >= 3:
                                                doi = f"{parts[0]}.{parts[1]}/{'.'.join(parts[2:])}"
                                            else:
                                                doi = filename
                                        
                                        keys.append({
                                            'key': key,
                                            'doi': doi,
                                            'category': month,
                                            'server': srv
                                        })
                                        
                                        if max_papers and len(keys) >= max_papers:
                                            return keys
                    except Exception as e:
                        logger.debug(f"Error listing {prefix}: {e}")
                else:
                    # All months for server: extractions/{server}/
                    prefix = f"extractions/{srv}/"
                    print(f"  Listing all months in: {prefix}")
                    try:
                        # First get list of months
                        response = wasabi.client.list_objects_v2(
                            Bucket=wasabi.bucket,
                            Prefix=prefix,
                            Delimiter='/'
                        )
                        
                        months = []
                        if 'CommonPrefixes' in response:
                            for prefix_info in response['CommonPrefixes']:
                                folder = prefix_info['Prefix'].split('/')[-2]
                                months.append(folder)
                        
                        print(f"    Found {len(months)} months: {months[:5]}...")
                        
                        for m in months:
                            month_prefix = f"extractions/{srv}/{m}/"
                            paginator = wasabi.client.get_paginator('list_objects_v2')
                            for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=month_prefix):
                                if 'Contents' in page:
                                    for obj in page['Contents']:
                                        key = obj['Key']
                                        if key.endswith('.json'):
                                            filename = key.split('/')[-1].replace('.json', '')
                                            
                                            if filename.startswith('uuid_'):
                                                doi = filename
                                            else:
                                                parts = filename.split('_')
                                                if len(parts) >= 3:
                                                    doi = f"{parts[0]}.{parts[1]}/{'.'.join(parts[2:])}"
                                                else:
                                                    doi = filename
                                            
                                            keys.append({
                                                'key': key,
                                                'doi': doi,
                                                'category': m,
                                                'server': srv
                                            })
                                            
                                            if max_papers and len(keys) >= max_papers:
                                                return keys
                    except Exception as e:
                        logger.debug(f"Error listing {prefix}: {e}")
            
            return keys
        
        # Fallback: category-based structure (legacy)
        if categories:
            for category in categories:
                safe_cat = category.replace(" ", "_").lower()
                prefix = f"extractions/{safe_cat}/"
                
                try:
                    paginator = wasabi.client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                key = obj['Key']
                                if key.endswith('.json'):
                                    filename = key.split('/')[-1].replace('.json', '')
                                    parts = filename.split('_')
                                    if len(parts) >= 3:
                                        doi = f"{parts[0]}.{parts[1]}/{'.'.join(parts[2:])}"
                                    else:
                                        doi = filename
                                    
                                    keys.append({
                                        'key': key,
                                        'doi': doi,
                                        'category': safe_cat
                                    })
                                    
                                    if max_papers and len(keys) >= max_papers:
                                        return keys
                except Exception as e:
                    logger.debug(f"Error listing {prefix}: {e}")
        
        return keys
        
    except Exception as e:
        logger.debug(f"Failed to list keys: {e}")
        return []


def _download_extraction_batch(keys: List[dict], wasabi, max_workers: int = 16) -> List[dict]:
    """
    Download a batch of extraction JSONs from Wasabi IN PARALLEL.
    
    Args:
        keys: List of {key, doi, category} dicts
        wasabi: WasabiClient instance
        max_workers: Number of parallel download threads (default 16)
    
    Returns:
        List of extraction data dicts
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def download_one(key_info):
        """Download a single JSON file"""
        try:
            response = wasabi.client.get_object(
                Bucket=wasabi.bucket,
                Key=key_info['key']
            )
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data
        except Exception as e:
            logger.debug(f"Failed to load {key_info['key']}: {e}")
            return None
    
    extractions = []
    
    # Parallel download using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, k): k for k in keys}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                extractions.append(result)
    
    return extractions


def _load_extractions_from_wasabi(categories: List[str], max_papers: Optional[int] = None) -> List[dict]:
    """
    Load extraction JSONs from Wasabi S3.
    Used when running embed step separately from extract step (e.g., Colab).
    
    Args:
        categories: Categories to load
        max_papers: Maximum number of papers to load (None = all)
    """
    import json
    
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        
        if not wasabi.access_key or not wasabi.secret_key:
            print("❌ Wasabi not configured")
            return []
        
        print(f"📥 Loading extractions from Wasabi: s3://{wasabi.bucket}/extractions/")
        if max_papers:
            print(f"   (Limited to {max_papers} papers)")
        
        extractions = []
        files_found = 0
        
        for category in categories:
            safe_cat = category.replace(" ", "_").lower()
            prefix = f"extractions/{safe_cat}/"
            
            try:
                paginator = wasabi.client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            # Check if we've hit max limit
                            if max_papers and len(extractions) >= max_papers:
                                print(f"✅ Loaded {len(extractions)} extraction JSONs (limited by --max {max_papers})")
                                return extractions
                            
                            key = obj['Key']
                            if key.endswith('.json'):
                                try:
                                    response = wasabi.client.get_object(
                                        Bucket=wasabi.bucket,
                                        Key=key
                                    )
                                    data = json.loads(response['Body'].read().decode('utf-8'))
                                    extractions.append(data)
                                    files_found += 1
                                    
                                    # Progress indicator every 50 files
                                    if files_found % 50 == 0:
                                        print(f"   📄 Loaded {files_found} files...")
                                        
                                except Exception as e:
                                    logger.debug(f"Failed to load {key}: {e}")
            except Exception as e:
                logger.debug(f"Error listing {prefix}: {e}")
        
        print(f"✅ Loaded {len(extractions)} extraction JSONs from Wasabi")
        return extractions
        
    except Exception as e:
        print(f"❌ Failed to load from Wasabi: {e}")
        return []


def step4_smart_embed(
    categories: List[str],
    batch_size: int = 500,
    gpu_batch: int = 32,
    force: bool = False,
    max_papers: Optional[int] = None,
    embed_mode: str = 'all',  # 'all', 'text', 'images'
    month: Optional[str] = None,
    server: Optional[str] = None
):
    """
    SMART EMBEDDING - Memory efficient, parallel GPU, resumable.
    
    Args:
        embed_mode: 'all' (text+images), 'text' (text only), 'images' (images only)
        month: Filter by specific month
        server: Filter by server ('biorxiv' or 'medrxiv')
    
    Strategy:
    1. Get existing DOIs from Qdrant FIRST (before loading anything)
    2. List JSON keys from Wasabi (don't download yet!)
    3. Filter to only NEW papers
    4. Process in batches (download → embed → clear memory)
    5. Parallel GPU embedding within each batch
    
    Args:
        categories: Categories to process
        batch_size: Papers to process per batch (default 500)
        gpu_batch: Texts to embed in parallel on GPU (default 32)
        force: Re-embed even if exists
        max_papers: Maximum papers to process
        embed_mode: 'all', 'text', or 'images'
    """
    import time
    import gc
    
    mode_desc = {
        'all': 'TEXT + IMAGES',
        'text': 'TEXT ONLY',
        'images': 'IMAGES ONLY'
    }.get(embed_mode, 'ALL')
    
    print("\n" + "="*70)
    print(f"STEP 4: SMART EMBEDDING - {mode_desc}")
    print("="*70)
    print(f"  Embed mode: {embed_mode.upper()}")
    print(f"  Strategy: Stream batches of {batch_size} papers")
    print(f"  GPU batch: {gpu_batch} texts in parallel")
    print(f"  Max papers: {max_papers or 'all'}")
    if month:
        print(f"  Month filter: {month}")
    if server:
        print(f"  Server filter: {server}")
    if force:
        print(f"  Mode: FORCE (re-embed all)")
    else:
        print(f"  Mode: SKIP existing papers")
    
    start_time = time.time()
    
    # ===== STEP 1: Connect to Services =====
    print("\n📡 Connecting to services...")
    
    # Qdrant
    qdrant = None
    try:
        from storage.qdrant_client import QdrantClient
        qdrant = QdrantClient()
        print("  ✓ Qdrant connected")
    except Exception as e:
        print(f"  ✗ Qdrant failed: {e}")
        return
    
    # Wasabi
    wasabi = None
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        if wasabi.access_key:
            print("  ✓ Wasabi connected")
        else:
            print("  ✗ Wasabi not configured")
            return
    except Exception as e:
        print(f"  ✗ Wasabi failed: {e}")
        return
    
    # ===== STEP 2: Get Existing DOIs from Qdrant AND PostgreSQL =====
    print("\n📋 Checking already embedded papers...")
    existing_dois = set()
    pg_status = {}  # {doi: {text: bool, figures: bool, tables: bool}}
    
    # Check PostgreSQL first (more reliable)
    pg_tracker = None
    try:
        from db.progress import ProgressTracker
        pg_tracker = ProgressTracker()
        
        # Get papers that are fully embedded in PostgreSQL
        stats = pg_tracker.get_stats()
        print(f"  PostgreSQL: {stats.get('fully_embedded', 0):,} fully embedded papers")
        
        # Get fully embedded DOIs from PostgreSQL
        if not force:
            from db.models import Paper, get_session
            from sqlalchemy import and_
            session = get_session()
            fully_embedded = session.query(Paper.doi).filter(
                and_(
                    Paper.text_embedded == True,
                    Paper.figures_embedded == True,
                    Paper.tables_embedded == True
                )
            ).all()
            pg_embedded_dois = {p.doi for p in fully_embedded}
            existing_dois.update(pg_embedded_dois)
            session.close()
            print(f"  ✓ Found {len(pg_embedded_dois):,} fully embedded papers in PostgreSQL")
    except ImportError:
        print("  PostgreSQL tracker not available, using Qdrant only")
    except Exception as e:
        logger.debug(f"PostgreSQL check failed: {e}")
    
    if not force:
        try:
            if qdrant.collection_exists("bio_abstracts"):
                # Scroll through ALL points to get existing DOIs
                offset = None
                while True:
                    scroll_result = qdrant.client.scroll(
                        collection_name="bio_abstracts",
                        limit=10000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    points, offset = scroll_result
                    
                    for point in points:
                        if point.payload and 'doi' in point.payload:
                            existing_dois.add(point.payload['doi'])
                    
                    if offset is None:
                        break
                        
                print(f"  ✓ Found {len(existing_dois):,} total already embedded papers (Qdrant + PostgreSQL)")
        except Exception as e:
            logger.debug(f"Could not get existing DOIs: {e}")
    
    # ===== STEP 3: List JSON Keys from Wasabi (NO DOWNLOAD YET!) =====
    print("\n📂 Listing extraction files in Wasabi...")
    all_keys = _list_extraction_keys_from_wasabi(
        categories=categories, 
        max_papers=max_papers,
        month=month,
        server=server
    )
    print(f"  ✓ Found {len(all_keys):,} extraction files")
    
    if not all_keys:
        print("  ✗ No extraction files found!")
        return
    
    # ===== STEP 4: Filter to Only NEW Papers =====
    if force:
        new_keys = all_keys
    else:
        new_keys = [k for k in all_keys if k['doi'] not in existing_dois]
    
    skipped_count = len(all_keys) - len(new_keys)
    
    print(f"\n{'='*60}")
    print(f"📊 EMBEDDING PLAN")
    print(f"{'='*60}")
    print(f"  Total in Wasabi:    {len(all_keys):,}")
    print(f"  Already embedded:   {skipped_count:,} (will skip)")
    print(f"  To embed:           {len(new_keys):,}")
    print(f"  Batches:            {(len(new_keys) + batch_size - 1) // batch_size}")
    print(f"{'='*60}")
    
    if not new_keys:
        print("\n✅ All papers already embedded! Nothing to do.")
        return
    
    # ===== STEP 5: Load Models =====
    print("\n🧠 Loading AI models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Check if GPU available
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  ✓ Text model loaded (device: {device})")
    except Exception as e:
        print(f"  ✗ Failed to load text model: {e}")
        return
    
    image_extractor = None
    try:
        from detection.image_feature_extractor import MultiMethodFeatureExtractor
        image_extractor = MultiMethodFeatureExtractor()
        print("  ✓ Image extractor loaded")
    except Exception as e:
        print(f"  ⚠ Image extractor not available: {e}")
    
    # ===== STEP 6: Process in Batches =====
    print(f"\n🚀 Starting batch processing...")
    print(f"  ⚡ OPTIMIZATIONS ENABLED:")
    print(f"     • Parallel JSON downloads (16 threads)")
    print(f"     • Parallel image downloads (8 threads per paper)")
    print(f"     • Batch GPU embedding ({gpu_batch} texts at once)")
    print(f"     • Batch Qdrant uploads (500 points per API call)")
    
    total_embedded = 0
    total_chunks = 0
    total_figures = 0
    total_tables = 0
    total_failed = 0
    
    num_batches = (len(new_keys) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(new_keys))
        batch_keys = new_keys[batch_start:batch_end]
        
        print(f"\n{'─'*50}")
        print(f"📦 BATCH {batch_idx + 1}/{num_batches} (papers {batch_start + 1}-{batch_end})")
        print(f"{'─'*50}")
        
        # 6a. Download this batch (PARALLEL!)
        print(f"  📥 Downloading {len(batch_keys)} JSONs (parallel)...")
        batch_data = _download_extraction_batch(batch_keys, wasabi)
        print(f"  ✓ Downloaded {len(batch_data)} papers")
        
        # 6b. Process papers - collect all points, then batch upload
        abstracts_points = []
        chunks_points = []
        figures_points = []
        tables_points = []
        
        print(f"  🧠 Embedding papers...")
        
        for paper_idx, paper in enumerate(batch_data):
            try:
                doi = paper.get('doi', f'unknown_{total_embedded}')
                safe_doi = doi.replace('/', '_').replace('.', '_')
                short_doi = doi.split('/')[-1][:25] if '/' in doi else doi[:25]
                
                # === LOGGING: Show which paper is being processed ===
                print(f"\n  📄 [{paper_idx+1}/{len(batch_data)}] Processing: {doi}")
                
                # Get server from paper data for unique ID prefix
                paper_server = paper.get('server', 'biorxiv')
                
                chunk_count = 0
                fig_count = 0
                table_count = 0
                
                # === TEXT EMBEDDING (Parallel GPU) ===
                if embed_mode in ('all', 'text'):
                    full_text = paper.get('full_text', '')
                    abstract = paper.get('abstract', '') or (full_text[:1000] if full_text else '')
                    title = paper.get('title', '')
                    
                    if abstract:
                        # Abstract embedding - add to batch
                        # Use server prefix to prevent collision: biorxiv_uuid_xxx or medrxiv_uuid_xxx
                        unique_id = f"{paper_server}_{doi}"
                        abstract_vector = text_model.encode(abstract).tolist()
                        abstracts_points.append({
                            'id': unique_id,
                            'vector': abstract_vector,
                            'payload': {
                                "doi": doi,
                                "paper_id": doi,
                                "safe_id": safe_doi,
                                "title": title,
                                "abstract": abstract[:500],
                                "server": paper_server
                            }
                        })
                    
                    # Text chunks - BATCH EMBED for GPU efficiency
                    if full_text:
                        # Smart chunking: no hard limit, but filter aggressively
                        # This ensures Discussion/Conclusion are always included
                        chunks = _create_chunks(full_text, chunk_size=200)
                        
                        # Warn if paper is unusually large (but still process all)
                        if len(chunks) > 150:
                            print(f"      ⚠️  Large paper: {len(chunks)} chunks")
                        
                        # No hard limit - process all filtered chunks
                        chunk_count = len(chunks)
                        
                        if chunks:
                            # Batch encode all chunks at once (GPU parallel!)
                            chunk_vectors = text_model.encode(chunks, batch_size=gpu_batch)
                            
                            # Add all chunks to batch
                            for i, (chunk, vector) in enumerate(zip(chunks, chunk_vectors)):
                                chunks_points.append({
                                    'id': f"{paper_server}_{doi}_chunk_{i}",
                                    'vector': vector.tolist(),
                                    'payload': {
                                        "doi": doi,
                                        "paper_id": doi,
                                        "chunk_index": i,
                                        "text": chunk,
                                        "server": paper_server
                                    }
                                })
                else:
                    full_text = paper.get('full_text', '')
                
                # === IMAGE EMBEDDING (PARALLEL DOWNLOAD) ===
                if embed_mode in ('all', 'images'):
                    figure_paths = paper.get('figure_paths', [])
                    if figure_paths and image_extractor:
                        from PIL import Image
                        from io import BytesIO
                        from concurrent.futures import ThreadPoolExecutor
                        
                        def download_image(path_info):
                            """Download single image from Wasabi"""
                            idx, path = path_info
                            try:
                                response = wasabi.client.get_object(
                                    Bucket=wasabi.bucket,
                                    Key=path
                                )
                                img = Image.open(BytesIO(response['Body'].read()))
                                return (idx, path, img)
                            except Exception as e:
                                return (idx, path, None)
                        
                        # Parallel download all figures
                        fig_items = list(enumerate(figure_paths[:10]))
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            results = list(executor.map(download_image, fig_items))
                        
                        # Process downloaded images
                        for idx, fig_path, img in results:
                            if img is not None:
                                try:
                                    features = image_extractor.extract_all_features(img, doi, f"fig_{idx}")
                                    if features.cnn_vector is not None:
                                        figures_points.append({
                                            'id': f"{paper_server}_{doi}_fig_{idx}",
                                            'vector': features.cnn_vector.tolist(),
                                            'payload': {
                                                "doi": doi,
                                                "paper_id": doi,
                                                "safe_id": safe_doi,
                                                "figure_index": idx,
                                                "type": "figure",
                                                "s3_path": fig_path,
                                                "perceptual_hashes": features.orientation_hashes,
                                                "original_id": f"{paper_server}_{doi}_fig_{idx}",
                                                "server": paper_server
                                            }
                                        })
                                        fig_count += 1
                                except Exception as e:
                                    logger.debug(f"Failed to process figure: {e}")
                
                    # === TABLE EMBEDDING (PARALLEL DOWNLOAD) ===
                    table_paths = paper.get('table_paths', [])
                    if table_paths and image_extractor:
                        from PIL import Image
                        from io import BytesIO
                        from concurrent.futures import ThreadPoolExecutor
                        
                        def download_table(path_info):
                            """Download single table image from Wasabi"""
                            idx, path = path_info
                            try:
                                response = wasabi.client.get_object(
                                    Bucket=wasabi.bucket,
                                    Key=path
                                )
                                img = Image.open(BytesIO(response['Body'].read()))
                                return (idx, path, img)
                            except Exception as e:
                                return (idx, path, None)
                        
                        # Parallel download all tables
                        tbl_items = list(enumerate(table_paths[:10]))
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            results = list(executor.map(download_table, tbl_items))
                        
                        # Process downloaded tables
                        for idx, tbl_path, img in results:
                            if img is not None:
                                try:
                                    features = image_extractor.extract_all_features(img, doi, f"table_{idx}")
                                    if features.cnn_vector is not None:
                                        # Build OCR data dict for table matching
                                        ocr_regions_count = len(features.ocr_regions) if features.ocr_regions else 0
                                        full_text_extracted = ' '.join([r.get('text', '') for r in features.ocr_regions]) if features.ocr_regions else ''
                                        ocr_data = {
                                            "has_text": ocr_regions_count > 0,
                                            "text_regions": features.ocr_regions if features.ocr_regions else [],
                                            "text_hash": features.text_hash if hasattr(features, 'text_hash') else '',
                                            "full_text": full_text_extracted
                                        }
                                        
                                        # === LOGGING: Show table OCR extraction ===
                                        ocr_preview = full_text_extracted[:80] if full_text_extracted else 'NO TEXT EXTRACTED'
                                        print(f"      📊 Table {idx}: {ocr_regions_count} OCR regions | '{ocr_preview}...'")
                                        
                                        tables_points.append({
                                            'id': f"{paper_server}_{doi}_table_{idx}",
                                            'vector': features.cnn_vector.tolist(),
                                            'payload': {
                                                "doi": doi,
                                                "paper_id": doi,
                                                "safe_id": safe_doi,
                                                "table_index": idx,
                                                "type": "table",
                                                "s3_path": tbl_path,
                                                "perceptual_hashes": features.orientation_hashes,
                                                "original_id": f"{paper_server}_{doi}_table_{idx}",
                                                "ocr_data": ocr_data,
                                                "ocr_regions": features.ocr_regions if features.ocr_regions else [],
                                                "server": paper_server
                                            }
                                        })
                                        table_count += 1
                                except Exception as e:
                                    logger.debug(f"Failed to process table: {e}")
                
                total_embedded += 1
                total_chunks += chunk_count
                total_figures += fig_count
                total_tables += table_count
                
                # Show summary for this paper
                print(f"      ✅ Done: {chunk_count} chunks, {fig_count} figures, {table_count} tables")
                
                # Progress within batch (less verbose)
                if (paper_idx + 1) % 10 == 0 or paper_idx == len(batch_data) - 1:
                    elapsed = time.time() - start_time
                    rate = total_embedded / elapsed if elapsed > 0 else 0
                    remaining = len(new_keys) - total_embedded
                    eta_sec = remaining / rate if rate > 0 else 0
                    eta_min = int(eta_sec // 60)
                    print(f"    [{paper_idx+1:3d}/{len(batch_data)}] Embedded {total_embedded}/{len(new_keys)} papers | ETA: {eta_min}m")
                
            except Exception as e:
                total_failed += 1
                logger.debug(f"Failed to process paper: {e}")
        
        # 6c. BATCH UPLOAD to Qdrant (FAST!)
        print(f"  📤 Uploading to Qdrant...")
        upload_start = time.time()
        
        if abstracts_points:
            qdrant.upsert_batch("bio_abstracts", abstracts_points)
            
        if chunks_points:
            # Split chunks into smaller batches if too large
            chunk_batch_size = 500
            for i in range(0, len(chunks_points), chunk_batch_size):
                batch = chunks_points[i:i + chunk_batch_size]
                qdrant.upsert_batch("bio_fulltext_chunks", batch)
        
        if figures_points:
            qdrant.upsert_batch("bio_figures", figures_points)
            
        if tables_points:
            qdrant.upsert_batch("bio_tables", tables_points)
        
        upload_time = time.time() - upload_start
        print(f"  ✓ Uploaded in {upload_time:.1f}s: {len(abstracts_points)} abstracts, {len(chunks_points)} chunks, {len(figures_points)} figs, {len(tables_points)} tables")
        
        # UPDATE POSTGRESQL: Mark papers as embedded
        try:
            from db.progress import ProgressTracker
            tracker = ProgressTracker()
            
            # Get DOIs from this batch
            batch_dois = [p.get('doi') for p in batch_data if p.get('doi')]
            
            for paper in batch_data:
                doi = paper.get('doi')
                if doi:
                    # Count what was embedded for this paper
                    text_vecs = sum(1 for p in chunks_points if p['payload'].get('doi') == doi)
                    fig_vecs = sum(1 for p in figures_points if p['payload'].get('doi') == doi)
                    tbl_vecs = sum(1 for p in tables_points if p['payload'].get('doi') == doi)
                    
                    tracker.mark_fully_embedded(
                        doi=doi,
                        text_vectors=text_vecs + 1,  # +1 for abstract
                        figure_vectors=fig_vecs,
                        table_vectors=tbl_vecs
                    )
            
            tracker.close()
            print(f"  📊 PostgreSQL updated: {len(batch_dois)} papers marked as embedded")
        except Exception as e:
            logger.debug(f"Failed to update embedding status in PostgreSQL: {e}")
        
        # 6d. Clear memory after each batch
        del batch_data
        del abstracts_points
        del chunks_points
        del figures_points
        del tables_points
        gc.collect()
        
        print(f"  🧹 Memory cleared")
    
    # ===== FINAL SUMMARY =====
    total_time = time.time() - start_time
    total_min = int(total_time // 60)
    total_sec = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"🎉 SMART EMBEDDING COMPLETE!")
    print(f"{'='*60}")
    print(f"  ✅ Papers embedded:     {total_embedded:,}")
    print(f"  ⏭️  Papers skipped:      {skipped_count:,}")
    print(f"  ❌ Papers failed:       {total_failed:,}")
    print(f"  {'─'*40}")
    print(f"  📝 Text chunks:         {total_chunks:,}")
    print(f"  🖼️  Figures:             {total_figures:,}")
    print(f"  📊 Tables:              {total_tables:,}")
    print(f"  {'─'*40}")
    print(f"  ⏱️  Total time:          {total_min}m {total_sec}s")
    if total_embedded > 0:
        print(f"  ⚡ Speed:               {total_embedded / total_time:.1f} papers/sec")
    print(f"{'='*60}")


def step4_embed_missing_images(
    categories: List[str],
    batch_size: int = 100,
    max_papers: Optional[int] = None
):
    """
    Embed ONLY missing images for papers that already have text embeddings.
    
    This function:
    1. Finds papers that have text chunks but NO figures/tables in Qdrant
    2. Downloads their extraction JSONs from Wasabi
    3. Embeds only the images (figures + tables)
    4. Does NOT touch existing text embeddings
    
    Args:
        categories: Categories to process
        batch_size: Papers per batch (default 100)
        max_papers: Maximum papers to process
    """
    import time
    import gc
    import uuid
    
    print("\n" + "="*70)
    print("STEP 4B: EMBED MISSING IMAGES ONLY")
    print("="*70)
    print(f"  Mode: Find papers with text but NO images, embed images only")
    print(f"  Categories: {categories}")
    print(f"  Batch size: {batch_size}")
    if max_papers:
        print(f"  Max papers: {max_papers}")
    
    start_time = time.time()
    
    # ===== STEP 1: Connect to services =====
    print("\n📡 Connecting to services...")
    
    qdrant = None
    try:
        from storage.qdrant_client import QdrantClient
        qdrant = QdrantClient()
        if qdrant.client:
            print("  ✓ Qdrant connected")
        else:
            print("  ✗ Qdrant not connected")
            return
    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
        return
    
    wasabi = None
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        if wasabi.access_key:
            print("  ✓ Wasabi connected")
        else:
            print("  ✗ Wasabi not configured")
            return
    except Exception as e:
        print(f"  ✗ Wasabi error: {e}")
        return
    
    # ===== STEP 2: Find papers with text but no images =====
    print("\n🔍 Finding papers with missing images...")
    
    # Get all DOIs that have text chunks
    print("  Scanning bio_fulltext_chunks...")
    text_dois = set()
    offset = None
    while True:
        try:
            points, offset = qdrant.client.scroll(
                collection_name="bio_fulltext_chunks",
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for p in points:
                doi = p.payload.get('doi') or p.payload.get('paper_id')
                if doi:
                    text_dois.add(doi)
            if offset is None:
                break
        except Exception as e:
            print(f"  Error scanning chunks: {e}")
            break
    print(f"  ✓ Found {len(text_dois):,} papers with text chunks")
    
    # Get all DOIs that have figures
    print("  Scanning bio_figures...")
    figure_dois = set()
    offset = None
    while True:
        try:
            points, offset = qdrant.client.scroll(
                collection_name="bio_figures",
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for p in points:
                doi = p.payload.get('doi') or p.payload.get('paper_id')
                if doi:
                    figure_dois.add(doi)
            if offset is None:
                break
        except Exception as e:
            print(f"  Error scanning figures: {e}")
            break
    print(f"  ✓ Found {len(figure_dois):,} papers with figures")
    
    # Find papers missing images
    missing_image_dois = text_dois - figure_dois
    print(f"\n  📊 Papers with text but NO images: {len(missing_image_dois):,}")
    
    if not missing_image_dois:
        print("\n✅ All papers already have images embedded! Nothing to do.")
        return
    
    # Apply max limit
    if max_papers and len(missing_image_dois) > max_papers:
        missing_image_dois = set(list(missing_image_dois)[:max_papers])
        print(f"  (Limited to {max_papers} papers)")
    
    # ===== STEP 3: Load extraction JSONs for missing papers =====
    print(f"\n📂 Finding extraction files for {len(missing_image_dois):,} papers...")
    
    # List all extraction keys and filter to missing DOIs
    all_keys = _list_extraction_keys_from_wasabi(categories, max_papers=None)
    
    # Filter to only papers missing images
    missing_keys = [k for k in all_keys if k['doi'] in missing_image_dois]
    print(f"  ✓ Found {len(missing_keys):,} extraction files to process")
    
    if not missing_keys:
        print("\n⚠️ No extraction files found for papers missing images.")
        print("   These papers may be from a different category.")
        return
    
    # ===== STEP 4: Load image extractor =====
    print("\n🧠 Loading image extractor...")
    
    image_extractor = None
    try:
        from detection.image_feature_extractor import MultiMethodFeatureExtractor
        image_extractor = MultiMethodFeatureExtractor()
        print("  ✓ Image extractor loaded")
    except Exception as e:
        print(f"  ✗ Image extractor failed: {e}")
        print("  Cannot proceed without image extractor!")
        return
    
    # ===== STEP 5: Process in batches =====
    print(f"\n🚀 Embedding missing images...")
    print(f"  ⚡ Parallel image downloads enabled (8 threads)")
    
    total_figures = 0
    total_tables = 0
    total_processed = 0
    total_failed = 0
    
    num_batches = (len(missing_keys) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(missing_keys))
        batch_keys = missing_keys[batch_start:batch_end]
        
        print(f"\n{'─'*50}")
        print(f"📦 BATCH {batch_idx + 1}/{num_batches} (papers {batch_start + 1}-{batch_end})")
        print(f"{'─'*50}")
        
        # Download extraction JSONs
        print(f"  📥 Downloading {len(batch_keys)} JSONs...")
        batch_data = _download_extraction_batch(batch_keys, wasabi)
        print(f"  ✓ Downloaded {len(batch_data)} papers")
        
        # Collect image points
        figures_points = []
        tables_points = []
        
        print(f"  🖼️ Processing images...")
        
        for paper_idx, paper in enumerate(batch_data):
            try:
                doi = paper.get('doi', f'unknown_{total_processed}')
                safe_doi = doi.replace('/', '_').replace('.', '_')
                
                fig_count = 0
                table_count = 0
                
                # === FIGURE EMBEDDING (PARALLEL DOWNLOAD) ===
                figure_paths = paper.get('figure_paths', [])
                if figure_paths:
                    from PIL import Image
                    from io import BytesIO
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def download_image(path_info):
                        idx, path = path_info
                        try:
                            response = wasabi.client.get_object(
                                Bucket=wasabi.bucket,
                                Key=path
                            )
                            img = Image.open(BytesIO(response['Body'].read()))
                            return (idx, path, img)
                        except Exception as e:
                            return (idx, path, None)
                    
                    # Parallel download
                    fig_items = list(enumerate(figure_paths[:10]))
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        results = list(executor.map(download_image, fig_items))
                    
                    # Process downloaded images
                    for idx, fig_path, img in results:
                        if img is not None:
                            try:
                                features = image_extractor.extract_all_features(img, doi, f"fig_{idx}")
                                if features.cnn_vector is not None:
                                    figures_points.append({
                                        'id': f"{doi}_fig_{idx}",
                                        'vector': features.cnn_vector.tolist(),
                                        'payload': {
                                            "doi": doi,
                                            "paper_id": doi,
                                            "safe_id": safe_doi,
                                            "figure_index": idx,
                                            "type": "figure",
                                            "s3_path": fig_path,
                                            "perceptual_hashes": features.orientation_hashes,
                                            "original_id": f"{doi}_fig_{idx}"
                                        }
                                    })
                                    fig_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to process figure: {e}")
                
                # === TABLE EMBEDDING (PARALLEL DOWNLOAD) ===
                table_paths = paper.get('table_paths', [])
                if table_paths:
                    from PIL import Image
                    from io import BytesIO
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def download_table(path_info):
                        idx, path = path_info
                        try:
                            response = wasabi.client.get_object(
                                Bucket=wasabi.bucket,
                                Key=path
                            )
                            img = Image.open(BytesIO(response['Body'].read()))
                            return (idx, path, img)
                        except Exception as e:
                            return (idx, path, None)
                    
                    # Parallel download
                    tbl_items = list(enumerate(table_paths[:10]))
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        results = list(executor.map(download_table, tbl_items))
                    
                    # Process downloaded tables
                    for idx, tbl_path, img in results:
                        if img is not None:
                            try:
                                features = image_extractor.extract_all_features(img, doi, f"table_{idx}")
                                if features.cnn_vector is not None:
                                    # Build OCR data dict for table matching
                                    ocr_data = {
                                        "has_text": len(features.ocr_regions) > 0 if features.ocr_regions else False,
                                        "text_regions": features.ocr_regions if features.ocr_regions else [],
                                        "text_hash": features.text_hash if hasattr(features, 'text_hash') else '',
                                        "full_text": ' '.join([r.get('text', '') for r in features.ocr_regions]) if features.ocr_regions else ''
                                    }
                                    
                                    tables_points.append({
                                        'id': f"{doi}_table_{idx}",
                                        'vector': features.cnn_vector.tolist(),
                                        'payload': {
                                            "doi": doi,
                                            "paper_id": doi,
                                            "safe_id": safe_doi,
                                            "table_index": idx,
                                            "type": "table",
                                            "s3_path": tbl_path,
                                            "perceptual_hashes": features.orientation_hashes,
                                            "original_id": f"{doi}_table_{idx}",
                                            "ocr_data": ocr_data,
                                            "ocr_regions": features.ocr_regions if features.ocr_regions else []
                                        }
                                    })
                                    table_count += 1
                            except Exception as e:
                                logger.debug(f"Failed to process table: {e}")
                
                total_processed += 1
                total_figures += fig_count
                total_tables += table_count
                
                # Progress
                if (paper_idx + 1) % 10 == 0 or paper_idx == len(batch_data) - 1:
                    print(f"    [{paper_idx+1:3d}/{len(batch_data)}] Processed {total_processed} papers | {total_figures} figs, {total_tables} tables")
                
            except Exception as e:
                total_failed += 1
                logger.debug(f"Failed to process paper: {e}")
        
        # Batch upload to Qdrant
        print(f"  📤 Uploading to Qdrant...")
        upload_start = time.time()
        
        if figures_points:
            qdrant.upsert_batch("bio_figures", figures_points)
        
        if tables_points:
            qdrant.upsert_batch("bio_tables", tables_points)
        
        upload_time = time.time() - upload_start
        print(f"  ✓ Uploaded in {upload_time:.1f}s: {len(figures_points)} figures, {len(tables_points)} tables")
        
        # Clear memory
        del batch_data
        del figures_points
        del tables_points
        gc.collect()
        print(f"  🧹 Memory cleared")
    
    # ===== FINAL SUMMARY =====
    total_time = time.time() - start_time
    total_min = int(total_time // 60)
    total_sec = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"🎉 MISSING IMAGES EMBEDDING COMPLETE!")
    print(f"{'='*60}")
    print(f"  ✅ Papers processed:    {total_processed:,}")
    print(f"  ❌ Papers failed:       {total_failed:,}")
    print(f"  {'─'*40}")
    print(f"  🖼️  Figures added:       {total_figures:,}")
    print(f"  📊 Tables added:        {total_tables:,}")
    print(f"  {'─'*40}")
    print(f"  ⏱️  Total time:          {total_min}m {total_sec}s")
    print(f"{'='*60}")


def step4_build_embeddings(
    extraction_results: List = None,
    categories: List[str] = None,
    batch_size: int = 32,
    force: bool = False,  # Force re-embed even if exists
    max_papers: Optional[int] = None  # Limit number of papers
):
    """
    Generate embeddings and store in proper locations:
    - Text embeddings → Qdrant (vectors for searching)
    - Image feature vectors → Qdrant (vectors for searching)
    - Actual image files → Wasabi S3 (or local if no Wasabi)
    
    Args:
        extraction_results: List of extraction results from step3 (optional)
        categories: Categories to load from Wasabi if no extraction_results
        batch_size: Batch size for embedding
        force: If True, re-embed even if DOI already exists in Qdrant
        max_papers: Maximum number of papers to process
    """
    print("\n" + "="*70)
    print("STEP 4: BUILD EMBEDDINGS & STORE")
    print("="*70)
    print("Architecture:")
    print("  - Text embeddings → Qdrant")
    print("  - Image vectors → Qdrant")
    print("  - Image files → Wasabi S3 (or local)")
    if force:
        print("  - Force mode: Re-embedding all papers")
    else:
        print("  - Skip mode: Skipping already embedded papers")
    if max_papers:
        print(f"  - Max papers: {max_papers}")
    
    # If no extraction results provided, try to load from Wasabi
    if not extraction_results and categories:
        print("\n📥 No extraction results in memory. Loading from Wasabi...")
        extraction_results = _load_extractions_from_wasabi(categories, max_papers=max_papers)
    
    # Apply max_papers limit if provided and results came from elsewhere
    if extraction_results and max_papers and len(extraction_results) > max_papers:
        print(f"📋 Limiting to {max_papers} papers (--max flag)")
        extraction_results = extraction_results[:max_papers]
    
    if not extraction_results:
        print("No extraction results to process!")
        print("\nOptions:")
        print("  1. Run --extract first (same session)")
        print("  2. Run --extract on Windows to save JSONs to Wasabi")
        print("  3. Check Wasabi has extractions/ folder")
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
        import uuid
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install sentence-transformers")
        return
    
    # ===== INITIALIZE STORAGE =====
    
    # Qdrant (for vectors)
    qdrant = None
    existing_dois = set()  # Track DOIs already in Qdrant
    try:
        from storage.qdrant_client import QdrantClient
        qdrant = QdrantClient()
        print("✓ Connected to Qdrant (vectors)")
        
        # Get existing DOIs from bio_abstracts collection (if not force mode)
        if not force:
            try:
                if qdrant.collection_exists("bio_abstracts"):
                    # Scroll through all points to get existing DOIs
                    scroll_result = qdrant.client.scroll(
                        collection_name="bio_abstracts",
                        limit=10000,
                        with_payload=True,
                        with_vectors=False
                    )
                    points, _ = scroll_result
                    for point in points:
                        if point.payload and 'doi' in point.payload:
                            existing_dois.add(point.payload['doi'])
                    if existing_dois:
                        print(f"✓ Found {len(existing_dois)} already embedded papers")
            except Exception as e:
                logger.debug(f"Could not get existing DOIs: {e}")
                
    except Exception as e:
        print(f"✗ Qdrant not available: {e}")
        print("  Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    
    # Wasabi S3 (for images)
    wasabi = None
    use_local_storage = True
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        # Test connection
        if wasabi.access_key and wasabi.secret_key:
            print("✓ Connected to Wasabi S3 (images)")
            use_local_storage = False
        else:
            print("✓ Wasabi not configured - using local storage for images")
    except Exception as e:
        print(f"✓ Using local storage for images: {e}")
    
    # Local fallback for images
    local_figures_dir = PROJECT_ROOT / "data" / "figures"
    local_tables_dir = PROJECT_ROOT / "data" / "tables"
    
    # ===== LOAD MODELS =====
    
    # Text embedding model
    print("\nLoading text embedding model (BGE-large-en-v1.5)...")
    text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    # Image feature extractor
    image_extractor = None
    try:
        from detection.image_feature_extractor import MultiMethodFeatureExtractor
        image_extractor = MultiMethodFeatureExtractor()
        print("✓ Image feature extractor loaded")
    except Exception as e:
        print(f"✗ Image extractor not available: {e}")
    
    # ===== PROCESS PAPERS =====
    
    total_papers = len(extraction_results)
    papers_to_process = total_papers - len([r for r in extraction_results if (r.get('doi') if isinstance(r, dict) else getattr(r, 'doi', None)) in existing_dois])
    
    print(f"\n{'='*60}")
    print(f"📊 EMBEDDING STATISTICS")
    print(f"{'='*60}")
    print(f"  Total papers in queue: {total_papers}")
    print(f"  Already embedded:      {len(existing_dois)}")
    print(f"  Papers to embed:       {papers_to_process}")
    print(f"{'='*60}\n")
    
    import time
    start_time = time.time()
    
    stored_papers = 0
    skipped_papers = 0
    stored_figures = 0
    stored_tables = 0
    stored_chunks = 0
    failed_papers = 0
    
    for idx, result in enumerate(extraction_results, 1):
        try:
            # Handle both object (from step3) and dict (from Wasabi JSON)
            is_dict = isinstance(result, dict)
            
            # Get DOI
            if is_dict:
                doi = result.get('doi') or result.get('paper_id')
            else:
                doi = getattr(result, 'doi', None) or getattr(result, 'paper_id', None)
            
            if not doi:
                doi = f"unknown_{stored_papers}"
            
            short_doi = doi.split('/')[-1] if '/' in doi else doi
            short_doi = short_doi[:30]
            
            # Skip if already embedded (unless force mode)
            if doi in existing_dois and not force:
                skipped_papers += 1
                # Only print every 50 skipped to reduce noise
                if skipped_papers <= 5 or skipped_papers % 50 == 0:
                    print(f"  [{idx:4d}/{total_papers}] {short_doi:<30} ⏭️  SKIPPED (exists)")
                    if skipped_papers == 5 and papers_to_process > 0:
                        print(f"  ... (skipping remaining already-embedded papers)")
                continue
            
            # Create safe ID for filenames (replace / and . with _)
            safe_doi = doi.replace('/', '_').replace('.', '_')
            
            # Get data based on format
            if is_dict:
                abstract = result.get('abstract', '') or result.get('full_text', '')[:1000] if result.get('full_text') else ''
                full_text = result.get('full_text', '')
                title = result.get('title', '')
                figure_paths = result.get('figure_paths', [])
                table_paths = result.get('table_paths', [])
            else:
                abstract = getattr(result, 'abstract', '') or (getattr(result, 'text', '')[:1000] if getattr(result, 'text', '') else '')
                full_text = getattr(result, 'full_text', '') or getattr(result, 'text', '')
                title = getattr(result, 'title', '')
                figure_paths = []  # Will process from result.figures
                table_paths = []   # Will process from result.tables
            
            chunk_count = 0
            fig_count = 0
            table_count = 0
            
            # ===== TEXT → QDRANT =====
            if qdrant:
                # Abstract embedding
                if abstract:
                    abstract_vector = text_model.encode(abstract).tolist()
                    
                    qdrant.upsert_point(
                        collection="bio_abstracts",
                        point_id=doi,  # Use DOI as point ID
                        vector=abstract_vector,
                        payload={
                            "doi": doi,
                            "paper_id": doi,
                            "safe_id": safe_doi,
                            "title": title,
                            "abstract": abstract[:500],
                            "server": "biorxiv"
                        }
                    )
                
                # Text chunks
                if full_text:
                    chunks = _create_chunks(full_text, chunk_size=200)
                    chunk_count = len(chunks)  # Track actual count
                    
                    # Warn if paper is unusually large
                    if chunk_count > 150:
                        print(f"      ⚠️  Large paper: {chunk_count} chunks")
                    
                    stored_chunks += chunk_count
                    for i, chunk in enumerate(chunks):  # No limit - process all
                        chunk_vector = text_model.encode(chunk).tolist()
                        qdrant.upsert_point(
                            collection="bio_fulltext_chunks",
                            point_id=f"{doi}_chunk_{i}",  # DOI + chunk index
                            vector=chunk_vector,
                            payload={
                                "doi": doi,
                                "paper_id": doi,
                                "chunk_index": i, 
                                "text": chunk
                            }
                        )
            
            # ===== FIGURES → WASABI + QDRANT =====
            # Handle figures from objects (step3 same session) or from Wasabi paths (JSON)
            if is_dict and figure_paths:
                # Load figures from Wasabi and embed
                for i, fig_path in enumerate(figure_paths):
                    try:
                        figure_id = f"{doi}_fig_{i}"
                        
                        # Download image from Wasabi
                        img_obj = wasabi.client.get_object(Bucket=wasabi.bucket, Key=fig_path)
                        from PIL import Image
                        from io import BytesIO
                        img = Image.open(BytesIO(img_obj['Body'].read()))
                        
                        # Store feature vectors → Qdrant
                        if qdrant and image_extractor:
                            features = image_extractor.extract_all_features(img, doi, f"fig_{i}")
                            if features.cnn_vector is not None:
                                qdrant.upsert_point(
                                    collection="bio_figures",
                                    point_id=figure_id,
                                    vector=features.cnn_vector.tolist(),
                                    payload={
                                        "doi": doi,
                                        "paper_id": doi,
                                        "safe_id": safe_doi,
                                        "figure_index": i,
                                        "type": "figure",
                                        "label": f"Figure {i+1}",
                                        "s3_path": fig_path,
                                        "perceptual_hashes": features.orientation_hashes
                                    }
                                )
                        stored_figures += 1
                        fig_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to process figure from Wasabi: {e}")
            
            elif not is_dict and hasattr(result, 'figures') and result.figures:
                for i, fig in enumerate(result.figures):
                    if fig.image is None:
                        continue
                    
                    figure_id = f"{doi}_fig_{i}"  # DOI + fig index
                    
                    # Store actual image → Wasabi S3 (or local)
                    if not use_local_storage and wasabi:
                        try:
                            s3_key = f"figures/{safe_doi}/fig_{i+1}.png"
                            wasabi.upload_image(fig.image, s3_key)
                        except Exception as e:
                            logger.debug(f"Wasabi upload failed, using local: {e}")
                            # Fallback to local
                            fig_dir = local_figures_dir / safe_doi
                            fig_dir.mkdir(parents=True, exist_ok=True)
                            fig.image.save(str(fig_dir / f"fig_{i}.png"))
                    else:
                        # Local storage
                        fig_dir = local_figures_dir / safe_doi
                        fig_dir.mkdir(parents=True, exist_ok=True)
                        if fig.image.mode in ('RGBA', 'P'):
                            fig.image.convert('RGB').save(str(fig_dir / f"fig_{i}.png"))
                        else:
                            fig.image.save(str(fig_dir / f"fig_{i}.png"))
                    
                    # Store feature vectors → Qdrant
                    if qdrant and image_extractor:
                        try:
                            features = image_extractor.extract_all_features(fig.image, doi, f"fig_{i}")
                            if features.cnn_vector is not None:
                                qdrant.upsert_point(
                                    collection="bio_figures",
                                    point_id=figure_id,
                                    vector=features.cnn_vector.tolist(),
                                    payload={
                                        "doi": doi,
                                        "paper_id": doi,
                                        "safe_id": safe_doi,
                                        "figure_index": i,
                                        "type": "figure",
                                        "label": getattr(fig, 'label', f"Figure {i+1}"),
                                        "s3_path": f"figures/{safe_doi}/fig_{i+1}.png",
                                        "perceptual_hashes": features.orientation_hashes
                                    }
                                )
                        except Exception as e:
                            logger.debug(f"Failed to embed figure: {e}")
                    
                    stored_figures += 1
                    fig_count += 1
            
            # ===== TABLES → WASABI + QDRANT =====
            # Handle tables from objects (step3 same session) or from Wasabi paths (JSON)
            if is_dict and table_paths:
                # Load tables from Wasabi and embed
                for i, tbl_path in enumerate(table_paths):
                    try:
                        table_id = f"{doi}_table_{i}"
                        
                        # Download image from Wasabi
                        img_obj = wasabi.client.get_object(Bucket=wasabi.bucket, Key=tbl_path)
                        from PIL import Image
                        from io import BytesIO
                        img = Image.open(BytesIO(img_obj['Body'].read()))
                        
                        # Store feature vectors → Qdrant
                        if qdrant and image_extractor:
                            features = image_extractor.extract_all_features(img, doi, f"table_{i}")
                            if features.cnn_vector is not None:
                                # Build OCR data dict for table matching
                                ocr_data = {
                                    "has_text": len(features.ocr_regions) > 0 if features.ocr_regions else False,
                                    "text_regions": features.ocr_regions if features.ocr_regions else [],
                                    "text_hash": features.text_hash if hasattr(features, 'text_hash') else '',
                                    "full_text": ' '.join([r.get('text', '') for r in features.ocr_regions]) if features.ocr_regions else ''
                                }
                                
                                qdrant.upsert_point(
                                    collection="bio_tables",
                                    point_id=table_id,
                                    vector=features.cnn_vector.tolist(),
                                    payload={
                                        "doi": doi,
                                        "paper_id": doi,
                                        "safe_id": safe_doi,
                                        "table_index": i,
                                        "type": "table",
                                        "label": f"Table {i+1}",
                                        "s3_path": tbl_path,
                                        "perceptual_hashes": features.orientation_hashes,
                                        "ocr_data": ocr_data,
                                        "ocr_regions": features.ocr_regions if features.ocr_regions else []
                                    }
                                )
                        stored_tables += 1
                        table_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to process table from Wasabi: {e}")
            
            elif not is_dict and hasattr(result, 'tables') and result.tables:
                for i, table in enumerate(result.tables):
                    if not hasattr(table, 'image') or table.image is None:
                        continue
                    
                    table_id = f"{doi}_table_{i}"  # DOI + table index
                    
                    # Store actual image → Wasabi S3 (or local)
                    if not use_local_storage and wasabi:
                        try:
                            s3_key = f"tables/{safe_doi}/table_{i+1}.png"
                            wasabi.upload_image(table.image, s3_key)
                        except Exception as e:
                            logger.debug(f"Wasabi upload failed, using local: {e}")
                            tbl_dir = local_tables_dir / safe_doi
                            tbl_dir.mkdir(parents=True, exist_ok=True)
                            table.image.save(str(tbl_dir / f"table_{i}.png"))
                    else:
                        # Local storage
                        tbl_dir = local_tables_dir / safe_doi
                        tbl_dir.mkdir(parents=True, exist_ok=True)
                        if table.image.mode in ('RGBA', 'P'):
                            table.image.convert('RGB').save(str(tbl_dir / f"table_{i}.png"))
                        else:
                            table.image.save(str(tbl_dir / f"table_{i}.png"))
                    
                    # Store feature vectors → Qdrant
                    if qdrant and image_extractor:
                        try:
                            features = image_extractor.extract_all_features(table.image, doi, f"table_{i}")
                            if features.cnn_vector is not None:
                                # Build OCR data dict for table matching
                                ocr_data = {
                                    "has_text": len(features.ocr_regions) > 0 if features.ocr_regions else False,
                                    "text_regions": features.ocr_regions if features.ocr_regions else [],
                                    "text_hash": features.text_hash if hasattr(features, 'text_hash') else '',
                                    "full_text": ' '.join([r.get('text', '') for r in features.ocr_regions]) if features.ocr_regions else ''
                                }
                                
                                qdrant.upsert_point(
                                    collection="bio_tables",
                                    point_id=table_id,
                                    vector=features.cnn_vector.tolist(),
                                    payload={
                                        "doi": doi,
                                        "paper_id": doi,
                                        "safe_id": safe_doi,
                                        "table_index": i,
                                        "type": "table",
                                        "label": getattr(table, 'label', f"Table {i+1}"),
                                        "s3_path": f"tables/{safe_doi}/table_{i+1}.png",
                                        "perceptual_hashes": features.orientation_hashes,
                                        "ocr_data": ocr_data,
                                        "ocr_regions": features.ocr_regions if features.ocr_regions else []
                                    }
                                )
                        except Exception as e:
                            logger.debug(f"Failed to embed table: {e}")
                    
                    stored_tables += 1
                    table_count += 1
            
            stored_papers += 1
            
            # Calculate progress and ETA
            elapsed = time.time() - start_time
            papers_done = stored_papers
            if papers_done > 0:
                avg_time = elapsed / papers_done
                remaining = papers_to_process - papers_done
                eta_seconds = remaining * avg_time
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s" if remaining > 0 else "Done!"
            else:
                eta_str = "Calculating..."
            
            progress_pct = (stored_papers / papers_to_process * 100) if papers_to_process > 0 else 100
            
            print(f"  [{idx:4d}/{total_papers}] {short_doi:<30} ✅ {chunk_count} chunks, {fig_count} figs, {table_count} tables | Progress: {stored_papers}/{papers_to_process} ({progress_pct:.1f}%) | {eta_str}")
            
        except Exception as e:
            failed_papers += 1
            print(f"  [{idx:4d}/{total_papers}] {short_doi:<30} ❌ {str(e)[:30]}")
            logger.debug(f"Failed to process DOI {doi}: {e}")
    
    # Calculate total time
    total_time = time.time() - start_time
    total_min = int(total_time // 60)
    total_sec = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"📊 EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Papers embedded:     {stored_papers}")
    print(f"  ⏭️  Papers skipped:      {skipped_papers}")
    print(f"  ❌ Papers failed:       {failed_papers}")
    print(f"  {'─'*40}")
    print(f"  📝 Text chunks stored:  {stored_chunks:,}")
    print(f"  🖼️  Figures embedded:    {stored_figures}")
    print(f"  📊 Tables embedded:     {stored_tables}")
    print(f"  {'─'*40}")
    print(f"  ⏱️  Total time:          {total_min}m {total_sec}s")
    if stored_papers > 0:
        avg_per_paper = total_time / stored_papers
        print(f"  ⚡ Avg time/paper:      {avg_per_paper:.1f}s")
    print(f"{'='*60}")
    print(f"\n📍 Storage locations:")
    print(f"  Text vectors:  Qdrant (bio_abstracts, bio_fulltext_chunks)")
    print(f"  Image vectors: Qdrant (bio_figures, bio_tables)")
    if use_local_storage:
        print(f"  Image files:   Local ({local_figures_dir})")
    else:
        print(f"  Image files:   Wasabi S3")
    
    if skipped_papers > 0:
        print(f"\n💡 Tip: Use --force to re-embed already processed papers")


def _create_chunks(text: str, chunk_size: int = 200, overlap: float = 0.5, filter_sections: bool = True) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Number of words per chunk
        overlap: Overlap ratio between chunks (0.5 = 50%)
        filter_sections: If True, remove references, authors, etc.
        
    Returns:
        List of text chunks
    """
    # Filter out non-content sections if requested
    if filter_sections:
        try:
            from utils.text_filter import TextSectionFilter
            filter = TextSectionFilter()
            result = filter.filter_text(text)
            text = result.filtered_text
        except ImportError:
            pass  # Filter not available, use original text
    
    words = text.split()
    chunks = []
    
    step = int(chunk_size * (1 - overlap))
    
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) >= chunk_size // 2:  # Skip tiny chunks
            chunks.append(chunk)
    
    return chunks


# =============================================================================
# STEP 5: USER SUBMISSION → PLAGIARISM REPORT
# =============================================================================

def step5_check_plagiarism(
    input_file: str,
    output_report: str = None,
    mode: str = 'full'  # 'full', 'text', or 'images'
):
    """
    Check a user-submitted document for plagiarism.
    
    Args:
        input_file: Path to PDF file
        output_report: Path to save JSON report
        mode: 'full' (text + images), 'text' (text only), 'images' (images only)
    
    Supports: PDF files
    """
    mode_names = {'full': 'FULL CHECK', 'text': 'TEXT ONLY', 'images': 'IMAGES ONLY'}
    
    print("\n" + "="*70)
    print(f"STEP 5: PLAGIARISM CHECK ({mode_names.get(mode, 'FULL CHECK')})")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Mode: {mode}")
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return None
    
    if input_path.suffix.lower() != '.pdf':
        print(f"Error: Only PDF files are supported. Got: {input_path.suffix}")
        return None
    
    # Initialize clients
    print("\nInitializing detector...")
    
    # Qdrant client
    qdrant = None
    try:
        from storage.qdrant_client import QdrantClient
        qdrant = QdrantClient()
        if qdrant.client:
            print("  ✓ Connected to Qdrant")
        else:
            print("  ✗ Could not connect to Qdrant")
    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
    
    # Wasabi client (optional)
    wasabi = None
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        if wasabi.access_key:
            print("  ✓ Connected to Wasabi S3")
        else:
            print("  ⚠ Wasabi not configured (image retrieval disabled)")
            wasabi = None
    except Exception as e:
        print(f"  ⚠ Wasabi not available: {e}")
        wasabi = None
    
    # Embedding model
    embedding_model = None
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding model...")
        embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        print("  ✓ Embedding model loaded")
    except Exception as e:
        print(f"  ⚠ Embedding model not available: {e}")
    
    # Initialize detector
    from detection.world_class_detector import WorldClassDetector, DetectorConfig
    
    config = DetectorConfig(
        run_splice_detection=False,  # Disable for faster checks
        run_ai_detection=False,
        max_text_matches=100,  # Show more text matches
        max_image_matches=50,  # Show more image matches
        text_moderate_threshold=0.4  # Lower threshold to catch more matches
    )
    
    detector = WorldClassDetector(
        qdrant_client=qdrant,
        wasabi_client=wasabi,
        embedding_model=embedding_model,
        config=config
    )
    
    print("\nAnalyzing paper for plagiarism...")
    if mode == 'text':
        print("  - Extracting text")
        print("  - Searching for text matches in Qdrant")
        print("  (Skipping image analysis)")
    elif mode == 'images':
        print("  - Extracting figures and tables")
        print("  - Searching for image matches in Qdrant")
        print("  (Skipping text analysis)")
    else:
        print("  - Extracting text and figures")
        print("  - Searching for text matches in Qdrant")
        print("  - Searching for image matches in Qdrant")
    
    try:
        report = detector.analyze_paper(str(input_path), mode=mode)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Print summary
    print("\n" + "="*70)
    print("PLAGIARISM REPORT")
    print("="*70)
    print(f"Paper: {report.query_title[:60] if report.query_title else 'Unknown'}...")
    print(f"Analyzed at: {report.analyzed_at}")
    print(f"Mode: {mode}")
    print(f"\nOverall Score: {report.overall_score * 100:.1f}%")
    print(f"Risk Level: {report.risk_level}")
    
    if mode in ['full', 'text']:
        print(f"\nText Plagiarism: {report.text_plagiarism_score * 100:.1f}%")
        print(f"  Matches found: {len(report.text_matches)}")
    
    if mode in ['full', 'images']:
        print(f"\nImage Plagiarism: {report.image_plagiarism_score * 100:.1f}%")
        print(f"  Matches found: {len(report.image_matches)}")
        print(f"\nIntegrity Issues: {len(report.integrity_issues)}")
    
    # Top text matches
    if report.text_matches and mode in ['full', 'text']:
        print("\nTop Text Matches:")
        for match in report.text_matches[:5]:
            source = getattr(match, 'source_title', None) or getattr(match, 'source_doi', 'Unknown')
            score = getattr(match, 'similarity_score', 0) or getattr(match, 'score', 0)
            if isinstance(source, str) and len(source) > 50:
                source = source[:50] + "..."
            print(f"  - {source} ({score * 100:.1f}%)")
    
    # Top image matches
    if report.image_matches:
        print("\nTop Image Matches:")
        for match in report.image_matches[:5]:
            source_doi = getattr(match, 'source_doi', 'Unknown')
            source_label = getattr(match, 'source_label', '')
            score = getattr(match, 'combined_score', 0) or getattr(match, 'score', 0)
            print(f"  - {source_doi} {source_label} ({score * 100:.1f}%)")
    
    # Integrity issues
    if report.integrity_issues:
        print("\nIntegrity Issues Detected:")
        for issue in report.integrity_issues[:5]:
            issue_type = getattr(issue, 'issue_type', 'Unknown')
            confidence = getattr(issue, 'confidence', 0)
            print(f"  - {issue_type} (confidence: {confidence * 100:.1f}%)")
    
    # Save report
    if output_report:
        output_path = Path(output_report)
        try:
            # Convert report to dict
            if hasattr(report, 'to_dict'):
                report_dict = report.to_dict()
            else:
                from dataclasses import asdict
                report_dict = asdict(report)
            
            with open(output_path, 'w') as f:
                import json
                json.dump(report_dict, f, indent=2, default=str)
            print(f"\nReport saved to: {output_path}")
        except Exception as e:
            print(f"\nCould not save report: {e}")
    
    return report


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def run_complete_pipeline(
    categories: List[str],
    max_papers: Optional[int] = None,
    skip_index: bool = False,
    skip_download: bool = False,
    skip_extract: bool = False,
    skip_embed: bool = False,
    workers: int = 4,
    from_wasabi: bool = False
):
    """
    Run the complete pipeline from start to finish.
    """
    print("\n" + "="*70)
    print("COMPLETE PLAGIARISM DETECTION PIPELINE")
    print("="*70)
    print(f"Categories: {categories}")
    print(f"Max papers: {max_papers or 'all'}")
    print(f"Workers: {workers}")
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Build/Update Index
    if not skip_index:
        step1_build_index(update_only=True, days=7)
    else:
        print("\n[Skipping Step 1: Index]")
    
    # Step 2: Download
    if not skip_download:
        step2_download_papers(categories, max_papers, workers=workers)
    else:
        print("\n[Skipping Step 2: Download]")
    
    # Step 3: Extract
    if not skip_extract:
        results = step3_extract_content(
            categories,
            from_wasabi=from_wasabi,
            workers=workers,
            max_papers=max_papers
        )
    else:
        print("\n[Skipping Step 3: Extract]")
        results = []
    
    # Step 4: Embed
    if not skip_embed and results:
        step4_build_embeddings(results, max_papers=max_papers)
    else:
        print("\n[Skipping Step 4: Embed]")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Finished at: {datetime.now()}")
    print("="*70)
    print("\nYou can now check documents with:")
    print("  python run_system.py --check path/to/document.pdf")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete Plagiarism Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMPLETE WORKFLOW:
==================

1. First-time setup (run once):
   python run_system.py --setup --categories "cancer biology"

2. Daily updates (run via cron):
   python run_system.py --update

3. Check a document:
   python run_system.py --check document.pdf --report report.json

INDIVIDUAL STEPS:
=================

Step 1 - Build index:
   python run_system.py --build-index

Step 2 - Download papers:
   python run_system.py --download --categories "cancer biology" --max 100

Step 3 - Extract content:
   python run_system.py --extract --categories "cancer biology"

Step 4 - Build embeddings:
   python run_system.py --embed --categories "cancer biology"

Step 5 - Check document:
   python run_system.py --check document.pdf
        """
    )
    
    # Main operations
    parser.add_argument("--setup", action="store_true",
                        help="Run complete first-time setup")
    parser.add_argument("--update", action="store_true",
                        help="Run daily update (index + new papers)")
    parser.add_argument("--check", type=str, metavar="FILE",
                        help="Check a document for plagiarism (full check)")
    parser.add_argument("--check-text", type=str, metavar="FILE",
                        help="Check text similarity only (faster)")
    parser.add_argument("--check-images", type=str, metavar="FILE",
                        help="Check image/table similarity only")
    
    # Individual steps
    parser.add_argument("--build-index", action="store_true",
                        help="Step 1: Build category index")
    parser.add_argument("--download", action="store_true",
                        help="Step 2: Download from S3")
    parser.add_argument("--extract", action="store_true",
                        help="Step 3: Extract content")
    parser.add_argument("--embed", action="store_true",
                        help="Step 4: Build ALL embeddings (text + images)")
    parser.add_argument("--embed-text", action="store_true",
                        help="Step 4A: Embed TEXT only (faster, skip images)")
    parser.add_argument("--embed-images", action="store_true",
                        help="Step 4B: Embed IMAGES only (skip text)")
    parser.add_argument("--smart-embed", action="store_true",
                        help="Step 4: SMART embedding (memory efficient, parallel GPU, for large datasets)")
    parser.add_argument("--embed-missing-images", action="store_true",
                        help="Step 4C: Embed ONLY missing images (for papers that have text but no images)")
    
    # Options
    parser.add_argument("--categories", nargs="+", default=["cancer biology"],
                        help="Categories to process")
    parser.add_argument("--month", type=str,
                        help="Process specific month (e.g., April_2019)")
    parser.add_argument("--server", type=str, choices=['biorxiv', 'medrxiv'],
                        help="Process specific server only")
    parser.add_argument("--max", type=int,
                        help="Maximum papers to process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for smart-embed (default 500)")
    parser.add_argument("--gpu-batch", type=int, default=32,
                        help="GPU batch size for parallel embedding (default 32)")
    parser.add_argument("--report", type=str,
                        help="Output path for plagiarism report")
    parser.add_argument("--days", type=int, default=7,
                        help="Days to look back for updates")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if file exists")
    parser.add_argument("--from-wasabi", action="store_true",
                        help="Stream PDFs from Wasabi S3 (skip local files)")
    parser.add_argument("--get-file", type=str,
                        help="Download a file from Wasabi S3 (e.g., figures/biorxiv/April_2019/uuid_.../fig_1.png)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path for downloaded file (default: filename from S3 key)")
    
    # Skip options
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    
    args = parser.parse_args()
    
    # Execute
    if args.get_file:
        # Download a single file from Wasabi S3
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        
        s3_key = args.get_file
        output_path = args.output if args.output else s3_key.split("/")[-1]
        
        print(f"Downloading: {s3_key}")
        print(f"Output: {output_path}")
        
        success = wasabi.download_file(s3_key, output_path)
        if success:
            print(f"✓ Downloaded successfully: {output_path}")
        else:
            print(f"✗ Download failed")
    
    elif args.setup:
        run_complete_pipeline(
            categories=args.categories,
            max_papers=args.max,
            skip_index=args.skip_index,
            skip_download=args.skip_download,
            skip_extract=args.skip_extract,
            skip_embed=args.skip_embed,
            workers=args.workers,
            from_wasabi=args.from_wasabi
        )
    
    elif args.update:
        step1_build_index(update_only=True, days=args.days)
        step2_download_papers(args.categories, args.max, args.workers, force=args.force)
        results = step3_extract_content(
            args.categories,
            from_wasabi=args.from_wasabi,
            workers=args.workers,
            max_papers=args.max
        )
        if results:
            step4_build_embeddings(
                results, 
                force=args.force,
                max_papers=args.max
            )
    
    elif args.check:
        step5_check_plagiarism(args.check, args.report, mode='full')
    
    elif args.check_text:
        step5_check_plagiarism(args.check_text, args.report, mode='text')
    
    elif args.check_images:
        step5_check_plagiarism(args.check_images, args.report, mode='images')
    
    elif args.build_index:
        step1_build_index(update_only=False)
    
    elif args.download:
        step2_download_papers(args.categories, args.max, args.workers, force=args.force)
    
    elif args.extract:
        step3_extract_content(
            args.categories, 
            from_wasabi=args.from_wasabi,
            workers=args.workers,
            max_papers=args.max,
            month=args.month,
            server=args.server
        )
    
    elif args.embed:
        # Try to load from Wasabi JSONs first (for Colab workflow)
        # If not found, fall back to extract
        step4_build_embeddings(
            extraction_results=None,  # Will try to load from Wasabi
            categories=args.categories,
            force=args.force,
            max_papers=args.max
        )
    
    elif args.embed_text:
        # TEXT ONLY embedding
        step4_smart_embed(
            categories=args.categories,
            batch_size=args.batch_size,
            gpu_batch=args.gpu_batch,
            force=args.force,
            max_papers=args.max,
            embed_mode='text',
            month=args.month,
            server=args.server
        )
    
    elif args.embed_images:
        # IMAGES ONLY embedding
        step4_smart_embed(
            categories=args.categories,
            batch_size=args.batch_size,
            gpu_batch=args.gpu_batch,
            force=args.force,
            max_papers=args.max,
            embed_mode='images',
            month=args.month,
            server=args.server
        )
    
    elif args.smart_embed:
        # SMART embedding - memory efficient, parallel GPU (ALL)
        step4_smart_embed(
            categories=args.categories,
            batch_size=args.batch_size,
            gpu_batch=args.gpu_batch,
            force=args.force,
            max_papers=args.max,
            embed_mode='all',
            month=args.month,
            server=args.server
        )
    
    elif args.embed_missing_images:
        # Embed ONLY missing images
        step4_embed_missing_images(
            categories=args.categories,
            batch_size=args.batch_size,
            max_papers=args.max
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
