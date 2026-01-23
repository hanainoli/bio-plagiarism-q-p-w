#!/usr/bin/env python3
"""
Pipeline Verification Script
Tests all components of the UyarAI plagiarism detection system.
"""

import os
import sys
from pathlib import Path

def check_imports():
    """Check all required imports"""
    print("\n" + "="*60)
    print("STEP 1: CHECKING IMPORTS")
    print("="*60)
    
    results = []
    
    # Core libraries
    core_libs = [
        ('boto3', 'AWS/Wasabi S3'),
        ('dotenv', 'Environment variables'),
        ('tqdm', 'Progress bars'),
    ]
    
    for lib, desc in core_libs:
        try:
            __import__(lib)
            print(f"  ✓ {lib} ({desc})")
            results.append((lib, True))
        except ImportError:
            print(f"  ✗ {lib} ({desc}) - pip install {lib}")
            results.append((lib, False))
    
    # ML libraries (optional in some environments)
    ml_libs = [
        ('torch', 'PyTorch'),
        ('sentence_transformers', 'Text embeddings'),
        ('PIL', 'Image processing'),
        ('fitz', 'PDF extraction (PyMuPDF)'),
    ]
    
    for lib, desc in ml_libs:
        try:
            __import__(lib)
            print(f"  ✓ {lib} ({desc})")
            results.append((lib, True))
        except ImportError:
            print(f"  ⚠ {lib} ({desc}) - optional/Colab-only")
            results.append((lib, False))
    
    return results


def check_env_file():
    """Check .env file configuration"""
    print("\n" + "="*60)
    print("STEP 2: CHECKING .env CONFIGURATION")
    print("="*60)
    
    env_path = Path('.env')
    if not env_path.exists():
        print("  ✗ .env file not found!")
        return False
    
    print("  ✓ .env file exists")
    
    # Load and check required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        ('WASABI_ACCESS_KEY', 'Wasabi access key'),
        ('WASABI_SECRET_KEY', 'Wasabi secret key'),
        ('WASABI_BUCKET', 'Wasabi bucket name'),
        ('QDRANT_HOST', 'Qdrant host URL'),
        ('QDRANT_API_KEY', 'Qdrant API key'),
    ]
    
    optional_vars = [
        ('AWS_ACCESS_KEY_ID', 'AWS access key (for S3 download)'),
        ('AWS_SECRET_ACCESS_KEY', 'AWS secret key'),
        ('DATABASE_URL', 'PostgreSQL URL'),
    ]
    
    all_ok = True
    for var, desc in required_vars:
        value = os.environ.get(var)
        if value:
            masked = value[:4] + '...' + value[-4:] if len(value) > 10 else '***'
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  ✗ {var}: NOT SET ({desc})")
            all_ok = False
    
    print("\n  Optional variables:")
    for var, desc in optional_vars:
        value = os.environ.get(var)
        if value:
            masked = value[:4] + '...' + value[-4:] if len(value) > 10 else '***'
            print(f"  ✓ {var}: {masked}")
        else:
            print(f"  - {var}: not set ({desc})")
    
    return all_ok


def check_wasabi_connection():
    """Check Wasabi S3 connection"""
    print("\n" + "="*60)
    print("STEP 3: CHECKING WASABI CONNECTION")
    print("="*60)
    
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        
        if not wasabi.access_key:
            print("  ✗ Wasabi not configured")
            return False
        
        print(f"  ✓ Wasabi client initialized")
        print(f"    Bucket: {wasabi.bucket}")
        print(f"    Region: {wasabi.region}")
        
        # Try to list objects
        response = wasabi.client.list_objects_v2(
            Bucket=wasabi.bucket,
            Prefix='pdfs/',
            MaxKeys=5
        )
        
        count = response.get('KeyCount', 0)
        print(f"  ✓ Connection successful (found {count} objects in pdfs/)")
        
        # Check structure
        prefixes = ['pdfs/', 'extractions/', 'figures/', 'tables/']
        for prefix in prefixes:
            response = wasabi.client.list_objects_v2(
                Bucket=wasabi.bucket,
                Prefix=prefix,
                MaxKeys=1
            )
            has_content = response.get('KeyCount', 0) > 0
            status = "✓" if has_content else "-"
            print(f"    {status} {prefix}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Wasabi error: {e}")
        return False


def check_qdrant_connection():
    """Check Qdrant connection"""
    print("\n" + "="*60)
    print("STEP 4: CHECKING QDRANT CONNECTION")
    print("="*60)
    
    try:
        from storage.qdrant_client import QdrantClient
        qdrant = QdrantClient()
        
        if not qdrant.client:
            print("  ✗ Qdrant not connected")
            return False
        
        print(f"  ✓ Qdrant client initialized")
        
        # Check collections
        collections = ['bio_abstracts', 'bio_fulltext_chunks', 'bio_figures', 'bio_tables']
        
        for coll in collections:
            try:
                exists = qdrant.collection_exists(coll)
                if exists:
                    # Get count
                    info = qdrant.client.get_collection(coll)
                    count = info.points_count
                    print(f"    ✓ {coll}: {count:,} vectors")
                else:
                    print(f"    - {coll}: not created yet")
            except Exception as e:
                print(f"    ✗ {coll}: error - {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
        return False


def check_extraction_files():
    """Check extraction files in Wasabi"""
    print("\n" + "="*60)
    print("STEP 5: CHECKING EXTRACTION FILES")
    print("="*60)
    
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        
        if not wasabi.access_key:
            print("  ✗ Wasabi not configured")
            return False
        
        # Check biorxiv extractions
        servers = ['biorxiv', 'medrxiv']
        for server in servers:
            prefix = f"extractions/{server}/"
            
            paginator = wasabi.client.get_paginator('list_objects_v2')
            count = 0
            months = set()
            
            for page in paginator.paginate(Bucket=wasabi.bucket, Prefix=prefix, MaxKeys=1000):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.json'):
                            count += 1
                            parts = obj['Key'].split('/')
                            if len(parts) > 2:
                                months.add(parts[2])
                if count >= 1000:  # Limit for quick check
                    break
            
            if count > 0:
                print(f"  ✓ {server}: {count:,}+ extraction files")
                print(f"    Months: {', '.join(sorted(months)[:5])}...")
            else:
                print(f"  - {server}: no extraction files yet")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def print_summary():
    """Print command summary"""
    print("\n" + "="*60)
    print("COMMAND REFERENCE")
    print("="*60)
    
    commands = """
DOWNLOAD (Local PC):
  python download_all_s3.py --workers 150

EXTRACT:
  python run_system.py --extract --from-wasabi --server biorxiv --month April_2019 --workers 16

EMBED TEXT:
  python run_system.py --embed-text --server biorxiv --month April_2019

EMBED IMAGES:
  python run_system.py --embed-images --server biorxiv --month April_2019

CHECK PLAGIARISM:
  python run_system.py --check document.pdf --report report.json
  python run_system.py --check-text document.pdf --report report.json
  python run_system.py --check-images document.pdf --report report.json

FLAGS:
  --force         Re-process everything (ignore skip checks)
  --max N         Limit to N papers
  --workers N     Parallel workers for extraction
"""
    print(commands)


def main():
    print("\n" + "="*60)
    print("UYARAI PIPELINE VERIFICATION")
    print("="*60)
    
    # Run checks
    import_results = check_imports()
    env_ok = check_env_file()
    wasabi_ok = check_wasabi_connection()
    qdrant_ok = check_qdrant_connection()
    extraction_ok = check_extraction_files()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_core_ok = all(ok for lib, ok in import_results if lib in ['boto3', 'dotenv', 'tqdm'])
    
    print(f"  Core imports:     {'✓ OK' if all_core_ok else '✗ FAIL'}")
    print(f"  Environment:      {'✓ OK' if env_ok else '✗ FAIL'}")
    print(f"  Wasabi:           {'✓ OK' if wasabi_ok else '✗ FAIL'}")
    print(f"  Qdrant:           {'✓ OK' if qdrant_ok else '✗ FAIL'}")
    print(f"  Extractions:      {'✓ OK' if extraction_ok else '- Check needed'}")
    
    if all_core_ok and env_ok and wasabi_ok and qdrant_ok:
        print("\n✅ Pipeline is ready to use!")
    else:
        print("\n⚠ Some issues need to be resolved.")
    
    print_summary()


if __name__ == "__main__":
    main()
