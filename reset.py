#!/usr/bin/env python3
"""
UyarAI Reset & Fresh Start
===========================
Delete all data and start fresh.

Usage:
    python reset.py --all              # Delete everything (Wasabi + Qdrant + local)
    python reset.py --wasabi           # Delete Wasabi only
    python reset.py --qdrant           # Delete Qdrant collections only
    python reset.py --local            # Delete local data only
    python reset.py --dry-run          # Show what would be deleted
"""

import os
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_wasabi_client():
    """Get Wasabi S3 client with correct region"""
    import boto3
    
    endpoint = os.getenv('WASABI_ENDPOINT', 'https://s3.us-east-1.wasabisys.com')
    
    # Extract region from endpoint
    # e.g., https://s3.ap-southeast-1.wasabisys.com -> ap-southeast-1
    region = 'us-east-1'
    if 'wasabisys.com' in endpoint:
        parts = endpoint.replace('https://', '').replace('http://', '').split('.')
        if len(parts) >= 2 and parts[0] == 's3':
            region = parts[1]
    
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv('WASABI_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('WASABI_SECRET_KEY'),
        region_name=region
    )


def delete_wasabi(dry_run: bool = False):
    """Delete all data from Wasabi S3"""
    print("\n" + "="*50)
    print("🗑️  DELETING WASABI DATA")
    print("="*50)
    
    try:
        bucket = os.getenv('WASABI_BUCKET')
        if not bucket:
            print("   ❌ WASABI_BUCKET not set")
            return
        
        s3 = get_wasabi_client()
        
        # List all objects with pagination
        prefixes = ['index/', 'papers/', 'figures/', 'tables/', 'extracted/']
        total_deleted = 0
        
        for prefix in prefixes:
            try:
                deleted_count = 0
                
                # Use paginator to get ALL objects (not just 1000)
                paginator = s3.get_paginator('list_objects_v2')
                
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if 'Contents' not in page:
                        continue
                    
                    objects = page['Contents']
                    
                    if dry_run:
                        deleted_count += len(objects)
                    else:
                        # Delete in batches of 1000
                        delete_keys = [{'Key': obj['Key']} for obj in objects]
                        
                        if delete_keys:
                            s3.delete_objects(
                                Bucket=bucket,
                                Delete={'Objects': delete_keys}
                            )
                            deleted_count += len(delete_keys)
                            
                            if deleted_count % 1000 == 0:
                                print(f"   🗑️  {prefix}: deleted {deleted_count} files...")
                
                if deleted_count > 0:
                    if dry_run:
                        print(f"   📁 {prefix}: {deleted_count} files (would delete)")
                    else:
                        print(f"   ✅ {prefix}: deleted {deleted_count} files")
                    total_deleted += deleted_count
                else:
                    print(f"   📁 {prefix}: empty")
                    
            except Exception as e:
                print(f"   ⚠️ {prefix}: {e}")
        
        if not dry_run:
            print(f"\n   Total deleted from Wasabi: {total_deleted} files")
        
    except ImportError:
        print("   ❌ boto3 not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def delete_qdrant(dry_run: bool = False):
    """Delete all Qdrant collections"""
    print("\n" + "="*50)
    print("🗑️  DELETING QDRANT COLLECTIONS")
    print("="*50)
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = ['bio_fulltext_chunks', 'bio_figures', 'bio_tables', 'bio_abstracts']
        
        for col in collections:
            try:
                info = client.get_collection(col)
                count = info.points_count or 0
                
                if dry_run:
                    print(f"   📦 {col}: {count:,} vectors (would delete)")
                else:
                    client.delete_collection(col)
                    print(f"   ✅ {col}: deleted ({count:,} vectors)")
                    
            except Exception as e:
                print(f"   ⚠️ {col}: not found or error")
        
    except ImportError:
        print("   ❌ qdrant-client not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def delete_local(dry_run: bool = False):
    """Delete local data directory"""
    print("\n" + "="*50)
    print("🗑️  DELETING LOCAL DATA")
    print("="*50)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("   📁 data/: not found")
        return
    
    # Count files
    total_files = sum(1 for _ in data_dir.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    if dry_run:
        print(f"   📁 data/: {total_files} files, {size_mb:.1f} MB (would delete)")
    else:
        shutil.rmtree(data_dir)
        data_dir.mkdir(exist_ok=True)
        print(f"   ✅ data/: deleted {total_files} files ({size_mb:.1f} MB)")


def sync_from_wasabi():
    """Download all data from Wasabi"""
    print("\n" + "="*50)
    print("📥 DOWNLOADING FROM WASABI")
    print("="*50)
    
    try:
        bucket = os.getenv('WASABI_BUCKET')
        if not bucket:
            print("   ❌ WASABI_BUCKET not set")
            return
        
        s3 = get_wasabi_client()
        
        # Download all objects
        paginator = s3.get_paginator('list_objects_v2')
        total_downloaded = 0
        
        print(f"\n   Bucket: {bucket}")
        
        for page in paginator.paginate(Bucket=bucket):
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                key = obj['Key']
                local_path = Path("data") / key
                
                # Create directory
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download
                s3.download_file(bucket, key, str(local_path))
                total_downloaded += 1
                
                if total_downloaded % 100 == 0:
                    print(f"   Downloaded: {total_downloaded} files...")
        
        print(f"\n   ✅ Total downloaded: {total_downloaded} files")
        
    except ImportError:
        print("   ❌ boto3 not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def sync_to_wasabi():
    """Upload all local data to Wasabi"""
    print("\n" + "="*50)
    print("📤 UPLOADING TO WASABI")
    print("="*50)
    
    try:
        bucket = os.getenv('WASABI_BUCKET')
        if not bucket:
            print("   ❌ WASABI_BUCKET not set")
            return
        
        s3 = get_wasabi_client()
        
        data_dir = Path("data")
        if not data_dir.exists():
            print("   ❌ No local data/ directory")
            return
        
        total_uploaded = 0
        
        for local_path in data_dir.rglob("*"):
            if local_path.is_file():
                key = str(local_path.relative_to(data_dir))
                s3.upload_file(str(local_path), bucket, key)
                total_uploaded += 1
                
                if total_uploaded % 100 == 0:
                    print(f"   Uploaded: {total_uploaded} files...")
        
        print(f"\n   ✅ Total uploaded: {total_uploaded} files")
        
    except ImportError:
        print("   ❌ boto3 not installed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


def create_fresh_collections():
    print("\n" + "="*50)
    print("🆕 CREATING FRESH QDRANT COLLECTIONS")
    print("="*50)
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = {
            'bio_fulltext_chunks': 1024,  # BGE-large-en-v1.5 text embeddings
            'bio_figures': 2048,          # ResNet50 image embeddings
            'bio_tables': 2048,           # ResNet50 image embeddings
            'bio_abstracts': 1024,        # BGE-large-en-v1.5 text embeddings
        }
        
        for col, dim in collections.items():
            try:
                client.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
                print(f"   ✅ Created {col} (dim={dim})")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"   ⚠️ {col}: already exists")
                else:
                    print(f"   ❌ {col}: {e}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reset UyarAI Data")
    parser.add_argument("--all", action="store_true", help="Delete everything")
    parser.add_argument("--wasabi", action="store_true", help="Delete Wasabi only")
    parser.add_argument("--qdrant", action="store_true", help="Delete Qdrant only")
    parser.add_argument("--local", action="store_true", help="Delete local only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--create-collections", action="store_true", help="Create fresh Qdrant collections")
    parser.add_argument("--sync-from-wasabi", action="store_true", help="Download all from Wasabi")
    parser.add_argument("--sync-to-wasabi", action="store_true", help="Upload all to Wasabi")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n⚠️  DRY RUN - Nothing will be deleted\n")
    
    if args.sync_from_wasabi:
        sync_from_wasabi()
    
    elif args.sync_to_wasabi:
        sync_to_wasabi()
    
    elif args.all:
        if not args.dry_run:
            confirm = input("⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Cancelled.")
                return
        
        delete_wasabi(args.dry_run)
        delete_qdrant(args.dry_run)
        delete_local(args.dry_run)
        
        if not args.dry_run:
            create_fresh_collections()
            print("\n" + "="*50)
            print("✅ RESET COMPLETE!")
            print("="*50)
            print("\nNext steps:")
            print("   python run_system.py --build-index")
    
    elif args.wasabi:
        delete_wasabi(args.dry_run)
    
    elif args.qdrant:
        delete_qdrant(args.dry_run)
        if not args.dry_run:
            create_fresh_collections()
    
    elif args.local:
        delete_local(args.dry_run)
    
    elif args.create_collections:
        create_fresh_collections()
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("   python reset.py --dry-run --all       # See what would be deleted")
        print("   python reset.py --all                 # Delete everything & create fresh")
        print("   python reset.py --qdrant              # Reset Qdrant only")
        print("   python reset.py --sync-from-wasabi    # Download all from Wasabi")
        print("   python reset.py --sync-to-wasabi      # Upload all to Wasabi")


if __name__ == "__main__":
    main()
