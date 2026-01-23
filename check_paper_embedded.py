"""
Check if a paper is already embedded in Qdrant.
Helps detect repetitive/duplicate embedding.

Usage:
    python check_paper_embedded.py <paper_id>
    python check_paper_embedded.py uuid_12f95942-6c13-1014-88fd-bc29ba074db8
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env
    env_paths = [
        Path(__file__).parent / '.env',  # Same directory as script
        Path.cwd() / '.env',              # Current working directory
        Path(__file__).parent.parent / '.env',  # Parent directory
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ Loaded .env from: {env_path}")
            break
except ImportError:
    print("Warning: python-dotenv not installed, using system environment only")


def check_paper_embedding(paper_id: str):
    """Check if a paper is already embedded in Qdrant collections."""
    
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    
    # Connect to Qdrant - support both URL and HOST formats
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_host = os.getenv('QDRANT_HOST')
    qdrant_port = os.getenv('QDRANT_PORT', '6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    qdrant_https = os.getenv('QDRANT_HTTPS', 'false').lower() == 'true'
    
    # Build URL if not provided directly
    if not qdrant_url:
        if qdrant_host:
            # Remove https:// if already in host
            clean_host = qdrant_host.replace('https://', '').replace('http://', '')
            protocol = 'https' if qdrant_https else 'http'
            qdrant_url = f"{protocol}://{clean_host}:{qdrant_port}"
        else:
            print("❌ Neither QDRANT_URL nor QDRANT_HOST set in environment")
            print("   Please set in .env file:")
            print("   QDRANT_HOST=your-cluster.qdrant.io")
            print("   QDRANT_PORT=6333")
            print("   QDRANT_API_KEY=your-api-key")
            print("   QDRANT_HTTPS=true")
            return
    
    print(f"Checking paper: {paper_id}")
    print(f"Connecting to Qdrant: {qdrant_url[:60]}...")
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    print("✓ Connected to Qdrant\n")
    
    # Collections to check - use actual collection names
    collections = {
        'bio_fulltext_chunks': 'text chunks',
        'bio_figures': 'figures',
        'bio_tables': 'tables',
        'bio_abstracts': 'abstracts'
    }
    
    # Try multiple variations of the paper ID
    id_variations = [
        paper_id,                              # Original: uuid_12f95942...
        f"biorxiv_{paper_id}",                 # With prefix: biorxiv_uuid_12f95942...
        f"medrxiv_{paper_id}",                 # With prefix: medrxiv_uuid_12f95942...
        paper_id.replace("uuid_", ""),         # Without uuid_: 12f95942...
        paper_id.replace("uuid_", "10.1101/"), # As DOI: 10.1101/12f95942...
    ]
    
    print(f"  Searching with ID variations:")
    for var in id_variations:
        print(f"    - {var}")
    
    results = {}
    
    for collection_name, desc in collections.items():
        try:
            # Check if collection exists
            collections_list = client.get_collections()
            if not any(c.name == collection_name for c in collections_list.collections):
                results[collection_name] = {'exists': False, 'count': 0}
                continue
            
            # Search for paper in collection
            # Try both 'doi' and 'paper_id' fields with all ID variations
            count = 0
            matched_id = None
            
            for test_id in id_variations:
                if count > 0:
                    break
                    
                for field in ['doi', 'paper_id']:
                    try:
                        scroll_result = client.scroll(
                            collection_name=collection_name,
                            scroll_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key=field,
                                        match=models.MatchValue(value=test_id)
                                    )
                                ]
                            ),
                            limit=100,
                            with_payload=True
                        )
                        
                        points = scroll_result[0]
                        if points:
                            count = len(points)
                            matched_id = test_id
                            break
                    except Exception:
                        continue
            
            results[collection_name] = {'exists': True, 'count': count, 'matched_id': matched_id}
            
        except Exception as e:
            results[collection_name] = {'exists': False, 'count': 0, 'error': str(e)}
    
    # Print results
    print("=" * 60)
    print(f"EMBEDDING STATUS FOR: {paper_id}")
    print("=" * 60)
    
    total_vectors = 0
    
    for collection_name, data in results.items():
        count = data['count']
        total_vectors += count
        matched_id = data.get('matched_id')
        
        if count > 0:
            status = f"✅ {count} vectors"
            if matched_id and matched_id != paper_id:
                status += f" (as: {matched_id[:40]}...)"
        else:
            status = "❌ Not found"
        
        print(f"  {collections[collection_name]:15} : {status}")
    
    print("-" * 60)
    
    if total_vectors > 0:
        print(f"📊 RESULT: Paper IS embedded ({total_vectors} total vectors)")
        print(f"   Status: WILL BE SKIPPED on re-embed (unless --force)")
    else:
        print(f"📊 RESULT: Paper is NOT embedded")
        print(f"   Status: WILL BE EMBEDDED on next run")
    
    print("=" * 60)
    
    return results


def check_all_embedded_papers():
    """List all unique paper IDs in Qdrant."""
    
    from qdrant_client import QdrantClient
    
    # Connect to Qdrant - support both URL and HOST formats
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_host = os.getenv('QDRANT_HOST')
    qdrant_port = os.getenv('QDRANT_PORT', '6333')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    qdrant_https = os.getenv('QDRANT_HTTPS', 'false').lower() == 'true'
    
    if not qdrant_url:
        if qdrant_host:
            # Remove https:// if already in host
            clean_host = qdrant_host.replace('https://', '').replace('http://', '')
            protocol = 'https' if qdrant_https else 'http'
            qdrant_url = f"{protocol}://{clean_host}:{qdrant_port}"
        else:
            print("❌ Neither QDRANT_URL nor QDRANT_HOST set in environment")
            return set()
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    print("Fetching all embedded papers from Qdrant...")
    
    all_dois = set()
    
    for collection in ['bio_fulltext_chunks', 'bio_figures', 'bio_tables']:
        try:
            offset = None
            while True:
                scroll_result = client.scroll(
                    collection_name=collection,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = scroll_result
                
                for point in points:
                    if point.payload:
                        doi = point.payload.get('doi') or point.payload.get('paper_id')
                        if doi:
                            all_dois.add(doi)
                
                if offset is None:
                    break
        except Exception as e:
            print(f"  Error with {collection}: {e}")
    
    print(f"\n✓ Found {len(all_dois)} unique papers embedded\n")
    
    # Show sample
    print("Sample papers (first 20):")
    for i, doi in enumerate(sorted(all_dois)[:20]):
        print(f"  {i+1}. {doi}")
    
    if len(all_dois) > 20:
        print(f"  ... and {len(all_dois) - 20} more")
    
    return all_dois


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nChecking all embedded papers...")
        check_all_embedded_papers()
    elif sys.argv[1] == "--all":
        check_all_embedded_papers()
    else:
        paper_id = sys.argv[1]
        check_paper_embedding(paper_id)
