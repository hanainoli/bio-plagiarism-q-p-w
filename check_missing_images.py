#!/usr/bin/env python3
"""
Check which papers have text embeddings but no image embeddings.
This identifies papers that need image re-embedding.
"""

import os
import sys
from collections import defaultdict

def main():
    # Read .env
    env_vars = {}
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip().strip('"').strip("'")
    
    # Import qdrant
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        print("ERROR: qdrant-client not installed")
        print("Run: pip install qdrant-client")
        sys.exit(1)
    
    # Connect
    host = env_vars.get('QDRANT_HOST', env_vars.get('QDRANT_URL', ''))
    if host and not host.startswith('http'):
        host = f"https://{host}"
    api_key = env_vars.get('QDRANT_API_KEY', '')
    
    print(f"Connecting to: {host[:60]}...")
    client = QdrantClient(url=host, api_key=api_key, timeout=60)
    
    # Get all DOIs from text collection
    print("\n1. Getting DOIs from bio_fulltext_chunks...")
    text_dois = set()
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name='bio_fulltext_chunks',
                limit=1000,
                with_payload=['doi'],
                offset=offset
            )
            points, offset = result
            if not points:
                break
            for p in points:
                doi = p.payload.get('doi', '')
                if doi:
                    text_dois.add(doi)
            if offset is None:
                break
        print(f"   Found {len(text_dois)} unique DOIs with text")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Get all DOIs from figures collection  
    print("\n2. Getting DOIs from bio_figures...")
    figure_dois = set()
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name='bio_figures',
                limit=1000,
                with_payload=['doi'],
                offset=offset
            )
            points, offset = result
            if not points:
                break
            for p in points:
                doi = p.payload.get('doi', '')
                if doi:
                    figure_dois.add(doi)
            if offset is None:
                break
        print(f"   Found {len(figure_dois)} unique DOIs with figures")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Get all DOIs from tables collection
    print("\n3. Getting DOIs from bio_tables...")
    table_dois = set()
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name='bio_tables',
                limit=1000,
                with_payload=['doi'],
                offset=offset
            )
            points, offset = result
            if not points:
                break
            for p in points:
                doi = p.payload.get('doi', '')
                if doi:
                    table_dois.add(doi)
            if offset is None:
                break
        print(f"   Found {len(table_dois)} unique DOIs with tables")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Find papers with text but no images
    image_dois = figure_dois | table_dois
    missing_images = text_dois - image_dois
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Papers with text only: {len(missing_images)}")
    print(f"  Papers with text + images: {len(text_dois & image_dois)}")
    print(f"  Papers with images only: {len(image_dois - text_dois)}")
    
    if missing_images:
        print(f"\nPapers missing images (first 20):")
        for doi in sorted(missing_images)[:20]:
            print(f"  - {doi}")
        
        if len(missing_images) > 20:
            print(f"  ... and {len(missing_images) - 20} more")
    
    # Check specific paper
    target = "uuid_12f95942-6c13-1014-88fd-bc29ba074db8"
    print(f"\n{'='*60}")
    print(f"CGGBP1 paper status ({target}):")
    print(f"  Has text: {'YES' if target in text