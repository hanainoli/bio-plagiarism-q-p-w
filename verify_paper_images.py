#!/usr/bin/env python3
"""
Verify if a paper has text AND images in Qdrant.
"""

import sys
import os

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("ERROR: pip install qdrant-client")
        sys.exit(1)
    
    host = os.getenv('QDRANT_HOST', os.getenv('QDRANT_URL', ''))
    if host and not host.startswith('http'):
        host = f"https://{host}"
    api_key = os.getenv('QDRANT_API_KEY', '')
    
    print(f"Connecting to Qdrant...")
    client = QdrantClient(url=host, api_key=api_key, timeout=120)
    
    paper_id = "uuid_12f95942-6c13-1014-88fd-bc29ba074db8"
    
    print(f"\nSearching for: {paper_id}")
    print("="*60)
    
    collections = [
        ('bio_fulltext_chunks', 'TEXT'),
        ('bio_figures', 'FIGURES'),
        ('bio_tables', 'TABLES')
    ]
    
    results = {}
    
    for collection, label in collections:
        print(f"\n{label} ({collection}):")
        print(f"  Scanning...", end=" ", flush=True)
        
        found = []
        offset = None
        scanned = 0
        
        try:
            while True:
                result = client.scroll(
                    collection_name=collection,
                    limit=500,
                    offset=offset,
                    with_payload=True
                )
                points, next_offset = result
                
                if not points:
                    break
                
                scanned += len(points)
                
                # Check each point
                for p in points:
                    doi = p.payload.get('doi', '')
                    if paper_id in doi:
                        found.append(p)
                
                # Show progress every 2000
                if scanned % 2000 == 0:
                    print(f"{scanned}...", end=" ", flush=True)
                
                offset = next_offset
                if offset is None:
                    break
            
            results[label] = len(found)
            
            if found:
                print(f"\n  ✓ FOUND: {len(found)} items (scanned {scanned})")
                for p in found[:2]:
                    print(f"    - ID: {p.id}")
                    print(f"      DOI: {p.payload.get('doi')}")
                    if label in ['FIGURES', 'TABLES']:
                        has_hash = bool(p.payload.get('perceptual_hashes'))
                        print(f"      perceptual_hashes: {'YES' if has_hash else 'NO'}")
                if len(found) > 2:
                    print(f"    ... and {len(found)-2} more")
            else:
                print(f"\n  ✗ NOT FOUND (scanned {scanned})")
                
        except Exception as e:
            print(f"\n  Error: {e}")
            results[label] = -1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    
    text = results.get('TEXT', 0)
    figs = results.get('FIGURES', 0)
    tbls = results.get('TABLES', 0)
    
    print(f"  Text chunks: {text}")
    print(f"  Figures:     {figs}")
    print(f"  Tables:      {tbls}")
    
    if text > 0 and figs == 0 and tbls == 0:
        print(f"\n⚠️  PROBLEM: Paper has TEXT but NO IMAGES!")
        print(f"   This is why image detection returns false positives.")
    elif text > 0 and (figs > 0 or tbls > 0):
        print(f"\n✓ Paper has TEXT + IMAGES - should work correctly")
    else:
        print(f"\n⚠️  Paper not found in database")

if __name__ == '__main__':
    main()
