#!/usr/bin/env python3
"""
Find papers with similar content in Qdrant to test multi-source plagiarism detection.
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("ERROR: pip install qdrant-client")
    exit(1)

host = os.getenv('QDRANT_HOST', os.getenv('QDRANT_URL', ''))
if host and not host.startswith('http'):
    host = f"https://{host}"
api_key = os.getenv('QDRANT_API_KEY', '')

print("Connecting to Qdrant...")
client = QdrantClient(url=host, api_key=api_key, timeout=120)

# Get sample papers from database
print("\nScanning papers in database...")
papers = {}
offset = None
count = 0

while count < 5000:  # Scan up to 5000 chunks
    result = client.scroll(
        collection_name='bio_fulltext_chunks',
        limit=500,
        offset=offset,
        with_payload=True
    )
    points, next_offset = result
    
    if not points:
        break
    
    for p in points:
        doi = p.payload.get('doi', '')
        if doi and doi not in papers:
            # Get first chunk text as sample
            text = p.payload.get('text', '')[:200]
            papers[doi] = {
                'doi': doi,
                'sample_text': text,
                'chunk_count': 1
            }
        elif doi in papers:
            papers[doi]['chunk_count'] += 1
    
    count += len(points)
    offset = next_offset
    if offset is None:
        break

print(f"\nFound {len(papers)} unique papers")
print("\n" + "="*70)
print("SAMPLE PAPERS (sorted by chunk count):")
print("="*70)

# Sort by chunk count (more chunks = more content)
sorted_papers = sorted(papers.values(), key=lambda x: x['chunk_count'], reverse=True)

for i, p in enumerate(sorted_papers[:20]):
    print(f"\n{i+1}. DOI: {p['doi']}")
    print(f"   Chunks: {p['chunk_count']}")
    print(f"   Sample: {p['sample_text'][:100]}...")

print("\n" + "="*70)
print("TO CREATE A MULTI-SOURCE TEST PAPER:")
print("="*70)
print("""
1. Pick 2 papers from the list above
2. Create a new PDF combining:
   - First half from Paper A
   - Second half from Paper B
3. Run detection on this combined PDF
4. Report should show matches from BOTH sources!
""")
