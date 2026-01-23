#!/usr/bin/env python3
"""Check if Discussion content exists in Qdrant bio_fulltext_chunks"""

from storage.qdrant_client import QdrantClient
import re

qdrant = QdrantClient()

doi = 'uuid_0ef37b7b-6c48-1014-9d11-d0c8d178dc67'
print(f'Checking bio_fulltext_chunks for: {doi}')

# First, scroll ALL points and filter manually (no index required)
print('Scrolling through collection (this may take a moment)...')

chunks = []
offset = None
total_scanned = 0

while True:
    result = qdrant.client.scroll(
        collection_name='bio_fulltext_chunks',
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    points, offset = result
    total_scanned += len(points)
    
    # Filter for our DOI
    for p in points:
        if p.payload.get('doi') == doi or p.payload.get('paper_id') == doi:
            chunks.append(p.payload)
    
    if offset is None:
        break
    
    if total_scanned % 5000 == 0:
        print(f'  Scanned {total_scanned} points, found {len(chunks)} for this DOI...')

print(f'\nTotal scanned: {total_scanned}')
print(f'Found {len(chunks)} chunks for DOI: {doi}')

if not chunks:
    print('\n⚠️  No chunks found for this DOI!')
    exit()

# Sort chunks by index
chunks.sort(key=lambda x: x.get('chunk_index', 0))

# Check for specific Discussion phrases
discussion_phrases = [
    'Discussion',
    'Phylogeny and diversification',
    'topological results',
    'evolutionary durability',
    'morphological conservatism',
    'mass-extinction events'
]

print('\n' + '='*60)
print('SEARCHING FOR DISCUSSION PHRASES:')
print('='*60)

for phrase in discussion_phrases:
    found_chunks = [c for c in chunks if phrase.lower() in c.get('text', '').lower()]
    if found_chunks:
        print(f'\n✓ "{phrase}" found in {len(found_chunks)} chunk(s):')
        for c in found_chunks[:1]:
            text = c.get('text', '')
            # Find position of phrase
            pos = text.lower().find(phrase.lower())
            start = max(0, pos - 50)
            end = min(len(text), pos + len(phrase) + 100)
            print(f'  Chunk {c.get("chunk_index")}: ...{text[start:end]}...')
    else:
        print(f'\n✗ "{phrase}" NOT FOUND in any chunk!')

# Show all chunks for inspection
print('\n' + '='*60)
print('ALL 50 CHUNKS (first 150 chars each):')
print('='*60)
for c in chunks:
    text = c.get('text', '')[:150]
    print(f'\nChunk {c.get("chunk_index", "?")}:')
    print(f'  {text}...')
