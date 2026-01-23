#!/usr/bin/env python3
"""
Find papers that have BOTH text AND images in Qdrant.
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

# Step 1: Get all papers with TEXT
print("\n1. Scanning text chunks...")
text_papers = {}
offset = None

while True:
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
        if doi:
            if doi not in text_papers:
                text_papers[doi] = {'text_count': 0, 'sample': ''}
            text_papers[doi]['text_count'] += 1
            if not text_papers[doi]['sample']:
                text_papers[doi]['sample'] = p.payload.get('text', '')[:150]
    
    offset = next_offset
    if offset is None:
        break

print(f"   Found {len(text_papers)} papers with text")

# Step 2: Get all papers with FIGURES
print("\n2. Scanning figures...")
figure_papers = {}
offset = None

while True:
    result = client.scroll(
        collection_name='bio_figures',
        limit=500,
        offset=offset,
        with_payload=True
    )
    points, next_offset = result
    
    if not points:
        break
    
    for p in points:
        doi = p.payload.get('doi', '')
        if doi:
            if doi not in figure_papers:
                figure_papers[doi] = 0
            figure_papers[doi] += 1
    
    offset = next_offset
    if offset is None:
        break

print(f"   Found {len(figure_papers)} papers with figures")

# Step 3: Get all papers with TABLES
print("\n3. Scanning tables...")
table_papers = {}
offset = None

while True:
    result = client.scroll(
        collection_name='bio_tables',
        limit=500,
        offset=offset,
        with_payload=True
    )
    points, next_offset = result
    
    if not points:
        break
    
    for p in points:
        doi = p.payload.get('doi', '')
        if doi:
            if doi not in table_papers:
                table_papers[doi] = 0
            table_papers[doi] += 1
    
    offset = next_offset
    if offset is None:
        break

print(f"   Found {len(table_papers)} papers with tables")

# Step 4: Find papers with BOTH text AND images
print("\n" + "="*70)
print("PAPERS WITH TEXT + IMAGES (figures or tables):")
print("="*70)

complete_papers = []
for doi in text_papers:
    figs = figure_papers.get(doi, 0)
    tbls = table_papers.get(doi, 0)
    if figs > 0 or tbls > 0:
        complete_papers.append({
            'doi': doi,
            'text_count': text_papers[doi]['text_count'],
            'fig_count': figs,
            'table_count': tbls,
            'sample': text_papers[doi]['sample']
        })

# Sort by total content
complete_papers.sort(key=lambda x: x['text_count'] + x['fig_count'] + x['table_count'], reverse=True)

print(f"\nFound {len(complete_papers)} papers with BOTH text AND images!\n")

for i, p in enumerate(complete_papers[:15]):
    print(f"{i+1}. DOI: {p['doi']}")
    print(f"   Text: {p['text_count']} chunks | Figures: {p['fig_count']} | Tables: {p['table_count']}")
    print(f"   Sample: {p['sample'][:80]}...")
    print()

if len(complete_papers) >= 2:
    print("="*70)
    print("RECOMMENDED PAIR FOR TESTING:")
    print("="*70)
    print(f"\nPaper A: {complete_papers[0]['doi']}")
    print(f"Paper B: {complete_papers[1]['doi']}")
