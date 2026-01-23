#!/usr/bin/env python3
"""
Quick debug: Check Qdrant chunk sizes directly.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient as QC
import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

# Connect to Qdrant
url = os.getenv('QDRANT_URL')
api_key = os.getenv('QDRANT_API_KEY')

print(f"Connecting to Qdrant...")
client = QC(url=url, api_key=api_key)

# Get some chunks from each paper
turtle_id = "uuid_0ef37b7b-6c48-1014-9d11-d0c8d178dc67"
cggbp1_id = "uuid_12f95942-6c13-1014-88fd-bc29ba074db8"

from qdrant_client.models import Filter, FieldCondition, MatchValue

print(f"\n{'='*70}")
print("TURTLE PAPER CHUNKS")
print('='*70)

results = client.scroll(
    collection_name="bio_fulltext_chunks",
    scroll_filter=Filter(
        must=[FieldCondition(key="paper_id", match=MatchValue(value=turtle_id))]
    ),
    limit=5,
    with_payload=True,
    with_vectors=False
)

for i, point in enumerate(results[0]):
    text = point.payload.get('text', '')
    tokens = tokenize(text)
    print(f"\n{i+1}. Chunk length: {len(text)} chars, {len(tokens)} tokens")
    print(f"   Text: {text[:150]}...")

print(f"\n{'='*70}")
print("CGGBP1 PAPER CHUNKS")
print('='*70)

results = client.scroll(
    collection_name="bio_fulltext_chunks",
    scroll_filter=Filter(
        must=[FieldCondition(key="paper_id", match=MatchValue(value=cggbp1_id))]
    ),
    limit=5,
    with_payload=True,
    with_vectors=False
)

for i, point in enumerate(results[0]):
    text = point.payload.get('text', '')
    tokens = tokenize(text)
    print(f"\n{i+1}. Chunk length: {len(text)} chars, {len(tokens)} tokens")
    print(f"   Text: {text[:150]}...")

print(f"\n{'='*70}")
print("KEY INSIGHT")
print('='*70)
print("If chunks are very long (1000+ tokens), they will have many 5-grams")
print("that can match common scientific phrases from ANY paper.")
