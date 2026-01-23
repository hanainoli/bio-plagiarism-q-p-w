#!/usr/bin/env python3
"""
Check actual Qdrant chunk content to understand phrase matching.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from storage.qdrant_client import QdrantClient
import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def count_phrase_matches(q_tokens, s_tokens):
    q_5grams = get_ngrams(q_tokens, 5)
    s_5grams = get_ngrams(s_tokens, 5)
    return len(q_5grams & s_5grams)

print("Loading model...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print("Connecting to Qdrant...")
qdrant = QdrantClient()

# Query: Turtle Abstract
query = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside a comprehensive"""

print(f"\nQuery: {query[:100]}...")
print(f"Query length: {len(query)} chars")

q_tokens = tokenize(query)
print(f"Query tokens: {len(q_tokens)}")

vector = model.encode([query])[0]
results = qdrant.search_chunks(vector, limit=10, score_threshold=0.3)

print(f"\n{'='*70}")
print("TOP 10 QDRANT RESULTS")
print('='*70)

for i, r in enumerate(results):
    text = r.payload.get('text', '')
    paper = r.payload.get('paper_id', '')
    
    s_tokens = tokenize(text)
    phrase_matches = count_phrase_matches(q_tokens, s_tokens)
    
    # Word overlap
    q_set = set(q_tokens)
    s_set = set(s_tokens)
    word_overlap = len(q_set & s_set) / len(q_set) if q_set else 0
    
    print(f"\n{i+1}. Paper: {paper[:45]}")
    print(f"   Semantic score: {r.score:.4f}")
    print(f"   Text length: {len(text)} chars, {len(s_tokens)} tokens")
    print(f"   Phrase matches (5-grams): {phrase_matches}")
    print(f"   Word overlap: {word_overlap:.1%}")
    print(f"   Text: {text[:80]}...")
    
    # Check if this is turtle or CGGBP1
    if 'turtle' in text.lower() or 'crocodil' in text.lower():
        print(f"   ✅ Contains turtle/crocodilian keywords")
    if 'cggbp1' in text.lower() or 'ctcf' in text.lower():
        print(f"   ⚠️ Contains CGGBP1/CTCF keywords")
