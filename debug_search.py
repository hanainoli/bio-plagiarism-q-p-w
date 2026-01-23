#!/usr/bin/env python3
"""
Debug script to see what Qdrant returns for a specific query.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from storage.qdrant_client import QdrantClient

# The exact text from Discussion section that should match
QUERY_TEXT = """Our topological results are closely aligned to previous work on turtles and crocodilians (15, 44, 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification and the phylogenetic and spatial distribution of extinction risk"""

# What we expect to find
EXPECTED_TEXT = "Our topological results are closely aligned"

def main():
    print("=" * 70)
    print("DEBUG: What does Qdrant return for Discussion query?")
    print("=" * 70)
    
    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    qdrant = QdrantClient()
    
    # Embed query
    print(f"\nQuery text (first 100 chars): {QUERY_TEXT[:100]}...")
    vector = model.encode([QUERY_TEXT])[0]
    
    # Search with different thresholds
    for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
        print(f"\n{'=' * 70}")
        print(f"Searching with threshold={threshold}, limit=100")
        print("=" * 70)
        
        results = qdrant.search_chunks(vector, limit=100, score_threshold=threshold)
        
        print(f"Results returned: {len(results)}")
        
        # Check if expected text is in results
        found_expected = False
        for i, r in enumerate(results):
            text = r.payload.get('text', '')[:150]
            if EXPECTED_TEXT.lower() in text.lower():
                found_expected = True
                print(f"\n✅ FOUND at position {i+1}!")
                print(f"   Score: {r.score:.4f}")
                print(f"   Text: {text}...")
                break
        
        if not found_expected:
            print(f"\n❌ Expected text NOT FOUND in {len(results)} results")
            
            # Show top 5 results
            print("\nTop 5 results returned:")
            for i, r in enumerate(results[:5]):
                text = r.payload.get('text', '')[:100]
                paper_id = r.payload.get('paper_id', 'unknown')
                print(f"  {i+1}. Score={r.score:.4f} Paper={paper_id[:30]}...")
                print(f"     Text: {text}...")
    
    # Now try searching for the exact expected text
    print("\n" + "=" * 70)
    print("Searching for EXACT expected text...")
    print("=" * 70)
    
    exact_query = "Our topological results are closely aligned to previous work on turtles and crocodilians"
    exact_vector = model.encode([exact_query])[0]
    
    results = qdrant.search_chunks(exact_vector, limit=100, score_threshold=0.1)
    print(f"Results: {len(results)}")
    
    for i, r in enumerate(results[:10]):
        text = r.payload.get('text', '')[:150]
        paper_id = r.payload.get('paper_id', 'unknown')
        print(f"  {i+1}. Score={r.score:.4f}")
        print(f"     Paper: {paper_id}")
        print(f"     Text: {text}...")
        print()

if __name__ == "__main__":
    main()
