#!/usr/bin/env python3
"""
Debug: Check lexical similarity between query and top Qdrant result.
"""

def tokenize(text):
    """Simple tokenization"""
    import re
    return re.findall(r'\b\w+\b', text.lower())

def word_overlap(query_tokens, source_tokens):
    """Calculate word overlap percentage"""
    query_set = set(query_tokens)
    source_set = set(source_tokens)
    overlap = query_set & source_set
    return len(overlap) / len(query_set) if query_set else 0

def containment(query_tokens, source_tokens, n=5):
    """Calculate n-gram containment"""
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    query_ngrams = get_ngrams(query_tokens, n)
    source_ngrams = get_ngrams(source_tokens, n)
    
    if not query_ngrams:
        return 0
    
    overlap = query_ngrams & source_ngrams
    return len(overlap) / len(query_ngrams)

# Query text (from PDF)
query = """Our topological results are closely aligned to previous work on turtles and crocodilians (15, 44, 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification and the phylogenetic and spatial distribution of extinction risk"""

# Source text (from Qdrant - top result)
source = """aligned to previous work on turtles and crocodilians (15, 44, 388 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification and the phylogenetic and spatial distribution of extinction risk"""

print("=" * 70)
print("LEXICAL COMPARISON")
print("=" * 70)

print(f"\nQuery ({len(query)} chars):")
print(query[:100] + "...")

print(f"\nSource ({len(source)} chars):")
print(source[:100] + "...")

query_tokens = tokenize(query)
source_tokens = tokenize(source)

print(f"\nQuery tokens: {len(query_tokens)}")
print(f"Source tokens: {len(source_tokens)}")

overlap = word_overlap(query_tokens, source_tokens)
print(f"\nWord overlap: {overlap:.2%}")

for n in [3, 5, 7]:
    cont = containment(query_tokens, source_tokens, n)
    print(f"{n}-gram containment: {cont:.2%}")

# Show common words
query_set = set(query_tokens)
source_set = set(source_tokens)
common = query_set & source_set
missing = query_set - source_set

print(f"\nCommon words ({len(common)}): {sorted(common)[:20]}...")
print(f"Missing from source ({len(missing)}): {sorted(missing)}")
