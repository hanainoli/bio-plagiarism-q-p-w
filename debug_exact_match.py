#!/usr/bin/env python3
"""
Debug: Check why we're getting 90 phrase matches between different texts.
"""

import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

# EXACT texts from the report

query_chunk_0 = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongsi"""

source_chunk_0 = """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites. INTRODUCTION Human CGGBP1 is a ubiquitously expressed protein with important functions in heat shock stress response, cell growth, proliferation and mitigation of endogenous"""

print("="*70)
print("EXACT TEXT COMPARISON FROM REPORT")
print("="*70)

q_tokens = tokenize(query_chunk_0)
s_tokens = tokenize(source_chunk_0)

print(f"\nQuery tokens: {len(q_tokens)}")
print(f"Source tokens: {len(s_tokens)}")

q_5grams = get_ngrams(q_tokens, 5)
s_5grams = get_ngrams(source_chunk_0.lower().split(), 5)  # Different tokenization?

print(f"\nQuery 5-grams: {len(q_5grams)}")
print(f"Source 5-grams: {len(s_5grams)}")

matches = q_5grams & s_5grams
print(f"\nMatching 5-grams: {len(matches)}")

if matches:
    print("\nMatches found:")
    for m in list(matches)[:10]:
        print(f"  '{m}'")
else:
    print("\n✅ NO MATCHES - as expected!")

# Now check word overlap
q_set = set(q_tokens)
s_set = set(s_tokens)
overlap = q_set & s_set
word_overlap = len(overlap) / len(q_set) if q_set else 0

print(f"\nWord overlap: {word_overlap:.1%}")
print(f"Common words: {sorted(overlap)}")

print("\n" + "="*70)
print("WHY DOES THE SYSTEM REPORT 90 PHRASE MATCHES?")
print("="*70)
print("""
Possible issues:
1. The source text in Qdrant is DIFFERENT from what's shown in the report
2. There's a bug in how we're tokenizing
3. The phrase matching is being done on different text
""")
