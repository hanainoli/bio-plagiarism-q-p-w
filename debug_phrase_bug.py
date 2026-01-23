#!/usr/bin/env python3
"""
Debug: Why are we getting 90 phrase matches between turtle abstract and CGGBP1 text?
"""

import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

# Query: Turtle Abstract (from combined PDF)
query = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside a comprehensive"""

# Source 1: CGGBP1 paper chunk (from debug - should NOT match)
source_cggbp1 = """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites. INTRODUCTION Human CGGBP1 is a ubiquitously expressed protein with important functions in heat shock stress response, cell growth, proliferation and mitigation of endogenous DNA damage"""

# Source 2: Turtle paper chunk (from debug - SHOULD match)
source_turtle = """Abstract 12 The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside"""

print("="*70)
print("DEBUGGING PHRASE MATCHING")
print("="*70)

q_tokens = tokenize(query)
print(f"\nQuery tokens: {len(q_tokens)}")
print(f"Query first 10: {q_tokens[:10]}")

# Test with CGGBP1 source
s1_tokens = tokenize(source_cggbp1)
q_5grams = get_ngrams(q_tokens, 5)
s1_5grams = get_ngrams(s1_tokens, 5)
matches1 = q_5grams & s1_5grams

print(f"\n--- vs CGGBP1 source ---")
print(f"Source tokens: {len(s1_tokens)}")
print(f"Query 5-grams: {len(q_5grams)}")
print(f"Source 5-grams: {len(s1_5grams)}")
print(f"Matching 5-grams: {len(matches1)}")
if matches1:
    print(f"Matches: {list(matches1)[:5]}")

# Test with Turtle source
s2_tokens = tokenize(source_turtle)
s2_5grams = get_ngrams(s2_tokens, 5)
matches2 = q_5grams & s2_5grams

print(f"\n--- vs Turtle source ---")
print(f"Source tokens: {len(s2_tokens)}")
print(f"Query 5-grams: {len(q_5grams)}")
print(f"Source 5-grams: {len(s2_5grams)}")
print(f"Matching 5-grams: {len(matches2)}")
if matches2:
    print(f"Matches (first 5): {list(matches2)[:5]}")

print("\n" + "="*70)
print("EXPECTED RESULT:")
print("  CGGBP1: 0 matches (different text)")
print("  Turtle: 50+ matches (same text)")
print("="*70)

if len(matches1) > len(matches2):
    print("\n❌ BUG: CGGBP1 has MORE matches than Turtle!")
elif len(matches2) > len(matches1):
    print("\n✅ CORRECT: Turtle has more matches")
