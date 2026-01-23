#!/usr/bin/env python3
"""
Diagnostic: Check phrase matching between query and source chunks.
"""

import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def analyze_phrase_match(query, source, name=""):
    """Analyze phrase-level matching between query and source"""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {name}")
    print('='*70)
    
    print(f"\nQuery (first 100 chars): {query[:100]}...")
    print(f"Source (first 100 chars): {source[:100]}...")
    
    q_tokens = tokenize(query)
    s_tokens = tokenize(source)
    
    print(f"\nQuery tokens: {len(q_tokens)}")
    print(f"Source tokens: {len(s_tokens)}")
    
    # Word overlap
    q_set = set(q_tokens)
    s_set = set(s_tokens)
    overlap = q_set & s_set
    word_overlap = len(overlap) / len(q_set) if q_set else 0
    print(f"\nWord overlap: {word_overlap:.1%} ({len(overlap)} common words)")
    
    # Phrase matches at different n-gram sizes
    for n in [3, 4, 5, 7]:
        q_ngrams = get_ngrams(q_tokens, n)
        s_ngrams = get_ngrams(s_tokens, n)
        matches = q_ngrams & s_ngrams
        
        if matches:
            print(f"\n{n}-gram matches ({len(matches)}):")
            for m in list(matches)[:5]:
                print(f"  '{m}'")
            if len(matches) > 5:
                print(f"  ... and {len(matches)-5} more")
        else:
            print(f"\n{n}-gram matches: NONE")
    
    # Check if this is a real match
    q_5grams = get_ngrams(q_tokens, 5)
    s_5grams = get_ngrams(s_tokens, 5)
    matched_5grams = len(q_5grams & s_5grams)
    
    if matched_5grams >= 3:
        print(f"\n✅ REAL MATCH: {matched_5grams} matching 5-word phrases")
    else:
        print(f"\n❌ FALSE POSITIVE: Only {matched_5grams} matching 5-word phrases")

# Test cases from the report

# Case 1: Chunk 0 - Turtle abstract matched to CGGBP1 (WRONG)
query1 = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss."""

source1_wrong = """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites."""

analyze_phrase_match(query1, source1_wrong, "Chunk 0: Turtle Abstract vs CGGBP1 (WRONG MATCH)")

# Case 2: Chunk 2 - Turtle text matched to Turtle (CORRECT)
query2 = """highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell, and the size of some marine and terrestrial species (1.4m and 400kg on land, 2m and 1,000kg in the sea)."""

source2_correct = """Both groups date to the Triassic, and exhibit highly derived, yet highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell."""

analyze_phrase_match(query2, source2_correct, "Chunk 2: Turtle text vs Turtle source (CORRECT)")

# Case 3: Chunk 8 - Discussion matched to wrong source
query3 = """Our topological results are closely aligned to previous work on turtles and crocodilians (15, 44, 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification"""

source3_wrong = """distinctiveness of the high number of 378 questionably-delimited deirochelyine species in that area. Interestingly, there are also modest 379 concentrations of non-threatened species"""

analyze_phrase_match(query3, source3_wrong, "Chunk 8: Discussion vs Wrong Source")

# What SHOULD be the correct source for Chunk 8
source3_correct = """aligned to previous work on turtles and crocodilians (15, 44, 388 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification"""

analyze_phrase_match(query3, source3_correct, "Chunk 8: Discussion vs CORRECT Source (what it should match)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
The FALSE POSITIVES pass our current filters because:
1. Word overlap is ~55% (above 35% threshold) due to common words
2. BUT phrase matches are ZERO or very few

The REAL MATCHES have:
1. Word overlap ~60%+
2. AND many 5-gram phrase matches (3+)

FIX NEEDED: Require BOTH word overlap AND phrase matches!
""")
