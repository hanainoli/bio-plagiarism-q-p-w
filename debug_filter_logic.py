#!/usr/bin/env python3
"""
Debug: Test the exact filtering logic with real examples.
"""

import re

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def count_phrase_matches(query_tokens, source_tokens):
    if len(query_tokens) < 5 or len(source_tokens) < 5:
        return 0
    q_5grams = get_ngrams(query_tokens, 5)
    s_5grams = get_ngrams(source_tokens, 5)
    return len(q_5grams & s_5grams)

def calculate_word_overlap(query_tokens, source_tokens):
    if not query_tokens:
        return 0.0
    q_set = set(query_tokens)
    s_set = set(source_tokens)
    return len(q_set & s_set) / len(q_set)

# Test cases from the actual report

test_cases = [
    {
        "name": "Chunk 0: Turtle Abstract vs CGGBP1 (WRONG)",
        "query": """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss.""",
        "source": """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites.""",
        "expected": "REJECT"
    },
    {
        "name": "Chunk 2: Turtle text vs Turtle source (CORRECT)",
        "query": """highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell""",
        "source": """Both groups date to the Triassic, and exhibit highly derived, yet highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell""",
        "expected": "INCLUDE"
    },
    {
        "name": "Chunk 8: Discussion vs Wrong Source (WRONG)",
        "query": """Our topological results are closely aligned to previous work on turtles and crocodilians (15, 44, 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification""",
        "source": """distinctiveness of the high number of 378 questionably-delimited deirochelyine species in that area. Interestingly, there are also modest 379 concentrations of non-threatened species""",
        "expected": "REJECT"
    },
    {
        "name": "Chunk 11: CGGBP1 text vs Turtle paper (WRONG)",
        "query": """We have shown the interactions between CTCF and CGGBP1 through co-immunoprecipitation and PLA on endogenously expressed proteins at native levels""",
        "source": """recent species of turtles and 27 species-level crocodilian lineages. We test for significant temporal variation in speciation and extinction rates related to the K-Pg boundary""",
        "expected": "REJECT"
    }
]

print("="*70)
print("TESTING PHRASE FILTERING LOGIC")
print("="*70)

for tc in test_cases:
    query_tokens = tokenize(tc["query"])
    source_tokens = tokenize(tc["source"])
    
    word_overlap = calculate_word_overlap(query_tokens, source_tokens)
    phrase_matches = count_phrase_matches(query_tokens, source_tokens)
    
    # Apply our filtering logic
    is_real_match = (
        phrase_matches >= 3 and word_overlap >= 0.40
    ) or (
        phrase_matches >= 5
    ) or (
        word_overlap >= 0.70 and phrase_matches >= 1
    )
    
    result = "INCLUDE" if is_real_match else "REJECT"
    status = "✅" if result == tc["expected"] else "❌ WRONG!"
    
    print(f"\n{tc['name']}")
    print(f"  Word overlap: {word_overlap:.1%}")
    print(f"  Phrase matches: {phrase_matches}")
    print(f"  Decision: {result}")
    print(f"  Expected: {tc['expected']}")
    print(f"  Status: {status}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
