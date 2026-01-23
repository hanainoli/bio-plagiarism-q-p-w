#!/usr/bin/env python3
"""
Unit test for two-stage segment extraction logic.
Tests the core algorithm before running full system.
"""

import re

def tokenize(text):
    """Simple word tokenization"""
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    """Get n-grams from token list"""
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def calculate_overlap_score(query_tokens, segment_tokens):
    """Calculate overlap score between query and segment"""
    if not query_tokens or not segment_tokens:
        return 0.0, 0, 0.0
    
    # Word overlap
    q_set = set(query_tokens)
    s_set = set(segment_tokens)
    common_words = q_set & s_set
    word_overlap = len(common_words) / len(q_set)
    
    # Phrase matches (5-grams)
    phrase_count = 0
    phrase_ratio = 0.0
    if len(query_tokens) >= 5 and len(segment_tokens) >= 5:
        q_5grams = get_ngrams(query_tokens, 5)
        s_5grams = get_ngrams(segment_tokens, 5)
        matching_phrases = q_5grams & s_5grams
        phrase_count = len(matching_phrases)
        if len(q_5grams) > 0:
            phrase_ratio = len(matching_phrases) / len(q_5grams)
    
    # Combined lexical score - word overlap is primary
    lexical_score = word_overlap * 0.6 + phrase_ratio * 0.4
    
    return word_overlap, phrase_count, lexical_score

def find_best_matching_segment(query_tokens, source_text, source_tokens):
    """Slide window to find best matching segment"""
    query_len = len(query_tokens)
    source_len = len(source_tokens)
    
    if source_len <= query_len:
        word_overlap, phrase_count, lexical_score = calculate_overlap_score(query_tokens, source_tokens)
        return source_text, source_tokens, word_overlap, phrase_count, lexical_score
    
    window_size = query_len
    step = max(5, query_len // 10)
    
    best_score = 0
    best_start = 0
    best_metrics = (0.0, 0, 0.0)
    
    for start in range(0, source_len - window_size + 1, step):
        end = start + window_size
        segment_tokens = source_tokens[start:end]
        
        word_overlap, phrase_count, lexical_score = calculate_overlap_score(query_tokens, segment_tokens)
        
        if lexical_score > best_score:
            best_score = lexical_score
            best_start = start
            best_metrics = (word_overlap, phrase_count, lexical_score)
    
    best_end = min(best_start + window_size, source_len)
    best_segment_tokens = source_tokens[best_start:best_end]
    best_segment_text = ' '.join(best_segment_tokens)
    
    return best_segment_text, best_segment_tokens, best_metrics[0], best_metrics[1], best_metrics[2]


# ============= TEST CASES =============

print("="*70)
print("UNIT TEST: TWO-STAGE SEGMENT EXTRACTION")
print("="*70)

# Test 1: Query matches beginning of source
print("\n--- TEST 1: Query matches BEGINNING of source ---")
query1 = "The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic"
source1 = "The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success extant species diversity is low. Additional unrelated text here about other topics that should not match."

q1_tokens = tokenize(query1)
s1_tokens = tokenize(source1)

seg_text, seg_tokens, overlap, phrases, score = find_best_matching_segment(q1_tokens, source1, s1_tokens)

print(f"Query: {query1[:50]}...")
print(f"Source length: {len(s1_tokens)} tokens")
print(f"Best segment: {seg_text[:50]}...")
print(f"Word overlap: {overlap:.1%}")
print(f"Phrase matches: {phrases}")
print(f"Lexical score: {score:.3f}")

if overlap > 0.8 and phrases > 10:
    print("✅ PASS - Found correct segment at beginning")
else:
    print("❌ FAIL")


# Test 2: Query matches MIDDLE of source
print("\n--- TEST 2: Query matches MIDDLE of source ---")
query2 = "Crocodilians are famous for their extraordinary size up to 6m and 1000kg long snouts and tails"
source2 = "Unrelated text about geography and habitat. Some more filler content here. Crocodilians are famous for their extraordinary size up to 6m and 1000kg long snouts and tails and bony armor under the skin. More text about other topics."

q2_tokens = tokenize(query2)
s2_tokens = tokenize(source2)

seg_text, seg_tokens, overlap, phrases, score = find_best_matching_segment(q2_tokens, source2, s2_tokens)

print(f"Query: {query2[:50]}...")
print(f"Source length: {len(s2_tokens)} tokens")
print(f"Best segment: {seg_text[:50]}...")
print(f"Word overlap: {overlap:.1%}")
print(f"Phrase matches: {phrases}")
print(f"Lexical score: {score:.3f}")

if "crocodilians" in seg_text.lower() and overlap > 0.7:
    print("✅ PASS - Found correct segment in middle")
else:
    print("❌ FAIL")


# Test 3: Query does NOT match source (different topic)
print("\n--- TEST 3: Query does NOT match source (false positive check) ---")
query3 = "The origin of turtles and crocodiles and their easily recognized body forms"
source3 = "CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions. ChIP sequencing for histone modifications reveals insulator activities. Human CGGBP1 is ubiquitously expressed protein."

q3_tokens = tokenize(query3)
s3_tokens = tokenize(source3)

seg_text, seg_tokens, overlap, phrases, score = find_best_matching_segment(q3_tokens, source3, s3_tokens)

print(f"Query: {query3[:50]}...")
print(f"Source length: {len(s3_tokens)} tokens")
print(f"Best segment: {seg_text[:50]}...")
print(f"Word overlap: {overlap:.1%}")
print(f"Phrase matches: {phrases}")
print(f"Lexical score: {score:.3f}")

if phrases < 2 and score < 0.15:
    print("✅ PASS - Correctly identified as NON-MATCH")
else:
    print("❌ FAIL - Should have low score for different topic")


# Test 4: Compare two candidates - should rank correct one higher
print("\n--- TEST 4: Compare two candidates, rank by lexical score ---")
query4 = "The origin of turtles and crocodiles dates to the Triassic period"

# Candidate A: Turtle paper (should match)
source_a = "Abstract The origin of turtles and crocodiles dates to the Triassic period. Both groups exhibit highly conserved body forms."

# Candidate B: CGGBP1 paper (should NOT match)
source_b = "CGGBP1 regulates CTCF binding patterns. The protein is involved in chromatin organization and gene regulation."

q4_tokens = tokenize(query4)
sa_tokens = tokenize(source_a)
sb_tokens = tokenize(source_b)

_, _, overlap_a, phrases_a, score_a = find_best_matching_segment(q4_tokens, source_a, sa_tokens)
_, _, overlap_b, phrases_b, score_b = find_best_matching_segment(q4_tokens, source_b, sb_tokens)

print(f"Query: {query4}")
print(f"\nCandidate A (Turtle paper):")
print(f"  Overlap: {overlap_a:.1%}, Phrases: {phrases_a}, Lexical: {score_a:.3f}")
print(f"\nCandidate B (CGGBP1 paper):")
print(f"  Overlap: {overlap_b:.1%}, Phrases: {phrases_b}, Lexical: {score_b:.3f}")

if score_a > score_b and phrases_a > phrases_b:
    print("\n✅ PASS - Turtle paper ranked higher (correct!)")
else:
    print("\n❌ FAIL - Wrong ranking")


print("\n" + "="*70)
print("UNIT TESTS COMPLETE")
print("="*70)
