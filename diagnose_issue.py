#!/usr/bin/env python3
"""
REALISTIC QDRANT SIMULATION TEST

This test simulates EXACTLY what happens with real Qdrant data:
- Source chunks are ~200 tokens (1300 chars)
- Query chunks are ~74 tokens (500 chars)
- Multiple source chunks from same paper
- Common scientific phrases appear in many chunks

The test output showed:
- CGGBP1 chunk had lexical=0.473, phrases=90 (WRONG - ranked first)
- Turtle chunk had lexical=0.323, phrases=48 (CORRECT - but ranked second)

This test will:
1. Reproduce this exact scenario
2. Find and fix the root cause
3. Verify the fix works
"""

import re

# ============= CORE FUNCTIONS =============

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def calculate_overlap_score_v1(query_tokens, segment_tokens):
    """ORIGINAL VERSION - has the bug"""
    if not query_tokens or not segment_tokens:
        return 0.0, 0, 0.0
    
    q_set = set(query_tokens)
    s_set = set(segment_tokens)
    word_overlap = len(q_set & s_set) / len(q_set)
    
    phrase_count = 0
    phrase_ratio = 0.0
    if len(query_tokens) >= 5 and len(segment_tokens) >= 5:
        q_5grams = get_ngrams(query_tokens, 5)
        s_5grams = get_ngrams(segment_tokens, 5)
        matching_phrases = q_5grams & s_5grams
        phrase_count = len(matching_phrases)
        if len(q_5grams) > 0:
            phrase_ratio = len(matching_phrases) / len(q_5grams)
    
    lexical_score = word_overlap * 0.6 + phrase_ratio * 0.4
    return word_overlap, phrase_count, lexical_score

def calculate_overlap_score_v2(query_tokens, segment_tokens):
    """FIXED VERSION - prioritizes actual text match"""
    if not query_tokens or not segment_tokens:
        return 0.0, 0, 0.0
    
    q_set = set(query_tokens)
    s_set = set(segment_tokens)
    common = q_set & s_set
    word_overlap = len(common) / len(q_set)
    
    # Remove common stopwords from overlap calculation for better accuracy
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                 'that', 'this', 'these', 'those', 'it', 'its', 'we', 'our', 'their'}
    
    q_content = q_set - stopwords
    s_content = s_set - stopwords
    content_overlap = len(q_content & s_content) / len(q_content) if q_content else 0
    
    # Phrase matches
    phrase_count = 0
    phrase_ratio = 0.0
    if len(query_tokens) >= 5 and len(segment_tokens) >= 5:
        q_5grams = get_ngrams(query_tokens, 5)
        s_5grams = get_ngrams(segment_tokens, 5)
        matching_phrases = q_5grams & s_5grams
        phrase_count = len(matching_phrases)
        if len(q_5grams) > 0:
            phrase_ratio = len(matching_phrases) / len(q_5grams)
    
    # CONTENT OVERLAP is most important - actual unique words matching
    # Then phrase ratio validates it
    lexical_score = content_overlap * 0.5 + word_overlap * 0.2 + phrase_ratio * 0.3
    
    return word_overlap, phrase_count, lexical_score, content_overlap

def find_best_segment(query_tokens, source_tokens, calc_func):
    """Find best matching segment using given calculation function"""
    query_len = len(query_tokens)
    source_len = len(source_tokens)
    
    if source_len <= query_len:
        result = calc_func(query_tokens, source_tokens)
        if len(result) == 4:
            return source_tokens, result[0], result[1], result[2], result[3]
        return source_tokens, result[0], result[1], result[2], 0
    
    window_size = query_len
    step = max(5, query_len // 10)
    
    best_score = 0
    best_start = 0
    best_metrics = None
    
    for start in range(0, source_len - window_size + 1, step):
        segment = source_tokens[start:start + window_size]
        result = calc_func(query_tokens, segment)
        score = result[2]  # lexical_score
        
        if score > best_score:
            best_score = score
            best_start = start
            best_metrics = result
    
    best_segment = source_tokens[best_start:best_start + window_size]
    if best_metrics is None:
        best_metrics = calc_func(query_tokens, best_segment)
    
    if len(best_metrics) == 4:
        return best_segment, best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3]
    return best_segment, best_metrics[0], best_metrics[1], best_metrics[2], 0


# ============= REALISTIC TEST DATA =============
# These need to match ACTUAL Qdrant chunk sizes:
# - Source chunks: ~1400 chars, ~200 tokens
# - Query chunks: ~500 chars, ~74 tokens

QUERY_TURTLE_ABSTRACT = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside"""

# Source chunk from CGGBP1 paper (~1400 chars, ~200 tokens - ACTUAL Qdrant size)
SOURCE_CGGBP1_LONG = """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites. INTRODUCTION Human CGGBP1 is a ubiquitously expressed protein with important functions in heat shock stress response, cell growth, proliferation and mitigation of endogenous DNA damage. The protein has evolved in amniotes with conservation in homeotherms. Yet the involvement of CGGBP1 in highly conserved cellular processes such as cell cycle maintenance of genomic integrity and cytosine methylation regulation suggests that CGGBP1 fine-tunes these processes in homeothermic organisms to meet the challenges of their terrestrial habitats. CGGBP1 has no known paralogs in the human genome its expression in human tissues is ubiquitous and RNAi against CGGBP1 causes G1 S arrest or G2 M arrest and heat shock stress response like gene expression changes with variable effects in different cell lines. CGGBP1 acts as a cis regulator of transcription for tRNA genes Alu elements FMR1 CDKN1A HSF1 and cytosine methylation regulatory genes including DNMT1. In comparison to the number of genes regulated by CGGBP1 its direct targets are few."""

# Source chunk from Turtle paper (~1400 chars, ~200 tokens - ACTUAL Qdrant size)
SOURCE_TURTLE_LONG = """Abstract 12 The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside a comprehensive assessment of extinction risk. We find that weights based on posterior probabilities of topological hypotheses improve the accuracy of phylogenetic comparative models. Our findings highlight the important role of geographic determinants of extinction risk particularly those resulting from anthropogenic habitat disturbance which affect species across body sizes and ecologies. Both groups date to the Triassic and exhibit highly derived yet highly conserved body forms. Crocodilians are famous for their extraordinary size up to 6m and 1000kg long snouts and tails and bony armor under the skin and turtles for their bony or cartilaginous shell and the size of some marine and terrestrial species."""


def run_diagnostic():
    """Diagnose exactly what's happening with the real data"""
    
    print("="*80)
    print("DIAGNOSTIC: Why is CGGBP1 ranking higher than Turtle?")
    print("="*80)
    
    query_tokens = tokenize(QUERY_TURTLE_ABSTRACT)
    cggbp1_tokens = tokenize(SOURCE_CGGBP1_LONG)
    turtle_tokens = tokenize(SOURCE_TURTLE_LONG)
    
    print(f"\nQuery length: {len(query_tokens)} tokens")
    print(f"CGGBP1 source length: {len(cggbp1_tokens)} tokens")
    print(f"Turtle source length: {len(turtle_tokens)} tokens")
    
    # Find best segments using V1 (buggy)
    print("\n" + "-"*80)
    print("VERSION 1 (Current - may have bug)")
    print("-"*80)
    
    seg_cggbp1, wo1_c, pc1_c, ls1_c, _ = find_best_segment(query_tokens, cggbp1_tokens, 
        lambda q, s: calculate_overlap_score_v1(q, s) + (0,))  # Add dummy for content_overlap
    seg_turtle, wo1_t, pc1_t, ls1_t, _ = find_best_segment(query_tokens, turtle_tokens,
        lambda q, s: calculate_overlap_score_v1(q, s) + (0,))
    
    print(f"\nCGGBP1 best segment:")
    print(f"  Word overlap: {wo1_c:.1%}")
    print(f"  Phrase count: {pc1_c}")
    print(f"  Lexical score: {ls1_c:.3f}")
    print(f"  Segment: {' '.join(seg_cggbp1[:15])}...")
    
    print(f"\nTurtle best segment:")
    print(f"  Word overlap: {wo1_t:.1%}")
    print(f"  Phrase count: {pc1_t}")
    print(f"  Lexical score: {ls1_t:.3f}")
    print(f"  Segment: {' '.join(seg_turtle[:15])}...")
    
    if ls1_c > ls1_t:
        print(f"\n❌ BUG CONFIRMED: CGGBP1 ({ls1_c:.3f}) > Turtle ({ls1_t:.3f})")
    else:
        print(f"\n✅ OK: Turtle ({ls1_t:.3f}) > CGGBP1 ({ls1_c:.3f})")
    
    # Now test V2 (fixed)
    print("\n" + "-"*80)
    print("VERSION 2 (Fixed - with content overlap)")
    print("-"*80)
    
    seg_cggbp1, wo2_c, pc2_c, ls2_c, co2_c = find_best_segment(query_tokens, cggbp1_tokens, calculate_overlap_score_v2)
    seg_turtle, wo2_t, pc2_t, ls2_t, co2_t = find_best_segment(query_tokens, turtle_tokens, calculate_overlap_score_v2)
    
    print(f"\nCGGBP1 best segment:")
    print(f"  Content overlap (no stopwords): {co2_c:.1%}")
    print(f"  Word overlap: {wo2_c:.1%}")
    print(f"  Phrase count: {pc2_c}")
    print(f"  Lexical score: {ls2_c:.3f}")
    
    print(f"\nTurtle best segment:")
    print(f"  Content overlap (no stopwords): {co2_t:.1%}")
    print(f"  Word overlap: {wo2_t:.1%}")
    print(f"  Phrase count: {pc2_t}")
    print(f"  Lexical score: {ls2_t:.3f}")
    
    if ls2_t > ls2_c:
        print(f"\n✅ FIXED: Turtle ({ls2_t:.3f}) > CGGBP1 ({ls2_c:.3f})")
    else:
        print(f"\n❌ STILL BROKEN: CGGBP1 ({ls2_c:.3f}) > Turtle ({ls2_t:.3f})")
    
    # Analyze WHY the scores are what they are
    print("\n" + "-"*80)
    print("ROOT CAUSE ANALYSIS")
    print("-"*80)
    
    # Check phrase overlap details
    q_5grams = get_ngrams(query_tokens, 5)
    c_5grams = get_ngrams(cggbp1_tokens, 5)
    t_5grams = get_ngrams(turtle_tokens, 5)
    
    print(f"\nQuery has {len(q_5grams)} unique 5-grams")
    print(f"CGGBP1 has {len(c_5grams)} unique 5-grams")
    print(f"Turtle has {len(t_5grams)} unique 5-grams")
    
    c_matches = q_5grams & c_5grams
    t_matches = q_5grams & t_5grams
    
    print(f"\nMatching 5-grams with CGGBP1: {len(c_matches)}")
    if c_matches:
        print(f"  Examples: {list(c_matches)[:3]}")
    
    print(f"\nMatching 5-grams with Turtle: {len(t_matches)}")
    if t_matches:
        print(f"  Examples: {list(t_matches)[:3]}")
    
    # Check word overlap details
    q_words = set(query_tokens)
    c_words = set(cggbp1_tokens)
    t_words = set(turtle_tokens)
    
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                 'that', 'this', 'these', 'those', 'it', 'its', 'we', 'our', 'their'}
    
    q_content = q_words - stopwords
    c_content = c_words - stopwords
    t_content = t_words - stopwords
    
    print(f"\n\nContent words (excluding stopwords):")
    print(f"Query: {len(q_content)} content words")
    print(f"CGGBP1: {len(q_content & c_content)} matching ({len(q_content & c_content)/len(q_content):.1%})")
    print(f"Turtle: {len(q_content & t_content)} matching ({len(q_content & t_content)/len(q_content):.1%})")
    
    # Show unique matches
    c_unique = (q_content & c_content) - (q_content & t_content)
    t_unique = (q_content & t_content) - (q_content & c_content)
    
    print(f"\nWords matching ONLY CGGBP1: {c_unique}")
    print(f"Words matching ONLY Turtle: {t_unique}")
    
    return ls2_t > ls2_c


if __name__ == "__main__":
    success = run_diagnostic()
    
    print("\n" + "="*80)
    if success:
        print("✅ SOLUTION FOUND: Use content overlap (excluding stopwords)")
    else:
        print("❌ NEED MORE INVESTIGATION")
    print("="*80)
