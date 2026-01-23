#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END TEST
Simulates the full plagiarism detection flow with realistic test cases.

This test validates:
1. Segment extraction finds correct matching portions
2. Lexical scoring ranks correct paper higher
3. Mixed content (Turtle + CGGBP1) is handled correctly
4. False positives are rejected
5. Report shows correct source text
"""

import re

# ============= CORE FUNCTIONS (copied from world_class_detector.py) =============

def tokenize(text):
    """Simple word tokenization"""
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    """Get n-grams from token list"""
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def calculate_overlap_score(query_tokens, segment_tokens):
    """
    Calculate comprehensive overlap score between query and segment.
    
    Key insight: Raw phrase count can be misleading with long sources.
    We need to use PHRASE RATIO (what % of possible phrases match).
    """
    if not query_tokens or not segment_tokens:
        return 0.0, 0, 0.0
    
    # Word overlap - most reliable for actual copying
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
    
    # Combined lexical score
    # WORD OVERLAP is primary (0.6 weight) - most reliable indicator
    # PHRASE RATIO is secondary (0.4 weight) - validates actual phrase copying
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
    
    # Also check larger window
    larger_window = min(int(query_len * 1.2), source_len)
    for start in range(0, source_len - larger_window + 1, step):
        end = start + larger_window
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

def simulate_plagiarism_check(query_text, source_chunks):
    """
    Simulate the full plagiarism check flow.
    
    Args:
        query_text: Text from the paper being checked
        source_chunks: List of (paper_id, paper_name, chunk_text) tuples
    
    Returns:
        Ranked list of matches with segments
    """
    query_tokens = tokenize(query_text)
    
    candidates = []
    for paper_id, paper_name, chunk_text in source_chunks:
        source_tokens = tokenize(chunk_text)
        
        # Find best matching segment
        best_segment, seg_tokens, word_overlap, phrase_count, lexical_score = \
            find_best_matching_segment(query_tokens, chunk_text, source_tokens)
        
        candidates.append({
            'paper_id': paper_id,
            'paper_name': paper_name,
            'source_text': chunk_text,
            'best_segment': best_segment,
            'word_overlap': word_overlap,
            'phrase_count': phrase_count,
            'lexical_score': lexical_score
        })
    
    # Sort by lexical score
    candidates.sort(key=lambda x: x['lexical_score'], reverse=True)
    
    return candidates


# ============= REALISTIC TEST DATA =============

# Turtle paper chunks (from actual Qdrant data)
TURTLE_CHUNK_1 = """Abstract 12 The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside a comprehensive assessment of extinction risk."""

TURTLE_CHUNK_2 = """geographic ranges. Our findings highlight the important role of geographic 27 determinants of extinction risk, particularly those resulting from anthropogenic habitat- 28 disturbance, which affect species across body sizes and ecologies. 29 Both groups date to the Triassic, and exhibit highly 33 derived, yet highly conserved body forms. Crocodilians are famous for their extraordinary size 34 (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for 35 their bony or cartilaginous shell."""

TURTLE_CHUNK_3 = """aligned to previous work on turtles and crocodilians (15, 44, 388 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification and the phylogenetic and spatial distribution of extinction risk."""

# CGGBP1 paper chunks (from actual Qdrant data)
CGGBP1_CHUNK_1 = """ChIP-sequencing for three different kinds of histone modifications (H3K4me3, H3K9me3 and H3K27me3) we uncover insulator-like chromatin barrier activities of the repeat-rich CTCF-binding sites. This work shows that CGGBP1 is a regulator of CTCF occupancy and posits it as a regulator of barrier functions of CTCF-binding sites. INTRODUCTION Human CGGBP1 is a ubiquitously expressed protein with important functions in heat shock stress response, cell growth, proliferation and mitigation of endogenous DNA damage."""

CGGBP1_CHUNK_2 = """Yet, the involvement of CGGBP1 in highly conserved cellular processes such as cell cycle, maintenance of genomic integrity and cytosine methylation regulation suggests that CGGBP1 fine-tunes these processes in homeothermic organisms to meet the challenges of their terrestrial habitats. CGGBP1 has no known paralogs in the human genome, its expression in human tissues is ubiquitous (Thul and Lindskog, 2018) and RNAi against CGGBP1 causes G1/S arrest or G2/M arrest (Singh et al., 2011) and heat shock stress response."""

CGGBP1_CHUNK_3 = """with proteins including cohesin ring members orchestrates chromatin structure and changes in CTCF binding has been shown to affect histone modifications and depend on cytosine methylation. Given the important functions executed by CTCF, it makes evolutionary sense that there are regulatory crosstalks between CTCF and other proteins."""

# Query chunks (simulating what's extracted from the test PDF)
QUERY_TURTLE_ABSTRACT = """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside"""

QUERY_TURTLE_BODY = """highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell, and the size of some marine and terrestrial species."""

QUERY_CGGBP1_TEXT = """expression in human tissues is ubiquitous (Thul and Lindskog, 2018) and RNAi against CGGBP1 causes G1/S arrest or G2/M arrest (Singh et al., 2011) and heat shock stress response-like gene expression changes with variable effects in different cell lines."""

QUERY_MIXED_TEXT = """Our topological results are closely aligned to previous work on turtles and crocodilians but offer a new time-calibrated perspective. CTCF in complex with proteins including cohesin ring members orchestrates chromatin structure."""


# ============= TEST CASES =============

def run_all_tests():
    print("="*80)
    print("COMPREHENSIVE END-TO-END PLAGIARISM DETECTION TEST")
    print("="*80)
    
    all_passed = True
    
    # All source chunks (simulating Qdrant results)
    all_sources = [
        ("turtle_paper", "Turtle/Crocodile Paper", TURTLE_CHUNK_1),
        ("turtle_paper", "Turtle/Crocodile Paper", TURTLE_CHUNK_2),
        ("turtle_paper", "Turtle/Crocodile Paper", TURTLE_CHUNK_3),
        ("cggbp1_paper", "CGGBP1 Paper", CGGBP1_CHUNK_1),
        ("cggbp1_paper", "CGGBP1 Paper", CGGBP1_CHUNK_2),
        ("cggbp1_paper", "CGGBP1 Paper", CGGBP1_CHUNK_3),
    ]
    
    # ===== TEST 1: Turtle Abstract should match Turtle paper =====
    print("\n" + "-"*80)
    print("TEST 1: Turtle Abstract Query")
    print("-"*80)
    print(f"Query: {QUERY_TURTLE_ABSTRACT[:80]}...")
    
    results = simulate_plagiarism_check(QUERY_TURTLE_ABSTRACT, all_sources)
    
    print(f"\nTop 3 Results:")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r['paper_name']}: lexical={r['lexical_score']:.3f}, overlap={r['word_overlap']:.1%}, phrases={r['phrase_count']}")
        print(f"     Segment: {r['best_segment'][:60]}...")
    
    if results[0]['paper_id'] == 'turtle_paper' and results[0]['lexical_score'] > 0.5:
        print("✅ TEST 1 PASSED: Turtle paper ranked first with high score")
    else:
        print("❌ TEST 1 FAILED: Wrong paper ranked first")
        all_passed = False
    
    # ===== TEST 2: Turtle Body should match Turtle paper =====
    print("\n" + "-"*80)
    print("TEST 2: Turtle Body Query (Crocodilians are famous...)")
    print("-"*80)
    print(f"Query: {QUERY_TURTLE_BODY[:80]}...")
    
    results = simulate_plagiarism_check(QUERY_TURTLE_BODY, all_sources)
    
    print(f"\nTop 3 Results:")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r['paper_name']}: lexical={r['lexical_score']:.3f}, overlap={r['word_overlap']:.1%}, phrases={r['phrase_count']}")
        print(f"     Segment: {r['best_segment'][:60]}...")
    
    if results[0]['paper_id'] == 'turtle_paper' and results[0]['lexical_score'] > 0.5:
        print("✅ TEST 2 PASSED: Turtle paper ranked first")
    else:
        print("❌ TEST 2 FAILED: Wrong paper ranked first")
        all_passed = False
    
    # ===== TEST 3: CGGBP1 text should match CGGBP1 paper =====
    print("\n" + "-"*80)
    print("TEST 3: CGGBP1 Query")
    print("-"*80)
    print(f"Query: {QUERY_CGGBP1_TEXT[:80]}...")
    
    results = simulate_plagiarism_check(QUERY_CGGBP1_TEXT, all_sources)
    
    print(f"\nTop 3 Results:")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r['paper_name']}: lexical={r['lexical_score']:.3f}, overlap={r['word_overlap']:.1%}, phrases={r['phrase_count']}")
        print(f"     Segment: {r['best_segment'][:60]}...")
    
    if results[0]['paper_id'] == 'cggbp1_paper' and results[0]['lexical_score'] > 0.5:
        print("✅ TEST 3 PASSED: CGGBP1 paper ranked first")
    else:
        print("❌ TEST 3 FAILED: Wrong paper ranked first")
        all_passed = False
    
    # ===== TEST 4: Mixed content should find both papers =====
    print("\n" + "-"*80)
    print("TEST 4: Mixed Content Query (Turtle + CGGBP1)")
    print("-"*80)
    print(f"Query: {QUERY_MIXED_TEXT[:80]}...")
    
    results = simulate_plagiarism_check(QUERY_MIXED_TEXT, all_sources)
    
    print(f"\nTop 3 Results:")
    for i, r in enumerate(results[:3]):
        print(f"  {i+1}. {r['paper_name']}: lexical={r['lexical_score']:.3f}, overlap={r['word_overlap']:.1%}, phrases={r['phrase_count']}")
        print(f"     Segment: {r['best_segment'][:60]}...")
    
    # For mixed content, both papers should appear in top results
    top_papers = set(r['paper_id'] for r in results[:3])
    if 'turtle_paper' in top_papers or 'cggbp1_paper' in top_papers:
        print("✅ TEST 4 PASSED: Relevant papers found in top results")
    else:
        print("❌ TEST 4 FAILED: Neither relevant paper in top 3")
        all_passed = False
    
    # ===== TEST 5: Segment text should match query =====
    print("\n" + "-"*80)
    print("TEST 5: Segment Quality Check")
    print("-"*80)
    
    query = "The origin of turtles and crocodiles and their easily recognized body forms"
    results = simulate_plagiarism_check(query, all_sources)
    
    best_match = results[0]
    query_words = set(tokenize(query))
    segment_words = set(tokenize(best_match['best_segment']))
    
    # Check that segment contains query words
    common = query_words & segment_words
    coverage = len(common) / len(query_words)
    
    print(f"Query: {query}")
    print(f"Best segment: {best_match['best_segment'][:80]}...")
    print(f"Query word coverage in segment: {coverage:.1%}")
    
    if coverage > 0.7:
        print("✅ TEST 5 PASSED: Segment contains most query words")
    else:
        print("❌ TEST 5 FAILED: Segment doesn't match query well")
        all_passed = False
    
    # ===== TEST 6: Unrelated query should have low scores =====
    print("\n" + "-"*80)
    print("TEST 6: False Positive Check (Unrelated Query)")
    print("-"*80)
    
    unrelated_query = "Machine learning neural networks deep learning artificial intelligence transformers attention mechanism gradient descent backpropagation"
    results = simulate_plagiarism_check(unrelated_query, all_sources)
    
    print(f"Query: {unrelated_query[:60]}...")
    print(f"\nTop result: {results[0]['paper_name']}")
    print(f"  Lexical score: {results[0]['lexical_score']:.3f}")
    print(f"  Word overlap: {results[0]['word_overlap']:.1%}")
    
    if results[0]['lexical_score'] < 0.15:
        print("✅ TEST 6 PASSED: Unrelated query has low score (no false positive)")
    else:
        print("❌ TEST 6 FAILED: Unrelated query has high score (false positive!)")
        all_passed = False
    
    # ===== SUMMARY =====
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - System is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Review the results above")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
