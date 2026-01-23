#!/usr/bin/env python3
"""
Test with ACTUAL combined paper content to verify detection works correctly.
"""

import re

# ============= CORE FUNCTIONS (same as world_class_detector.py) =============

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def get_ngrams(tokens, n):
    if len(tokens) < n:
        return set()
    return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'that', 'this', 'these', 'those', 'it', 'its', 'we', 'our', 'their',
    'he', 'she', 'they', 'them', 'his', 'her', 'you', 'your', 'i', 'me',
    'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'then', 'if', 'else'
}

def calculate_overlap_score(query_tokens, segment_tokens):
    if not query_tokens or not segment_tokens:
        return 0.0, 0, 0.0, 0.0
    
    q_set = set(query_tokens)
    s_set = set(segment_tokens)
    word_overlap = len(q_set & s_set) / len(q_set) if q_set else 0
    
    q_content = q_set - STOPWORDS
    s_content = s_set - STOPWORDS
    content_overlap = len(q_content & s_content) / len(q_content) if q_content else 0
    
    phrase_count = 0
    phrase_ratio = 0.0
    if len(query_tokens) >= 5 and len(segment_tokens) >= 5:
        q_5grams = get_ngrams(query_tokens, 5)
        s_5grams = get_ngrams(segment_tokens, 5)
        matching_phrases = q_5grams & s_5grams
        phrase_count = len(matching_phrases)
        if len(q_5grams) > 0:
            phrase_ratio = len(matching_phrases) / len(q_5grams)
    
    lexical_score = content_overlap * 0.5 + phrase_ratio * 0.35 + word_overlap * 0.15
    
    return word_overlap, phrase_count, lexical_score, content_overlap

def find_best_segment(query_tokens, source_tokens):
    query_len = len(query_tokens)
    source_len = len(source_tokens)
    
    if source_len <= query_len:
        wo, pc, ls, co = calculate_overlap_score(query_tokens, source_tokens)
        return source_tokens, wo, pc, ls, co
    
    window_size = query_len
    step = max(5, query_len // 10)
    
    best_score = 0
    best_start = 0
    best_metrics = (0.0, 0, 0.0, 0.0)
    
    for start in range(0, source_len - window_size + 1, step):
        segment = source_tokens[start:start + window_size]
        wo, pc, ls, co = calculate_overlap_score(query_tokens, segment)
        
        if ls > best_score:
            best_score = ls
            best_start = start
            best_metrics = (wo, pc, ls, co)
    
    best_segment = source_tokens[best_start:best_start + window_size]
    return best_segment, best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3]


# ============= ACTUAL PAPER CHUNKS (from the combined PDF) =============

# These are chunks that would be extracted from the combined paper
QUERY_CHUNKS = [
    # Chunk 1: Abstract (Turtle content)
    """ABSTRACT The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside""",
    
    # Chunk 2: Introduction (CGGBP1 content)
    """Human CGGBP1 is a ubiquitously expressed protein with important functions in heat shock stress response, cell growth, proliferation and mitigation of endogenous DNA damage (Agarwal et al., 2016; Singh and Westermark, 2011, 2015; Singh et al., 2011, 2014). CGGBP1 has evolved in the amniotes with >98% conservation in homeotherms (Singh and Westermark, 2015). Yet, the involvement of CGGBP1 in highly conserved cellular processes such as cell cycle""",
    
    # Chunk 3: Turtle body description
    """Both groups date to the Triassic, and exhibit highly derived, yet highly conserved body forms. Crocodilians are famous for their extraordinary size (up to 6m and 1,000kg), long snouts and tails, and bony armor under the skin; and turtles for their bony or cartilaginous shell, and the size of some marine and terrestrial species (1.4m and 400kg on land, 2m and 1,000kg in the sea). Both groups are well-represented in the fossil record""",
    
    # Chunk 4: CGGBP1 expression
    """CGGBP1 has no known paralogs in the human genome, its expression in human tissues is ubiquitous (Thul and Lindskog, 2018) and RNAi against CGGBP1 causes G1/S arrest or G2/M arrest (Singh et al., 2011) and heat shock stress response-like gene expression changes with variable effects in different cell lines (Singh et al., 2009, 2011). CGGBP1 acts as a cis-regulator of transcription for tRNA genes""",
    
    # Chunk 5: Diversification results (Turtle)
    """Our topological results are closely aligned to previous work on turtles and crocodilians (15, 44, 45, 74-76) but offer a new time-calibrated perspective on the tempo and mode of evolutionary diversification and the phylogenetic and spatial distribution of extinction risk""",
    
    # Chunk 6: CTCF discussion (CGGBP1)
    """Human CTCF is a multifunctional protein pivotal to the functional organization of chromatin (Ong and Corces, 2014). Some well established functions of CTCF are regulation of insulators and boundary elements, regulation of topologically associating domains and higher order chromatin structure. CTCF in complex with proteins including cohesin ring members orchestrates chromatin structure"""
]

# Source chunks that SHOULD match (from Turtle paper in Qdrant)
TURTLE_SOURCE = """Abstract 12 The origin of turtles and crocodiles and their easily recognized body forms dates to the Triassic. Despite their long-term success, extant species diversity is low, and endangerment is extremely high compared to other terrestrial vertebrate groups, with ~ 65% of ~25 crocodilian and ~360 turtle species now threatened by exploitation and habitat loss. Here, we combine available molecular and morphological evidence with machine learning algorithms to present a phylogenetic analysis alongside a comprehensive assessment of extinction risk. We find that weights based on posterior probabilities of topological hypotheses improve the accuracy of phylogenetic comparative models. Our findings highlight the important role of geographic determinants of extinction risk particularly those resulting from anthropogenic habitat disturbance which affect species across body sizes and ecologies. Both groups date to the Triassic and exhibit highly derived yet highly conserved body forms."""

# Source chunks that SHOULD match (from CGGBP1 paper in Qdrant)
CGGBP1_SOURCE = """Yet, the involvement of CGGBP1 in highly conserved cellular processes such as cell cycle, maintenance of genomic integrity and cytosine methylation regulation suggests that CGGBP1 fine-tunes these processes in homeothermic organisms to meet the challenges of their terrestrial habitats. CGGBP1 has no known paralogs in the human genome, its expression in human tissues is ubiquitous (Thul and Lindskog, 2018) and RNAi against CGGBP1 causes G1/S arrest or G2/M arrest (Singh et al., 2011) and heat shock stress response-like gene expression changes with variable effects in different cell lines. CGGBP1 acts as a cis-regulator of transcription for tRNA genes, Alu elements (Agarwal et al., 2016), FMR1, CDKN1A, HSF1 and cytosine methylation regulatory genes including DNMT1."""


def test_combined_paper():
    print("="*80)
    print("TESTING COMBINED PAPER CHUNKS")
    print("="*80)
    
    turtle_tokens = tokenize(TURTLE_SOURCE)
    cggbp1_tokens = tokenize(CGGBP1_SOURCE)
    
    results = []
    
    for i, chunk in enumerate(QUERY_CHUNKS):
        print(f"\n{'='*80}")
        print(f"CHUNK {i+1}: {chunk[:60]}...")
        print("="*80)
        
        query_tokens = tokenize(chunk)
        
        # Test against both sources
        seg_t, wo_t, pc_t, ls_t, co_t = find_best_segment(query_tokens, turtle_tokens)
        seg_c, wo_c, pc_c, ls_c, co_c = find_best_segment(query_tokens, cggbp1_tokens)
        
        print(f"\nVs TURTLE paper:")
        print(f"  Content overlap: {co_t:.1%}")
        print(f"  Word overlap: {wo_t:.1%}")
        print(f"  Phrase matches: {pc_t}")
        print(f"  Lexical score: {ls_t:.3f}")
        print(f"  Segment: {' '.join(seg_t[:12])}...")
        
        print(f"\nVs CGGBP1 paper:")
        print(f"  Content overlap: {co_c:.1%}")
        print(f"  Word overlap: {wo_c:.1%}")
        print(f"  Phrase matches: {pc_c}")
        print(f"  Lexical score: {ls_c:.3f}")
        print(f"  Segment: {' '.join(seg_c[:12])}...")
        
        # Determine expected source
        if 'turtle' in chunk.lower() or 'crocodil' in chunk.lower() or 'triassic' in chunk.lower():
            if 'cggbp1' not in chunk.lower() and 'ctcf' not in chunk.lower():
                expected = "TURTLE"
            else:
                expected = "EITHER"
        elif 'cggbp1' in chunk.lower() or 'ctcf' in chunk.lower():
            expected = "CGGBP1"
        else:
            expected = "EITHER"
        
        # Check result
        if ls_t > ls_c:
            selected = "TURTLE"
        else:
            selected = "CGGBP1"
        
        if expected == "EITHER" or selected == expected:
            status = "✅ CORRECT"
        else:
            status = f"❌ WRONG (expected {expected})"
        
        print(f"\n  Selected: {selected} ({max(ls_t, ls_c):.3f})")
        print(f"  {status}")
        
        results.append({
            'chunk': i+1,
            'expected': expected,
            'selected': selected,
            'correct': expected == "EITHER" or selected == expected,
            'turtle_score': ls_t,
            'cggbp1_score': ls_c
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    
    print(f"\nResults: {correct}/{total} correct")
    print("\nDetailed:")
    for r in results:
        mark = "✅" if r['correct'] else "❌"
        print(f"  Chunk {r['chunk']}: {mark} Selected {r['selected']} (T:{r['turtle_score']:.2f}, C:{r['cggbp1_score']:.2f})")
    
    if correct == total:
        print("\n✅ ALL CHUNKS CORRECTLY MATCHED!")
    else:
        print(f"\n⚠️ {total - correct} chunks incorrectly matched")
    
    return correct == total


if __name__ == "__main__":
    test_combined_paper()
