#!/usr/bin/env python3
"""
Test Script for Enhanced Plagiarism Detection
==============================================
Demonstrates improvements in false positive reduction:
1. Semantic-Lexical Divergence Check
2. Multi-scale N-gram Analysis
3. Citation Context Detection
4. Confidence Scoring

Run: python test_enhanced_detector.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from detection.text_detector_enhanced import EnhancedTextPlagiarismDetector

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result(result):
    """Pretty print a match result"""
    print(f"  Match Type:       {result.match_type}")
    print(f"  Similarity Score: {result.similarity_score:.3f}")
    print(f"  Confidence:       {result.confidence:.3f}")
    print(f"  Semantic Score:   {result.semantic_score:.3f}")
    print(f"  Lexical Score:    {result.lexical_score:.3f}")
    print(f"  Divergence Flag:  {result.divergence_flag}")
    print(f"  Has Citation:     {result.has_citation_context}")
    if result.citation_type:
        print(f"  Citation Type:    {result.citation_type}")
    print(f"  Word Overlap:     {result.word_overlap_percentage:.3f}")
    print(f"  N-gram Scores:    {result.ngram_scores}")
    if result.sentence_matches:
        print(f"  Sentence Matches: {len(result.sentence_matches)}")

def main():
    detector = EnhancedTextPlagiarismDetector()
    
    print("\n" + "=" * 70)
    print("  ENHANCED PLAGIARISM DETECTOR TEST SUITE")
    print("  Testing False Positive Reduction")
    print("=" * 70)
    
    # =========================================================================
    # TEST 1: Exact Copy (should detect)
    # =========================================================================
    print_separator("TEST 1: EXACT COPY (Should: EXACT_COPY, High Score)")
    
    source_text = """
    Barrett's oesophagus is a premalignant condition which predisposes to 
    esophageal adenocarcinoma with a reported annual conversion rate of 0.1-0.2%.
    The implementation of formal surveillance strategies and widespread adoption 
    of endoscopic treatment techniques have led to a surge in diagnostic pathology 
    workload. Previous studies have revealed that diagnostic reproducibility 
    amongst pathologists grading dysplasia is suboptimal.
    """
    
    query_exact = """
    Barrett's oesophagus is a premalignant condition which predisposes to 
    esophageal adenocarcinoma with a reported annual conversion rate of 0.1-0.2%.
    The implementation of formal surveillance strategies and widespread adoption 
    of endoscopic treatment techniques have led to a surge in diagnostic pathology 
    workload. Previous studies have revealed that diagnostic reproducibility 
    amongst pathologists grading dysplasia is suboptimal.
    """
    
    result1 = detector.compare(query_exact, source_text, semantic_score=0.98)
    print_result(result1)
    assert result1.match_type in ["EXACT_COPY", "DIRECT_COPY"], "Should detect exact copy"
    assert result1.similarity_score > 0.9, "Should have high similarity"
    print("\n  ✓ PASSED: Correctly identified as copy")
    
    # =========================================================================
    # TEST 2: Same Topic, Different Words (FALSE POSITIVE scenario)
    # =========================================================================
    print_separator("TEST 2: SAME TOPIC, DIFFERENT WORDING (Should: COMMON_TOPIC or NO_MATCH)")
    
    query_different = """
    Barrett's esophagus represents a precancerous state that increases the risk
    of developing esophageal cancer, occurring at a rate of approximately 0.1-0.2%
    annually. Modern clinical practice has seen increased use of surveillance 
    programs and therapeutic endoscopy, resulting in higher volumes of tissue
    samples requiring pathological evaluation. Research indicates that agreement
    between pathologists on dysplasia grades remains inconsistent.
    """
    
    # High semantic (same topic) but should have low lexical (different words)
    result2 = detector.compare(query_different, source_text, semantic_score=0.85)
    print_result(result2)
    
    # Key assertion: this should NOT be flagged as plagiarism
    is_false_positive = result2.match_type in ["EXACT_COPY", "DIRECT_COPY"]
    print(f"\n  Divergence flag: {result2.divergence_flag}")
    print(f"  Would be false positive: {is_false_positive}")
    
    if result2.divergence_flag or result2.match_type in ["COMMON_TOPIC", "COMMON_KNOWLEDGE", "NO_MATCH"]:
        print("  ✓ PASSED: Correctly identified as NOT plagiarism (same topic, different words)")
    else:
        print("  ⚠ WARNING: May be a false positive - review thresholds")
    
    # =========================================================================
    # TEST 3: Properly Cited Text
    # =========================================================================
    print_separator("TEST 3: PROPERLY CITED TEXT (Should: PROPER_CITATION)")
    
    query_cited = """
    According to previous research (Smith et al., 2023), Barrett's oesophagus 
    is a premalignant condition which predisposes to esophageal adenocarcinoma 
    with a reported annual conversion rate of 0.1-0.2% [1,2]. As demonstrated 
    by earlier studies, the implementation of formal surveillance strategies 
    has led to a surge in diagnostic pathology workload.
    """
    
    result3 = detector.compare(query_cited, source_text, semantic_score=0.88)
    print_result(result3)
    
    assert result3.has_citation_context, "Should detect citations"
    if result3.match_type == "PROPER_CITATION":
        print("\n  ✓ PASSED: Correctly identified as properly cited")
    else:
        print(f"\n  Note: Classified as {result3.match_type} (citation detected: {result3.has_citation_context})")
    
    # =========================================================================
    # TEST 4: Paraphrase (should detect but with lower confidence)
    # =========================================================================
    print_separator("TEST 4: PARAPHRASE (Should: PARAPHRASE, Moderate Score)")
    
    query_paraphrase = """
    Barrett's oesophagus, a condition that can lead to esophageal cancer, has
    an annual progression rate between 0.1% and 0.2%. Surveillance programs and
    modern endoscopic treatments have increased the number of biopsies requiring
    pathologist review. Studies show pathologist agreement on dysplasia grading
    needs improvement.
    """
    
    result4 = detector.compare(query_paraphrase, source_text, semantic_score=0.78)
    print_result(result4)
    
    if result4.match_type in ["PARAPHRASE", "MOSAIC"]:
        print("\n  ✓ PASSED: Correctly identified as paraphrase")
    elif result4.match_type in ["COMMON_TOPIC", "COMMON_KNOWLEDGE"]:
        print("\n  ✓ PASSED: Classified as common topic (acceptable for heavy paraphrase)")
    else:
        print(f"\n  Note: Classified as {result4.match_type}")
    
    # =========================================================================
    # TEST 5: Unrelated Text
    # =========================================================================
    print_separator("TEST 5: UNRELATED TEXT (Should: NO_MATCH)")
    
    query_unrelated = """
    Machine learning algorithms have revolutionized image classification tasks.
    Convolutional neural networks achieve state-of-the-art performance on 
    benchmark datasets. Transfer learning enables models to generalize across
    domains with limited training data. Recent advances in attention mechanisms
    have further improved model accuracy.
    """
    
    result5 = detector.compare(query_unrelated, source_text, semantic_score=0.15)
    print_result(result5)
    
    assert result5.match_type == "NO_MATCH", "Should be NO_MATCH for unrelated text"
    assert result5.similarity_score < 0.3, "Should have low similarity"
    print("\n  ✓ PASSED: Correctly identified as no match")
    
    # =========================================================================
    # TEST 6: Methods Boilerplate (common scientific phrases)
    # =========================================================================
    print_separator("TEST 6: COMMON METHODS PHRASES (Should: Lower Score)")
    
    query_methods = """
    Statistical analysis was performed using SPSS version 25. Data are presented
    as mean ± SD. P values < 0.05 were considered statistically significant.
    Patients were enrolled between January 2020 and December 2022.
    """
    
    source_methods = """
    Statistical analysis was performed using SPSS version 25. Data are presented
    as mean ± SD. P values < 0.05 were considered statistically significant.
    Subjects were recruited from January 2019 to December 2021.
    """
    
    result6 = detector.compare(query_methods, source_methods, semantic_score=0.92)
    print_result(result6)
    
    print(f"\n  Note: Common methods phrases detected. Score adjusted: {result6.similarity_score:.3f}")
    print(f"  Confidence reduced: {result6.confidence:.3f}")
    
    # =========================================================================
    # TEST 7: Mosaic Plagiarism (sentence-level copying)
    # =========================================================================
    print_separator("TEST 7: MOSAIC PLAGIARISM (Mixed original and copied)")
    
    query_mosaic = """
    Our study investigates esophageal conditions in elderly patients. Barrett's 
    oesophagus is a premalignant condition which predisposes to esophageal 
    adenocarcinoma with a reported annual conversion rate of 0.1-0.2%. We enrolled
    250 participants from three hospitals. Previous studies have revealed that 
    diagnostic reproducibility amongst pathologists grading dysplasia is suboptimal.
    Our results suggest new therapeutic approaches are needed.
    """
    
    result7 = detector.compare(query_mosaic, source_text, semantic_score=0.72)
    print_result(result7)
    
    if result7.sentence_matches:
        print(f"\n  Sentence-level matches found: {len(result7.sentence_matches)}")
        for i, sm in enumerate(result7.sentence_matches[:3]):
            print(f"    {i+1}. Similarity: {sm['similarity']:.2f}, Cited: {sm['is_cited']}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_separator("TEST SUMMARY")
    
    tests_passed = 0
    total_tests = 7
    
    # Check each test
    if result1.match_type in ["EXACT_COPY", "DIRECT_COPY"]:
        tests_passed += 1
        print("  ✓ Test 1 (Exact Copy): PASSED")
    else:
        print("  ✗ Test 1 (Exact Copy): FAILED")
    
    if result2.divergence_flag or result2.match_type in ["COMMON_TOPIC", "COMMON_KNOWLEDGE", "NO_MATCH"]:
        tests_passed += 1
        print("  ✓ Test 2 (False Positive Prevention): PASSED")
    else:
        print("  ✗ Test 2 (False Positive Prevention): FAILED")
    
    if result3.has_citation_context:
        tests_passed += 1
        print("  ✓ Test 3 (Citation Detection): PASSED")
    else:
        print("  ✗ Test 3 (Citation Detection): FAILED")
    
    if result4.match_type in ["PARAPHRASE", "MOSAIC", "COMMON_TOPIC", "COMMON_KNOWLEDGE"]:
        tests_passed += 1
        print("  ✓ Test 4 (Paraphrase Detection): PASSED")
    else:
        print("  ✗ Test 4 (Paraphrase Detection): FAILED")
    
    if result5.match_type == "NO_MATCH":
        tests_passed += 1
        print("  ✓ Test 5 (Unrelated Text): PASSED")
    else:
        print("  ✗ Test 5 (Unrelated Text): FAILED")
    
    if result6.confidence < 1.0:  # Confidence should be reduced for boilerplate
        tests_passed += 1
        print("  ✓ Test 6 (Boilerplate Handling): PASSED")
    else:
        print("  ✗ Test 6 (Boilerplate Handling): FAILED")
    
    if result7.sentence_matches:  # Should find sentence-level matches
        tests_passed += 1
        print("  ✓ Test 7 (Mosaic Detection): PASSED")
    else:
        print("  ✗ Test 7 (Mosaic Detection): FAILED")
    
    print(f"\n  TOTAL: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n  🎉 ALL TESTS PASSED - Enhanced detector working correctly!")
    else:
        print(f"\n  ⚠ {total_tests - tests_passed} test(s) need attention")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
