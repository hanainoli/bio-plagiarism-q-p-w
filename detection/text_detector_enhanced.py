"""
Enhanced Text Plagiarism Detector.
=================================
Improvements over base text_detector.py:
1. Semantic-Lexical Divergence Check - reduces false positives
2. Multi-scale N-gram Analysis - catches paraphrases better
3. Sentence-Level Granularity - pinpoints exact plagiarized sentences
4. Citation Context Detection - identifies properly cited text
5. Confidence Scoring - provides reliability measure

Author: UyarAI
Version: 2.0
"""

import re
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import hashlib


@dataclass
class EnhancedTextMatch:
    """Enhanced result of text comparison with confidence scoring"""
    source_text: str
    query_text: str
    
    # Core scores
    similarity_score: float
    confidence: float  # NEW: How confident we are in this match
    match_type: str
    
    # Detailed metrics
    semantic_score: float  # NEW: From embedding similarity
    lexical_score: float   # NEW: Combined lexical metrics
    divergence_flag: bool  # NEW: True if semantic high but lexical low
    
    # Traditional metrics
    word_overlap_percentage: float
    containment: float
    jaccard: float
    char_similarity: float
    
    # Match details (required - no defaults)
    matched_word_count: int
    total_query_words: int
    
    # Fields with defaults MUST come after required fields
    # Multi-scale n-gram scores (NEW)
    ngram_scores: Dict[int, float] = field(default_factory=dict)
    best_ngram_scale: int = 5
    
    matched_spans: List[Tuple[int, int]] = field(default_factory=list)
    matched_ngrams: List[str] = field(default_factory=list)
    
    # Sentence-level analysis (NEW)
    sentence_matches: List[Dict] = field(default_factory=list)
    
    # Citation detection (NEW)
    has_citation_context: bool = False
    citation_type: Optional[str] = None


@dataclass
class SentenceMatch:
    """Individual sentence match result"""
    query_sentence: str
    source_sentence: str
    similarity: float
    is_cited: bool
    start_pos: int
    end_pos: int


class EnhancedTextPlagiarismDetector:
    """
    Enhanced text plagiarism detection with false positive reduction.
    
    Key improvements:
    - Semantic-lexical divergence detection
    - Multi-scale n-gram analysis
    - Sentence-level matching
    - Citation context awareness
    """
    
    def __init__(
        self,
        ngram_sizes: List[int] = [3, 4, 5, 7],  # Multi-scale
        min_match_length: int = 4,
        exact_threshold: float = 0.95,
        high_threshold: float = 0.80,
        moderate_threshold: float = 0.50,
        divergence_threshold: float = 0.45,  # NEW: semantic-lexical gap threshold
        citation_boost: float = 0.3  # NEW: reduce score if properly cited
    ):
        """
        Initialize enhanced detector.
        
        Args:
            ngram_sizes: List of n-gram sizes for multi-scale analysis
            min_match_length: Minimum consecutive words for a match
            exact_threshold: Threshold for exact copy
            high_threshold: Threshold for high similarity
            moderate_threshold: Threshold for moderate similarity
            divergence_threshold: Max allowed gap between semantic and lexical
            citation_boost: Score reduction for properly cited text
        """
        self.ngram_sizes = ngram_sizes
        self.primary_ngram_size = 5
        self.min_match_length = min_match_length
        self.exact_threshold = exact_threshold
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.divergence_threshold = divergence_threshold
        self.citation_boost = citation_boost
        
        # Citation patterns
        self.citation_patterns = [
            r'\(\s*\d{4}\s*\)',           # (2023)
            r'\(\s*[A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4}\s*\)',  # (Smith et al., 2023)
            r'\[\s*\d+\s*\]',             # [1], [23]
            r'\[\s*\d+\s*[-,]\s*\d+\s*\]', # [1-3], [1,2]
            r'\bet\s+al\.?\b',            # et al.
            r'\baccording\s+to\b',
            r'\bas\s+(?:shown|demonstrated|reported|described)\s+(?:by|in)\b',
            r'\bprevious(?:ly)?\s+(?:shown|demonstrated|reported)\b',
            r'\b(?:cited|reported)\s+(?:by|in)\b',
        ]
        
        # Boilerplate patterns (enhanced)
        self.boilerplate_patterns = [
            r'all\s+rights\s+reserved',
            r'copyright\s+\d{4}',
            r'licensed\s+under',
            r'creative\s+commons',
            r'methods?\s+were\s+performed',
            r'data\s+(?:are|is|was|were)\s+available',
            r'statistical(?:ly)?\s+(?:significant|analysis)',
            r'p\s*[<>=≤≥]\s*0\.\d+',
            r'mean\s*[±+]\s*(?:SD|SEM|se)',
            r'(?:figure|table|fig\.?|tab\.?)\s*\d+',
            r'supplementary\s+(?:figure|table|material)',
            r'we\s+(?:used|performed|conducted|analyzed)',  # Common methods phrases
            r'patients?\s+(?:were|was)\s+(?:enrolled|recruited|included)',
        ]
        
        # Scientific common phrases (should have lower weight)
        self.common_scientific_phrases = [
            r'in\s+this\s+study',
            r'the\s+results\s+(?:show|demonstrate|indicate|suggest)',
            r'(?:our|these)\s+(?:results|findings|data)',
            r'was\s+(?:significantly|statistically)',
            r'compared\s+(?:to|with)\s+(?:the\s+)?control',
            r'no\s+significant\s+difference',
            r'p\s*(?:value)?\s*(?:was|=|<|>)',
        ]
    
    # =========================================================================
    # TEXT PREPROCESSING
    # =========================================================================
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in text.split() if len(t) > 1]
        return tokens
    
    def sentence_tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into sentences with positions.
        
        Returns:
            List of (sentence, start_char, end_char)
        """
        # Simple sentence splitter - handles academic text well
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = []
        last_end = 0
        
        for match in re.finditer(sentence_pattern, text):
            sent = text[last_end:match.start()].strip()
            if len(sent) > 20:  # Minimum sentence length
                sentences.append((sent, last_end, match.start()))
            last_end = match.end()
        
        # Add last sentence
        if last_end < len(text):
            sent = text[last_end:].strip()
            if len(sent) > 20:
                sentences.append((sent, last_end, len(text)))
        
        return sentences
    
    def generate_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generate n-grams of size n"""
        if len(tokens) < n:
            return set()
        
        return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}
    
    def generate_ngram_positions(self, tokens: List[str], n: int) -> Dict[str, List[int]]:
        """Generate n-grams with positions"""
        if len(tokens) < n:
            return {}
        
        positions = {}
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            if ngram not in positions:
                positions[ngram] = []
            positions[ngram].append(i)
        
        return positions
    
    # =========================================================================
    # CITATION DETECTION
    # =========================================================================
    
    def detect_citation_context(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text appears to be properly cited.
        
        Returns:
            (has_citation, citation_type)
        """
        for pattern in self.citation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Determine citation type
                if re.search(r'\[\d+\]', text):
                    return True, "numbered"
                elif re.search(r'\(\d{4}\)', text):
                    return True, "author_year"
                elif re.search(r'et\s+al\.', text, re.IGNORECASE):
                    return True, "author_reference"
                else:
                    return True, "contextual"
        
        return False, None
    
    def is_boilerplate(self, text: str) -> bool:
        """Check if text is likely boilerplate"""
        text_lower = text.lower()
        for pattern in self.boilerplate_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def is_common_scientific_phrase(self, text: str) -> bool:
        """Check if text is a common scientific phrase"""
        text_lower = text.lower()
        for pattern in self.common_scientific_phrases:
            if re.search(pattern, text_lower):
                return True
        return False
    
    # =========================================================================
    # MULTI-SCALE N-GRAM ANALYSIS
    # =========================================================================
    
    def multi_scale_containment(
        self, 
        query_tokens: List[str], 
        source_tokens: List[str]
    ) -> Dict[int, float]:
        """
        Calculate containment at multiple n-gram scales.
        
        Returns:
            Dict mapping n-gram size to containment score
        """
        scores = {}
        
        for n in self.ngram_sizes:
            query_ngrams = self.generate_ngrams(query_tokens, n)
            source_ngrams = self.generate_ngrams(source_tokens, n)
            
            if not query_ngrams:
                scores[n] = 0.0
            else:
                intersection = len(query_ngrams & source_ngrams)
                scores[n] = intersection / len(query_ngrams)
        
        return scores
    
    def best_ngram_score(self, ngram_scores: Dict[int, float]) -> Tuple[float, int]:
        """Get best n-gram score and its scale"""
        if not ngram_scores:
            return 0.0, self.primary_ngram_size
        
        best_n = max(ngram_scores.keys(), key=lambda k: ngram_scores[k])
        return ngram_scores[best_n], best_n
    
    # =========================================================================
    # SIMILARITY METRICS
    # =========================================================================
    
    def calculate_jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def calculate_containment(self, query_set: Set[str], source_set: Set[str]) -> float:
        """Calculate containment of query in source"""
        if not query_set:
            return 0.0
        intersection = len(query_set & source_set)
        return intersection / len(query_set)
    
    def calculate_word_overlap(
        self, 
        query_tokens: List[str], 
        source_tokens: List[str]
    ) -> Tuple[float, int]:
        """Calculate word-level overlap"""
        if not query_tokens:
            return 0.0, 0
        
        query_counter = Counter(query_tokens)
        source_counter = Counter(source_tokens)
        
        matched = sum(
            min(count, source_counter.get(word, 0))
            for word, count in query_counter.items()
        )
        
        return matched / len(query_tokens), matched
    
    def calculate_char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity (optimized)"""
        # Use simpler approach for long texts
        if len(text1) > 1000 or len(text2) > 1000:
            # Approximate with word-level comparison
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / max(len(words1), len(words2))
        
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    # =========================================================================
    # SEMANTIC-LEXICAL DIVERGENCE CHECK (KEY INNOVATION)
    # =========================================================================
    
    def check_semantic_lexical_divergence(
        self,
        semantic_score: float,
        lexical_score: float
    ) -> Tuple[bool, str]:
        """
        Check for divergence between semantic and lexical similarity.
        
        High semantic + low lexical = same topic, different words = NOT plagiarism
        High lexical + low semantic = unusual, might be coincidental phrases
        
        Args:
            semantic_score: Cosine similarity from embeddings (0-1)
            lexical_score: Combined lexical metric (0-1)
        
        Returns:
            (is_divergent, explanation)
        """
        gap = semantic_score - lexical_score
        
        # Case 1: High semantic, low lexical - LIKELY FALSE POSITIVE
        if semantic_score > 0.70 and lexical_score < 0.30:
            return True, "COMMON_TOPIC: Same subject matter but different wording"
        
        # Case 2: Very high semantic, moderate lexical - CHECK CAREFULLY
        if semantic_score > 0.85 and lexical_score < 0.50:
            return True, "TOPIC_OVERLAP: Related content, may be paraphrase or common knowledge"
        
        # Case 3: Large gap in general
        if gap > self.divergence_threshold:
            return True, f"DIVERGENT: Semantic-lexical gap of {gap:.2f}"
        
        # Case 4: Both high - LIKELY PLAGIARISM
        if semantic_score > 0.80 and lexical_score > 0.60:
            return False, "CONSISTENT: Both metrics indicate similarity"
        
        # Case 5: Both low - NO MATCH
        if semantic_score < 0.50 and lexical_score < 0.30:
            return False, "NO_MATCH: Low similarity on both metrics"
        
        return False, "NORMAL: Metrics are consistent"
    
    def calculate_adjusted_score(
        self,
        semantic_score: float,
        lexical_score: float,
        has_citation: bool,
        is_boilerplate: bool,
        is_common_phrase: bool
    ) -> Tuple[float, float]:
        """
        Calculate adjusted similarity score with confidence.
        
        IMPORTANT: This returns an adjusted score but the original lexical_score
        is preserved in the match result for threshold comparisons.
        
        Returns:
            (adjusted_score, confidence)
        """
        # Base combination - weight lexical heavily for plagiarism detection
        # High lexical = definite match, high semantic alone = possible paraphrase
        raw_score = (
            semantic_score * 0.30 +
            lexical_score * 0.70  # Lexical is primary indicator
        )
        
        # Boost score when BOTH semantic and lexical agree (high confidence match)
        if semantic_score >= 0.60 and lexical_score >= 0.60:
            # Both indicators agree - boost confidence
            raw_score = max(raw_score, (semantic_score + lexical_score) / 2 * 1.1)
        
        # Confidence starts at 1.0
        confidence = 1.0
        
        # Check divergence
        is_divergent, _ = self.check_semantic_lexical_divergence(
            semantic_score, lexical_score
        )
        
        if is_divergent:
            # High semantic but low lexical could be:
            # 1. Paraphrase (legitimate detection)
            # 2. False positive (common topic)
            # Use lexical as the deciding factor
            if lexical_score >= 0.5:
                raw_score *= 0.85  # Minor reduction - lexical evidence is strong
                confidence *= 0.75
            elif lexical_score >= 0.35:
                raw_score *= 0.70  # Moderate reduction
                confidence *= 0.60
            else:
                raw_score *= 0.5  # Large reduction - weak lexical = likely false positive
                confidence *= 0.4
        
        # Citation reduces plagiarism score slightly
        if has_citation:
            raw_score *= (1 - self.citation_boost * 0.5)  # Less aggressive
            confidence *= 0.85
        
        # Boilerplate reduces score but not as aggressively
        if is_boilerplate:
            raw_score *= 0.5
            confidence *= 0.5
        
        # Common scientific phrases - minimal reduction
        if is_common_phrase:
            raw_score *= 0.85
            confidence *= 0.8
        
        return min(1.0, raw_score), confidence
    
    # =========================================================================
    # SENTENCE-LEVEL ANALYSIS
    # =========================================================================
    
    def analyze_sentences(
        self,
        query_text: str,
        source_text: str,
        threshold: float = 0.6
    ) -> List[SentenceMatch]:
        """
        Perform sentence-level comparison.
        
        Args:
            query_text: Query document text
            source_text: Source document text
            threshold: Minimum similarity for a sentence match
        
        Returns:
            List of sentence-level matches
        """
        query_sentences = self.sentence_tokenize(query_text)
        source_sentences = self.sentence_tokenize(source_text)
        
        if not query_sentences or not source_sentences:
            return []
        
        matches = []
        
        for q_sent, q_start, q_end in query_sentences:
            q_tokens = self.tokenize(q_sent)
            if len(q_tokens) < 5:
                continue
            
            best_match = None
            best_score = 0.0
            
            for s_sent, _, _ in source_sentences:
                s_tokens = self.tokenize(s_sent)
                if len(s_tokens) < 5:
                    continue
                
                # Quick word overlap check
                overlap, _ = self.calculate_word_overlap(q_tokens, s_tokens)
                
                if overlap > best_score:
                    best_score = overlap
                    best_match = s_sent
            
            if best_score >= threshold and best_match:
                has_citation, _ = self.detect_citation_context(q_sent)
                
                matches.append(SentenceMatch(
                    query_sentence=q_sent,
                    source_sentence=best_match,
                    similarity=best_score,
                    is_cited=has_citation,
                    start_pos=q_start,
                    end_pos=q_end
                ))
        
        return matches
    
    # =========================================================================
    # MATCH CLASSIFICATION
    # =========================================================================
    
    def classify_match(
        self,
        adjusted_score: float,
        confidence: float,
        lexical_score: float,
        semantic_score: float,
        has_citation: bool,
        is_divergent: bool,
        sentence_matches: List[SentenceMatch]
    ) -> str:
        """
        Classify the type of match with enhanced logic.
        
        Uses BOTH adjusted_score and lexical_score for classification.
        lexical_score is more reliable for actual plagiarism detection.
        """
        # Count high-confidence sentence matches
        strong_sentence_matches = [
            m for m in sentence_matches 
            if m.similarity > 0.8 and not m.is_cited
        ]
        
        # Use the higher of adjusted or lexical for classification
        effective_score = max(adjusted_score, lexical_score)
        
        # Only flag as COMMON_TOPIC if lexical is truly low
        if is_divergent and lexical_score < 0.30:  # Was 0.40
            if semantic_score > 0.75:
                return "COMMON_TOPIC"
            else:
                return "NO_MATCH"
        
        # Properly cited text is a special case
        # IMPORTANT: Having citations IN the text doesn't mean it's properly cited
        # Properly cited means: the source is referenced AND it's a quote or paraphrase with attribution
        # For now, we should NOT auto-classify as PROPER_CITATION based on citation patterns alone
        # because copied text often contains citations from the original paper
        
        # Only mark as PROPER_CITATION if:
        # 1. Very low lexical overlap (just common phrases) AND
        # 2. Has citation pattern AND
        # 3. Low semantic score (not really similar)
        if has_citation and lexical_score < 0.35 and semantic_score < 0.50:
            return "PROPER_CITATION"
        
        # Exact or near-exact copy (lexical is the definitive indicator)
        if lexical_score >= 0.95:
            return "EXACT_COPY"
        
        if lexical_score >= 0.85 or effective_score >= self.exact_threshold:
            return "DIRECT_COPY"
        
        # High similarity
        if lexical_score >= 0.70 or effective_score >= self.high_threshold:
            if lexical_score > 0.75:
                return "DIRECT_COPY"
            elif len(strong_sentence_matches) > 3:
                return "MOSAIC"
            else:
                return "PARAPHRASE"
        
        # Moderate similarity
        if lexical_score >= 0.50 or effective_score >= self.moderate_threshold:
            if len(strong_sentence_matches) > 2:
                return "MOSAIC"
            elif confidence < 0.4:  # Was 0.5
                return "COMMON_KNOWLEDGE"
            else:
                return "PARAPHRASE"
        
        # Low similarity - still might be paraphrase
        if lexical_score > 0.30 or effective_score > 0.35:
            return "COMMON_KNOWLEDGE"
        
        return "NO_MATCH"
    
    # =========================================================================
    # MAIN COMPARISON METHOD
    # =========================================================================
    
    def compare(
        self, 
        query_text: str, 
        source_text: str,
        semantic_score: float = None  # Optional: from embedding search
    ) -> EnhancedTextMatch:
        """
        Compare query text against source text with enhanced analysis.
        
        Args:
            query_text: Text to check for plagiarism
            source_text: Source text to compare against
            semantic_score: Optional pre-computed semantic similarity
        
        Returns:
            EnhancedTextMatch with all metrics and classification
        """
        # Tokenize
        query_tokens = self.tokenize(query_text)
        source_tokens = self.tokenize(source_text)
        
        if not query_tokens or not source_tokens:
            return self._empty_match(query_text, source_text)
        
        # Multi-scale n-gram analysis
        ngram_scores = self.multi_scale_containment(query_tokens, source_tokens)
        best_ngram, best_scale = self.best_ngram_score(ngram_scores)
        
        # Traditional metrics with primary n-gram size
        query_ngrams = self.generate_ngrams(query_tokens, self.primary_ngram_size)
        source_ngrams = self.generate_ngrams(source_tokens, self.primary_ngram_size)
        
        containment = self.calculate_containment(query_ngrams, source_ngrams)
        jaccard = self.calculate_jaccard(query_ngrams, source_ngrams)
        word_overlap, matched_count = self.calculate_word_overlap(query_tokens, source_tokens)
        char_sim = self.calculate_char_similarity(query_text, source_text)
        
        # Combined lexical score (use best n-gram score)
        lexical_score = (
            best_ngram * 0.35 +
            word_overlap * 0.35 +
            char_sim * 0.20 +
            jaccard * 0.10
        )
        
        # If no semantic score provided, estimate from lexical
        if semantic_score is None:
            # Rough estimate - in production, use actual embeddings
            semantic_score = (lexical_score + char_sim) / 2
        
        # Citation detection
        has_citation, citation_type = self.detect_citation_context(query_text)
        
        # Boilerplate and common phrase detection
        is_boilerplate = self.is_boilerplate(query_text)
        is_common = self.is_common_scientific_phrase(query_text)
        
        # Check semantic-lexical divergence
        is_divergent, divergence_reason = self.check_semantic_lexical_divergence(
            semantic_score, lexical_score
        )
        
        # Calculate adjusted score and confidence
        adjusted_score, confidence = self.calculate_adjusted_score(
            semantic_score,
            lexical_score,
            has_citation,
            is_boilerplate,
            is_common
        )
        
        # Sentence-level analysis - always do this if semantic score is high
        # This catches cases where query chunk has mixed content but some sentences match perfectly
        sentence_matches = []
        if adjusted_score > 0.3 or lexical_score > 0.4 or semantic_score > 0.6:
            sentence_matches_raw = self.analyze_sentences(
                query_text, source_text, threshold=0.6
            )
            sentence_matches = [
                {
                    'query': m.query_sentence[:200],
                    'source': m.source_sentence[:200],
                    'similarity': m.similarity,
                    'is_cited': m.is_cited,
                    'position': (m.start_pos, m.end_pos)
                }
                for m in sentence_matches_raw
            ]
            
            # BOOST: If we have strong sentence matches, boost the score
            # This handles cases where query chunk has mixed content but some sentences are copied
            if sentence_matches:
                # Get best sentence match similarity
                best_sentence_sim = max(m['similarity'] for m in sentence_matches)
                num_strong_matches = sum(1 for m in sentence_matches if m['similarity'] >= 0.7)
                
                # If we have very strong sentence matches, boost the adjusted score
                if best_sentence_sim >= 0.8 and num_strong_matches >= 1:
                    # Blend in the best sentence similarity
                    sentence_boost = best_sentence_sim * 0.3 + adjusted_score * 0.7
                    adjusted_score = max(adjusted_score, sentence_boost)
                    confidence = min(1.0, confidence + 0.1 * num_strong_matches)
        
        # Classify match type
        match_type = self.classify_match(
            adjusted_score,
            confidence,
            lexical_score,
            semantic_score,
            has_citation,
            is_divergent,
            [SentenceMatch(**{
                'query_sentence': m['query'],
                'source_sentence': m['source'],
                'similarity': m['similarity'],
                'is_cited': m['is_cited'],
                'start_pos': m['position'][0],
                'end_pos': m['position'][1]
            }) for m in sentence_matches]
        )
        
        # Find matching n-grams and spans
        query_ngram_pos = self.generate_ngram_positions(query_tokens, best_scale)
        matching_ngrams = [ng for ng in query_ngram_pos.keys() if ng in source_ngrams]
        matched_spans = self._merge_spans(query_tokens, matching_ngrams, query_ngram_pos, best_scale)
        
        return EnhancedTextMatch(
            source_text=source_text,
            query_text=query_text,
            similarity_score=adjusted_score,
            confidence=confidence,
            match_type=match_type,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            divergence_flag=is_divergent,
            word_overlap_percentage=word_overlap,
            containment=containment,
            jaccard=jaccard,
            char_similarity=char_sim,
            ngram_scores=ngram_scores,
            best_ngram_scale=best_scale,
            matched_word_count=matched_count,
            total_query_words=len(query_tokens),
            matched_spans=matched_spans,
            matched_ngrams=matching_ngrams[:20],
            sentence_matches=sentence_matches,
            has_citation_context=has_citation,
            citation_type=citation_type
        )
    
    def _empty_match(self, query_text: str, source_text: str) -> EnhancedTextMatch:
        """Return empty match result"""
        return EnhancedTextMatch(
            source_text=source_text,
            query_text=query_text,
            similarity_score=0.0,
            confidence=0.0,
            match_type="NO_MATCH",
            semantic_score=0.0,
            lexical_score=0.0,
            divergence_flag=False,
            word_overlap_percentage=0.0,
            containment=0.0,
            jaccard=0.0,
            char_similarity=0.0,
            ngram_scores={},
            best_ngram_scale=5,
            matched_word_count=0,
            total_query_words=0,
            matched_spans=[],
            matched_ngrams=[],
            sentence_matches=[],
            has_citation_context=False,
            citation_type=None
        )
    
    def _merge_spans(
        self,
        tokens: List[str],
        matching_ngrams: List[str],
        ngram_positions: Dict[str, List[int]],
        ngram_size: int
    ) -> List[Tuple[int, int]]:
        """Merge consecutive matching n-grams into spans"""
        if not matching_ngrams:
            return []
        
        match_positions = set()
        for ngram in matching_ngrams:
            for pos in ngram_positions.get(ngram, []):
                for i in range(ngram_size):
                    if pos + i < len(tokens):
                        match_positions.add(pos + i)
        
        if not match_positions:
            return []
        
        positions = sorted(match_positions)
        spans = []
        start = end = positions[0]
        
        for pos in positions[1:]:
            if pos <= end + 1:
                end = pos
            else:
                if end - start + 1 >= self.min_match_length:
                    spans.append((start, end + 1))
                start = end = pos
        
        if end - start + 1 >= self.min_match_length:
            spans.append((start, end + 1))
        
        return spans
    
    # =========================================================================
    # BATCH COMPARISON
    # =========================================================================
    
    def batch_compare(
        self,
        query_text: str,
        source_texts: List[Dict],
        semantic_scores: List[float] = None,
        top_k: int = 10,
        min_score: float = 0.2
    ) -> List[Dict]:
        """
        Compare query against multiple sources.
        
        Args:
            query_text: Text to check
            source_texts: List of dicts with 'text' and metadata
            semantic_scores: Optional list of semantic scores from embedding search
            top_k: Number of top matches to return
            min_score: Minimum score to include
        
        Returns:
            Top matches sorted by score
        """
        results = []
        
        for i, source in enumerate(source_texts):
            source_text = source.get('text', '')
            if not source_text:
                continue
            
            sem_score = semantic_scores[i] if semantic_scores and i < len(semantic_scores) else None
            
            match = self.compare(query_text, source_text, semantic_score=sem_score)
            
            # Filter by adjusted score AND confidence
            if match.similarity_score >= min_score and match.confidence >= 0.3:
                # Skip if flagged as common topic with low lexical
                if match.match_type == "COMMON_TOPIC":
                    continue
                    
                results.append({
                    'match': match,
                    'metadata': {k: v for k, v in source.items() if k != 'text'}
                })
        
        # Sort by similarity score (adjusted)
        results.sort(key=lambda x: x['match'].similarity_score, reverse=True)
        
        return results[:top_k]


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# =============================================================================

class TextPlagiarismDetector(EnhancedTextPlagiarismDetector):
    """
    Backward-compatible wrapper for enhanced detector.
    Maintains same interface as original TextPlagiarismDetector.
    """
    
    def __init__(
        self,
        ngram_size: int = 5,
        min_match_length: int = 5,
        exact_threshold: float = 0.95,
        high_threshold: float = 0.80,
        moderate_threshold: float = 0.50
    ):
        # Convert single ngram_size to multi-scale
        ngram_sizes = [3, 4, ngram_size, 7] if ngram_size == 5 else [ngram_size - 1, ngram_size, ngram_size + 2]
        
        super().__init__(
            ngram_sizes=ngram_sizes,
            min_match_length=min_match_length,
            exact_threshold=exact_threshold,
            high_threshold=high_threshold,
            moderate_threshold=moderate_threshold
        )
        
        self.ngram_size = ngram_size  # Keep for compatibility


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    detector = EnhancedTextPlagiarismDetector()
    
    # Test case 1: Exact copy
    print("=" * 70)
    print("TEST 1: EXACT COPY")
    print("=" * 70)
    
    query1 = """
    Animal behavior is a valuable endpoint to assess brain function in healthy 
    and disease states. From insects to rodents to non-human primates, behavioral 
    outputs have been instrumental in understanding neural circuit function.
    """
    
    source1 = """
    Animal behavior is a valuable endpoint to assess brain function in healthy 
    and disease states. From insects to rodents to non-human primates, behavioral 
    outputs have been instrumental in understanding neural circuit function in 
    normal and pathological conditions.
    """
    
    result1 = detector.compare(query1, source1, semantic_score=0.95)
    print(f"Match Type: {result1.match_type}")
    print(f"Adjusted Score: {result1.similarity_score:.3f}")
    print(f"Confidence: {result1.confidence:.3f}")
    print(f"Lexical Score: {result1.lexical_score:.3f}")
    print(f"Divergence Flag: {result1.divergence_flag}")
    
    # Test case 2: Same topic, different words (should NOT be plagiarism)
    print("\n" + "=" * 70)
    print("TEST 2: SAME TOPIC, DIFFERENT WORDS (should be COMMON_TOPIC)")
    print("=" * 70)
    
    query2 = """
    Machine learning algorithms have revolutionized medical image analysis.
    Deep neural networks can now detect cancer in radiology scans with high accuracy.
    These AI systems are transforming clinical practice worldwide.
    """
    
    source2 = """
    Artificial intelligence is changing how doctors interpret medical images.
    Computer vision models trained on large datasets achieve remarkable performance
    in identifying malignant tumors from X-rays and CT scans. Healthcare is being
    transformed by these technological advances.
    """
    
    result2 = detector.compare(query2, source2, semantic_score=0.82)
    print(f"Match Type: {result2.match_type}")
    print(f"Adjusted Score: {result2.similarity_score:.3f}")
    print(f"Confidence: {result2.confidence:.3f}")
    print(f"Lexical Score: {result2.lexical_score:.3f}")
    print(f"Semantic Score: {result2.semantic_score:.3f}")
    print(f"Divergence Flag: {result2.divergence_flag}")
    
    # Test case 3: Properly cited text
    print("\n" + "=" * 70)
    print("TEST 3: PROPERLY CITED TEXT")
    print("=" * 70)
    
    query3 = """
    According to Smith et al. (2023), animal behavior is a valuable endpoint 
    to assess brain function in healthy and disease states. As reported by 
    previous studies [1,2], behavioral outputs have been instrumental in 
    understanding neural circuit function.
    """
    
    result3 = detector.compare(query3, source1, semantic_score=0.88)
    print(f"Match Type: {result3.match_type}")
    print(f"Adjusted Score: {result3.similarity_score:.3f}")
    print(f"Confidence: {result3.confidence:.3f}")
    print(f"Has Citation: {result3.has_citation_context}")
    print(f"Citation Type: {result3.citation_type}")
    
    # Test case 4: Paraphrase
    print("\n" + "=" * 70)
    print("TEST 4: PARAPHRASE")
    print("=" * 70)
    
    query4 = """
    Studying animal behavior provides crucial insights into how the brain works
    in both healthy conditions and when disease is present. Research spanning
    from insect models through rodent experiments to primate studies has shown
    that behavioral measurements are key to understanding how neural circuits operate.
    """
    
    result4 = detector.compare(query4, source1, semantic_score=0.78)
    print(f"Match Type: {result4.match_type}")
    print(f"Adjusted Score: {result4.similarity_score:.3f}")
    print(f"Confidence: {result4.confidence:.3f}")
    print(f"Lexical Score: {result4.lexical_score:.3f}")
    print(f"N-gram Scores: {result4.ngram_scores}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
