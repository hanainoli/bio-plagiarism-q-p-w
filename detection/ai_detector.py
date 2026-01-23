#!/usr/bin/env python3
"""
AI Content Detector
====================
Detects AI-generated text in academic papers using multiple techniques:
1. Perplexity analysis
2. Burstiness detection
3. Vocabulary patterns
4. Sentence structure analysis

Usage:
    from detection.ai_detector import AIDetector
    
    detector = AIDetector()
    result = detector.analyze_text(text)
    print(f"AI probability: {result['ai_probability']}")
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AIDetectionResult:
    """Result of AI content detection"""
    ai_probability: float  # 0-1 probability text is AI-generated
    confidence: float  # Confidence in the prediction
    indicators: Dict[str, float]  # Individual indicator scores
    flags: List[str]  # Specific flags raised
    verdict: str  # "likely_human", "uncertain", "likely_ai"


class AIDetector:
    """
    Detect AI-generated content in text.
    
    Uses statistical analysis to identify patterns typical of AI-generated text:
    - Unusually consistent sentence lengths
    - Low vocabulary diversity
    - Repetitive phrase patterns
    - Lack of "burstiness" in writing style
    """
    
    def __init__(self):
        """Initialize the AI detector"""
        # Common AI writing patterns
        self.ai_phrases = [
            "it's important to note",
            "it is worth noting",
            "in conclusion",
            "to summarize",
            "as mentioned earlier",
            "it should be noted",
            "furthermore",
            "moreover",
            "in this context",
            "it is essential",
            "plays a crucial role",
            "significant impact",
            "comprehensive analysis",
            "in-depth understanding",
            "leveraging",
            "utilizing",
            "implementing",
            "facilitating",
            "encompassing",
            "streamlining",
        ]
        
        # Academic filler phrases often overused by AI
        self.filler_phrases = [
            "in order to",
            "due to the fact that",
            "it is clear that",
            "there is no doubt that",
            "it goes without saying",
            "needless to say",
            "as a matter of fact",
            "for the purpose of",
            "in the event that",
            "at the present time",
        ]
        
        logger.info("AI Detector initialized")
    
    def analyze_text(self, text: str) -> AIDetectionResult:
        """
        Analyze text for AI-generated content.
        
        Args:
            text: Text to analyze
            
        Returns:
            AIDetectionResult with probability and indicators
        """
        if not text or len(text) < 100:
            return AIDetectionResult(
                ai_probability=0.0,
                confidence=0.0,
                indicators={},
                flags=["text_too_short"],
                verdict="uncertain"
            )
        
        indicators = {}
        flags = []
        
        # 1. Sentence length consistency (AI tends to be more uniform)
        sent_variance = self._analyze_sentence_variance(text)
        indicators["sentence_variance"] = sent_variance
        if sent_variance < 0.3:
            flags.append("low_sentence_variance")
        
        # 2. Vocabulary diversity
        vocab_diversity = self._analyze_vocabulary_diversity(text)
        indicators["vocabulary_diversity"] = vocab_diversity
        if vocab_diversity < 0.4:
            flags.append("low_vocabulary_diversity")
        
        # 3. AI phrase detection
        ai_phrase_score = self._detect_ai_phrases(text)
        indicators["ai_phrase_score"] = ai_phrase_score
        if ai_phrase_score > 0.3:
            flags.append("high_ai_phrase_usage")
        
        # 4. Burstiness analysis
        burstiness = self._analyze_burstiness(text)
        indicators["burstiness"] = burstiness
        if burstiness < 0.3:
            flags.append("low_burstiness")
        
        # 5. Repetition patterns
        repetition = self._analyze_repetition(text)
        indicators["repetition_score"] = repetition
        if repetition > 0.4:
            flags.append("high_repetition")
        
        # 6. Paragraph structure
        para_uniformity = self._analyze_paragraph_structure(text)
        indicators["paragraph_uniformity"] = para_uniformity
        if para_uniformity > 0.7:
            flags.append("uniform_paragraphs")
        
        # Calculate overall AI probability
        ai_probability = self._calculate_ai_probability(indicators)
        
        # Determine confidence based on text length and indicator agreement
        confidence = self._calculate_confidence(text, indicators)
        
        # Determine verdict
        if ai_probability > 0.7:
            verdict = "likely_ai"
        elif ai_probability < 0.3:
            verdict = "likely_human"
        else:
            verdict = "uncertain"
        
        return AIDetectionResult(
            ai_probability=round(ai_probability, 3),
            confidence=round(confidence, 3),
            indicators={k: round(v, 3) for k, v in indicators.items()},
            flags=flags,
            verdict=verdict
        )
    
    def _analyze_sentence_variance(self, text: str) -> float:
        """
        Analyze variance in sentence lengths.
        Human writing tends to have more variance.
        
        Returns:
            Normalized variance score (0-1, higher = more human-like)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return 0.5
        
        lengths = [len(s.split()) for s in sentences]
        
        if not lengths:
            return 0.5
        
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.5
        
        # Coefficient of variation
        cv = np.std(lengths) / mean_len
        
        # Normalize to 0-1 range (CV typically 0.3-0.8 for human text)
        normalized = min(cv / 0.6, 1.0)
        
        return normalized
    
    def _analyze_vocabulary_diversity(self, text: str) -> float:
        """
        Analyze vocabulary diversity using type-token ratio.
        
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        if len(words) < 50:
            return 0.5
        
        # Type-token ratio (unique words / total words)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # Adjusted for text length (TTR decreases with length)
        # Use root TTR for better normalization
        root_ttr = len(unique_words) / math.sqrt(len(words))
        
        # Normalize (typical range 5-15 for academic text)
        normalized = min(root_ttr / 10, 1.0)
        
        return normalized
    
    def _detect_ai_phrases(self, text: str) -> float:
        """
        Detect common AI-generated phrases.
        
        Returns:
            Score (0-1, higher = more AI phrases detected)
        """
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count < 50:
            return 0.0
        
        # Count AI phrases
        ai_count = sum(1 for phrase in self.ai_phrases if phrase in text_lower)
        filler_count = sum(1 for phrase in self.filler_phrases if phrase in text_lower)
        
        total_phrases = ai_count + filler_count
        
        # Normalize by word count (expect ~1 per 200 words for human text)
        score = (total_phrases / word_count) * 200
        
        return min(score, 1.0)
    
    def _analyze_burstiness(self, text: str) -> float:
        """
        Analyze burstiness (variation in complexity across text).
        Human writing tends to be more "bursty".
        
        Returns:
            Burstiness score (0-1, higher = more human-like)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 5:
            return 0.5
        
        # Calculate complexity per sentence (using word length as proxy)
        complexities = []
        for sent in sentences:
            words = sent.split()
            if words:
                avg_word_len = np.mean([len(w) for w in words])
                complexities.append(avg_word_len)
        
        if len(complexities) < 3:
            return 0.5
        
        # Calculate variance in complexity
        complexity_var = np.std(complexities)
        
        # Normalize (typical range 0.5-2.0 for human text)
        normalized = min(complexity_var / 1.5, 1.0)
        
        return normalized
    
    def _analyze_repetition(self, text: str) -> float:
        """
        Analyze n-gram repetition patterns.
        AI tends to repeat phrases more.
        
        Returns:
            Repetition score (0-1, higher = more repetition)
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        if len(words) < 50:
            return 0.0
        
        # Count trigrams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        
        # Count repeated trigrams
        repeated = sum(1 for count in trigram_counts.values() if count > 1)
        
        # Normalize
        repetition_ratio = repeated / len(trigrams) if trigrams else 0
        
        # Scale (typical range 0.01-0.1 for human text)
        return min(repetition_ratio * 10, 1.0)
    
    def _analyze_paragraph_structure(self, text: str) -> float:
        """
        Analyze paragraph structure uniformity.
        AI tends to create more uniform paragraphs.
        
        Returns:
            Uniformity score (0-1, higher = more uniform)
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        if len(paragraphs) < 3:
            return 0.5
        
        lengths = [len(p.split()) for p in paragraphs]
        
        mean_len = np.mean(lengths)
        if mean_len == 0:
            return 0.5
        
        # Coefficient of variation (lower = more uniform)
        cv = np.std(lengths) / mean_len
        
        # Invert and normalize (lower CV = higher uniformity score)
        uniformity = 1 - min(cv, 1.0)
        
        return uniformity
    
    def _calculate_ai_probability(self, indicators: Dict[str, float]) -> float:
        """
        Calculate overall AI probability from indicators.
        """
        weights = {
            "sentence_variance": -0.2,  # Higher variance = less AI
            "vocabulary_diversity": -0.15,  # Higher diversity = less AI
            "ai_phrase_score": 0.25,  # More AI phrases = more AI
            "burstiness": -0.2,  # Higher burstiness = less AI
            "repetition_score": 0.15,  # More repetition = more AI
            "paragraph_uniformity": 0.15,  # More uniform = more AI
        }
        
        score = 0.5  # Start at neutral
        
        for indicator, weight in weights.items():
            if indicator in indicators:
                value = indicators[indicator]
                # Adjust score based on deviation from neutral (0.5)
                deviation = value - 0.5
                score += weight * deviation * 2
        
        # Clamp to 0-1
        return max(0, min(1, score))
    
    def _calculate_confidence(self, text: str, indicators: Dict[str, float]) -> float:
        """
        Calculate confidence in the prediction.
        """
        # Base confidence on text length
        word_count = len(text.split())
        length_confidence = min(word_count / 500, 1.0)
        
        # Check indicator agreement
        values = list(indicators.values())
        if not values:
            return 0.3
        
        # If indicators mostly agree, higher confidence
        mean_val = np.mean(values)
        indicator_std = np.std(values)
        agreement_confidence = 1 - min(indicator_std * 2, 0.5)
        
        # Combine
        confidence = (length_confidence * 0.4) + (agreement_confidence * 0.6)
        
        return confidence
    
    def analyze_document(self, chunks: List[str]) -> Dict:
        """
        Analyze an entire document (multiple chunks).
        
        Args:
            chunks: List of text chunks from document
            
        Returns:
            Aggregated analysis results
        """
        if not chunks:
            return {
                "overall_ai_probability": 0.0,
                "confidence": 0.0,
                "verdict": "uncertain",
                "chunk_results": []
            }
        
        chunk_results = []
        probabilities = []
        confidences = []
        
        for i, chunk in enumerate(chunks):
            result = self.analyze_text(chunk)
            chunk_results.append({
                "chunk_index": i,
                "ai_probability": result.ai_probability,
                "confidence": result.confidence,
                "verdict": result.verdict,
                "flags": result.flags
            })
            probabilities.append(result.ai_probability)
            confidences.append(result.confidence)
        
        # Weighted average by confidence
        if sum(confidences) > 0:
            overall_prob = sum(p * c for p, c in zip(probabilities, confidences)) / sum(confidences)
        else:
            overall_prob = np.mean(probabilities)
        
        overall_confidence = np.mean(confidences)
        
        if overall_prob > 0.7:
            verdict = "likely_ai"
        elif overall_prob < 0.3:
            verdict = "likely_human"
        else:
            verdict = "uncertain"
        
        return {
            "overall_ai_probability": round(overall_prob, 3),
            "confidence": round(overall_confidence, 3),
            "verdict": verdict,
            "chunks_analyzed": len(chunks),
            "chunks_flagged_ai": sum(1 for r in chunk_results if r["verdict"] == "likely_ai"),
            "chunk_results": chunk_results
        }


# Convenience function
def detect_ai_content(text: str) -> Dict:
    """
    Quick function to detect AI content.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with ai_probability, confidence, verdict
    """
    detector = AIDetector()
    result = detector.analyze_text(text)
    return {
        "ai_probability": result.ai_probability,
        "confidence": result.confidence,
        "verdict": result.verdict,
        "flags": result.flags
    }


if __name__ == "__main__":
    # Test
    test_text = """
    In this comprehensive study, we present a thorough analysis of the molecular 
    mechanisms underlying cellular differentiation. It is important to note that 
    our findings have significant implications for the field. Furthermore, we 
    demonstrate that leveraging advanced computational methods facilitates a 
    deeper understanding of these complex processes. Moreover, our results 
    encompass a wide range of experimental conditions.
    """
    
    detector = AIDetector()
    result = detector.analyze_text(test_text)
    
    print(f"AI Probability: {result.ai_probability}")
    print(f"Confidence: {result.confidence}")
    print(f"Verdict: {result.verdict}")
    print(f"Flags: {result.flags}")
    print(f"Indicators: {result.indicators}")
