"""
Text Plagiarism Detector.
Detects and classifies text plagiarism including:
- Exact copy detection
- Paraphrase detection
- Mosaic plagiarism
- N-gram matching with containment metrics
"""

import re
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import hashlib


@dataclass
class TextMatch:
    """Result of text comparison"""
    source_text: str
    query_text: str
    
    similarity_score: float
    match_type: str
    
    word_overlap_percentage: float
    containment: float
    jaccard: float
    char_similarity: float
    
    matched_word_count: int
    total_query_words: int
    
    matched_spans: List[Tuple[int, int]] = field(default_factory=list)
    matched_ngrams: List[str] = field(default_factory=list)


class TextPlagiarismDetector:
    """Detect text plagiarism using multiple methods"""
    
    def __init__(
        self,
        ngram_size: int = 5,
        min_match_length: int = 5,
        exact_threshold: float = 0.95,
        high_threshold: float = 0.80,
        moderate_threshold: float = 0.50
    ):
        """
        Initialize the text plagiarism detector.
        
        Args:
            ngram_size: Size of n-grams for matching
            min_match_length: Minimum consecutive words for a match
            exact_threshold: Threshold for exact copy classification
            high_threshold: Threshold for high similarity classification
            moderate_threshold: Threshold for moderate similarity classification
        """
        self.ngram_size = ngram_size
        self.min_match_length = min_match_length
        self.exact_threshold = exact_threshold
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        
        # Common boilerplate phrases to filter
        self.boilerplate_patterns = [
            r'all rights reserved',
            r'copyright \d{4}',
            r'methods?\s+were\s+performed',
            r'data\s+(are|is)\s+available',
            r'statistical\s+analysis',
            r'p\s*[<>=]\s*0\.\d+',
            r'mean\s*[±+]\s*\w+',
            r'(figure|table|fig\.?|tab\.?)\s*\d+',
        ]
    
    # =========================================================================
    # TEXT PREPROCESSING
    # =========================================================================
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
        
        Returns:
            List of lowercase word tokens
        """
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split and filter empty tokens
        tokens = [t for t in text.split() if len(t) > 1]
        
        return tokens
    
    def generate_ngrams(self, tokens: List[str], n: int = None) -> Set[str]:
        """
        Generate n-grams from token list.
        
        Args:
            tokens: List of word tokens
            n: N-gram size (default: self.ngram_size)
        
        Returns:
            Set of n-gram strings
        """
        if n is None:
            n = self.ngram_size
        
        if len(tokens) < n:
            return set()
        
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.add(ngram)
        
        return ngrams
    
    def generate_ngram_positions(
        self, 
        tokens: List[str], 
        n: int = None
    ) -> Dict[str, List[int]]:
        """
        Generate n-grams with their positions.
        
        Args:
            tokens: List of word tokens
            n: N-gram size
        
        Returns:
            Dict mapping n-gram to list of start positions
        """
        if n is None:
            n = self.ngram_size
        
        if len(tokens) < n:
            return {}
        
        ngram_positions = {}
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            if ngram not in ngram_positions:
                ngram_positions[ngram] = []
            ngram_positions[ngram].append(i)
        
        return ngram_positions
    
    def is_boilerplate(self, text: str) -> bool:
        """Check if text is likely boilerplate content"""
        text_lower = text.lower()
        
        for pattern in self.boilerplate_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    # =========================================================================
    # SIMILARITY METRICS
    # =========================================================================
    
    def calculate_jaccard(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
        
        Returns:
            Jaccard similarity (0-1)
        """
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_containment(self, query_set: Set[str], source_set: Set[str]) -> float:
        """
        Calculate containment of query in source.
        Containment(A, B) = |A ∩ B| / |A|
        
        Args:
            query_set: Query n-grams
            source_set: Source n-grams
        
        Returns:
            Containment score (0-1)
        """
        if not query_set:
            return 0.0
        
        intersection = len(query_set & source_set)
        return intersection / len(query_set)
    
    def calculate_word_overlap(
        self, 
        query_tokens: List[str], 
        source_tokens: List[str]
    ) -> Tuple[float, int]:
        """
        Calculate word-level overlap.
        
        Args:
            query_tokens: Query word tokens
            source_tokens: Source word tokens
        
        Returns:
            Tuple of (overlap_percentage, matched_count)
        """
        if not query_tokens:
            return 0.0, 0
        
        query_counter = Counter(query_tokens)
        source_counter = Counter(source_tokens)
        
        # Count matching words
        matched = 0
        for word, count in query_counter.items():
            if word in source_counter:
                matched += min(count, source_counter[word])
        
        overlap = matched / len(query_tokens)
        
        return overlap, matched
    
    def calculate_char_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate character-level similarity using sequence matching.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        from difflib import SequenceMatcher
        
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    # =========================================================================
    # MATCH DETECTION
    # =========================================================================
    
    def find_matching_ngrams(
        self,
        query_ngrams: Dict[str, List[int]],
        source_ngrams: Set[str]
    ) -> List[str]:
        """
        Find n-grams that appear in both query and source.
        
        Args:
            query_ngrams: Query n-grams with positions
            source_ngrams: Source n-gram set
        
        Returns:
            List of matching n-grams
        """
        return [ng for ng in query_ngrams.keys() if ng in source_ngrams]
    
    def merge_consecutive_matches(
        self,
        query_tokens: List[str],
        matching_ngrams: List[str],
        query_ngram_positions: Dict[str, List[int]]
    ) -> List[Tuple[int, int]]:
        """
        Merge consecutive matching n-grams into spans.
        
        Args:
            query_tokens: Query word tokens
            matching_ngrams: List of matching n-grams
            query_ngram_positions: N-gram positions in query
        
        Returns:
            List of (start, end) spans in word indices
        """
        if not matching_ngrams:
            return []
        
        # Get all match positions
        match_positions = set()
        for ngram in matching_ngrams:
            for pos in query_ngram_positions.get(ngram, []):
                for i in range(self.ngram_size):
                    if pos + i < len(query_tokens):
                        match_positions.add(pos + i)
        
        if not match_positions:
            return []
        
        # Sort positions and merge consecutive
        positions = sorted(match_positions)
        spans = []
        start = positions[0]
        end = positions[0]
        
        for pos in positions[1:]:
            if pos <= end + 1:
                end = pos
            else:
                if end - start + 1 >= self.min_match_length:
                    spans.append((start, end + 1))
                start = pos
                end = pos
        
        # Add last span
        if end - start + 1 >= self.min_match_length:
            spans.append((start, end + 1))
        
        return spans
    
    # =========================================================================
    # MATCH CLASSIFICATION
    # =========================================================================
    
    def classify_match(
        self,
        containment: float,
        word_overlap: float,
        char_similarity: float,
        jaccard: float
    ) -> str:
        """
        Classify the type of match based on metrics.
        
        Args:
            containment: Containment score
            word_overlap: Word overlap percentage
            char_similarity: Character similarity
            jaccard: Jaccard similarity
        
        Returns:
            Match type string
        """
        # Calculate combined score
        combined = (
            containment * 0.4 +
            word_overlap * 0.3 +
            char_similarity * 0.2 +
            jaccard * 0.1
        )
        
        # Exact or near-exact copy
        if combined >= self.exact_threshold:
            if char_similarity > 0.95:
                return "EXACT_COPY"
            else:
                return "DIRECT_COPY"
        
        # High similarity (likely paraphrase)
        if combined >= self.high_threshold:
            if word_overlap > 0.7:
                return "DIRECT_COPY"
            else:
                return "PARAPHRASE"
        
        # Moderate similarity (mosaic or light paraphrase)
        if combined >= self.moderate_threshold:
            if containment > 0.6 and word_overlap < 0.5:
                return "MOSAIC"
            else:
                return "PARAPHRASE"
        
        # Low similarity but some overlap
        if containment > 0.2 or word_overlap > 0.2:
            return "COMMON_KNOWLEDGE"
        
        return "NO_MATCH"
    
    # =========================================================================
    # COMPLETE COMPARISON
    # =========================================================================
    
    def compare(self, query_text: str, source_text: str) -> TextMatch:
        """
        Compare query text against source text.
        
        Args:
            query_text: Text to check for plagiarism
            source_text: Source text to compare against
        
        Returns:
            TextMatch with all metrics and classification
        """
        # Tokenize
        query_tokens = self.tokenize(query_text)
        source_tokens = self.tokenize(source_text)
        
        if not query_tokens or not source_tokens:
            return TextMatch(
                source_text=source_text,
                query_text=query_text,
                similarity_score=0.0,
                match_type="NO_MATCH",
                word_overlap_percentage=0.0,
                containment=0.0,
                jaccard=0.0,
                char_similarity=0.0,
                matched_word_count=0,
                total_query_words=len(query_tokens)
            )
        
        # Generate n-grams
        query_ngram_positions = self.generate_ngram_positions(query_tokens)
        query_ngrams = set(query_ngram_positions.keys())
        source_ngrams = self.generate_ngrams(source_tokens)
        
        # Calculate metrics
        containment = self.calculate_containment(query_ngrams, source_ngrams)
        jaccard = self.calculate_jaccard(query_ngrams, source_ngrams)
        word_overlap, matched_count = self.calculate_word_overlap(
            query_tokens, source_tokens
        )
        char_similarity = self.calculate_char_similarity(query_text, source_text)
        
        # Find matching spans
        matching_ngrams = self.find_matching_ngrams(
            query_ngram_positions, source_ngrams
        )
        matched_spans = self.merge_consecutive_matches(
            query_tokens, matching_ngrams, query_ngram_positions
        )
        
        # Classify match type
        match_type = self.classify_match(
            containment, word_overlap, char_similarity, jaccard
        )
        
        # Calculate overall similarity score
        similarity_score = (
            containment * 0.4 +
            word_overlap * 0.3 +
            char_similarity * 0.2 +
            jaccard * 0.1
        )
        
        return TextMatch(
            source_text=source_text,
            query_text=query_text,
            similarity_score=similarity_score,
            match_type=match_type,
            word_overlap_percentage=word_overlap,
            containment=containment,
            jaccard=jaccard,
            char_similarity=char_similarity,
            matched_word_count=matched_count,
            total_query_words=len(query_tokens),
            matched_spans=matched_spans,
            matched_ngrams=matching_ngrams[:20]  # Limit to top 20
        )
    
    def batch_compare(
        self,
        query_text: str,
        source_texts: List[Dict],
        top_k: int = 10,
        min_score: float = 0.2
    ) -> List[Dict]:
        """
        Compare query against multiple sources.
        
        Args:
            query_text: Text to check
            source_texts: List of dicts with 'text' and metadata
            top_k: Number of top matches to return
            min_score: Minimum score to include
        
        Returns:
            Top matches sorted by score
        """
        results = []
        
        for source in source_texts:
            source_text = source.get('text', '')
            if not source_text:
                continue
            
            match = self.compare(query_text, source_text)
            
            if match.similarity_score >= min_score:
                results.append({
                    'match': match,
                    'metadata': {k: v for k, v in source.items() if k != 'text'}
                })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['match'].similarity_score, reverse=True)
        
        return results[:top_k]


class SemanticTextDetector:
    """Semantic-based plagiarism detection using embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with sentence transformer model.
        
        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not available")
                self._model = None
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            return None
        
        return self.model.encode(texts, convert_to_numpy=True)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.model is None:
            return 0.0
        
        embeddings = self.encode([text1, text2])
        
        # Cosine similarity
        sim = np.dot(embeddings[0], embeddings[1])
        sim /= (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8)
        
        return float(sim)


# Import numpy for SemanticTextDetector
try:
    import numpy as np
except ImportError:
    np = None


if __name__ == "__main__":
    # Test the detector
    detector = TextPlagiarismDetector()
    
    query = """
    Animal behavior is a valuable endpoint to assess brain function in healthy 
    and disease states. From insects to rodents to non-human primates, behavioral 
    outputs have been instrumental in understanding neural circuit function.
    """
    
    source = """
    Animal behavior is a valuable endpoint to assess brain function in healthy 
    and disease states. From insects to rodents to non-human primates, behavioral 
    outputs have been instrumental in understanding neural circuit function in 
    normal and pathological conditions. The development of high-resolution imaging 
    has enabled researchers to quantify behavior with unprecedented precision.
    """
    
    result = detector.compare(query, source)
    
    print(f"Match Type: {result.match_type}")
    print(f"Similarity Score: {result.similarity_score:.3f}")
    print(f"Containment: {result.containment:.3f}")
    print(f"Word Overlap: {result.word_overlap_percentage:.3f}")
    print(f"Jaccard: {result.jaccard:.3f}")
    print(f"Char Similarity: {result.char_similarity:.3f}")
    print(f"Matched Words: {result.matched_word_count}/{result.total_query_words}")
    print(f"Matched Spans: {result.matched_spans}")
