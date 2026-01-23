"""
World-Class Plagiarism Detector.
Main orchestrator for comprehensive plagiarism detection:
- Text plagiarism (exact, paraphrase, mosaic)
- Image plagiarism (copy, rotation, splicing)
- AI-generated content detection
- Multi-modal analysis
"""

import logging
# Suppress verbose HTTP request logs from httpx/qdrant
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

# Handle imports for both package and standalone usage
try:
    from ..models.schemas import (
        Paper, PaperID, Figure, Table, TextChunk,
        PlagiarismReport, TextMatch, ImageMatch, IntegrityResult,
        MatchType, ImageMatchType
    )
    from ..utils.id_generator import (
        generate_paper_id, generate_figure_id, generate_table_id, generate_chunk_id
    )
    from ..extraction.pdf_extractor import PDFExtractor, TextChunker
except ImportError:
    from models.schemas import (
        Paper, PaperID, Figure, Table, TextChunk,
        PlagiarismReport, TextMatch, ImageMatch, IntegrityResult,
        MatchType, ImageMatchType
    )
    from utils.id_generator import (
        generate_paper_id, generate_figure_id, generate_table_id, generate_chunk_id
    )
    from extraction.pdf_extractor import PDFExtractor, TextChunker

from .image_feature_extractor import MultiMethodFeatureExtractor
from .image_comparator import MultiMethodComparator
from .splice_detector import SpliceDetector

# Use enhanced text detector with false positive reduction
try:
    from .text_detector_enhanced import EnhancedTextPlagiarismDetector as TextPlagiarismDetector
    USING_ENHANCED_DETECTOR = True
except ImportError:
    from .text_detector import TextPlagiarismDetector
    USING_ENHANCED_DETECTOR = False

# Safe import for AIGeneratedDetector (may not exist)
try:
    from .ai_detector import AIGeneratedDetector
except ImportError:
    AIGeneratedDetector = None


@dataclass
class DetectorConfig:
    """Configuration for plagiarism detector"""
    # Text detection thresholds
    text_exact_threshold: float = 0.95
    text_high_threshold: float = 0.80
    text_moderate_threshold: float = 0.50
    text_ngram_size: int = 5
    
    # Enhanced detection options (NEW)
    use_enhanced_detection: bool = True  # Use semantic-lexical divergence check
    divergence_threshold: float = 0.45   # Gap threshold for false positive detection
    citation_boost: float = 0.3          # Score reduction for cited text
    min_confidence: float = 0.3          # Minimum confidence to report match
    
    # Image detection thresholds
    image_hash_threshold: int = 10
    image_cnn_threshold: float = 0.85
    image_combined_threshold: float = 0.75
    
    # Splice detection
    run_splice_detection: bool = True
    splice_threshold: float = 0.3
    
    # AI detection
    run_ai_detection: bool = True
    ai_threshold: float = 0.5
    
    # Processing options
    use_gpu: bool = True
    batch_size: int = 32
    max_text_matches: int = 500  # Increased to allow full paper coverage
    max_image_matches: int = 20


class WorldClassDetector:
    """Comprehensive plagiarism detection system"""
    
    def __init__(
        self,
        qdrant_client,
        wasabi_client,
        embedding_model=None,
        config: Optional[DetectorConfig] = None
    ):
        """
        Initialize the detector.
        
        Args:
            qdrant_client: Qdrant database client
            wasabi_client: Wasabi storage client
            embedding_model: Text embedding model
            config: Detector configuration
        """
        self.qdrant = qdrant_client
        self.wasabi = wasabi_client
        self.embedding_model = embedding_model
        self.config = config or DetectorConfig()
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.text_chunker = TextChunker(chunk_size=200, overlap=0.5)
        
        # Initialize text detector with enhanced options if available
        if USING_ENHANCED_DETECTOR and self.config.use_enhanced_detection:
            self.text_detector = TextPlagiarismDetector(
                ngram_sizes=[3, 4, self.config.text_ngram_size, 7],
                min_match_length=4,
                exact_threshold=self.config.text_exact_threshold,
                high_threshold=self.config.text_high_threshold,
                moderate_threshold=self.config.text_moderate_threshold,
                divergence_threshold=self.config.divergence_threshold,
                citation_boost=self.config.citation_boost
            )
            print("[INFO] Using ENHANCED text detector with false positive reduction")
        else:
            self.text_detector = TextPlagiarismDetector(
                ngram_size=self.config.text_ngram_size,
                exact_threshold=self.config.text_exact_threshold,
                high_threshold=self.config.text_high_threshold,
                moderate_threshold=self.config.text_moderate_threshold
            )
            print("[INFO] Using standard text detector")
        
        self.image_extractor = MultiMethodFeatureExtractor(use_gpu=self.config.use_gpu)
        self.image_comparator = MultiMethodComparator(
            phash_threshold=self.config.image_hash_threshold,
            cnn_threshold=self.config.image_cnn_threshold,
            combined_threshold=self.config.image_combined_threshold
        )
        
        if self.config.run_splice_detection:
            self.splice_detector = SpliceDetector()
        else:
            self.splice_detector = None
        
        if self.config.run_ai_detection:
            self.ai_detector = AIGeneratedDetector(use_gpu=self.config.use_gpu)
        else:
            self.ai_detector = None
    
    # =========================================================================
    # MAIN DETECTION METHODS
    # =========================================================================
    
    def analyze_paper(self, pdf_path: str, paper_id: Optional[str] = None, mode: str = 'full') -> PlagiarismReport:
        """
        Analyze a paper for plagiarism.
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper ID (generated if not provided)
            mode: 'full' (text + images), 'text' (text only), 'images' (images only)
        
        Returns:
            Complete PlagiarismReport
        """
        # Extract content from PDF
        extraction = self.pdf_extractor.extract(pdf_path)
        
        if not paper_id:
            paper_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize results
        text_matches = []
        image_matches = []
        integrity_issues = []
        text_score = 0.0
        image_score = 0.0
        
        # Analyze text (if mode is 'full' or 'text')
        if mode in ['full', 'text']:
            text_matches = self._analyze_text(extraction, paper_id)
            text_score = self._calculate_text_score(text_matches)
        
        # Analyze images (if mode is 'full' or 'images')
        if mode in ['full', 'images']:
            image_matches, integrity_issues = self._analyze_images(extraction, paper_id)
            image_score = self._calculate_image_score(image_matches)
        
        # Overall score and risk level
        if mode == 'text':
            overall_score = text_score
        elif mode == 'images':
            overall_score = image_score
            if integrity_issues:
                overall_score = max(overall_score, 0.5)
        else:
            overall_score = max(text_score, image_score)
            if integrity_issues:
                overall_score = max(overall_score, 0.5)
        
        risk_level = self._determine_risk_level(overall_score, text_matches, image_matches, integrity_issues)
        
        return PlagiarismReport(
            query_paper_id=paper_id,
            query_title=getattr(extraction, 'title', '') or pdf_path,
            analyzed_at=datetime.now(),
            text_matches=text_matches,
            text_plagiarism_score=text_score,
            image_matches=image_matches,
            image_plagiarism_score=image_score,
            integrity_issues=integrity_issues,
            overall_score=overall_score,
            risk_level=risk_level
        )
    
    def _convert_match_type(self, match_type_str: str) -> MatchType:
        """
        Convert match type string to MatchType enum with fallback handling.
        
        Maps enhanced detector types to schema types.
        """
        # Direct mapping for exact matches
        try:
            return MatchType(match_type_str)
        except ValueError:
            pass
        
        # Fallback mapping for types not in enum
        fallback_map = {
            'COMMON_TOPIC': MatchType.COMMON_KNOWLEDGE,  # Same topic = common knowledge
            'HIGH_SIMILARITY': MatchType.PARAPHRASE,
            'MODERATE_SIMILARITY': MatchType.PARAPHRASE,
            'LOW_SIMILARITY': MatchType.COMMON_KNOWLEDGE,
            'MINIMAL_SIMILARITY': MatchType.NO_MATCH,
        }
        
        return fallback_map.get(match_type_str, MatchType.PARAPHRASE)
    
    def analyze_text_only(self, text: str) -> List[TextMatch]:
        """
        TWO-STAGE TEXT ANALYSIS with SEGMENT EXTRACTION
        
        Stage 1: Semantic Search
        - Use embeddings to get candidate chunks from Qdrant
        - Fast retrieval based on meaning similarity
        
        Stage 2: Lexical Verification with Segment Extraction
        - For each candidate, find the BEST MATCHING SEGMENT
        - Calculate precise metrics on that segment
        - Re-rank by LEXICAL score (not semantic)
        - Report shows the ACTUAL MATCHING TEXT
        
        This ensures accurate matching regardless of chunk sizes.
        """
        import re
        
        # ===== HELPER FUNCTIONS =====
        
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
            Calculate overlap score between query and segment.
            
            ORIGINAL SIMPLE LOGIC - proven to work with single-topic papers.
            
            Returns (word_overlap, phrase_count, lexical_score)
            """
            if not query_tokens or not segment_tokens:
                return 0.0, 0, 0.0
            
            # Basic word sets
            q_set = set(query_tokens)
            s_set = set(segment_tokens)
            
            # Word overlap - simple and effective
            word_overlap = len(q_set & s_set) / len(q_set) if q_set else 0
            
            # Phrase matches (5-grams) - exact sequences prove copying
            phrase_count = 0
            phrase_ratio = 0.0
            if len(query_tokens) >= 5 and len(segment_tokens) >= 5:
                q_5grams = get_ngrams(query_tokens, 5)
                s_5grams = get_ngrams(segment_tokens, 5)
                matching_phrases = q_5grams & s_5grams
                phrase_count = len(matching_phrases)
                
                if len(q_5grams) > 0:
                    phrase_ratio = len(matching_phrases) / len(q_5grams)
            
            # LEXICAL SCORE: Original simple formula
            # Word overlap (60%) + Phrase ratio (40%)
            lexical_score = word_overlap * 0.6 + phrase_ratio * 0.4
            
            return word_overlap, phrase_count, lexical_score
        
        def find_best_matching_segment(query_tokens, source_text, source_tokens):
            """
            Slide a window across source to find the segment that best matches query.
            
            Returns: (best_segment_text, best_segment_tokens, word_overlap, phrase_count, lexical_score, matched_spans)
            """
            query_len = len(query_tokens)
            source_len = len(source_tokens)
            
            # If source is shorter than query, use entire source
            if source_len <= query_len:
                word_overlap, phrase_count, lexical_score = calculate_overlap_score(query_tokens, source_tokens)
                # For full match, span is entire query
                matched_spans = [(0, len(query_tokens))] if phrase_count > 0 else []
                return source_text, source_tokens, word_overlap, phrase_count, lexical_score, matched_spans
            
            # Sliding window - window size matches query length
            window_size = query_len
            step = max(5, query_len // 10)  # Step by 5 tokens or 10% of query
            
            best_score = 0
            best_start = 0
            best_window_size = window_size
            best_metrics = (0.0, 0, 0.0)
            
            for start in range(0, source_len - window_size + 1, step):
                end = start + window_size
                segment_tokens = source_tokens[start:end]
                
                word_overlap, phrase_count, lexical_score = calculate_overlap_score(query_tokens, segment_tokens)
                
                if lexical_score > best_score:
                    best_score = lexical_score
                    best_start = start
                    best_window_size = window_size
                    best_metrics = (word_overlap, phrase_count, lexical_score)
            
            # Also check a slightly larger window (1.2x query size) for partial matches
            larger_window = min(int(query_len * 1.2), source_len)
            for start in range(0, source_len - larger_window + 1, step):
                end = start + larger_window
                segment_tokens = source_tokens[start:end]
                
                word_overlap, phrase_count, lexical_score = calculate_overlap_score(query_tokens, segment_tokens)
                
                if lexical_score > best_score:
                    best_score = lexical_score
                    best_start = start
                    best_window_size = larger_window
                    best_metrics = (word_overlap, phrase_count, lexical_score)
            
            # Extract best segment text using the CORRECT window size
            best_end = min(best_start + best_window_size, source_len)
            best_segment_tokens = source_tokens[best_start:best_end]
            
            # Simple approach: join tokens back
            best_segment_text = ' '.join(best_segment_tokens)
            
            # Calculate matched_spans based on matching 5-grams positions
            matched_spans = []
            if best_metrics[1] > 0:  # phrase_count > 0
                q_5grams = get_ngrams(query_tokens, 5)
                s_5grams = get_ngrams(best_segment_tokens, 5)
                matching_phrases = q_5grams & s_5grams
                
                if matching_phrases:
                    # Find positions of matching phrases in query
                    match_positions = set()
                    for i in range(len(query_tokens) - 4):
                        phrase = ' '.join(query_tokens[i:i+5])
                        if phrase in matching_phrases:
                            for j in range(5):
                                match_positions.add(i + j)
                    
                    # Merge consecutive positions into spans
                    if match_positions:
                        positions = sorted(match_positions)
                        start_pos = end_pos = positions[0]
                        for pos in positions[1:]:
                            if pos <= end_pos + 1:
                                end_pos = pos
                            else:
                                matched_spans.append((start_pos, end_pos + 1))
                                start_pos = end_pos = pos
                        matched_spans.append((start_pos, end_pos + 1))
            
            return best_segment_text, best_segment_tokens, best_metrics[0], best_metrics[1], best_metrics[2], matched_spans
        
        # ===== STAGE 1: SEMANTIC SEARCH =====
        
        # Embed query text
        if self.embedding_model:
            vector = self.embedding_model.encode([text])[0]
        else:
            vector = np.random.randn(1024)  # Mock for testing
        
        # Search Qdrant - get many candidates
        results = self.qdrant.search_chunks(vector, limit=50, score_threshold=0.35)
        
        if not results:
            return []
        
        # ===== STAGE 2: LEXICAL VERIFICATION WITH SEGMENT EXTRACTION =====
        
        query_tokens = tokenize(text)
        all_candidates = []
        
        for result in results:
            source_text = result.payload.get('text', '')
            if not source_text:
                continue
            
            source_tokens = tokenize(source_text)
            semantic_score = result.score
            
            # FIND BEST MATCHING SEGMENT
            best_segment_text, best_segment_tokens, word_overlap, phrase_count, lexical_score, segment_matched_spans = \
                find_best_matching_segment(query_tokens, source_text, source_tokens)
            
            # Get additional info from text_detector
            if USING_ENHANCED_DETECTOR and hasattr(self.text_detector, 'compare'):
                match = self.text_detector.compare(text, best_segment_text, semantic_score=semantic_score)
            else:
                match = self.text_detector.compare(text, best_segment_text)
            
            all_candidates.append({
                'result': result,
                'match': match,
                'source_text': source_text,
                'best_segment': best_segment_text,
                'semantic_score': semantic_score,
                'word_overlap': word_overlap,
                'phrase_count': phrase_count,
                'lexical_score': lexical_score,
                'containment': match.containment,
                'matched_spans': segment_matched_spans  # Use our calculated spans
            })
        
        # ===== RE-RANK BY LEXICAL SCORE (not semantic!) =====
        all_candidates.sort(key=lambda x: (
            x['lexical_score'],     # Primary: lexical quality
            x['phrase_count'],      # Secondary: phrase matches
            x['word_overlap']       # Tertiary: word overlap
        ), reverse=True)
        
        # ===== BUILD FINAL MATCHES =====
        matches = []
        
        for cand in all_candidates:
            match = cand['match']
            result = cand['result']
            
            # FILTERING: Require meaningful lexical match
            # The key insight is that real plagiarism has HIGH content overlap
            # False positives have word overlap inflated by stopwords but low content overlap
            is_real_match = (
                cand['lexical_score'] >= 0.25 and cand['phrase_count'] >= 3
            ) or (
                cand['phrase_count'] >= 10  # Many phrases = definitely real
            ) or (
                cand['word_overlap'] >= 0.70 and cand['phrase_count'] >= 5
            )
            
            if not is_real_match:
                continue
            
            # Calculate final score
            effective_score = min(1.0, (
                cand['lexical_score'] * 0.5 +
                cand['word_overlap'] * 0.3 +
                cand['semantic_score'] * 0.2
            ))
            
            match_type_enum = self._convert_match_type(match.match_type)
            
            # USE BEST_SEGMENT in report (not full source_text!)
            match_data = {
                'query_text': text[:500],
                'query_section': "query",
                'source_paper_id': result.payload.get('paper_id', ''),
                'source_doi': result.payload.get('doi', ''),
                'source_title': result.payload.get('title', ''),
                'source_section': result.payload.get('section', ''),
                'source_text': cand['best_segment'][:500],  # MATCHED SEGMENT!
                'similarity_score': effective_score,
                'match_type': match_type_enum,
                'word_overlap_percentage': cand['word_overlap'],
                'containment': cand['containment'],
                'jaccard': match.jaccard,
                'matched_word_count': match.matched_word_count,
                'matched_spans': cand['matched_spans']  # Use our calculated spans!
            }
            
            matches.append(TextMatch(**match_data))
        
        # Sort by lexical quality
        matches.sort(key=lambda x: x.word_overlap_percentage, reverse=True)
        return matches[:20]
    
    def analyze_image_only(self, image, is_table: bool = False) -> Dict:
        """
        Analyze a single image.
        
        Args:
            image: PIL Image
            is_table: Whether this is a table image (searches bio_tables collection)
        
        Returns:
            Dict with matches, integrity, and AI detection results
        """
        # Extract features
        features = self.image_extractor.extract_all_features(
            image, "query_paper", "query_figure"
        )
        
        # Search database - search BOTH figures and tables
        all_results = []
        if features.cnn_vector is not None:
            # Search figures collection
            try:
                figure_results = self.qdrant.search_figures(features.cnn_vector, limit=50)
                for r in figure_results:
                    r.payload['_source_type'] = 'figure'
                all_results.extend(figure_results)
            except Exception as e:
                print(f"Error searching figures: {e}")
            
            # Search tables collection
            try:
                table_results = self.qdrant.search_tables(features.cnn_vector, limit=50)
                for r in table_results:
                    r.payload['_source_type'] = 'table'
                all_results.extend(table_results)
            except Exception as e:
                print(f"Error searching tables: {e}")
        
        # Compare with each result
        matches = []
        for result in all_results:
            # Use Qdrant search score as CNN similarity (it's from CNN vector search!)
            qdrant_cnn_score = result.score
            
            db_features = {
                'orientation_hashes': result.payload.get('perceptual_hashes', {}),
                'cnn_vector': None,  # Not stored in payload, but we have qdrant_cnn_score
                'ocr_regions': result.payload.get('ocr_data', {}).get('text_regions', []) or result.payload.get('ocr_regions', []),
                'ocr_data': result.payload.get('ocr_data', {}),  # Pass full ocr_data dict
                'radial_histogram': result.payload.get('rotation_invariant', {}).get('radial_histogram'),
                'zernike_moments': result.payload.get('rotation_invariant', {}).get('zernike_moments'),
                '_qdrant_cnn_score': qdrant_cnn_score,  # Pass the Qdrant score
                'type': result.payload.get('type', result.payload.get('_source_type', 'figure'))  # For table-specific weights
            }
            
            comparison = self.image_comparator.compare_images(
                self._features_to_dict(features),
                db_features
            )
            
            # Use Qdrant CNN score if comparison didn't have CNN vectors
            effective_cnn_score = comparison.cnn_similarity if comparison.cnn_similarity > 0 else qdrant_cnn_score
            effective_combined_score = max(comparison.combined_score, qdrant_cnn_score)
            
            # Check if this is a table comparison
            is_table = db_features.get('type') == 'table'
            
            # For tables: require OCR match to confirm plagiarism (unless exact hash match)
            # This prevents false positives from visually similar but different tables
            ocr_overlap = comparison.ocr_similarity
            hash_is_exact = comparison.hash_similarity >= 0.95
            
            if is_table and not hash_is_exact:
                # Table-specific plagiarism logic:
                # - If OCR data exists and overlaps < 0.3, likely false positive
                # - If OCR data missing from both sides, fall back to CNN+hash only
                query_has_ocr = len(features.ocr_regions) > 0 if features.ocr_regions else False
                db_has_ocr = len(db_features.get('ocr_regions', [])) > 0
                
                if query_has_ocr and db_has_ocr and ocr_overlap < 0.3:
                    # Low OCR overlap = different table content, skip this match
                    continue
                elif query_has_ocr and db_has_ocr:
                    # Use OCR-weighted scoring for tables
                    effective_combined_score = 0.3 * effective_cnn_score + 0.7 * ocr_overlap
            
            # Determine plagiarism using Qdrant score
            is_plagiarism = (
                comparison.is_plagiarism or
                qdrant_cnn_score >= 0.85 or  # High CNN similarity from Qdrant
                (qdrant_cnn_score >= 0.70 and comparison.hash_similarity > 0.5)  # Combined evidence
            )
            
            # Determine correct match_type based on effective scores
            if effective_cnn_score >= 0.99 or effective_combined_score >= 0.99:
                if comparison.detected_rotation not in ['original', None, '']:
                    match_type = 'ROTATED_COPY'
                elif comparison.is_flipped:
                    match_type = 'FLIPPED_COPY'
                else:
                    match_type = 'EXACT_COPY'
            elif effective_cnn_score >= 0.95 or effective_combined_score >= 0.95:
                match_type = 'NEAR_EXACT'
            elif effective_cnn_score >= 0.85:
                match_type = 'HIGH_SIMILARITY'
            elif effective_cnn_score >= 0.75:
                match_type = 'MODERATE_SIMILARITY'
            elif effective_cnn_score >= 0.70:
                match_type = 'LOW_SIMILARITY'
            else:
                match_type = 'MINIMAL_SIMILARITY'
            
            if is_plagiarism:
                source_type = result.payload.get('_source_type', 'figure')
                source_label = result.payload.get('label', '')
                if not source_label:
                    idx = result.payload.get('figure_index', result.payload.get('table_index', 0))
                    source_label = f"Table {idx + 1}" if source_type == 'table' else f"Figure {idx + 1}"
                
                matches.append({
                    'source_figure_id': result.id,
                    'source_paper_id': result.payload.get('paper_id'),
                    'source_doi': result.payload.get('doi'),
                    'source_label': source_label,
                    'source_type': source_type,
                    's3_path': result.payload.get('s3_path', ''),
                    'match_type': match_type,  # Use corrected match_type
                    'combined_score': effective_combined_score,
                    'phash_similarity': comparison.hash_similarity,
                    'phash_hamming': comparison.details.get('hash', {}).get('hamming_distance', 64),
                    'cnn_similarity': effective_cnn_score,
                    'qdrant_score': qdrant_cnn_score,  # Include raw Qdrant score
                    'ocr_similarity': comparison.ocr_similarity,
                    'detected_rotation': comparison.detected_rotation,
                    'is_flipped': comparison.is_flipped,
                    'details': comparison.details
                })
        
        # Sort by combined_score descending
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # LIMIT: Only return top 5 matches per query image (best matches only)
        matches = matches[:5]
        
        # Run integrity checks
        integrity = None
        if self.splice_detector:
            splice_result = self.splice_detector.analyze(image)
            integrity = {
                'is_manipulated': splice_result.is_manipulated,
                'manipulation_score': splice_result.manipulation_score,
                'ela_regions': len(splice_result.ela_suspicious_regions),
                'copy_move_pairs': len(splice_result.copy_move_pairs)
            }
        
        # Run AI detection
        ai_result = None
        if self.ai_detector:
            ai_detection = self.ai_detector.analyze(image)
            ai_result = {
                'is_ai_generated': ai_detection.is_ai_generated,
                'confidence': ai_detection.confidence,
                'risk_factors': ai_detection.risk_factors
            }
        
        return {
            'matches': matches,
            'integrity': integrity,
            'ai_detection': ai_result
        }
    
    # =========================================================================
    # INTERNAL ANALYSIS METHODS
    # =========================================================================
    
    def _analyze_text(self, extraction, paper_id: str) -> List[TextMatch]:
        """Analyze text content - find BEST match for each chunk"""
        all_matches = []
        
        # Get text to analyze
        full_text = getattr(extraction, 'full_text', '') or getattr(extraction, 'text', '')
        if not full_text:
            return []
        
        # Chunk the full text
        chunks = self.text_chunker.chunk_text(full_text, "body")
        
        print(f"  Analyzing {len(chunks)} text chunks...")
        
        for i, chunk in enumerate(chunks):
            chunk_matches = self.analyze_text_only(chunk['text'])
            
            # Keep only the BEST match for this chunk
            # ALWAYS prioritize LEXICAL match - it's the most accurate indicator of actual copying
            if chunk_matches:
                # Sort by word overlap - this is the most reliable metric
                chunk_matches.sort(key=lambda x: (
                    x.word_overlap_percentage,
                    x.containment,
                    x.similarity_score
                ), reverse=True)
                
                # Take the best lexical match
                best_match = chunk_matches[0]
                
                # Only include if there's meaningful overlap
                # This prevents false matches where semantic is high but text is different
                if best_match.word_overlap_percentage >= 0.30:
                    best_match.query_section = f"{chunk['section']}_chunk_{i}"
                    all_matches.append(best_match)
        
        # Sort final results by word overlap (most accurate)
        all_matches.sort(key=lambda x: x.word_overlap_percentage, reverse=True)
        
        print(f"  Found {len(all_matches)} chunk matches (1 per chunk)")
        
        return all_matches[:self.config.max_text_matches]
    
    def _analyze_images(self, extraction, paper_id: str) -> tuple:
        """
        Analyze images (figures AND tables) - SIMPLIFIED APPROACH
        
        For each query image:
        1. Extract features
        2. Search Qdrant for similar images
        3. Find the BEST match (highest score)
        4. Add ONE match to results (if above threshold)
        """
        all_matches = []
        integrity_issues = []
        
        # Initialize Wasabi for uploading query images
        wasabi = None
        try:
            from storage.wasabi_client import WasabiClient
            wasabi = WasabiClient()
            if not wasabi.access_key:
                wasabi = None
        except:
            wasabi = None
        
        # Generate unique session ID for this analysis
        import datetime
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Minimum score threshold for reporting a match
        # 0.75 = too many false positives (similar-looking charts)
        # 0.90 = balanced (catches copies with minor differences)
        # 0.95 = strict (only near-exact copies)
        MIN_SCORE_THRESHOLD = 0.90
        
        # For a PROPER match, we need confirmation beyond just CNN similarity
        # CNN alone gives false positives for similar-looking charts
        # Minimum 15% hash similarity required (unless CNN >= 98%)
        MIN_HASH_SIMILARITY = 0.15
        
        # =====================================================
        # ANALYZE FIGURES
        # =====================================================
        total_figures = len([f for f in extraction.figures if hasattr(f, 'image') and f.image is not None])
        if total_figures > 0:
            print(f"  Analyzing {total_figures} figures...")
        
        for fig_idx, fig in enumerate(extraction.figures):
            if not hasattr(fig, 'image') or fig.image is None:
                continue
            
            print(f"    Figure {fig_idx + 1}/{total_figures}: extracting features & searching...", end=" ", flush=True)
            
            # Create unique query figure ID
            page_num = getattr(fig, 'page_number', 0)
            fig_index = getattr(fig, 'figure_index', fig_idx)
            query_figure_id = f"query_{session_id}_fig_p{page_num}_{fig_index}"
            
            # Upload query image to Wasabi for verification
            query_s3_path = ""
            if wasabi:
                try:
                    safe_paper_id = paper_id.replace('/', '_').replace('.', '_').replace(':', '_')
                    query_s3_path = f"query_uploads/{safe_paper_id}/{session_id}/fig_{fig_idx}.png"
                    wasabi.upload_image(fig.image, query_s3_path)
                except Exception as e:
                    print(f"upload failed", end=" ")
            
            # Find BEST match for this figure
            best_match = self._find_best_match(fig.image, is_table=False)
            
            # =============================================================
            # VALIDATION: Only accept if it's a PROPER match
            # NO PROPER MATCH = REJECT (don't return wrong values)
            # =============================================================
            is_valid_match = False
            rejection_reason = None
            
            if not best_match:
                rejection_reason = "no candidates found"
            elif best_match['score'] < MIN_SCORE_THRESHOLD:
                rejection_reason = f"score too low ({best_match['score']:.1%} < {MIN_SCORE_THRESHOLD:.0%})"
            else:
                cnn_score = best_match['score']
                phash_sim = best_match.get('phash_similarity', 0.0)
                
                # A proper match requires BOTH visual AND structural similarity
                # Exception: very high CNN (>=98%) is accepted as likely exact copy
                if cnn_score >= 0.98:
                    is_valid_match = True  # Very high CNN = accept
                elif phash_sim >= MIN_HASH_SIMILARITY:
                    is_valid_match = True  # CNN + hash both confirm = accept
                else:
                    # CNN says similar but hash says different = FALSE POSITIVE
                    # This is NOT a proper match - REJECT IT
                    rejection_reason = f"false positive (CNN={cnn_score:.1%} but hash={phash_sim:.1%})"
            
            # Only add to results if it's a PROPER match
            if is_valid_match:
                print(f"✓ match found ({best_match['score']:.1%})")
                # Determine match type based on score
                match_type = self._score_to_match_type(
                    best_match['score'], 
                    best_match.get('detected_rotation', 'original'),
                    best_match.get('is_flipped', False)
                )
                
                all_matches.append(ImageMatch(
                    query_figure_id=query_figure_id,
                    source_figure_id=best_match['source_id'],
                    source_paper_id=best_match['paper_id'],
                    source_doi=best_match['doi'],
                    source_label=best_match['label'],
                    source_caption='',
                    combined_score=best_match['score'],
                    match_type=ImageMatchType(match_type),
                    phash_similarity=best_match.get('phash_similarity', 0.0),
                    phash_hamming=best_match.get('phash_hamming', 64),
                    cnn_similarity=best_match.get('cnn_score', best_match['score']),
                    ocr_text_overlap=best_match.get('ocr_similarity', 0.0),
                    detected_rotation=best_match.get('detected_rotation', 'original'),
                    is_flipped=best_match.get('is_flipped', False),
                    details={
                        'source_type': best_match.get('source_type', 'figure'),
                        's3_path': best_match.get('s3_path', ''),
                        'query_type': 'figure',
                        'query_s3_path': query_s3_path,
                        'query_page': page_num,
                        'query_index': fig_index
                    }
                ))
            else:
                # NO PROPER MATCH FOUND - reject (don't return wrong values)
                print(f"✗ rejected: {rejection_reason}")
            
            # Check integrity (splice detection, AI detection)
            if self.splice_detector:
                try:
                    splice_result = self.splice_detector.analyze(fig.image)
                    if splice_result.get('is_manipulated'):
                        integrity_issues.append(IntegrityResult(
                            figure_id=query_figure_id,
                            is_manipulated=True,
                            manipulation_score=splice_result.get('manipulation_score', 0)
                        ))
                except:
                    pass
        
        # =====================================================
        # ANALYZE TABLES
        # =====================================================
        total_tables = len([t for t in extraction.tables if hasattr(t, 'image') and t.image is not None])
        if total_tables > 0:
            print(f"  Analyzing {total_tables} tables...")
        
        for table_idx, table in enumerate(extraction.tables):
            if not hasattr(table, 'image') or table.image is None:
                continue
            
            print(f"    Table {table_idx + 1}/{total_tables}: extracting features & searching...", end=" ", flush=True)
            
            # Create unique query table ID
            page_num = getattr(table, 'page_number', 0)
            tbl_index = getattr(table, 'table_index', table_idx)
            query_table_id = f"query_{session_id}_table_p{page_num}_{tbl_index}"
            
            # Upload query table image to Wasabi
            query_s3_path = ""
            if wasabi:
                try:
                    safe_paper_id = paper_id.replace('/', '_').replace('.', '_').replace(':', '_')
                    query_s3_path = f"query_uploads/{safe_paper_id}/{session_id}/table_{table_idx}.png"
                    wasabi.upload_image(table.image, query_s3_path)
                except Exception as e:
                    print(f"upload failed", end=" ")
            
            # Find BEST match for this table
            best_match = self._find_best_match(table.image, is_table=True)
            
            # =============================================================
            # VALIDATION: Only accept if it's a PROPER match
            # NO PROPER MATCH = REJECT (don't return wrong values)
            # =============================================================
            is_valid_match = False
            rejection_reason = None
            
            if not best_match:
                rejection_reason = "no candidates found"
            elif best_match['score'] < MIN_SCORE_THRESHOLD:
                rejection_reason = f"score too low ({best_match['score']:.1%} < {MIN_SCORE_THRESHOLD:.0%})"
            else:
                cnn_score = best_match['score']
                phash_sim = best_match.get('phash_similarity', 0.0)
                ocr_sim = best_match.get('ocr_similarity', 0.0)
                
                # For tables, we have 3 ways to confirm a proper match:
                # 1. Very high CNN (>=98%) - likely exact copy
                # 2. OCR text matches (>=30%) - content is same
                # 3. CNN + hash both confirm similarity
                if cnn_score >= 0.98:
                    is_valid_match = True
                elif ocr_sim >= 0.3:
                    is_valid_match = True  # OCR confirms same content
                elif phash_sim >= MIN_HASH_SIMILARITY:
                    is_valid_match = True  # Hash confirms same structure
                else:
                    # CNN says similar but hash AND OCR say different = FALSE POSITIVE
                    rejection_reason = f"false positive (CNN={cnn_score:.1%}, hash={phash_sim:.1%}, OCR={ocr_sim:.1%})"
            
            # Only add to results if it's a PROPER match
            if is_valid_match:
                print(f"✓ match found ({best_match['score']:.1%})")
                match_type = self._score_to_match_type(
                    best_match['score'],
                    best_match.get('detected_rotation', 'original'),
                    best_match.get('is_flipped', False)
                )
                
                all_matches.append(ImageMatch(
                    query_figure_id=query_table_id,
                    source_figure_id=best_match['source_id'],
                    source_paper_id=best_match['paper_id'],
                    source_doi=best_match['doi'],
                    source_label=best_match['label'],
                    source_caption='',
                    combined_score=best_match['score'],
                    match_type=ImageMatchType(match_type),
                    phash_similarity=best_match.get('phash_similarity', 0.0),
                    phash_hamming=best_match.get('phash_hamming', 64),
                    cnn_similarity=best_match.get('cnn_score', best_match['score']),
                    ocr_text_overlap=best_match.get('ocr_similarity', 0.0),
                    detected_rotation=best_match.get('detected_rotation', 'original'),
                    is_flipped=best_match.get('is_flipped', False),
                    details={
                        'source_type': best_match.get('source_type', 'table'),
                        's3_path': best_match.get('s3_path', ''),
                        'query_type': 'table',
                        'query_s3_path': query_s3_path,
                        'query_page': page_num,
                        'query_index': tbl_index
                    }
                ))
            else:
                # NO PROPER MATCH FOUND - reject (don't return wrong values)
                print(f"✗ rejected: {rejection_reason}")
        
        # Summary
        fig_matches = len([m for m in all_matches if m.details.get('query_type') == 'figure'])
        tbl_matches = len([m for m in all_matches if m.details.get('query_type') == 'table'])
        print(f"  Found {fig_matches} figure matches, {tbl_matches} table matches")
        
        return all_matches, integrity_issues
    
    def _find_best_match(self, image, is_table: bool = False) -> Optional[Dict]:
        """
        Find the SINGLE best match for an image.
        
        Args:
            image: PIL Image object
            is_table: Whether to search tables collection
        
        Returns the highest-scoring match from Qdrant, or None if no good match.
        """
        # Extract features
        features = self.image_extractor.extract_all_features(image, "query", "query_img")
        
        if features.cnn_vector is None:
            return None
        
        # Search appropriate collection
        if is_table:
            results = self.qdrant.search_tables(features.cnn_vector, limit=10, score_threshold=0.5)
        else:
            results = self.qdrant.search_figures(features.cnn_vector, limit=10, score_threshold=0.5)
        
        if not results:
            return None
        
        # Prepare query features for OCR comparison
        query_ocr_regions = features.ocr_regions if features.ocr_regions else []
        query_has_ocr = len(query_ocr_regions) > 0
        
        # For tables, find the best match considering OCR content
        best_result = None
        best_effective_score = 0.0
        best_ocr_similarity = 0.0
        
        for result in results:
            payload = result.payload
            cnn_score = result.score
            
            # Get database OCR data
            db_ocr_data = payload.get('ocr_data', {})
            db_ocr_regions = db_ocr_data.get('text_regions', []) or payload.get('ocr_regions', [])
            db_has_ocr = len(db_ocr_regions) > 0
            
            # Compare OCR if both have it
            ocr_similarity = 0.0
            if query_has_ocr and db_has_ocr:
                # Prepare dicts for comparison
                query_dict = {
                    'ocr_regions': [{'text': r.get('text', ''), 'bbox': r.get('bbox', [])} 
                                   for r in query_ocr_regions]
                }
                db_dict = {
                    'ocr_regions': db_ocr_regions,
                    'ocr_data': db_ocr_data
                }
                ocr_result = self.image_comparator.compare_ocr_data(
                    query_dict.get('ocr_regions', []),
                    db_dict.get('ocr_regions', [])
                )
                ocr_similarity = ocr_result.get('text_overlap', 0.0)
            
            # Calculate effective score
            if is_table and query_has_ocr and db_has_ocr:
                # For tables with OCR: weight OCR heavily
                # If OCR overlap is very low, this is likely a false positive
                if ocr_similarity < 0.1:
                    # Skip this match - OCR says different table
                    effective_score = cnn_score * 0.3  # Heavily penalized
                else:
                    # Use OCR-weighted scoring
                    effective_score = 0.3 * cnn_score + 0.7 * ocr_similarity
            else:
                # For figures or when OCR unavailable, use CNN score
                effective_score = cnn_score
            
            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_result = result
                best_ocr_similarity = ocr_similarity
        
        if not best_result:
            return None
        
        # Extract info from best match
        payload = best_result.payload
        source_type = 'table' if is_table else 'figure'
        
        # Get label
        label = payload.get('label', '')
        if not label:
            idx = payload.get('figure_index', payload.get('table_index', 0))
            label = f"Table {idx + 1}" if is_table else f"Figure {idx + 1}"
        
        # Compare hashes if available (for rotation/flip detection)
        detected_rotation = 'original'
        is_flipped = False
        phash_similarity = 0.0
        phash_hamming = 64
        
        db_hashes = payload.get('perceptual_hashes', {})
        
        if db_hashes and features.orientation_hashes:
            # Convert orientation_hashes to dict if needed
            query_hashes = features.orientation_hashes
            if hasattr(query_hashes, 'to_dict'):
                query_hashes = query_hashes.to_dict()
            
            hash_result = self.image_comparator.compare_orientation_hashes(
                query_hashes,
                db_hashes
            )
            detected_rotation = hash_result.get('best_orientation', 'original')
            is_flipped = hash_result.get('is_flipped', False)
            phash_similarity = hash_result.get('similarity', 0.0)
            phash_hamming = hash_result.get('hamming_distance', 64)
        
        return {
            'source_id': best_result.id,
            'paper_id': payload.get('paper_id', payload.get('doi', '')),
            'doi': payload.get('doi', ''),
            'label': label,
            'score': best_effective_score,  # Use OCR-weighted score for tables
            'cnn_score': best_result.score,  # Keep raw CNN score
            'ocr_similarity': best_ocr_similarity,  # Add OCR similarity
            's3_path': payload.get('s3_path', ''),
            'source_type': source_type,
            'detected_rotation': detected_rotation,
            'is_flipped': is_flipped,
            'phash_similarity': phash_similarity,
            'phash_hamming': phash_hamming
        }
    
    def _score_to_match_type(self, score: float, rotation: str, is_flipped: bool) -> str:
        """Convert score to match type string."""
        if score >= 0.99:
            if rotation not in ['original', None, '']:
                return 'ROTATED_COPY'
            elif is_flipped:
                return 'FLIPPED_COPY'
            else:
                return 'EXACT_COPY'
        elif score >= 0.95:
            return 'NEAR_EXACT'
        elif score >= 0.85:
            return 'HIGH_SIMILARITY'
        elif score >= 0.75:
            return 'MODERATE_SIMILARITY'
        else:
            return 'LOW_SIMILARITY'
    
    def _features_to_dict(self, features) -> Dict:
        """Convert ImageFeatures to dict"""
        return {
            'orientation_hashes': features.orientation_hashes,
            'cnn_vector': features.cnn_vector,
            'ocr_regions': [
                {'text': r['text'], 'bbox': r['bbox']}
                for r in features.ocr_regions
            ] if features.ocr_regions else [],
            'radial_histogram': features.radial_histogram,
            'zernike_moments': features.zernike_moments
        }
    
    def _calculate_text_score(self, matches: List[TextMatch]) -> float:
        """
        Calculate overall text plagiarism score.
        
        Uses WORD OVERLAP as the primary indicator since it's the most
        accurate measure of actual text copying.
        """
        if not matches:
            return 0.0
        
        # Weight by match quality (word overlap) not just type
        scores = []
        for match in matches[:10]:  # Top 10 matches
            # Word overlap is the most reliable indicator
            word_overlap = match.word_overlap_percentage
            
            # High word overlap = definite plagiarism
            if word_overlap >= 0.70:
                score = word_overlap * 1.0  # Full weight
            elif word_overlap >= 0.50:
                score = word_overlap * 0.9  # High weight
            elif word_overlap >= 0.35:
                score = word_overlap * 0.7  # Medium weight
            else:
                score = word_overlap * 0.4  # Low weight (possible false positive)
            
            scores.append(score)
        
        # Return average of top scores, capped at 1.0
        if scores:
            return min(1.0, sum(scores) / min(5, len(scores)))
    
    def _calculate_image_score(self, matches: List[ImageMatch]) -> float:
        """Calculate overall image plagiarism score"""
        if not matches:
            return 0.0
        
        # Count by type
        exact_copies = sum(1 for m in matches if m.match_type in [
            ImageMatchType.EXACT_COPY, ImageMatchType.ROTATED_COPY, ImageMatchType.FLIPPED_COPY
        ])
        
        if exact_copies > 0:
            return min(1.0, 0.5 + exact_copies * 0.15)
        
        high_matches = sum(1 for m in matches if m.combined_score > 0.85)
        return min(1.0, high_matches * 0.2)
    
    def _determine_risk_level(
        self,
        overall_score: float,
        text_matches: List[TextMatch],
        image_matches: List[ImageMatch],
        integrity_issues: List[IntegrityResult]
    ) -> str:
        """Determine overall risk level"""
        # Critical: exact copies or AI-generated
        has_exact = any(
            m.match_type in [MatchType.EXACT_COPY, MatchType.DIRECT_COPY]
            for m in text_matches
        ) or any(
            m.match_type == ImageMatchType.EXACT_COPY
            for m in image_matches
        )
        
        has_ai = any(i.ai_generated for i in integrity_issues)
        
        if has_exact or has_ai or overall_score > 0.8:
            return "CRITICAL"
        elif overall_score > 0.5 or len(integrity_issues) > 0:
            return "HIGH"
        elif overall_score > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    # =========================================================================
    # INDEXING METHODS (for building database)
    # =========================================================================
    
    def index_paper(self, pdf_path: str, doi: str, server: str) -> bool:
        """
        Index a paper into the database.
        
        Args:
            pdf_path: Path to PDF
            doi: Paper DOI
            server: Server name (biorxiv/medrxiv)
        
        Returns:
            True if successful
        """
        try:
            # Extract content
            extraction = self.pdf_extractor.extract(pdf_path)
            
            # Generate IDs
            paper_id = generate_paper_id(doi, server)
            
            # Index text chunks
            chunks = self.text_chunker.chunk_text(extraction.full_text, "body")
            
            for chunk in chunks:
                chunk_id = generate_chunk_id(paper_id, chunk['chunk_index'])
                
                # Embed text
                if self.embedding_model:
                    vector = self.embedding_model.encode([chunk['text']])[0]
                else:
                    vector = np.random.randn(1024)
                
                self.qdrant.upsert_point(
                    collection='fulltext_chunks',
                    point_id=chunk_id,
                    vector=vector,
                    payload={
                        'paper_id': paper_id,
                        'doi': doi,
                        'server': server,
                        'title': extraction.title,
                        'section': chunk['section'],
                        'text': chunk['text'],
                        'chunk_index': chunk['chunk_index']
                    }
                )
            
            # Index figures
            for fig in extraction.figures:
                figure_id = generate_figure_id(paper_id, fig.page_number, fig.figure_index)
                
                # Extract features
                features = self.image_extractor.extract_all_features(
                    fig.image, paper_id, figure_id
                )
                
                # Upload to Wasabi
                wasabi_key = self.wasabi.build_figure_path_from_doi(
                    doi, server, fig.page_number, fig.figure_index
                )
                self.wasabi.upload_image(fig.image, wasabi_key)
                
                # Index in Qdrant
                self.qdrant.upsert_point(
                    collection='figures',
                    point_id=figure_id,
                    vector={
                        'cnn_vector': features.cnn_vector.tolist() if features.cnn_vector is not None else [0]*2048,
                        'clip_vector': [0]*512  # Placeholder
                    },
                    payload={
                        'paper_id': paper_id,
                        'doi': doi,
                        'server': server,
                        'label': fig.label,
                        'caption': fig.caption,
                        'page_number': fig.page_number,
                        'perceptual_hashes': features.orientation_hashes,
                        'rotation_invariant': {
                            'radial_histogram': features.radial_histogram.tolist(),
                            'zernike_moments': features.zernike_moments.tolist(),
                            'orb_keypoint_count': features.orb_keypoint_count
                        },
                        'ocr_data': {
                            'has_text': len(features.ocr_regions) > 0,
                            'text_regions': features.ocr_regions,
                            'text_hash': features.text_hash
                        },
                        'image_type': {
                            'classification': features.image_type,
                            'confidence': features.image_type_confidence
                        },
                        'wasabi_url': self.wasabi.get_s3_url(wasabi_key)
                    }
                )
            
            return True
        
        except Exception as e:
            print(f"Error indexing paper: {e}")
            return False


if __name__ == "__main__":
    print("WorldClassDetector module loaded successfully")
    print("Use with Qdrant and Wasabi clients for full functionality")