"""
Multi-Method Image Comparator.
Compares images using multiple methods for robust detection:
- Perceptual hash comparison (with rotation detection)
- CNN feature similarity
- OCR text and position matching
- Rotation-invariant feature comparison
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Result of image comparison"""
    combined_score: float
    match_type: str
    is_plagiarism: bool
    
    # Individual scores
    hash_similarity: float
    cnn_similarity: float
    ocr_similarity: float
    rotation_similarity: float
    
    # Rotation detection
    detected_rotation: str
    is_flipped: bool
    
    # Detailed results
    details: Dict


class MultiMethodComparator:
    """Compare images using multiple methods"""
    
    def __init__(
        self,
        phash_threshold: int = 10,
        cnn_threshold: float = 0.85,
        ocr_threshold: float = 0.7,
        combined_threshold: float = 0.75
    ):
        """
        Initialize the comparator.
        
        Args:
            phash_threshold: Maximum Hamming distance for hash match
            cnn_threshold: Minimum cosine similarity for CNN match
            ocr_threshold: Minimum text overlap for OCR match
            combined_threshold: Minimum combined score for plagiarism
        """
        self.phash_threshold = phash_threshold
        self.cnn_threshold = cnn_threshold
        self.ocr_threshold = ocr_threshold
        self.combined_threshold = combined_threshold
        
        # Weights for combined score (figures)
        self.weights = {
            'hash': 0.25,
            'cnn': 0.40,
            'ocr': 0.15,
            'rotation': 0.20
        }
        
        # Table-specific weights: prioritize TEXT content over visual appearance
        # Tables look visually similar (grids, white background) but have different content
        self.table_weights = {
            'hash': 0.10,      # Reduced - tables have similar visual structure
            'cnn': 0.15,       # Reduced - CNN sees all tables as similar
            'ocr': 0.55,       # INCREASED - text content is what differentiates tables
            'rotation': 0.20
        }
    
    # =========================================================================
    # HASH COMPARISON
    # =========================================================================
    
    def hex_to_binary(self, hex_hash: str) -> str:
        """
        Convert hex hash to binary string.
        
        Handles:
        - Pure hex strings
        - Hex strings with leading zeros stripped
        - imagehash objects converted to string
        """
        if not hex_hash:
            return '0' * 64
        
        try:
            # Clean the hash string
            clean_hash = str(hex_hash).strip().lower()
            
            # Remove any '0x' prefix if present
            if clean_hash.startswith('0x'):
                clean_hash = clean_hash[2:]
            
            # Ensure it's a valid hex string (only hex chars)
            if not all(c in '0123456789abcdef' for c in clean_hash):
                return '0' * 64
            
            # Pad to 16 characters if shorter (phash is 64 bits = 16 hex chars)
            clean_hash = clean_hash.zfill(16)
            
            # Convert to binary and pad to 64 bits
            return bin(int(clean_hash, 16))[2:].zfill(64)
        except (ValueError, TypeError):
            return '0' * 64
    
    def hamming_distance(self, hash1, hash2) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Args:
            hash1: First hash (hex string or imagehash object)
            hash2: Second hash (hex string or imagehash object)
        
        Returns:
            Hamming distance (0-64)
        """
        # Handle None/empty
        if hash1 is None or hash2 is None:
            return 64
        
        if not hash1 or not hash2:
            return 64
        
        try:
            # Method 1: Handle imagehash objects directly (they support subtraction)
            if hasattr(hash1, '__sub__') and hasattr(hash2, '__sub__'):
                return abs(hash1 - hash2)
            
            # Method 2: Handle imagehash objects with .hash attribute (numpy array)
            if hasattr(hash1, 'hash') and hasattr(hash2, 'hash'):
                return int(np.sum(hash1.hash != hash2.hash))
            
            # Method 3: Handle hex strings
            str1 = str(hash1).strip()
            str2 = str(hash2).strip()
            
            # Quick check: if strings are identical, distance is 0
            if str1 == str2:
                return 0
            
            # Convert to binary and compare
            bin1 = self.hex_to_binary(str1)
            bin2 = self.hex_to_binary(str2)
            
            # Count differing bits
            distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))
            return distance
            
        except Exception:
            return 64  # Maximum distance on error
    
    def compare_single_hash(self, hash1: str, hash2: str) -> Dict:
        """Compare two perceptual hashes"""
        distance = self.hamming_distance(hash1, hash2)
        similarity = 1 - (distance / 64)
        
        return {
            'hamming_distance': distance,
            'similarity': similarity,
            'is_match': distance < self.phash_threshold
        }
    
    def _extract_phash(self, hash_data) -> str:
        """
        Extract phash from various formats.
        
        Handles:
        - String directly: "a4e3f2c1d5b6e7f8"
        - Dict with phash key: {'phash': '...', 'dhash': '...'}
        - Dict with hash key: {'hash': '...'}
        """
        if hash_data is None:
            return ''
        
        # If it's already a string, return it
        if isinstance(hash_data, str):
            return hash_data
        
        # If it's a dict, try to get phash or hash
        if isinstance(hash_data, dict):
            return hash_data.get('phash') or hash_data.get('hash') or hash_data.get('perceptual_hash') or ''
        
        # Try converting to string
        return str(hash_data) if hash_data else ''
    
    def compare_orientation_hashes(
        self,
        query_hashes: Dict,
        db_hashes: Dict
    ) -> Dict:
        """
        Compare query hash against all orientation variants.
        
        Handles multiple formats:
        - {'original': {'phash': '...', 'dhash': '...'}, ...}
        - {'0': 'hash_string', '90': 'hash_string', ...}
        - {'original': 'hash_string', 'rot90': 'hash_string', ...}
        
        Args:
            query_hashes: Query image hashes (all orientations)
            db_hashes: Database image hashes (all orientations)
        
        Returns:
            Best match with rotation detection
        """
        best_distance = 64
        best_orientation = 'original'
        
        # Get query's original phash - handle multiple key formats
        query_original = query_hashes.get('original') or query_hashes.get('0') or {}
        query_phash = self._extract_phash(query_original)
        
        if not query_phash:
            return {
                'best_orientation': 'original',
                'hamming_distance': 64,
                'similarity': 0,
                'is_rotated': False,
                'is_flipped': False,
                'is_match': False
            }
        
        # Map of orientation keys to normalized names
        orientation_map = {
            'original': 'original', '0': 'original',
            'rot90': 'rot90', '90': 'rot90',
            'rot180': 'rot180', '180': 'rot180',
            'rot270': 'rot270', '270': 'rot270',
            'flip_h': 'flip_h', 'flip_v': 'flip_v',
            'flip_h_rot90': 'flip_h_rot90', 'flip_v_rot90': 'flip_v_rot90'
        }
        
        # Compare against all orientations in database
        for orientation, hash_data in db_hashes.items():
            db_phash = self._extract_phash(hash_data)
            if not db_phash:
                continue
            
            distance = self.hamming_distance(query_phash, db_phash)
            
            if distance < best_distance:
                best_distance = distance
                # Normalize orientation name
                best_orientation = orientation_map.get(orientation, orientation)
        
        similarity = 1 - (best_distance / 64)
        
        # Determine if rotated or flipped
        is_rotated = best_orientation not in ['original', '0']
        is_flipped = 'flip' in best_orientation
        
        return {
            'best_orientation': best_orientation,
            'hamming_distance': best_distance,
            'similarity': similarity,
            'is_rotated': is_rotated,
            'is_flipped': is_flipped,
            'is_match': best_distance < self.phash_threshold
        }
    
    # =========================================================================
    # CNN COMPARISON
    # =========================================================================
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity (-1 to 1)
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def compare_cnn_vectors(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> Dict:
        """Compare CNN feature vectors"""
        similarity = self.cosine_similarity(vec1, vec2)
        
        return {
            'cosine_similarity': similarity,
            'is_match': similarity > self.cnn_threshold
        }
    
    # =========================================================================
    # OCR COMPARISON
    # =========================================================================
    
    def _normalize_bbox(self, bbox) -> Dict:
        """
        Convert bbox to normalized dict format {x, y, w, h}.
        
        Handles:
        - Dict format: {'x': ..., 'y': ..., 'w': ..., 'h': ...}
        - List format: [x1, y1, x2, y2] or [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        - EasyOCR format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        """
        if not bbox:
            return {'x': 0, 'y': 0, 'w': 1, 'h': 1}
        
        # Already a dict with expected keys
        if isinstance(bbox, dict):
            return {
                'x': bbox.get('x', 0),
                'y': bbox.get('y', 0),
                'w': bbox.get('w', 1),
                'h': bbox.get('h', 1)
            }
        
        # List format
        if isinstance(bbox, (list, tuple)):
            # EasyOCR format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (4 corner points)
            if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                return {
                    'x': x1,
                    'y': y1,
                    'w': max(x2 - x1, 1),
                    'h': max(y2 - y1, 1)
                }
            # Simple list format: [x1, y1, x2, y2]
            elif len(bbox) == 4:
                return {
                    'x': bbox[0],
                    'y': bbox[1],
                    'w': max(bbox[2] - bbox[0], 1),
                    'h': max(bbox[3] - bbox[1], 1)
                }
            # [x, y, w, h] format
            elif len(bbox) >= 4:
                return {
                    'x': bbox[0],
                    'y': bbox[1],
                    'w': max(bbox[2], 1),
                    'h': max(bbox[3], 1)
                }
        
        return {'x': 0, 'y': 0, 'w': 1, 'h': 1}
    
    def _position_similarity(self, bbox1, bbox2) -> float:
        """Calculate position similarity between bounding boxes"""
        if not bbox1 or not bbox2:
            return 0
        
        # Normalize both bboxes to dict format
        b1 = self._normalize_bbox(bbox1)
        b2 = self._normalize_bbox(bbox2)
        
        x_diff = abs(b1['x'] - b2['x'])
        y_diff = abs(b1['y'] - b2['y'])
        
        max_w = max(b1['w'], b2['w'], 1)
        max_h = max(b1['h'], b2['h'], 1)
        
        x_sim = max(0, 1 - x_diff / max_w)
        y_sim = max(0, 1 - y_diff / max_h)
        
        return (x_sim + y_sim) / 2
    
    def compare_ocr_data(
        self,
        ocr1: List[Dict],
        ocr2: List[Dict]
    ) -> Dict:
        """
        Compare OCR text and positions.
        Enhanced to handle nested ocr_data structure and provide better table matching.
        
        Args:
            ocr1: First image's OCR regions (or ocr_data dict)
            ocr2: Second image's OCR regions (or ocr_data dict)
        
        Returns:
            Comparison results with text overlap and position matching
        """
        # Handle nested ocr_data structure
        if isinstance(ocr1, dict):
            ocr1 = ocr1.get('text_regions', []) or ocr1.get('ocr_regions', [])
        if isinstance(ocr2, dict):
            ocr2 = ocr2.get('text_regions', []) or ocr2.get('ocr_regions', [])
        
        if not ocr1 or not ocr2:
            return {
                'text_overlap': 0,
                'position_match': 0,
                'common_texts': [],
                'is_match': False
            }
        
        # Extract text from regions
        def get_texts(regions):
            texts = set()
            for r in regions:
                if isinstance(r, dict):
                    text = r.get('text', '')
                    if text:
                        # Normalize text
                        texts.add(text.lower().strip())
                elif isinstance(r, str):
                    texts.add(r.lower().strip())
            return texts
        
        texts1 = get_texts(ocr1)
        texts2 = get_texts(ocr2)
        
        if not texts1 or not texts2:
            return {
                'text_overlap': 0,
                'position_match': 0,
                'common_texts': [],
                'is_match': False
            }
        
        # Calculate text overlap (Jaccard similarity)
        intersection = texts1 & texts2
        union = texts1 | texts2
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Also calculate containment (what fraction of smaller set is in larger)
        containment = len(intersection) / min(len(texts1), len(texts2)) if min(len(texts1), len(texts2)) > 0 else 0
        
        # Combined overlap score (average of Jaccard and containment)
        combined_overlap = (jaccard + containment) / 2
        
        # Position matching for common texts
        position_scores = []
        for r1 in ocr1:
            if not isinstance(r1, dict):
                continue
            text1 = r1.get('text', '').lower().strip()
            if text1 not in intersection:
                continue
            
            for r2 in ocr2:
                if not isinstance(r2, dict):
                    continue
                text2 = r2.get('text', '').lower().strip()
                if text1 == text2:
                    pos_sim = self._position_similarity(
                        r1.get('bbox'),
                        r2.get('bbox')
                    )
                    position_scores.append(pos_sim)
        
        position_match = np.mean(position_scores) if position_scores else 0
        
        return {
            'text_overlap': float(combined_overlap),
            'position_match': float(position_match),
            'common_texts': list(intersection)[:20],  # Limit for payload size
            'is_match': combined_overlap > self.ocr_threshold
        }
    
    # =========================================================================
    # ROTATION-INVARIANT COMPARISON
    # =========================================================================
    
    def compare_radial_histograms(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray
    ) -> Dict:
        """Compare radial histograms"""
        if hist1 is None or hist2 is None or len(hist1) == 0 or len(hist2) == 0:
            return {'similarity': 0, 'is_match': False}
        
        # Histogram intersection
        min_len = min(len(hist1), len(hist2))
        h1, h2 = hist1[:min_len], hist2[:min_len]
        
        intersection = np.minimum(h1, h2).sum()
        total = np.sum(h1) + 1e-8
        
        similarity = intersection / total
        
        return {
            'intersection': float(intersection),
            'similarity': float(similarity),
            'is_match': similarity > 0.7
        }
    
    def compare_zernike_moments(
        self,
        moments1: np.ndarray,
        moments2: np.ndarray
    ) -> Dict:
        """Compare Zernike moments"""
        if moments1 is None or moments2 is None:
            return {'similarity': 0, 'is_match': False}
        
        min_len = min(len(moments1), len(moments2))
        m1, m2 = moments1[:min_len], moments2[:min_len]
        
        distance = np.linalg.norm(m1 - m2)
        max_mag = max(np.linalg.norm(m1), np.linalg.norm(m2), 1e-8)
        
        normalized_distance = distance / max_mag
        similarity = max(0, 1 - normalized_distance)
        
        return {
            'distance': float(distance),
            'similarity': float(similarity),
            'is_match': similarity > 0.8
        }
    
    def compare_rotation_invariant(
        self,
        radial1: np.ndarray,
        radial2: np.ndarray,
        zernike1: np.ndarray,
        zernike2: np.ndarray
    ) -> Dict:
        """Compare rotation-invariant features"""
        radial_result = self.compare_radial_histograms(radial1, radial2)
        zernike_result = self.compare_zernike_moments(zernike1, zernike2)
        
        combined = (
            0.5 * radial_result['similarity'] +
            0.5 * zernike_result['similarity']
        )
        
        return {
            'radial': radial_result,
            'zernike': zernike_result,
            'combined_similarity': float(combined),
            'is_match': combined > 0.75
        }
    
    # =========================================================================
    # COMPLETE COMPARISON
    # =========================================================================
    
    def compare_images(
        self,
        query_features: Dict,
        db_features: Dict,
        is_table: bool = False
    ) -> ComparisonResult:
        """
        Compare two images using multiple methods.
        
        Args:
            query_features: Query image features
            db_features: Database image features
            is_table: Whether comparing tables (uses different weights)
        
        Returns:
            ComparisonResult with all metrics
        """
        # Use table-specific weights if comparing tables
        active_weights = self.table_weights if is_table else self.weights
        
        # Hash comparison
        query_hashes = query_features.get('orientation_hashes', {})
        db_hashes = db_features.get('orientation_hashes', {})
        
        hash_result = self.compare_orientation_hashes(query_hashes, db_hashes)
        
        # CNN comparison
        cnn_result = self.compare_cnn_vectors(
            query_features.get('cnn_vector'),
            db_features.get('cnn_vector')
        )
        
        # OCR comparison - handle multiple formats
        query_ocr = query_features.get('ocr_regions') or query_features.get('ocr_data', {}).get('text_regions', [])
        db_ocr = db_features.get('ocr_regions') or db_features.get('ocr_data', {}).get('text_regions', [])
        
        ocr_result = self.compare_ocr_data(query_ocr, db_ocr)
        
        # Rotation-invariant comparison
        rotation_result = self.compare_rotation_invariant(
            query_features.get('radial_histogram'),
            db_features.get('radial_histogram'),
            query_features.get('zernike_moments'),
            db_features.get('zernike_moments')
        )
        
        # Dynamic weight calculation based on available features
        available_weights = {}
        total_weight = 0.0
        
        # Hash only if both have hashes
        if query_hashes and db_hashes:
            available_weights['hash'] = active_weights['hash']
            total_weight += active_weights['hash']
        
        # CNN only if both vectors available
        if query_features.get('cnn_vector') is not None and db_features.get('cnn_vector') is not None:
            available_weights['cnn'] = active_weights['cnn']
            total_weight += active_weights['cnn']
        
        # OCR only if both have OCR data
        if query_ocr and db_ocr:
            available_weights['ocr'] = active_weights['ocr']
            total_weight += active_weights['ocr']
        
        # Rotation only if both have rotation data
        if (query_features.get('radial_histogram') is not None and 
            db_features.get('radial_histogram') is not None):
            available_weights['rotation'] = active_weights['rotation']
            total_weight += active_weights['rotation']
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for key in available_weights:
                available_weights[key] /= total_weight
        else:
            # Fallback: use hash only
            available_weights['hash'] = 1.0
        
        # Calculate combined score with normalized weights
        combined_score = 0.0
        if 'hash' in available_weights:
            combined_score += available_weights['hash'] * hash_result['similarity']
        if 'cnn' in available_weights:
            combined_score += available_weights['cnn'] * cnn_result['cosine_similarity']
        if 'ocr' in available_weights:
            combined_score += available_weights['ocr'] * ocr_result['text_overlap']
        if 'rotation' in available_weights:
            combined_score += available_weights['rotation'] * rotation_result['combined_similarity']
        
        # Determine match type
        match_type = self._determine_match_type(
            hash_result,
            cnn_result,
            ocr_result,
            combined_score
        )
        
        # Determine if plagiarism - use hash-based detection if CNN not available
        # Hash distance < 10 is a strong match, < 5 is near-identical
        is_plagiarism = (
            combined_score > self.combined_threshold or
            hash_result['hamming_distance'] < 10 or  # Relaxed from 5 to 10
            hash_result['similarity'] > 0.85  # High hash similarity
        )
        
        return ComparisonResult(
            combined_score=combined_score,
            match_type=match_type,
            is_plagiarism=is_plagiarism,
            hash_similarity=hash_result['similarity'],
            cnn_similarity=cnn_result['cosine_similarity'],
            ocr_similarity=ocr_result['text_overlap'],
            rotation_similarity=rotation_result['combined_similarity'],
            detected_rotation=hash_result['best_orientation'],
            is_flipped=hash_result['is_flipped'],
            details={
                'hash': hash_result,
                'cnn': cnn_result,
                'ocr': ocr_result,
                'rotation': rotation_result
            }
        )
    
    def _determine_match_type(
        self,
        hash_result: Dict,
        cnn_result: Dict,
        ocr_result: Dict,
        combined_score: float
    ) -> str:
        """Determine the type of image match"""
        
        hamming = hash_result['hamming_distance']
        cnn_sim = cnn_result.get('cosine_similarity', 0)
        
        # Check for exact match via CNN (when hashes not available)
        # CNN score >= 0.99 is essentially identical image
        if cnn_sim >= 0.99:
            if hash_result.get('is_rotated'):
                return 'ROTATED_COPY'
            elif hash_result.get('is_flipped'):
                return 'FLIPPED_COPY'
            else:
                return 'EXACT_COPY'
        
        # Exact or near-exact copy via hash
        if hamming < 5:
            if hash_result.get('is_rotated'):
                return 'ROTATED_COPY'
            elif hash_result.get('is_flipped'):
                return 'FLIPPED_COPY'
            else:
                return 'EXACT_COPY'
        
        # Very high similarity (CNN or combined)
        if combined_score > 0.95 or cnn_sim > 0.95:
            return 'NEAR_EXACT'
        
        # High CNN but different text = possible relabeling
        if cnn_sim > 0.85 and ocr_result.get('text_overlap', 0) < 0.3:
            return 'RELABELED'
        
        # Standard similarity levels
        if combined_score > 0.85 or cnn_sim > 0.85:
            return 'HIGH_SIMILARITY'
        elif combined_score > 0.75 or cnn_sim > 0.75:
            return 'MODERATE_SIMILARITY'
        elif combined_score > 0.70 or cnn_sim > 0.70:
            return 'LOW_SIMILARITY'
        else:
            return 'MINIMAL_SIMILARITY'
    
    def batch_compare(
        self,
        query_features: Dict,
        db_features_list: List[Dict],
        top_k: int = 10
    ) -> List[ComparisonResult]:
        """
        Compare query against multiple database images.
        
        Args:
            query_features: Query image features
            db_features_list: List of database image features
            top_k: Number of top matches to return
        
        Returns:
            Top k matches sorted by combined score
        """
        results = []
        
        for db_features in db_features_list:
            result = self.compare_images(query_features, db_features)
            results.append(result)
        
        # Sort by combined score descending
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results[:top_k]


class RotationInvariantComparator:
    """Specialized comparator for rotation detection"""
    
    def __init__(self):
        try:
            import cv2
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        except ImportError:
            self.bf_matcher = None
    
    def compare_orb_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> Dict:
        """Compare ORB descriptors with geometric verification"""
        
        if self.bf_matcher is None:
            return {'num_matches': 0, 'similarity': 0, 'is_match': False}
        
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return {'num_matches': 0, 'similarity': 0, 'is_match': False}
        
        try:
            matches = self.bf_matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            num_matches = len(matches)
            max_possible = min(len(desc1), len(desc2))
            
            good_matches = [m for m in matches if m.distance < 50]
            
            similarity = len(good_matches) / max_possible if max_possible > 0 else 0
            
            return {
                'num_matches': num_matches,
                'num_good_matches': len(good_matches),
                'similarity': float(similarity),
                'is_match': len(good_matches) > 10
            }
        
        except Exception as e:
            print(f"ORB comparison failed: {e}")
            return {'num_matches': 0, 'similarity': 0, 'is_match': False}


if __name__ == "__main__":
    # Test the comparator
    comparator = MultiMethodComparator()
    
    # Test with string format (like your database)
    query = {
        'orientation_hashes': {
            '0': 'a4e3f2c1d5b6e7f8',
            '90': 'b5f4e3d2c6a7f8e9',
            '180': 'c6a7f8e9d5b6e7f8',
            '270': 'd7b8e9f0a4e3f2c1'
        },
        'cnn_vector': np.random.randn(2048),
        'ocr_regions': [{'text': 'Control', 'bbox': {'x': 100, 'y': 50, 'w': 80, 'h': 20}}],
        'radial_histogram': np.random.rand(32),
        'zernike_moments': np.random.rand(25)
    }
    
    # Same hashes = exact match
    db = {
        'orientation_hashes': {
            '0': 'a4e3f2c1d5b6e7f8',  # Same as query
            '90': 'b5f4e3d2c6a7f8e9',
            '180': 'c6a7f8e9d5b6e7f8',
            '270': 'd7b8e9f0a4e3f2c1'
        },
        'cnn_vector': query['cnn_vector'],
        'ocr_regions': [{'text': 'Control', 'bbox': {'x': 105, 'y': 48, 'w': 82, 'h': 22}}],
        'radial_histogram': query['radial_histogram'],
        'zernike_moments': query['zernike_moments']
    }
    
    result = comparator.compare_images(query, db)
    
    print(f"Combined Score: {result.combined_score:.3f}")
    print(f"Match Type: {result.match_type}")
    print(f"Is Plagiarism: {result.is_plagiarism}")
    print(f"Detected Rotation: {result.detected_rotation}")
    print(f"Hash Similarity: {result.hash_similarity:.3f}")
    print(f"Hash Hamming Distance: {result.details['hash']['hamming_distance']}")
    print(f"CNN Similarity: {result.cnn_similarity:.3f}")
    print(f"OCR Similarity: {result.ocr_similarity:.3f}")