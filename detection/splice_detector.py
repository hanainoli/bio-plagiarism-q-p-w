#!/usr/bin/env python3
"""
Image Splice/Manipulation Detector
====================================
Detects image manipulation, splicing, and duplication in scientific figures.

Techniques used:
1. Error Level Analysis (ELA)
2. Copy-move detection
3. Noise inconsistency analysis
4. JPEG artifact analysis
5. Edge inconsistency detection

Usage:
    from detection.splice_detector import SpliceDetector
    
    detector = SpliceDetector()
    result = detector.analyze_image(image_path)
    print(f"Manipulation probability: {result['manipulation_probability']}")
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import hashlib

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SpliceDetectionResult:
    """Result of splice/manipulation detection"""
    manipulation_probability: float  # 0-1 probability of manipulation
    confidence: float  # Confidence in the prediction
    techniques_used: List[str]  # Which techniques were applied
    findings: Dict[str, any]  # Detailed findings from each technique
    flags: List[str]  # Specific manipulation flags
    verdict: str  # "clean", "suspicious", "likely_manipulated"


class SpliceDetector:
    """
    Detect image manipulation and splicing.
    
    Uses multiple forensic techniques to identify:
    - Copy-move forgery (duplicated regions)
    - Splicing (content from different images)
    - Retouching and editing artifacts
    - Compression inconsistencies
    """
    
    def __init__(self, quality: int = 90):
        """
        Initialize the splice detector.
        
        Args:
            quality: JPEG quality for ELA analysis (lower = more sensitive)
        """
        self.ela_quality = quality
        self.min_region_size = 32  # Minimum region size for copy-move
        self.similarity_threshold = 0.95  # Threshold for duplicate detection
        
        logger.info("Splice Detector initialized")
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        detailed: bool = True
    ) -> SpliceDetectionResult:
        """
        Analyze an image for manipulation.
        
        Args:
            image: Image path, PIL Image, or numpy array
            detailed: Whether to run all detailed analyses
            
        Returns:
            SpliceDetectionResult with findings
        """
        # Load image
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return SpliceDetectionResult(
                manipulation_probability=0.0,
                confidence=0.0,
                techniques_used=[],
                findings={"error": str(e)},
                flags=["load_error"],
                verdict="error"
            )
        
        findings = {}
        flags = []
        techniques = []
        
        img_array = np.array(img)
        
        # 1. Error Level Analysis
        try:
            ela_result = self._error_level_analysis(img)
            findings["ela"] = ela_result
            techniques.append("ela")
            if ela_result["suspicious_regions"] > 0:
                flags.append("ela_inconsistency")
        except Exception as e:
            logger.warning(f"ELA failed: {e}")
            findings["ela"] = {"error": str(e)}
        
        # 2. Noise analysis
        try:
            noise_result = self._noise_analysis(img_array)
            findings["noise"] = noise_result
            techniques.append("noise_analysis")
            if noise_result["inconsistent"]:
                flags.append("noise_inconsistency")
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            findings["noise"] = {"error": str(e)}
        
        # 3. Copy-move detection (simplified)
        if detailed:
            try:
                copymove_result = self._copy_move_detection(img_array)
                findings["copy_move"] = copymove_result
                techniques.append("copy_move")
                if copymove_result["duplicates_found"]:
                    flags.append("duplicate_regions")
            except Exception as e:
                logger.warning(f"Copy-move detection failed: {e}")
                findings["copy_move"] = {"error": str(e)}
        
        # 4. Edge consistency
        try:
            edge_result = self._edge_consistency(img_array)
            findings["edge"] = edge_result
            techniques.append("edge_analysis")
            if edge_result["inconsistent"]:
                flags.append("edge_inconsistency")
        except Exception as e:
            logger.warning(f"Edge analysis failed: {e}")
            findings["edge"] = {"error": str(e)}
        
        # 5. Color distribution analysis
        try:
            color_result = self._color_analysis(img_array)
            findings["color"] = color_result
            techniques.append("color_analysis")
            if color_result["anomalies"]:
                flags.append("color_anomaly")
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            findings["color"] = {"error": str(e)}
        
        # Calculate overall probability
        manipulation_prob = self._calculate_manipulation_probability(findings, flags)
        confidence = self._calculate_confidence(findings, techniques)
        
        # Determine verdict
        if manipulation_prob > 0.7:
            verdict = "likely_manipulated"
        elif manipulation_prob > 0.4:
            verdict = "suspicious"
        else:
            verdict = "clean"
        
        return SpliceDetectionResult(
            manipulation_probability=round(manipulation_prob, 3),
            confidence=round(confidence, 3),
            techniques_used=techniques,
            findings=findings,
            flags=flags,
            verdict=verdict
        )
    
    def _error_level_analysis(self, img: Image.Image) -> Dict:
        """
        Perform Error Level Analysis (ELA).
        
        ELA detects areas that have been modified by looking at
        compression artifact differences.
        """
        import io
        
        # Save at specified quality
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.ela_quality)
        buffer.seek(0)
        
        # Reload
        recompressed = Image.open(buffer)
        
        # Calculate difference
        original_array = np.array(img).astype(float)
        recompressed_array = np.array(recompressed).astype(float)
        
        ela_image = np.abs(original_array - recompressed_array)
        
        # Analyze ELA
        ela_mean = np.mean(ela_image)
        ela_std = np.std(ela_image)
        ela_max = np.max(ela_image)
        
        # Find suspicious regions (high ELA values)
        threshold = ela_mean + 2 * ela_std
        suspicious_mask = ela_image > threshold
        suspicious_ratio = np.mean(suspicious_mask)
        
        # Count distinct suspicious regions
        suspicious_regions = 0
        if suspicious_ratio > 0.01:  # More than 1% suspicious
            suspicious_regions = 1
        if suspicious_ratio > 0.05:
            suspicious_regions = 2
        if suspicious_ratio > 0.1:
            suspicious_regions = 3
        
        return {
            "ela_mean": float(ela_mean),
            "ela_std": float(ela_std),
            "ela_max": float(ela_max),
            "suspicious_ratio": float(suspicious_ratio),
            "suspicious_regions": suspicious_regions
        }
    
    def _noise_analysis(self, img_array: np.ndarray) -> Dict:
        """
        Analyze noise patterns for inconsistencies.
        
        Different image sources have different noise patterns.
        Spliced regions may have inconsistent noise.
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate local noise using Laplacian
        from scipy import ndimage
        laplacian = ndimage.laplace(gray.astype(float))
        
        # Divide image into blocks
        block_size = 64
        h, w = gray.shape
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        if blocks_h < 2 or blocks_w < 2:
            return {
                "inconsistent": False,
                "noise_variance": float(np.var(laplacian)),
                "block_variances": []
            }
        
        block_variances = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = laplacian[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                block_variances.append(np.var(block))
        
        # Check for inconsistency
        mean_var = np.mean(block_variances)
        std_var = np.std(block_variances)
        
        # High variation in noise levels indicates inconsistency
        cv = std_var / mean_var if mean_var > 0 else 0
        inconsistent = cv > 0.5  # Coefficient of variation threshold
        
        return {
            "inconsistent": inconsistent,
            "noise_variance": float(mean_var),
            "noise_cv": float(cv),
            "block_count": len(block_variances)
        }
    
    def _copy_move_detection(self, img_array: np.ndarray) -> Dict:
        """
        Detect copy-move forgery using block matching.
        
        Simplified version - looks for duplicate regions.
        """
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array.astype(np.uint8)
        
        # Use block hashing
        block_size = self.min_region_size
        h, w = gray.shape
        
        if h < block_size * 2 or w < block_size * 2:
            return {
                "duplicates_found": False,
                "duplicate_count": 0,
                "message": "Image too small for copy-move detection"
            }
        
        # Sample blocks and compute hashes
        block_hashes = {}
        duplicate_pairs = []
        
        step = block_size // 2  # 50% overlap
        
        for i in range(0, h - block_size, step):
            for j in range(0, w - block_size, step):
                block = gray[i:i+block_size, j:j+block_size]
                
                # Simple hash using mean and std
                block_hash = (
                    int(np.mean(block)),
                    int(np.std(block)),
                    int(np.mean(block[:block_size//2, :])),
                    int(np.mean(block[block_size//2:, :]))
                )
                
                if block_hash in block_hashes:
                    # Check if it's actually similar (not just same hash)
                    prev_i, prev_j = block_hashes[block_hash]
                    
                    # Must be at least block_size apart
                    if abs(i - prev_i) > block_size or abs(j - prev_j) > block_size:
                        prev_block = gray[prev_i:prev_i+block_size, prev_j:prev_j+block_size]
                        
                        # Calculate actual similarity
                        similarity = 1 - np.mean(np.abs(block.astype(float) - prev_block.astype(float))) / 255
                        
                        if similarity > self.similarity_threshold:
                            duplicate_pairs.append({
                                "region1": (prev_i, prev_j),
                                "region2": (i, j),
                                "similarity": float(similarity)
                            })
                else:
                    block_hashes[block_hash] = (i, j)
        
        return {
            "duplicates_found": len(duplicate_pairs) > 0,
            "duplicate_count": len(duplicate_pairs),
            "duplicate_pairs": duplicate_pairs[:10]  # Limit output
        }
    
    def _edge_consistency(self, img_array: np.ndarray) -> Dict:
        """
        Analyze edge consistency across the image.
        
        Spliced regions may have inconsistent edge characteristics.
        """
        from scipy import ndimage
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Sobel edge detection
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Analyze edge distribution in blocks
        block_size = 64
        h, w = gray.shape
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        if blocks_h < 2 or blocks_w < 2:
            return {
                "inconsistent": False,
                "edge_density": float(np.mean(edges > 30))
            }
        
        block_densities = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = edges[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                density = np.mean(block > 30)  # Edge threshold
                block_densities.append(density)
        
        # Check for unusual variation
        mean_density = np.mean(block_densities)
        std_density = np.std(block_densities)
        
        # High variation could indicate manipulation
        cv = std_density / mean_density if mean_density > 0 else 0
        inconsistent = cv > 1.0
        
        return {
            "inconsistent": inconsistent,
            "edge_density": float(mean_density),
            "edge_cv": float(cv)
        }
    
    def _color_analysis(self, img_array: np.ndarray) -> Dict:
        """
        Analyze color distribution for anomalies.
        """
        if len(img_array.shape) != 3:
            return {"anomalies": False, "message": "Grayscale image"}
        
        # Analyze each channel
        anomalies = []
        
        for c, name in enumerate(['red', 'green', 'blue']):
            channel = img_array[:, :, c]
            
            # Check for unnatural peaks in histogram
            hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
            
            # Normalize
            hist = hist / hist.sum()
            
            # Check for unusual spikes (could indicate color manipulation)
            threshold = np.mean(hist) + 3 * np.std(hist)
            spikes = np.sum(hist > threshold)
            
            if spikes > 5:  # More than 5 unusual spikes
                anomalies.append(f"{name}_spikes")
        
        # Check color balance
        means = [np.mean(img_array[:, :, c]) for c in range(3)]
        color_imbalance = np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
        
        if color_imbalance > 0.5:
            anomalies.append("color_imbalance")
        
        return {
            "anomalies": len(anomalies) > 0,
            "anomaly_list": anomalies,
            "color_means": [float(m) for m in means],
            "color_imbalance": float(color_imbalance)
        }
    
    def _calculate_manipulation_probability(
        self,
        findings: Dict,
        flags: List[str]
    ) -> float:
        """
        Calculate overall manipulation probability.
        """
        score = 0.0
        weights = 0.0
        
        # ELA contribution
        if "ela" in findings and "error" not in findings["ela"]:
            ela = findings["ela"]
            ela_score = min(ela.get("suspicious_ratio", 0) * 5, 1.0)
            score += ela_score * 0.3
            weights += 0.3
        
        # Noise contribution
        if "noise" in findings and "error" not in findings["noise"]:
            noise = findings["noise"]
            if noise.get("inconsistent"):
                score += 0.25 * 0.6
            weights += 0.25
        
        # Copy-move contribution
        if "copy_move" in findings and "error" not in findings["copy_move"]:
            cm = findings["copy_move"]
            if cm.get("duplicates_found"):
                dup_score = min(cm.get("duplicate_count", 0) / 5, 1.0)
                score += dup_score * 0.25
            weights += 0.25
        
        # Edge contribution
        if "edge" in findings and "error" not in findings["edge"]:
            edge = findings["edge"]
            if edge.get("inconsistent"):
                score += 0.1 * 0.5
            weights += 0.1
        
        # Color contribution
        if "color" in findings and "error" not in findings["color"]:
            color = findings["color"]
            if color.get("anomalies"):
                score += 0.1 * 0.4
            weights += 0.1
        
        if weights > 0:
            return score / weights
        return 0.0
    
    def _calculate_confidence(
        self,
        findings: Dict,
        techniques: List[str]
    ) -> float:
        """
        Calculate confidence in the analysis.
        """
        # Base confidence on number of successful techniques
        technique_confidence = len(techniques) / 5  # 5 total techniques
        
        # Reduce confidence if there were errors
        errors = sum(1 for k, v in findings.items() 
                    if isinstance(v, dict) and "error" in v)
        error_penalty = errors * 0.1
        
        confidence = technique_confidence - error_penalty
        
        return max(0.1, min(1.0, confidence))
    
    def compare_images(
        self,
        image1: Union[str, Path, Image.Image],
        image2: Union[str, Path, Image.Image]
    ) -> Dict:
        """
        Compare two images for potential splicing/copying between them.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Comparison results
        """
        # Load images
        if isinstance(image1, (str, Path)):
            img1 = np.array(Image.open(image1).convert('RGB'))
        else:
            img1 = np.array(image1.convert('RGB'))
        
        if isinstance(image2, (str, Path)):
            img2 = np.array(Image.open(image2).convert('RGB'))
        else:
            img2 = np.array(image2.convert('RGB'))
        
        # Compare histograms
        hist_similarity = self._compare_histograms(img1, img2)
        
        # Compare noise patterns
        noise_similarity = self._compare_noise(img1, img2)
        
        # Overall similarity
        overall_similarity = (hist_similarity + noise_similarity) / 2
        
        return {
            "histogram_similarity": float(hist_similarity),
            "noise_similarity": float(noise_similarity),
            "overall_similarity": float(overall_similarity),
            "likely_same_source": overall_similarity > 0.8,
            "likely_related": overall_similarity > 0.6
        }
    
    def _compare_histograms(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """Compare color histograms of two images."""
        similarity = 0.0
        
        for c in range(3):
            hist1, _ = np.histogram(img1[:, :, c].flatten(), bins=256, range=(0, 256))
            hist2, _ = np.histogram(img2[:, :, c].flatten(), bins=256, range=(0, 256))
            
            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Correlation
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            similarity += max(0, correlation)
        
        return similarity / 3
    
    def _compare_noise(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """Compare noise patterns of two images."""
        from scipy import ndimage
        
        # Convert to grayscale
        gray1 = np.mean(img1, axis=2)
        gray2 = np.mean(img2, axis=2)
        
        # Extract noise using Laplacian
        noise1 = ndimage.laplace(gray1.astype(float))
        noise2 = ndimage.laplace(gray2.astype(float))
        
        # Compare noise statistics
        var1 = np.var(noise1)
        var2 = np.var(noise2)
        
        # Similarity based on variance ratio
        ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 1.0
        
        return ratio


# Convenience function
def detect_manipulation(image_path: str) -> Dict:
    """
    Quick function to detect image manipulation.
    
    Args:
        image_path: Path to image
        
    Returns:
        Dictionary with manipulation_probability, confidence, verdict
    """
    detector = SpliceDetector()
    result = detector.analyze_image(image_path)
    return {
        "manipulation_probability": result.manipulation_probability,
        "confidence": result.confidence,
        "verdict": result.verdict,
        "flags": result.flags
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detector = SpliceDetector()
        result = detector.analyze_image(image_path)
        
        print(f"Manipulation Probability: {result.manipulation_probability}")
        print(f"Confidence: {result.confidence}")
        print(f"Verdict: {result.verdict}")
        print(f"Flags: {result.flags}")
        print(f"Techniques: {result.techniques_used}")
    else:
        print("Usage: python splice_detector.py <image_path>")
