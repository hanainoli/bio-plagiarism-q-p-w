"""
Detection Module
================
Plagiarism and manipulation detection for biomedical papers.

Components:
- WorldClassDetector: Main orchestrator
- EnhancedTextPlagiarismDetector: Text similarity with false positive reduction (NEW)
- TextPlagiarismDetector: Text similarity detection  
- MultiMethodComparator: Image similarity detection
- ImageFeatureExtractor: Image embedding extraction
- AIDetector: AI-generated content detection
- SpliceDetector: Image manipulation detection
"""

# Enhanced text detector (NEW - with false positive reduction)
try:
    from .text_detector_enhanced import (
        EnhancedTextPlagiarismDetector,
        EnhancedTextMatch,
        SentenceMatch
    )
    ENHANCED_DETECTOR_AVAILABLE = True
except ImportError as e:
    EnhancedTextPlagiarismDetector = None
    EnhancedTextMatch = None
    SentenceMatch = None
    ENHANCED_DETECTOR_AVAILABLE = False
    print(f"Warning: Could not import text_detector_enhanced: {e}")

# Text detector - class is TextPlagiarismDetector
try:
    from .text_detector import TextPlagiarismDetector, TextMatch
    # Use enhanced detector if available, otherwise fall back to standard
    if ENHANCED_DETECTOR_AVAILABLE:
        TextDetector = EnhancedTextPlagiarismDetector
    else:
        TextDetector = TextPlagiarismDetector
except ImportError as e:
    TextPlagiarismDetector = None
    TextDetector = None
    TextMatch = None
    print(f"Warning: Could not import text_detector: {e}")

# Image feature extractor - class is ImageFeatureExtractor
try:
    from .image_feature_extractor import ImageFeatureExtractor, get_image_extractor
    # Alias for compatibility with old code expecting MultiMethodFeatureExtractor
    MultiMethodFeatureExtractor = ImageFeatureExtractor
except ImportError as e:
    ImageFeatureExtractor = None
    MultiMethodFeatureExtractor = None
    get_image_extractor = None
    print(f"Warning: Could not import image_feature_extractor: {e}")

# Image comparator - class is MultiMethodComparator
try:
    from .image_comparator import MultiMethodComparator, ComparisonResult
    # Alias for convenience
    ImageComparator = MultiMethodComparator
except ImportError as e:
    MultiMethodComparator = None
    ImageComparator = None
    ComparisonResult = None
    print(f"Warning: Could not import image_comparator: {e}")

# AI content detector - class is AIDetector
try:
    from .ai_detector import AIDetector, AIDetectionResult, detect_ai_content
    # Alias for compatibility
    AIGeneratedDetector = AIDetector
except ImportError as e:
    AIDetector = None
    AIGeneratedDetector = None
    AIDetectionResult = None
    detect_ai_content = None
    print(f"Warning: Could not import ai_detector: {e}")

# Splice/manipulation detector - class is SpliceDetector
try:
    from .splice_detector import SpliceDetector, SpliceDetectionResult, detect_manipulation
except ImportError as e:
    SpliceDetector = None
    SpliceDetectionResult = None
    detect_manipulation = None
    print(f"Warning: Could not import splice_detector: {e}")

# Main detector - class is WorldClassDetector
try:
    from .world_class_detector import WorldClassDetector, DetectorConfig
except ImportError as e:
    WorldClassDetector = None
    DetectorConfig = None
    print(f"Warning: Could not import world_class_detector: {e}")


__all__ = [
    # Enhanced text detection (NEW)
    "EnhancedTextPlagiarismDetector",
    "EnhancedTextMatch",
    "SentenceMatch",
    "ENHANCED_DETECTOR_AVAILABLE",
    
    # Text detection
    "TextPlagiarismDetector",
    "TextDetector",  # alias (points to enhanced if available)
    "TextMatch",
    
    # Image features
    "ImageFeatureExtractor", 
    "MultiMethodFeatureExtractor",  # alias
    "get_image_extractor",
    
    # Image comparison
    "MultiMethodComparator",
    "ImageComparator",  # alias
    "ComparisonResult",
    
    # AI detection
    "AIDetector",
    "AIGeneratedDetector",  # alias
    "AIDetectionResult",
    "detect_ai_content",
    
    # Splice detection
    "SpliceDetector",
    "SpliceDetectionResult",
    "detect_manipulation",
    
    # Main detector
    "WorldClassDetector",
    "DetectorConfig",
]
