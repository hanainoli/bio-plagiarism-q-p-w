"""
Data models for plagiarism detection system.
"""

from .schemas import (
    # Enums
    Server,
    MatchType,
    ImageMatchType,
    ImageType,
    
    # Paper identification
    PaperID,
    
    # Text structures
    TextChunk,
    Abstract,
    
    # Image structures
    BoundingBox,
    OCRRegion,
    OrientationHashes,
    RotationInvariantFeatures,
    IntegrityAnalysis,
    SubFigure,
    Figure,
    
    # Table structures
    Table,
    
    # Match results
    TextMatch,
    ImageMatch,
    IntegrityResult,
    
    # Complete structures
    Paper,
    PlagiarismReport,
)

__all__ = [
    # Enums
    'Server',
    'MatchType', 
    'ImageMatchType',
    'ImageType',
    
    # Paper identification
    'PaperID',
    
    # Text structures
    'TextChunk',
    'Abstract',
    
    # Image structures
    'BoundingBox',
    'OCRRegion',
    'OrientationHashes',
    'RotationInvariantFeatures',
    'IntegrityAnalysis',
    'SubFigure',
    'Figure',
    
    # Table structures
    'Table',
    
    # Match results
    'TextMatch',
    'ImageMatch',
    'IntegrityResult',
    
    # Complete structures
    'Paper',
    'PlagiarismReport',
]
