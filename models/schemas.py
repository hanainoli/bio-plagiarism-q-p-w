"""
Data models and schemas for plagiarism detection system.
Supports bioRxiv, medRxiv, and PMC papers with multi-modal detection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================

class Server(str, Enum):
    """Supported preprint servers"""
    BIORXIV = "biorxiv"
    MEDRXIV = "medrxiv"
    PMC = "pmc"


class MatchType(str, Enum):
    """Types of plagiarism matches"""
    EXACT_COPY = "EXACT_COPY"
    NEAR_EXACT = "NEAR_EXACT"
    DIRECT_COPY = "DIRECT_COPY"
    PARAPHRASE = "PARAPHRASE"
    MOSAIC = "MOSAIC"
    COMMON_KNOWLEDGE = "COMMON_KNOWLEDGE"
    PROPER_CITATION = "PROPER_CITATION"
    NO_MATCH = "NO_MATCH"
    # NEW: Enhanced detector match types
    COMMON_TOPIC = "COMMON_TOPIC"           # Same subject, different wording (not plagiarism)
    HIGH_SIMILARITY = "HIGH_SIMILARITY"     # High but not exact similarity
    MODERATE_SIMILARITY = "MODERATE_SIMILARITY"
    LOW_SIMILARITY = "LOW_SIMILARITY"


class ImageMatchType(str, Enum):
    """Types of image matches"""
    EXACT_COPY = "EXACT_COPY"
    NEAR_EXACT = "NEAR_EXACT"
    ROTATED_COPY = "ROTATED_COPY"
    FLIPPED_COPY = "FLIPPED_COPY"
    CROPPED = "CROPPED"
    RESIZED = "RESIZED"
    COLOR_MODIFIED = "COLOR_MODIFIED"
    RELABELED = "RELABELED"
    SPLICED = "SPLICED"
    AI_GENERATED = "AI_GENERATED"
    HIGH_SIMILARITY = "HIGH_SIMILARITY"
    MODERATE_SIMILARITY = "MODERATE_SIMILARITY"
    LOW_SIMILARITY = "LOW_SIMILARITY"
    MINIMAL_SIMILARITY = "MINIMAL_SIMILARITY"


class ImageType(str, Enum):
    """Types of scientific images"""
    WESTERN_BLOT = "western_blot"
    GEL = "gel"
    MICROSCOPY = "microscopy"
    FLOW_CYTOMETRY = "flow_cytometry"
    GRAPH = "graph"
    SCATTER_PLOT = "scatter_plot"
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    FLOWCHART = "flowchart"
    SCHEMATIC = "schematic"
    PHOTOGRAPH = "photograph"
    GENERAL = "general"


# =============================================================================
# PAPER IDENTIFICATION
# =============================================================================

@dataclass
class PaperID:
    """Unique paper identification"""
    doi: str
    server: Server
    paper_id: str  # Sanitized ID for storage
    
    @classmethod
    def from_doi(cls, doi: str, server: str) -> 'PaperID':
        """Create PaperID from DOI and server"""
        # Extract suffix after "10.1101/"
        if doi.startswith("10.1101/"):
            doi_suffix = doi[8:]
        else:
            doi_suffix = doi
        
        # Sanitize: replace . and / with _
        sanitized = doi_suffix.replace(".", "_").replace("/", "_")
        paper_id = f"{server}_{sanitized}"
        
        return cls(
            doi=doi,
            server=Server(server),
            paper_id=paper_id
        )
    
    def get_wasabi_base_path(self) -> str:
        """Get Wasabi storage path for this paper"""
        doi_suffix = self.doi.replace("10.1101/", "")
        parts = doi_suffix.split(".")
        
        if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 4:
            # New format: 2024.01.15.575685
            return f"{self.server.value}/{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"
        else:
            # Old format: UUID
            return f"{self.server.value}/legacy/{doi_suffix}"


# =============================================================================
# TEXT STRUCTURES
# =============================================================================

@dataclass
class TextChunk:
    """A chunk of text from a paper"""
    chunk_id: str
    paper_id: str
    chunk_index: int
    section: str
    text: str
    start_word: int
    end_word: int
    vector: Optional[np.ndarray] = None
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format"""
        return {
            "paper_id": self.paper_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "text": self.text,
            "start_word": self.start_word,
            "end_word": self.end_word
        }


@dataclass
class Abstract:
    """Paper abstract with metadata"""
    paper_id: str
    doi: str
    server: str
    title: str
    authors: List[str]
    posted_date: str
    category: str
    text: str
    word_count: int
    chunk_count: int
    figure_count: int
    table_count: int
    vector: Optional[np.ndarray] = None
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format"""
        return {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "server": self.server,
            "title": self.title,
            "authors": self.authors,
            "posted_date": self.posted_date,
            "category": self.category,
            "text": self.text,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "figure_count": self.figure_count,
            "table_count": self.table_count
        }


# =============================================================================
# IMAGE STRUCTURES
# =============================================================================

@dataclass
class BoundingBox:
    """Bounding box for regions"""
    x: int
    y: int
    w: int
    h: int
    
    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class OCRRegion:
    """OCR detected text region"""
    text: str
    bbox: BoundingBox
    confidence: float
    text_type: str  # scale_bar, figure_label, condition_label, axis_number, text


@dataclass
class OrientationHashes:
    """Perceptual hashes for all 8 orientations"""
    original: Dict[str, str]  # phash, dhash, ahash
    rot90: Dict[str, str]
    rot180: Dict[str, str]
    rot270: Dict[str, str]
    flip_h: Dict[str, str]
    flip_v: Dict[str, str]
    flip_h_rot90: Dict[str, str]
    flip_v_rot90: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        return {
            "original": self.original,
            "rot90": self.rot90,
            "rot180": self.rot180,
            "rot270": self.rot270,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "flip_h_rot90": self.flip_h_rot90,
            "flip_v_rot90": self.flip_v_rot90
        }


@dataclass
class RotationInvariantFeatures:
    """Rotation-invariant image features"""
    radial_histogram: List[float]  # 32-dim
    zernike_moments: List[float]   # 25-dim
    orb_keypoint_count: int


@dataclass 
class IntegrityAnalysis:
    """Image integrity analysis results"""
    ela_computed: bool = False
    ela_suspicious_regions: List[Dict] = field(default_factory=list)
    copy_move_checked: bool = False
    copy_move_pairs: List[Dict] = field(default_factory=list)
    noise_analyzed: bool = False
    noise_inconsistencies: List[Dict] = field(default_factory=list)
    ai_detection_score: float = 0.0
    ai_risk_factors: List[str] = field(default_factory=list)


@dataclass
class SubFigure:
    """Sub-panel of a composite figure"""
    sub_id: str
    label: str  # A, B, C, etc.
    bbox: BoundingBox
    phash: str
    cnn_vector: Optional[np.ndarray] = None


@dataclass
class Figure:
    """Complete figure with all features"""
    figure_id: str
    paper_id: str
    doi: str
    server: str
    label: str  # Figure 1, Figure 2, etc.
    caption: str
    page_number: int
    figure_index: int
    width: int
    height: int
    
    # Vectors
    cnn_vector: Optional[np.ndarray] = None  # 2048-dim
    clip_vector: Optional[np.ndarray] = None  # 512-dim
    
    # Hashes
    orientation_hashes: Optional[OrientationHashes] = None
    
    # Rotation-invariant features
    rotation_invariant: Optional[RotationInvariantFeatures] = None
    
    # OCR data
    ocr_regions: List[OCRRegion] = field(default_factory=list)
    text_hash: str = ""
    
    # Sub-figures
    sub_figures: List[SubFigure] = field(default_factory=list)
    
    # Image type classification
    image_type: ImageType = ImageType.GENERAL
    image_type_confidence: float = 0.0
    
    # Integrity analysis
    integrity: Optional[IntegrityAnalysis] = None
    
    # Storage URL
    wasabi_url: str = ""
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format"""
        payload = {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "server": self.server,
            "figure_id": self.figure_id,
            "label": self.label,
            "caption": self.caption,
            "page_number": self.page_number,
            "figure_index": self.figure_index,
            "image_metadata": {
                "width": self.width,
                "height": self.height
            },
            "image_type": {
                "classification": self.image_type.value,
                "confidence": self.image_type_confidence
            },
            "wasabi_url": self.wasabi_url
        }
        
        # Add orientation hashes
        if self.orientation_hashes:
            payload["perceptual_hashes"] = self.orientation_hashes.to_dict()
        
        # Add rotation-invariant features
        if self.rotation_invariant:
            payload["rotation_invariant"] = {
                "radial_histogram": self.rotation_invariant.radial_histogram,
                "zernike_moments": self.rotation_invariant.zernike_moments,
                "orb_keypoint_count": self.rotation_invariant.orb_keypoint_count
            }
        
        # Add OCR data
        payload["ocr_data"] = {
            "has_text": len(self.ocr_regions) > 0,
            "text_regions": [
                {
                    "text": r.text,
                    "bbox": r.bbox.to_dict(),
                    "confidence": r.confidence,
                    "type": r.text_type
                }
                for r in self.ocr_regions
            ],
            "text_hash": self.text_hash
        }
        
        # Add sub-figures
        if self.sub_figures:
            payload["sub_figures"] = [
                {
                    "sub_id": sf.sub_id,
                    "label": sf.label,
                    "bbox": sf.bbox.to_dict(),
                    "phash": sf.phash
                }
                for sf in self.sub_figures
            ]
        
        # Add integrity analysis
        if self.integrity:
            payload["integrity_analysis"] = {
                "ela_computed": self.integrity.ela_computed,
                "copy_move_checked": self.integrity.copy_move_checked,
                "noise_analyzed": self.integrity.noise_analyzed,
                "ai_detection_score": self.integrity.ai_detection_score
            }
        
        return payload


# =============================================================================
# TABLE STRUCTURES
# =============================================================================

@dataclass
class Table:
    """Table extracted from paper"""
    table_id: str
    paper_id: str
    doi: str
    server: str
    label: str  # Table 1, Table 2, etc.
    caption: str
    page_number: int
    table_index: int
    full_text: str
    row_count: int
    col_count: int
    structured_data: Optional[Dict] = None  # {headers: [], rows: [[]]}
    vector: Optional[np.ndarray] = None
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant payload format"""
        return {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "server": self.server,
            "table_id": self.table_id,
            "label": self.label,
            "caption": self.caption,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "full_text": self.full_text,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "structured_data": self.structured_data
        }


# =============================================================================
# MATCH RESULTS
# =============================================================================

@dataclass
class TextMatch:
    """Result of text plagiarism detection"""
    query_text: str
    query_section: str
    source_paper_id: str
    source_doi: str
    source_title: str
    source_section: str
    source_text: str
    
    similarity_score: float
    match_type: MatchType
    
    word_overlap_percentage: float
    containment: float
    jaccard: float
    matched_word_count: int
    
    matched_spans: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ImageMatch:
    """Result of image plagiarism detection"""
    query_figure_id: str
    source_figure_id: str
    source_paper_id: str
    source_doi: str
    source_label: str
    source_caption: str
    
    combined_score: float
    match_type: ImageMatchType
    
    # Individual method scores
    phash_similarity: float = 0.0
    phash_hamming: int = 64
    cnn_similarity: float = 0.0
    ocr_text_overlap: float = 0.0
    ocr_position_match: float = 0.0
    
    # Rotation detection
    detected_rotation: str = "original"
    is_flipped: bool = False
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityResult:
    """Result of image integrity analysis"""
    figure_id: str
    is_manipulated: bool
    manipulation_score: float
    
    ela_suspicious: bool = False
    ela_regions: List[Dict] = field(default_factory=list)
    
    copy_move_detected: bool = False
    copy_move_pairs: List[Dict] = field(default_factory=list)
    
    noise_inconsistent: bool = False
    noise_regions: List[Dict] = field(default_factory=list)
    
    ai_generated: bool = False
    ai_confidence: float = 0.0
    ai_risk_factors: List[str] = field(default_factory=list)


# =============================================================================
# COMPLETE PAPER
# =============================================================================

@dataclass
class Paper:
    """Complete paper with all extracted content"""
    paper_id: PaperID
    title: str
    authors: List[str]
    posted_date: str
    category: str
    
    abstract: Optional[Abstract] = None
    chunks: List[TextChunk] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    
    # Metadata
    processed_at: Optional[datetime] = None
    source_url: str = ""
    pdf_path: str = ""


# =============================================================================
# DETECTION REPORT
# =============================================================================

@dataclass
class PlagiarismReport:
    """Complete plagiarism detection report"""
    query_paper_id: str
    query_title: str
    analyzed_at: datetime
    
    # Text matches
    text_matches: List[TextMatch] = field(default_factory=list)
    text_plagiarism_score: float = 0.0
    
    # Image matches
    image_matches: List[ImageMatch] = field(default_factory=list)
    image_plagiarism_score: float = 0.0
    
    # Integrity issues
    integrity_issues: List[IntegrityResult] = field(default_factory=list)
    
    # Summary
    overall_score: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "query_paper_id": self.query_paper_id,
            "query_title": self.query_title,
            "analyzed_at": self.analyzed_at.isoformat(),
            "summary": {
                "overall_score": self.overall_score,
                "risk_level": self.risk_level,
                "text_plagiarism_score": self.text_plagiarism_score,
                "image_plagiarism_score": self.image_plagiarism_score,
                "total_text_matches": len(self.text_matches),
                "total_image_matches": len(self.image_matches),
                "integrity_issues": len(self.integrity_issues)
            },
            "text_matches": [
                {
                    # Source information
                    "source_doi": m.source_doi,
                    "source_paper_id": m.source_paper_id,
                    "source_title": m.source_title,
                    "source_section": m.source_section,
                    
                    # Match details
                    "match_type": m.match_type.value if hasattr(m.match_type, 'value') else str(m.match_type),
                    "similarity_score": m.similarity_score,
                    "word_overlap_percentage": m.word_overlap_percentage,
                    "containment": m.containment,
                    "jaccard": m.jaccard,
                    "matched_word_count": m.matched_word_count,
                    
                    # THE ACTUAL TEXT - This is what shows WHY it's a match
                    "query_text": m.query_text,
                    "query_section": m.query_section,
                    "source_text": m.source_text,
                    
                    # Matched spans (character positions)
                    "matched_spans": m.matched_spans
                }
                for m in self.text_matches
            ],
            "image_matches": [
                {
                    # Source information
                    "source_doi": m.source_doi,
                    "source_paper_id": m.source_paper_id,
                    "source_label": m.source_label,
                    "source_caption": m.source_caption,
                    "source_figure_id": m.source_figure_id,
                    
                    # Query information
                    "query_figure_id": m.query_figure_id,
                    
                    # Match details
                    "match_type": m.match_type.value if hasattr(m.match_type, 'value') else str(m.match_type),
                    "combined_score": m.combined_score,
                    
                    # Individual method scores - shows WHY it's a match
                    "phash_similarity": m.phash_similarity,
                    "phash_hamming_distance": m.phash_hamming,
                    "cnn_similarity": m.cnn_similarity,
                    "ocr_text_overlap": m.ocr_text_overlap,
                    "ocr_position_match": m.ocr_position_match,
                    
                    # Transformation detection
                    "detected_rotation": m.detected_rotation,
                    "is_flipped": m.is_flipped,
                    
                    # Additional details
                    "details": m.details
                }
                for m in self.image_matches
            ],
            "integrity_issues": [
                {
                    "figure_id": i.figure_id,
                    "is_manipulated": i.is_manipulated,
                    "manipulation_score": i.manipulation_score,
                    
                    # ELA analysis
                    "ela_suspicious": i.ela_suspicious,
                    "ela_regions": i.ela_regions,
                    
                    # Copy-move detection
                    "copy_move_detected": i.copy_move_detected,
                    "copy_move_pairs": i.copy_move_pairs,
                    
                    # Noise analysis
                    "noise_inconsistent": i.noise_inconsistent,
                    "noise_regions": i.noise_regions,
                    
                    # AI generation detection
                    "ai_generated": i.ai_generated,
                    "ai_confidence": i.ai_confidence,
                    "ai_risk_factors": i.ai_risk_factors
                }
                for i in self.integrity_issues
            ]
        }
