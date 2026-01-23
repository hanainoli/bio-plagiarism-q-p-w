"""
Configuration settings for plagiarism detection system.

Loads configuration from:
1. .env file (if exists)
2. Environment variables
3. Default values

Usage:
    from config.settings import load_config
    config = load_config()
    
    # Or load from specific .env file
    config = load_config(env_file=".env.production")
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    
    # Find .env file (check current dir and parent dirs)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    elif Path(".env").exists():
        load_dotenv(".env")
    
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def get_env(key: str, default: str = None) -> Optional[str]:
    """Get environment variable with fallback"""
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class QdrantConfig:
    """Qdrant database configuration"""
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    https: bool = False
    
    # Collection names
    abstracts_collection: str = "abstracts"
    chunks_collection: str = "fulltext_chunks"
    figures_collection: str = "figures"
    subfigures_collection: str = "sub_figures"
    tables_collection: str = "tables"
    
    @classmethod
    def from_env(cls) -> 'QdrantConfig':
        """Load configuration from environment variables"""
        return cls(
            host=get_env('QDRANT_HOST', 'localhost'),
            port=get_env_int('QDRANT_PORT', 6333),
            api_key=get_env('QDRANT_API_KEY'),
            https=get_env_bool('QDRANT_HTTPS', False)
        )
    
    @property
    def is_cloud(self) -> bool:
        """Check if using Qdrant Cloud"""
        return self.api_key is not None and 'qdrant.io' in self.host


@dataclass
class WasabiConfig:
    """Wasabi S3 storage configuration (OPTIONAL)"""
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket: str = "uyar-plagiarism"
    region: str = "us-east-1"
    endpoint_url: str = "https://s3.wasabisys.com"
    enabled: bool = False  # Disabled by default
    
    @classmethod
    def from_env(cls) -> 'WasabiConfig':
        """Load configuration from environment variables"""
        access_key = get_env('WASABI_ACCESS_KEY')
        secret_key = get_env('WASABI_SECRET_KEY')
        
        # Only enable if credentials are provided
        enabled = access_key is not None and secret_key is not None
        
        return cls(
            access_key=access_key,
            secret_key=secret_key,
            bucket=get_env('WASABI_BUCKET', 'uyar-plagiarism'),
            region=get_env('WASABI_REGION', 'us-east-1'),
            enabled=enabled
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Wasabi is properly configured"""
        return self.enabled and self.access_key and self.secret_key


@dataclass
class LocalStorageConfig:
    """Local file storage configuration (when Wasabi is not used)"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    meca_dir: Path = None
    figures_dir: Path = None
    index_path: Path = None
    logs_dir: Path = None
    
    def __post_init__(self):
        """Set up directory paths"""
        if self.meca_dir is None:
            self.meca_dir = self.base_dir / "meca_files"
        if self.figures_dir is None:
            self.figures_dir = self.base_dir / "figures"
        if self.index_path is None:
            self.index_path = self.base_dir / "biorxiv_index.json"
        if self.logs_dir is None:
            self.logs_dir = self.base_dir / "logs"
    
    @classmethod
    def from_env(cls) -> 'LocalStorageConfig':
        """Load from environment variables"""
        base_dir = Path(get_env('DATA_DIR', str(Path(__file__).parent.parent / "data")))
        
        return cls(
            base_dir=base_dir,
            meca_dir=Path(get_env('MECA_DIR')) if get_env('MECA_DIR') else None,
            figures_dir=Path(get_env('FIGURES_DIR')) if get_env('FIGURES_DIR') else None,
            index_path=Path(get_env('INDEX_PATH')) if get_env('INDEX_PATH') else None,
            logs_dir=Path(get_env('LOGS_DIR')) if get_env('LOGS_DIR') else None,
        )
    
    def ensure_directories(self):
        """Create all required directories"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.meca_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "reports").mkdir(parents=True, exist_ok=True)


@dataclass
class TextDetectionConfig:
    """Text plagiarism detection configuration"""
    # N-gram settings
    ngram_size: int = 5
    min_match_length: int = 5
    
    # Thresholds
    exact_threshold: float = 0.95
    high_threshold: float = 0.80
    moderate_threshold: float = 0.50
    
    # Chunking
    chunk_size: int = 200
    chunk_overlap: float = 0.50
    min_chunk_size: int = 50
    
    # Search limits
    max_search_results: int = 100
    max_returned_matches: int = 50


@dataclass
class ImageDetectionConfig:
    """Image plagiarism detection configuration"""
    # Perceptual hash settings
    phash_threshold: int = 10  # Hamming distance
    dhash_threshold: int = 12
    ahash_threshold: int = 15
    
    # CNN similarity
    cnn_threshold: float = 0.85
    
    # OCR settings
    ocr_threshold: float = 0.70
    min_ocr_confidence: float = 0.50
    
    # Combined score
    combined_threshold: float = 0.75
    
    # Weights for combined score
    hash_weight: float = 0.25
    cnn_weight: float = 0.40
    ocr_weight: float = 0.15
    rotation_weight: float = 0.20
    
    # Feature extraction
    min_figure_size: int = 100
    extraction_dpi: int = 150
    use_gpu: bool = True
    
    # Search limits
    max_search_results: int = 50
    max_returned_matches: int = 20


@dataclass
class IntegrityConfig:
    """Image integrity analysis configuration"""
    # ELA settings
    ela_quality: int = 95
    ela_threshold: float = 50
    
    # Copy-move detection
    block_size: int = 16
    min_match_distance: int = 50
    
    # Overall
    splice_threshold: float = 0.30
    run_splice_detection: bool = True


@dataclass
class AIDetectionConfig:
    """AI-generated image detection configuration"""
    run_ai_detection: bool = True
    ai_threshold: float = 0.50
    model_path: Optional[str] = None
    use_gpu: bool = True


@dataclass
class ProcessingConfig:
    """General processing configuration"""
    batch_size: int = 32
    max_workers: int = 4
    use_gpu: bool = True
    log_level: str = "INFO"
    
    # Supported servers
    supported_servers: tuple = ("biorxiv", "medrxiv", "pmc")
    
    # File size limits
    max_pdf_size_mb: int = 100
    max_image_dimension: int = 4096


@dataclass
class AppConfig:
    """Complete application configuration"""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    wasabi: WasabiConfig = field(default_factory=WasabiConfig)
    local_storage: LocalStorageConfig = field(default_factory=LocalStorageConfig)
    text_detection: TextDetectionConfig = field(default_factory=TextDetectionConfig)
    image_detection: ImageDetectionConfig = field(default_factory=ImageDetectionConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    ai_detection: AIDetectionConfig = field(default_factory=AIDetectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Load configuration from environment"""
        config = cls(
            qdrant=QdrantConfig.from_env(),
            wasabi=WasabiConfig.from_env(),
            local_storage=LocalStorageConfig.from_env()
        )
        
        # Update processing config from env
        config.processing.use_gpu = get_env_bool('USE_GPU', True)
        config.processing.max_workers = get_env_int('MAX_WORKERS', 4)
        config.processing.batch_size = get_env_int('BATCH_SIZE', 32)
        
        # Update detection thresholds from env
        if get_env('TEXT_EXACT_THRESHOLD'):
            config.text_detection.exact_threshold = get_env_float('TEXT_EXACT_THRESHOLD', 0.95)
        if get_env('TEXT_HIGH_THRESHOLD'):
            config.text_detection.high_threshold = get_env_float('TEXT_HIGH_THRESHOLD', 0.80)
        if get_env('IMAGE_CNN_THRESHOLD'):
            config.image_detection.cnn_threshold = get_env_float('IMAGE_CNN_THRESHOLD', 0.85)
        
        return config
    
    @property
    def use_wasabi(self) -> bool:
        """Check if Wasabi storage should be used"""
        return self.wasabi.is_configured
    
    @property
    def figures_path(self) -> Path:
        """Get path for figure storage"""
        return self.local_storage.figures_dir
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Load configuration from dictionary"""
        config = cls()
        
        if 'qdrant' in config_dict:
            config.qdrant = QdrantConfig(**config_dict['qdrant'])
        
        if 'wasabi' in config_dict:
            config.wasabi = WasabiConfig(**config_dict['wasabi'])
        
        if 'text_detection' in config_dict:
            config.text_detection = TextDetectionConfig(**config_dict['text_detection'])
        
        if 'image_detection' in config_dict:
            config.image_detection = ImageDetectionConfig(**config_dict['image_detection'])
        
        if 'integrity' in config_dict:
            config.integrity = IntegrityConfig(**config_dict['integrity'])
        
        if 'ai_detection' in config_dict:
            config.ai_detection = AIDetectionConfig(**config_dict['ai_detection'])
        
        if 'processing' in config_dict:
            config.processing = ProcessingConfig(**config_dict['processing'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# Default configuration instance
default_config = AppConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to YAML/JSON config file
    
    Returns:
        AppConfig instance
    """
    if config_path:
        import json
        try:
            import yaml
            
            with open(config_path) as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            return AppConfig.from_dict(config_dict)
        
        except ImportError:
            with open(config_path) as f:
                config_dict = json.load(f)
            return AppConfig.from_dict(config_dict)
    
    return AppConfig.from_env()


# Example configuration file content
EXAMPLE_CONFIG_YAML = """
# Plagiarism Detection Configuration

qdrant:
  host: localhost
  port: 6333
  https: false

wasabi:
  bucket: uyar-plagiarism
  region: us-east-1

text_detection:
  ngram_size: 5
  exact_threshold: 0.95
  high_threshold: 0.80
  moderate_threshold: 0.50
  chunk_size: 200
  chunk_overlap: 0.50

image_detection:
  phash_threshold: 10
  cnn_threshold: 0.85
  combined_threshold: 0.75
  use_gpu: true

integrity:
  run_splice_detection: true
  splice_threshold: 0.30

ai_detection:
  run_ai_detection: true
  ai_threshold: 0.50

processing:
  batch_size: 32
  max_workers: 4
  use_gpu: true
"""


if __name__ == "__main__":
    # Test configuration loading
    config = AppConfig()
    
    print("Default Configuration:")
    print(f"  Qdrant Host: {config.qdrant.host}:{config.qdrant.port}")
    print(f"  Wasabi Bucket: {config.wasabi.bucket}")
    print(f"  Text N-gram Size: {config.text_detection.ngram_size}")
    print(f"  Image CNN Threshold: {config.image_detection.cnn_threshold}")
    print(f"  Run Splice Detection: {config.integrity.run_splice_detection}")
    print(f"  Run AI Detection: {config.ai_detection.run_ai_detection}")
