"""
Configuration module for plagiarism detection system.

ALL configuration is loaded from .env file.

Usage:
    from config import get_config
    config = get_config()
    
    # Access Qdrant settings
    print(config.qdrant.host)
    
    # Access Wasabi settings
    print(config.wasabi.bucket)
    
    # Access detection thresholds
    print(config.text_detection.exact_threshold)
"""

import os
from pathlib import Path

# Load .env file FIRST before anything else
_env_loaded = False
try:
    from dotenv import load_dotenv
    
    # Find .env file
    env_paths = [
        Path(__file__).parent.parent / ".env",  # plagiarism_detection/.env
        Path.cwd() / ".env",                     # current directory
        Path(__file__).parent.parent.parent / ".env"  # parent directory
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            _env_loaded = True
            break
            
except ImportError:
    pass

# Now import settings (which will use the loaded env vars)
from .settings import (
    AppConfig,
    QdrantConfig,
    WasabiConfig,
    LocalStorageConfig,
    TextDetectionConfig,
    ImageDetectionConfig,
    IntegrityConfig,
    AIDetectionConfig,
    ProcessingConfig,
    load_config,
    default_config,
    get_env,
    get_env_bool,
    get_env_int,
    get_env_float
)

# Singleton config instance
_config = None

def get_config() -> AppConfig:
    """
    Get the global configuration instance.
    Loads from .env file and environment variables.
    """
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config

def reload_config() -> AppConfig:
    """Force reload configuration from environment"""
    global _config
    _config = AppConfig.from_env()
    return _config

def print_config_status():
    """Print current configuration status"""
    config = get_config()
    
    print("="*60)
    print("CONFIGURATION STATUS")
    print("="*60)
    
    # Qdrant
    print(f"\nQdrant:")
    print(f"  Host: {config.qdrant.host}")
    print(f"  Port: {config.qdrant.port}")
    print(f"  API Key: {'***' + config.qdrant.api_key[-4:] if config.qdrant.api_key else 'None'}")
    print(f"  HTTPS: {config.qdrant.https}")
    print(f"  Is Cloud: {config.qdrant.is_cloud}")
    
    # Wasabi
    print(f"\nWasabi S3:")
    print(f"  Configured: {config.wasabi.is_configured}")
    if config.wasabi.is_configured:
        print(f"  Bucket: {config.wasabi.bucket}")
        print(f"  Region: {config.wasabi.region}")
        print(f"  Access Key: {config.wasabi.access_key[:8]}..." if config.wasabi.access_key else "None")
    
    # Processing
    print(f"\nProcessing:")
    print(f"  Use GPU: {config.processing.use_gpu}")
    print(f"  Max Workers: {config.processing.max_workers}")
    print(f"  Batch Size: {config.processing.batch_size}")
    
    # Thresholds
    print(f"\nThresholds:")
    print(f"  Text Exact: {config.text_detection.exact_threshold}")
    print(f"  Text High: {config.text_detection.high_threshold}")
    print(f"  Image CNN: {config.image_detection.cnn_threshold}")
    
    print("="*60)

# Aliases for convenience
Settings = AppConfig
QdrantSettings = QdrantConfig
WasabiSettings = WasabiConfig
get_settings = load_config

__all__ = [
    'get_config',
    'reload_config',
    'print_config_status',
    'AppConfig',
    'QdrantConfig', 
    'WasabiConfig',
    'LocalStorageConfig',
    'TextDetectionConfig',
    'ImageDetectionConfig',
    'IntegrityConfig',
    'AIDetectionConfig',
    'ProcessingConfig',
    'load_config',
    'default_config',
    'get_env',
    'get_env_bool',
    'get_env_int',
    'get_env_float',
    # Aliases
    'Settings',
    'QdrantSettings',
    'WasabiSettings',
    'get_settings',
]
