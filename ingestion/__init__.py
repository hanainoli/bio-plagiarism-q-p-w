"""
Ingestion Module
================
Category-filtered ingestion pipeline for bioRxiv/medRxiv papers.

Components:
    - CategoryIndexBuilder: Builds index of papers organized by category
    - BioRxivS3Downloader: Downloads MECA files from bioRxiv S3 bucket
    - MECAExtractor: Extracts content from MECA files
    - CategoryFilteredPipeline: Complete ingestion pipeline

Quick Start:
    # Build category index (one-time, 2-3 hours)
    from ingestion import build_category_index
    build_category_index()
    
    # Run pipeline for specific categories
    from ingestion import run_pipeline
    stats = run_pipeline(
        categories=["cancer biology", "oncology"],
        max_papers=100,  # for testing
        store=False      # don't store yet
    )
    
    # Daily updates
    from ingestion import run_daily_update
    run_daily_update(categories=["cancer biology"], days=2)

CLI Usage:
    # Build index
    python -m ingestion.pipeline --build-index
    
    # Run pipeline
    python -m ingestion.pipeline --categories "cancer biology" --run
    
    # Daily update
    python -m ingestion.daily_update --days 2
"""

# Category index (requires: requests)
try:
    from .category_index import (
        CategoryIndexBuilder,
        CategoryIndex,
        build_index as build_category_index,
        update_index as update_category_index,
        get_category_dois,
        get_index_stats,
        BIORXIV_CATEGORIES,
        MEDRXIV_CATEGORIES,
    )
except ImportError as e:
    CategoryIndexBuilder = None
    CategoryIndex = None
    build_category_index = None
    update_category_index = None
    get_category_dois = None
    get_index_stats = None
    BIORXIV_CATEGORIES = []
    MEDRXIV_CATEGORIES = []
    print(f"Warning: Could not import category_index: {e}")

# S3 Downloader (requires: boto3)
try:
    from .s3_downloader import (
        BioRxivS3Downloader,
        download_category,
        download_dois,
        DownloadResult,
        DownloadStats,
    )
except ImportError as e:
    BioRxivS3Downloader = None
    download_category = None
    download_dois = None
    DownloadResult = None
    DownloadStats = None
    print(f"Warning: Could not import s3_downloader (install boto3): {e}")

# MECA Extractor (requires: lxml, PyMuPDF, pdfplumber)
try:
    from .meca_extractor import (
        MECAExtractor,
        MECAExtractionResult,
        ExtractedImage,
        ExtractedTable,
        ExtractedFormula,
        Author,
        extract_meca,
        extract_pdf,
    )
except ImportError as e:
    MECAExtractor = None
    MECAExtractionResult = None
    ExtractedImage = None
    ExtractedTable = None
    ExtractedFormula = None
    Author = None
    extract_meca = None
    extract_pdf = None
    print(f"Warning: Could not import meca_extractor: {e}")

# Pipeline
try:
    from .pipeline import (
        CategoryFilteredPipeline,
        PipelineConfig,
        PipelineStats,
        run_pipeline,
        process_directory,
    )
except ImportError as e:
    CategoryFilteredPipeline = None
    PipelineConfig = None
    PipelineStats = None
    run_pipeline = None
    process_directory = None
    print(f"Warning: Could not import pipeline: {e}")

# Daily update
try:
    from .daily_update import (
        run_daily_update,
    )
except ImportError as e:
    run_daily_update = None
    print(f"Warning: Could not import daily_update: {e}")

__all__ = [
    # Index
    'CategoryIndexBuilder',
    'CategoryIndex',
    'build_category_index',
    'update_category_index',
    'get_category_dois',
    'get_index_stats',
    'BIORXIV_CATEGORIES',
    'MEDRXIV_CATEGORIES',
    
    # Downloader
    'BioRxivS3Downloader',
    'download_category',
    'download_dois',
    'DownloadResult',
    'DownloadStats',
    
    # Extractor
    'MECAExtractor',
    'MECAExtractionResult',
    'ExtractedImage',
    'ExtractedTable',
    'ExtractedFormula',
    'Author',
    'extract_meca',
    'extract_pdf',
    
    # Pipeline
    'CategoryFilteredPipeline',
    'PipelineConfig',
    'PipelineStats',
    'run_pipeline',
    'process_directory',
    
    # Updates
    'run_daily_update',
]
