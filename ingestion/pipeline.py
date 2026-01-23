"""
Category-Filtered Ingestion Pipeline
=====================================
Complete pipeline for ingesting bioRxiv/medRxiv papers by category.

Workflow:
    1. Build/load category index (from API metadata)
    2. Filter papers by target categories
    3. Download MECA files from S3 (fast, no rate limit)
    4. Extract content (text, images, tables)
    5. Generate embeddings
    6. Store in Qdrant + Wasabi

Usage:
    # Full pipeline for cancer biology
    python pipeline.py --categories "cancer biology" "oncology" --run
    
    # Just build index
    python pipeline.py --build-index
    
    # Just download (no processing)
    python pipeline.py --categories "cancer biology" --download-only
    
    # Process already downloaded files
    python pipeline.py --process-local ./data/meca_files/cancer_biology
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import sys

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.category_index import CategoryIndexBuilder, CategoryIndex, DEFAULT_INDEX_PATH
from ingestion.s3_downloader import BioRxivS3Downloader, DownloadStats
from ingestion.meca_extractor import MECAExtractor, MECAExtractionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Target categories
    categories: List[str] = field(default_factory=list)
    
    # Paths
    index_path: Path = DEFAULT_INDEX_PATH
    download_dir: Path = Path(__file__).parent.parent / "data" / "meca_files"
    figure_dir: Path = Path(__file__).parent.parent / "data" / "figures"
    
    # Processing settings
    batch_size: int = 100
    download_workers: int = 4
    process_workers: int = 2
    
    # Options
    skip_existing: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    build_embeddings: bool = True
    upload_to_wasabi: bool = False
    store_to_qdrant: bool = True
    
    # Limits (for testing)
    max_papers: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "categories": self.categories,
            "index_path": str(self.index_path),
            "download_dir": str(self.download_dir),
            "figure_dir": str(self.figure_dir),
            "batch_size": self.batch_size,
            "download_workers": self.download_workers,
            "process_workers": self.process_workers,
            "skip_existing": self.skip_existing,
            "extract_images": self.extract_images,
            "extract_tables": self.extract_tables,
            "build_embeddings": self.build_embeddings,
            "upload_to_wasabi": self.upload_to_wasabi,
            "store_to_qdrant": self.store_to_qdrant,
            "max_papers": self.max_papers,
        }


@dataclass
class PipelineStats:
    """Pipeline execution statistics"""
    start_time: datetime = None
    end_time: datetime = None
    
    # Counts
    total_papers: int = 0
    downloaded: int = 0
    download_skipped: int = 0
    download_failed: int = 0
    processed: int = 0
    process_failed: int = 0
    stored: int = 0
    
    # Content stats
    total_figures: int = 0
    total_tables: int = 0
    total_words: int = 0
    
    # Size
    download_bytes: int = 0
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    @property
    def papers_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.processed / self.duration_seconds
        return 0
    
    def to_dict(self) -> dict:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "total_papers": self.total_papers,
            "downloaded": self.downloaded,
            "download_skipped": self.download_skipped,
            "download_failed": self.download_failed,
            "processed": self.processed,
            "process_failed": self.process_failed,
            "stored": self.stored,
            "total_figures": self.total_figures,
            "total_tables": self.total_tables,
            "total_words": self.total_words,
            "download_mb": round(self.download_bytes / 1024 / 1024, 2),
            "papers_per_second": round(self.papers_per_second, 4),
        }
    
    def print_summary(self):
        """Print summary to console"""
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {self.duration_seconds:.2f} seconds ({self.duration_seconds/60:.1f} minutes)")
        print(f"\nDownload:")
        print(f"  Total papers: {self.total_papers}")
        print(f"  Downloaded: {self.downloaded}")
        print(f"  Skipped (existing): {self.download_skipped}")
        print(f"  Failed: {self.download_failed}")
        print(f"  Total size: {self.download_bytes / 1024 / 1024:.2f} MB")
        print(f"\nProcessing:")
        print(f"  Processed: {self.processed}")
        print(f"  Failed: {self.process_failed}")
        print(f"  Stored: {self.stored}")
        print(f"\nContent extracted:")
        print(f"  Figures: {self.total_figures}")
        print(f"  Tables: {self.total_tables}")
        print(f"  Words: {self.total_words:,}")
        print(f"\nThroughput: {self.papers_per_second:.2f} papers/second")
        print(f"{'='*60}")


# =============================================================================
# PIPELINE
# =============================================================================

class CategoryFilteredPipeline:
    """
    Complete pipeline for category-filtered ingestion.
    
    Steps:
    1. Build/load category index
    2. Filter papers by category
    3. Download MECA files from S3
    4. Extract content
    5. (Optional) Generate embeddings
    6. (Optional) Store in Qdrant + Wasabi
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = PipelineStats()
        
        # Initialize components
        self.index_builder = CategoryIndexBuilder(config.index_path)
        self.downloader = BioRxivS3Downloader(
            output_dir=config.download_dir,
            skip_existing=config.skip_existing
        )
        self.extractor = MECAExtractor(
            extract_images=config.extract_images,
            extract_tables=config.extract_tables,
        )
        
        # Storage clients (lazy load)
        self._qdrant_client = None
        self._wasabi_client = None
        self._embedding_engine = None
        
        # Create directories
        config.download_dir.mkdir(parents=True, exist_ok=True)
        config.figure_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def qdrant_client(self):
        """Lazy load Qdrant client"""
        if self._qdrant_client is None and self.config.store_to_qdrant:
            try:
                from storage.qdrant_client import PlagiarismQdrantClient
                self._qdrant_client = PlagiarismQdrantClient()
                logger.info("Qdrant client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Qdrant client: {e}")
        return self._qdrant_client
    
    @property
    def wasabi_client(self):
        """Lazy load Wasabi client"""
        if self._wasabi_client is None and self.config.upload_to_wasabi:
            try:
                from storage.wasabi_client import WasabiClient
                self._wasabi_client = WasabiClient()
                logger.info("Wasabi client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Wasabi client: {e}")
        return self._wasabi_client
    
    @property
    def embedding_engine(self):
        """Lazy load embedding engine"""
        if self._embedding_engine is None and self.config.build_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_engine = SentenceTransformer('BAAI/bge-large-en-v1.5')
                logger.info("Embedding engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize embedding engine: {e}")
        return self._embedding_engine
    
    def build_index(self, update_only: bool = False, days: int = 7):
        """
        Build or update the category index.
        
        Args:
            update_only: If True, only fetch recent papers
            days: Days to look back for update
        """
        if update_only:
            logger.info(f"Updating index with papers from last {days} days...")
            self.index_builder.update_index(days=days)
        else:
            logger.info("Building full category index (this takes 2-3 hours)...")
            self.index_builder.build_full_index()
        
        self.index_builder.print_stats()
    
    def get_target_papers(self) -> List[dict]:
        """Get papers matching target categories"""
        if not self.config.categories:
            raise ValueError("No categories specified")
        
        papers = self.index_builder.get_papers_by_category(self.config.categories)
        
        if self.config.max_papers:
            papers = papers[:self.config.max_papers]
        
        return papers
    
    def download_papers(self, papers: List[dict] = None) -> List[Path]:
        """
        Download MECA files for target papers.
        
        Args:
            papers: Paper list (if None, fetches from index)
            
        Returns:
            List of downloaded file paths
        """
        if papers is None:
            papers = self.get_target_papers()
        
        self.stats.total_papers = len(papers)
        logger.info(f"Downloading {len(papers)} papers...")
        
        # Use downloader
        results, download_stats = self.downloader.download_category(
            categories=self.config.categories,
            index_path=self.config.index_path,
            workers=self.config.download_workers,
            max_papers=self.config.max_papers,
        )
        
        # Update stats
        self.stats.downloaded = download_stats.success
        self.stats.download_skipped = download_stats.skipped
        self.stats.download_failed = download_stats.failed
        self.stats.download_bytes = download_stats.total_bytes
        
        # Return paths of successful downloads
        return [r.local_path for r in results if r.success and r.local_path]
    
    def process_single(self, meca_path: Path) -> Optional[MECAExtractionResult]:
        """
        Process a single MECA file.
        
        Args:
            meca_path: Path to MECA file
            
        Returns:
            Extraction result or None if failed
        """
        try:
            result = self.extractor.extract(meca_path=meca_path)
            
            if not result.parse_success:
                logger.warning(f"Parse failed for {meca_path.name}: {result.extraction_errors}")
                return None
            
            # Save figures locally
            if result.figures:
                self._save_figures(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {meca_path.name}: {e}")
            return None
    
    def _save_figures(self, result: MECAExtractionResult):
        """Save extracted figures to disk"""
        if not result.figures:
            return
        
        # Create paper directory
        paper_dir = self.config.figure_dir / result.paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        for figure in result.figures:
            if figure.image_data:
                fig_path = paper_dir / f"{figure.image_id}.{figure.image_format}"
                fig_path.write_bytes(figure.image_data)
    
    def process_batch(
        self,
        meca_paths: List[Path],
        progress: bool = True
    ) -> List[MECAExtractionResult]:
        """
        Process a batch of MECA files.
        
        Args:
            meca_paths: List of MECA file paths
            progress: Show progress bar
            
        Returns:
            List of extraction results
        """
        results = []
        
        iterator = meca_paths
        if progress and HAS_TQDM:
            iterator = tqdm(meca_paths, desc="Processing")
        
        for meca_path in iterator:
            result = self.process_single(meca_path)
            
            if result:
                results.append(result)
                self.stats.processed += 1
                self.stats.total_figures += result.figure_count
                self.stats.total_tables += result.table_count
                self.stats.total_words += result.word_count
            else:
                self.stats.process_failed += 1
        
        return results
    
    def store_results(self, results: List[MECAExtractionResult]):
        """
        Store extraction results in Qdrant and Wasabi.
        
        Args:
            results: List of extraction results
        """
        if not results:
            return
        
        logger.info(f"Storing {len(results)} papers...")
        
        for result in tqdm(results, desc="Storing"):
            try:
                # Store in Qdrant
                if self.qdrant_client and self.config.store_to_qdrant:
                    self._store_to_qdrant(result)
                
                # Upload figures to Wasabi
                if self.wasabi_client and self.config.upload_to_wasabi:
                    self._upload_to_wasabi(result)
                
                self.stats.stored += 1
                
            except Exception as e:
                logger.error(f"Error storing {result.paper_id}: {e}")
    
    def _store_to_qdrant(self, result: MECAExtractionResult):
        """Store paper in Qdrant"""
        # This will be implemented based on your Qdrant schema
        # For now, just log
        logger.debug(f"Would store to Qdrant: {result.paper_id}")
        
        # Example structure (uncomment when Qdrant client is ready):
        # 
        # # Store abstract
        # if result.abstract and self.embedding_engine:
        #     abstract_vector = self.embedding_engine.encode(result.abstract)
        #     self.qdrant_client.store_abstract(
        #         paper_id=result.paper_id,
        #         abstract=result.abstract,
        #         vector=abstract_vector,
        #         metadata={
        #             "doi": result.doi,
        #             "title": result.title,
        #             "server": result.server,
        #             "category": result.category,
        #         }
        #     )
        # 
        # # Store text chunks
        # if result.full_text and self.embedding_engine:
        #     chunks = self._create_chunks(result.full_text)
        #     vectors = self.embedding_engine.encode(chunks)
        #     self.qdrant_client.store_chunks(
        #         paper_id=result.paper_id,
        #         chunks=chunks,
        #         vectors=vectors
        #     )
    
    def _upload_to_wasabi(self, result: MECAExtractionResult):
        """Upload figures to Wasabi"""
        # This will be implemented based on your Wasabi structure
        logger.debug(f"Would upload to Wasabi: {result.figure_count} figures")
    
    def run(
        self,
        download: bool = True,
        process: bool = True,
        store: bool = True
    ) -> PipelineStats:
        """
        Run the complete pipeline.
        
        Args:
            download: Download from S3
            process: Process MECA files
            store: Store in Qdrant/Wasabi
            
        Returns:
            Pipeline statistics
        """
        self.stats = PipelineStats()
        self.stats.start_time = datetime.now()
        
        logger.info(f"Starting pipeline for categories: {self.config.categories}")
        logger.info(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")
        
        # Get target papers
        papers = self.get_target_papers()
        self.stats.total_papers = len(papers)
        logger.info(f"Found {len(papers)} papers in target categories")
        
        if not papers:
            logger.warning("No papers to process")
            self.stats.end_time = datetime.now()
            return self.stats
        
        # Download
        if download:
            meca_paths = self.download_papers(papers)
        else:
            # Use existing files
            category_name = self.config.categories[0].replace(" ", "_").lower()
            meca_dir = self.config.download_dir / category_name
            meca_paths = list(meca_dir.glob("*.meca")) if meca_dir.exists() else []
            logger.info(f"Using {len(meca_paths)} existing MECA files")
        
        # Process
        if process and meca_paths:
            results = self.process_batch(meca_paths)
            
            # Store
            if store and results:
                self.store_results(results)
        
        self.stats.end_time = datetime.now()
        self.stats.print_summary()
        
        return self.stats
    
    def process_local_directory(
        self,
        directory: Path,
        store: bool = True
    ) -> PipelineStats:
        """
        Process already downloaded MECA files.
        
        Args:
            directory: Directory containing MECA files
            store: Store results in Qdrant/Wasabi
            
        Returns:
            Pipeline statistics
        """
        self.stats = PipelineStats()
        self.stats.start_time = datetime.now()
        
        # Find MECA files
        meca_paths = list(Path(directory).glob("**/*.meca"))
        self.stats.total_papers = len(meca_paths)
        
        logger.info(f"Processing {len(meca_paths)} MECA files from {directory}")
        
        if not meca_paths:
            logger.warning("No MECA files found")
            self.stats.end_time = datetime.now()
            return self.stats
        
        # Process
        results = self.process_batch(meca_paths)
        
        # Store
        if store and results:
            self.store_results(results)
        
        self.stats.end_time = datetime.now()
        self.stats.print_summary()
        
        return self.stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_pipeline(
    categories: List[str],
    max_papers: int = None,
    download: bool = True,
    process: bool = True,
    store: bool = False
) -> PipelineStats:
    """
    Run the ingestion pipeline for given categories.
    
    Args:
        categories: List of category names
        max_papers: Maximum papers to process (for testing)
        download: Download from S3
        process: Process MECA files
        store: Store in Qdrant/Wasabi
        
    Returns:
        Pipeline statistics
    """
    config = PipelineConfig(
        categories=categories,
        max_papers=max_papers,
        store_to_qdrant=store,
        upload_to_wasabi=store,
    )
    
    pipeline = CategoryFilteredPipeline(config)
    return pipeline.run(download=download, process=process, store=store)


def build_category_index(update_only: bool = False, days: int = 7):
    """Build or update the category index"""
    config = PipelineConfig()
    pipeline = CategoryFilteredPipeline(config)
    pipeline.build_index(update_only=update_only, days=days)


def process_directory(directory: str, store: bool = False) -> PipelineStats:
    """Process MECA files in a directory"""
    config = PipelineConfig(
        store_to_qdrant=store,
        upload_to_wasabi=store,
    )
    pipeline = CategoryFilteredPipeline(config)
    return pipeline.process_local_directory(Path(directory), store=store)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Category-Filtered Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build the category index (one-time, 2-3 hours)
    python pipeline.py --build-index
    
    # Update index with recent papers
    python pipeline.py --update-index --days 7
    
    # Download and process cancer biology papers
    python pipeline.py --categories "cancer biology" --run
    
    # Download multiple categories
    python pipeline.py --categories "cancer biology" "oncology" "pathology" --run
    
    # Test with limited papers
    python pipeline.py --categories "cancer biology" --max 100 --run
    
    # Download only (no processing)
    python pipeline.py --categories "cancer biology" --download-only
    
    # Process existing files
    python pipeline.py --process-local ./data/meca_files/cancer_biology
        """
    )
    
    # Index operations
    parser.add_argument("--build-index", action="store_true",
                        help="Build full category index (2-3 hours)")
    parser.add_argument("--update-index", action="store_true",
                        help="Update index with recent papers")
    parser.add_argument("--days", type=int, default=7,
                        help="Days to look back for update")
    parser.add_argument("--index-stats", action="store_true",
                        help="Print index statistics")
    
    # Pipeline operations
    parser.add_argument("--categories", nargs="+", type=str,
                        help="Categories to process")
    parser.add_argument("--run", action="store_true",
                        help="Run full pipeline (download + process + store)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download, don't process")
    parser.add_argument("--process-local", type=str,
                        help="Process MECA files in local directory")
    
    # Options
    parser.add_argument("--max", type=int,
                        help="Maximum papers to process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip image extraction")
    parser.add_argument("--store", action="store_true",
                        help="Store results in Qdrant/Wasabi")
    parser.add_argument("--output-dir", type=str,
                        help="Output directory for downloads")
    parser.add_argument("--save-report", type=str,
                        help="Save execution report to file")
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        categories=args.categories or [],
        max_papers=args.max,
        download_workers=args.workers,
        extract_images=not args.no_images,
        store_to_qdrant=args.store,
        upload_to_wasabi=args.store,
    )
    
    if args.output_dir:
        config.download_dir = Path(args.output_dir)
    
    # Execute
    if args.build_index:
        build_category_index(update_only=False)
    
    elif args.update_index:
        build_category_index(update_only=True, days=args.days)
    
    elif args.index_stats:
        pipeline = CategoryFilteredPipeline(config)
        pipeline.index_builder.print_stats()
    
    elif args.process_local:
        stats = process_directory(args.process_local, store=args.store)
        
        if args.save_report:
            with open(args.save_report, 'w') as f:
                json.dump(stats.to_dict(), f, indent=2)
    
    elif args.categories and (args.run or args.download_only):
        pipeline = CategoryFilteredPipeline(config)
        
        if args.download_only:
            pipeline.download_papers()
            print(f"\nDownloaded to: {config.download_dir}")
        else:
            stats = pipeline.run(store=args.store)
            
            if args.save_report:
                with open(args.save_report, 'w') as f:
                    json.dump(stats.to_dict(), f, indent=2)
    
    else:
        parser.print_help()
