#!/usr/bin/env python3
"""
Daily Update Script
===================
Runs as a cron job to ingest new papers from the last 24-48 hours.

Workflow:
    1. Update category index with recent papers (API)
    2. Download new MECA files for target categories (S3)
    3. Process and store new papers

Cron setup (run daily at 2 AM):
    0 2 * * * /path/to/venv/bin/python /path/to/daily_update.py >> /var/log/plagiarism_update.log 2>&1

Or using systemd timer for more reliability.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import CategoryFilteredPipeline, PipelineConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default categories to update
DEFAULT_CATEGORIES = [
    "cancer biology",
    "oncology",
    # Add more categories as needed
]

# Log configuration
LOG_FILE = Path(__file__).parent.parent / "logs" / "daily_update.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DAILY UPDATE
# =============================================================================

def run_daily_update(
    categories: list = None,
    days: int = 2,
    store: bool = True,
    dry_run: bool = False
):
    """
    Run daily update for specified categories.
    
    Args:
        categories: Categories to update (defaults to DEFAULT_CATEGORIES)
        days: Days to look back (default 2 for safety margin)
        store: Store results in Qdrant/Wasabi
        dry_run: Just show what would be done
    """
    categories = categories or DEFAULT_CATEGORIES
    
    logger.info("="*60)
    logger.info("DAILY UPDATE STARTED")
    logger.info(f"Categories: {categories}")
    logger.info(f"Looking back: {days} days")
    logger.info(f"Store results: {store}")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    try:
        # Configure pipeline
        config = PipelineConfig(
            categories=categories,
            store_to_qdrant=store,
            upload_to_wasabi=store,
            skip_existing=True,  # Important: skip already processed
        )
        
        pipeline = CategoryFilteredPipeline(config)
        
        # Step 1: Update category index
        logger.info("\n[Step 1/4] Updating category index...")
        if not dry_run:
            pipeline.index_builder.update_index(days=days)
        else:
            logger.info("  (dry run - skipping)")
        
        # Step 2: Get new papers
        logger.info("\n[Step 2/4] Finding new papers...")
        papers = pipeline.get_target_papers()
        
        # Filter to recent papers only
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent_papers = [
            p for p in papers
            if p.get("date", "") >= cutoff_date
        ]
        
        logger.info(f"  Total papers in categories: {len(papers)}")
        logger.info(f"  Recent papers (last {days} days): {len(recent_papers)}")
        
        if not recent_papers:
            logger.info("  No new papers to process")
            return
        
        if dry_run:
            logger.info(f"\n  Would process {len(recent_papers)} papers:")
            for p in recent_papers[:10]:
                logger.info(f"    - {p['doi']}: {p['title'][:50]}...")
            if len(recent_papers) > 10:
                logger.info(f"    ... and {len(recent_papers) - 10} more")
            return
        
        # Step 3: Download new papers
        logger.info("\n[Step 3/4] Downloading new MECA files...")
        # Override max_papers with just recent ones
        config.max_papers = len(recent_papers)
        meca_paths = pipeline.download_papers(recent_papers)
        logger.info(f"  Downloaded: {len(meca_paths)} files")
        
        # Step 4: Process and store
        logger.info("\n[Step 4/4] Processing and storing...")
        if meca_paths:
            results = pipeline.process_batch(meca_paths)
            logger.info(f"  Processed: {len(results)} papers")
            
            if store and results:
                pipeline.store_results(results)
                logger.info(f"  Stored: {len(results)} papers")
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("DAILY UPDATE COMPLETE")
        logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Papers processed: {pipeline.stats.processed}")
        logger.info(f"Figures extracted: {pipeline.stats.total_figures}")
        logger.info(f"Tables extracted: {pipeline.stats.total_tables}")
        logger.info("="*60)
        
        # Save report
        report_dir = Path(__file__).parent.parent / "logs" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "categories": categories,
            "days_back": days,
            "duration_seconds": duration,
            "stats": pipeline.stats.to_dict(),
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_file}")
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}", exc_info=True)
        raise


def check_index_exists():
    """Check if category index exists, build if not"""
    from ingestion.category_index import DEFAULT_INDEX_PATH
    
    if not DEFAULT_INDEX_PATH.exists():
        logger.warning("Category index not found!")
        logger.warning("Please build it first with: python pipeline.py --build-index")
        logger.warning("This is a one-time operation that takes 2-3 hours.")
        return False
    return True


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Ingestion Update")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help="Categories to update")
    parser.add_argument("--days", type=int, default=2,
                        help="Days to look back (default: 2)")
    parser.add_argument("--no-store", action="store_true",
                        help="Don't store results (just download/process)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    
    args = parser.parse_args()
    
    # Check index exists
    if not check_index_exists():
        sys.exit(1)
    
    run_daily_update(
        categories=args.categories,
        days=args.days,
        store=not args.no_store,
        dry_run=args.dry_run
    )
