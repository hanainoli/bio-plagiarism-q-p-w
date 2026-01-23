"""
bioRxiv/medRxiv Category Index Builder
======================================
Builds and maintains a local index of papers organized by category.
This enables filtered downloads from S3 (only papers you want).

Usage:
    # Build index (one-time, takes 2-3 hours)
    python category_index.py --build
    
    # Update index with recent papers
    python category_index.py --update --days 7
    
    # Get stats
    python category_index.py --stats
    
    # Get papers for a category
    python category_index.py --category "cancer biology" --output cancer_dois.json
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# bioRxiv API endpoints
BIORXIV_API_BASE = "https://api.biorxiv.org"
BIORXIV_DETAILS_URL = f"{BIORXIV_API_BASE}/details/biorxiv"
MEDRXIV_DETAILS_URL = f"{BIORXIV_API_BASE}/details/medrxiv"

# Rate limiting
API_RATE_LIMIT_DELAY = 0.35  # seconds between requests (~3 req/sec)
API_BATCH_SIZE = 100  # papers per request (API maximum)

# Default paths
DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "data" / "biorxiv_index.json"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# All known categories in bioRxiv/medRxiv
BIORXIV_CATEGORIES = [
    "animal behavior and cognition",
    "biochemistry",
    "bioengineering",
    "bioinformatics",
    "biophysics",
    "cancer biology",
    "cell biology",
    "clinical trials",
    "developmental biology",
    "ecology",
    "epidemiology",
    "evolutionary biology",
    "genetics",
    "genomics",
    "immunology",
    "microbiology",
    "molecular biology",
    "neuroscience",
    "paleontology",
    "pathology",
    "pharmacology and toxicology",
    "physiology",
    "plant biology",
    "scientific communication and education",
    "synthetic biology",
    "systems biology",
    "zoology",
]

MEDRXIV_CATEGORIES = [
    "addiction medicine",
    "allergy and immunology",
    "anesthesia",
    "cardiovascular medicine",
    "dentistry and oral medicine",
    "dermatology",
    "emergency medicine",
    "endocrinology",
    "epidemiology",
    "forensic medicine",
    "gastroenterology",
    "genetic and genomic medicine",
    "geriatric medicine",
    "health economics",
    "health informatics",
    "health policy",
    "health systems and quality improvement",
    "hematology",
    "hiv/aids",
    "infectious diseases",
    "intensive care and critical care medicine",
    "medical education",
    "medical ethics",
    "nephrology",
    "neurology",
    "nursing",
    "nutrition",
    "obstetrics and gynecology",
    "occupational and environmental health",
    "oncology",
    "ophthalmology",
    "orthopedics",
    "otolaryngology",
    "pain medicine",
    "palliative medicine",
    "pathology",
    "pediatrics",
    "pharmacology and therapeutics",
    "primary care research",
    "psychiatry and clinical psychology",
    "public and global health",
    "radiology and imaging",
    "rehabilitation medicine and physical therapy",
    "respiratory medicine",
    "rheumatology",
    "sexual and reproductive health",
    "sports medicine",
    "surgery",
    "toxicology",
    "transplantation",
    "urology",
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PaperMetadata:
    """Minimal paper metadata for indexing"""
    doi: str
    date: str
    title: str
    server: str  # "biorxiv" or "medrxiv"
    category: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PaperMetadata':
        return cls(**d)


@dataclass
class CategoryIndex:
    """Index of papers organized by category"""
    categories: Dict[str, List[dict]] = field(default_factory=dict)
    last_updated: str = ""
    total_papers: int = 0
    servers: Dict[str, int] = field(default_factory=dict)  # server -> count
    
    def add_paper(self, paper: PaperMetadata):
        """Add a paper to the index"""
        category = paper.category.lower()
        if category not in self.categories:
            self.categories[category] = []
        
        # Check for duplicates
        existing_dois = {p["doi"] for p in self.categories[category]}
        if paper.doi not in existing_dois:
            self.categories[category].append(paper.to_dict())
            self.total_papers += 1
            
            # Track server counts
            server = paper.server
            self.servers[server] = self.servers.get(server, 0) + 1
    
    def get_papers(self, category: str, fuzzy: bool = True) -> List[dict]:
        """Get papers for a category"""
        category_lower = category.lower()
        
        if fuzzy:
            # Fuzzy match - find categories containing the search term
            results = []
            for cat, papers in self.categories.items():
                if category_lower in cat.lower():
                    results.extend(papers)
            return results
        else:
            return self.categories.get(category_lower, [])
    
    def get_dois(self, category: str) -> List[str]:
        """Get just DOIs for a category"""
        papers = self.get_papers(category)
        return [p["doi"] for p in papers]
    
    def get_stats(self) -> Dict[str, any]:
        """Get index statistics"""
        return {
            "total_papers": self.total_papers,
            "total_categories": len(self.categories),
            "last_updated": self.last_updated,
            "servers": self.servers,
            "papers_by_category": {
                cat: len(papers) 
                for cat, papers in sorted(
                    self.categories.items(), 
                    key=lambda x: -len(x[1])
                )
            }
        }
    
    def save(self, path: Path, sync_to_wasabi: bool = True):
        """Save index to JSON file and optionally sync to Wasabi"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "categories": self.categories,
            "last_updated": self.last_updated,
            "total_papers": self.total_papers,
            "servers": self.servers,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved index to {path} ({self.total_papers} papers)")
        
        # Sync to Wasabi for backup
        if sync_to_wasabi:
            try:
                from storage.wasabi_client import WasabiClient
                wasabi = WasabiClient()
                if wasabi.client:
                    wasabi_key = "index/biorxiv_index.json"
                    wasabi.client.upload_file(
                        str(path),
                        wasabi.bucket,
                        wasabi_key
                    )
                    logger.info(f"✓ Synced index to Wasabi: s3://{wasabi.bucket}/{wasabi_key}")
            except Exception as e:
                logger.warning(f"Could not sync to Wasabi: {e}")
    
    @classmethod
    def load(cls, path: Path, try_wasabi: bool = True) -> 'CategoryIndex':
        """Load index from JSON file, with Wasabi fallback"""
        # Try local file first
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            
            index = cls(
                categories=data.get("categories", {}),
                last_updated=data.get("last_updated", ""),
                total_papers=data.get("total_papers", 0),
                servers=data.get("servers", {}),
            )
            
            logger.info(f"Loaded index from {path} ({index.total_papers} papers)")
            return index
        
        # Try Wasabi if local doesn't exist
        if try_wasabi:
            try:
                from storage.wasabi_client import WasabiClient
                wasabi = WasabiClient()
                if wasabi.client:
                    wasabi_key = "index/biorxiv_index.json"
                    logger.info(f"Local index not found, trying Wasabi: s3://{wasabi.bucket}/{wasabi_key}")
                    
                    # Download from Wasabi
                    path.parent.mkdir(parents=True, exist_ok=True)
                    wasabi.client.download_file(
                        wasabi.bucket,
                        wasabi_key,
                        str(path)
                    )
                    logger.info(f"✓ Downloaded index from Wasabi to {path}")
                    
                    # Now load it
                    with open(path) as f:
                        data = json.load(f)
                    
                    index = cls(
                        categories=data.get("categories", {}),
                        last_updated=data.get("last_updated", ""),
                        total_papers=data.get("total_papers", 0),
                        servers=data.get("servers", {}),
                    )
                    
                    logger.info(f"Loaded index from Wasabi ({index.total_papers} papers)")
                    return index
            except Exception as e:
                logger.warning(f"Could not load from Wasabi: {e}")
        
        logger.warning(f"Index file not found: {path}")
        return cls()


# =============================================================================
# INDEX BUILDER
# =============================================================================

class CategoryIndexBuilder:
    """
    Builds and maintains the category index by fetching from bioRxiv/medRxiv API.
    
    The API returns paper metadata including category, which we use to build
    a local index for filtered S3 downloads.
    """
    
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH):
        self.index_path = index_path
        self.index = CategoryIndex.load(index_path) if index_path.exists() else CategoryIndex()
        self._existing_dois: Set[str] = set()
        self._load_existing_dois()
    
    def _load_existing_dois(self):
        """Load existing DOIs to avoid duplicates"""
        for papers in self.index.categories.values():
            for paper in papers:
                self._existing_dois.add(paper["doi"])
        logger.info(f"Loaded {len(self._existing_dois)} existing DOIs")
    
    def _fetch_page(self, server: str, start_date: str, end_date: str, cursor: int) -> dict:
        """Fetch a single page of results from API"""
        if server == "biorxiv":
            url = f"{BIORXIV_DETAILS_URL}/{start_date}/{end_date}/{cursor}"
        else:
            url = f"{MEDRXIV_DETAILS_URL}/{start_date}/{end_date}/{cursor}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {"collection": [], "messages": [{"total": 0}]}
    
    def build_full_index(
        self,
        servers: List[str] = ["biorxiv", "medrxiv"],
        start_date: str = "2013-01-01",
        end_date: str = None,
        save_interval: int = 10000
    ):
        """
        Build complete index from scratch.
        
        Args:
            servers: Which servers to index
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (defaults to today)
            save_interval: Save progress every N papers
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Building full index from {start_date} to {end_date}")
        logger.info(f"Servers: {servers}")
        logger.info("This will take 2-3 hours due to API rate limits...")
        
        for server in servers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Indexing {server}...")
            logger.info(f"{'='*60}")
            
            cursor = 0
            server_count = 0
            
            while True:
                # Fetch page
                data = self._fetch_page(server, start_date, end_date, cursor)
                collection = data.get("collection", [])
                
                if not collection:
                    break
                
                # Process papers
                for paper_data in collection:
                    doi = paper_data.get("doi", "")
                    
                    # Skip if already indexed
                    if doi in self._existing_dois:
                        continue
                    
                    # Create paper metadata
                    paper = PaperMetadata(
                        doi=doi,
                        date=paper_data.get("date", ""),
                        title=paper_data.get("title", "")[:200],  # Truncate
                        server=server,
                        category=paper_data.get("category", "unknown").lower()
                    )
                    
                    # Add to index
                    self.index.add_paper(paper)
                    self._existing_dois.add(doi)
                    server_count += 1
                
                # Progress update
                cursor += API_BATCH_SIZE
                total_str = data.get("messages", [{}])[0].get("total", "0")
                total = int(total_str) if isinstance(total_str, str) else total_str
                
                if cursor % 1000 == 0:
                    logger.info(f"   {server}: {cursor}/{total} processed, {server_count} new papers")
                
                # Save periodically
                if self.index.total_papers % save_interval == 0:
                    self.index.last_updated = datetime.now().isoformat()
                    self.index.save(self.index_path)
                
                # Check if done
                if cursor >= total:
                    break
                
                # Rate limit
                time.sleep(API_RATE_LIMIT_DELAY)
            
            logger.info(f"   {server} complete: {server_count} papers indexed")
        
        # Final save
        self.index.last_updated = datetime.now().isoformat()
        self.index.save(self.index_path)
        
        logger.info(f"\n{'='*60}")
        logger.info("INDEX BUILD COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total papers: {self.index.total_papers}")
        logger.info(f"Categories: {len(self.index.categories)}")
    
    def update_index(
        self,
        servers: List[str] = ["biorxiv", "medrxiv"],
        days: int = 7
    ):
        """
        Update index with recent papers.
        
        Args:
            servers: Which servers to update
            days: How many days back to check
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        logger.info(f"Updating index for last {days} days ({start_date} to {end_date})")
        
        new_papers = 0
        
        for server in servers:
            cursor = 0
            
            while True:
                data = self._fetch_page(server, start_date, end_date, cursor)
                collection = data.get("collection", [])
                
                if not collection:
                    break
                
                for paper_data in collection:
                    doi = paper_data.get("doi", "")
                    
                    if doi in self._existing_dois:
                        continue
                    
                    paper = PaperMetadata(
                        doi=doi,
                        date=paper_data.get("date", ""),
                        title=paper_data.get("title", "")[:200],
                        server=server,
                        category=paper_data.get("category", "unknown").lower()
                    )
                    
                    self.index.add_paper(paper)
                    self._existing_dois.add(doi)
                    new_papers += 1
                
                cursor += API_BATCH_SIZE
                total_str = data.get("messages", [{}])[0].get("total", "0")
                total = int(total_str) if isinstance(total_str, str) else total_str
                
                if cursor >= total:
                    break
                
                time.sleep(API_RATE_LIMIT_DELAY)
        
        # Save
        self.index.last_updated = datetime.now().isoformat()
        self.index.save(self.index_path)
        
        logger.info(f"Update complete: {new_papers} new papers added")
        return new_papers
    
    def get_papers_by_category(
        self,
        categories: List[str],
        server: str = None
    ) -> List[dict]:
        """
        Get papers matching given categories.
        
        Args:
            categories: List of category names (fuzzy matched)
            server: Optional server filter ("biorxiv" or "medrxiv")
            
        Returns:
            List of paper metadata dicts
        """
        results = []
        seen_dois = set()
        
        for category in categories:
            papers = self.index.get_papers(category, fuzzy=True)
            for paper in papers:
                # Apply server filter
                if server and paper.get("server") != server:
                    continue
                
                # Deduplicate
                if paper["doi"] not in seen_dois:
                    results.append(paper)
                    seen_dois.add(paper["doi"])
        
        logger.info(f"Found {len(results)} papers matching {categories}")
        return results
    
    def export_dois(
        self,
        categories: List[str],
        output_path: Path,
        server: str = None
    ):
        """Export DOIs for given categories to a file"""
        papers = self.get_papers_by_category(categories, server)
        dois = [p["doi"] for p in papers]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dois, f, indent=2)
        
        logger.info(f"Exported {len(dois)} DOIs to {output_path}")
        return dois
    
    def print_stats(self):
        """Print index statistics"""
        stats = self.index.get_stats()
        
        print("\n" + "="*60)
        print("BIORXIV/MEDRXIV CATEGORY INDEX")
        print("="*60)
        print(f"Total Papers: {stats['total_papers']:,}")
        print(f"Categories: {stats['total_categories']}")
        print(f"Last Updated: {stats['last_updated']}")
        print(f"\nBy Server:")
        for server, count in stats['servers'].items():
            print(f"  {server}: {count:,}")
        print(f"\nBy Category (top 20):")
        for i, (cat, count) in enumerate(stats['papers_by_category'].items()):
            if i >= 20:
                print(f"  ... and {len(stats['papers_by_category']) - 20} more")
                break
            print(f"  {cat}: {count:,}")
        print("="*60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_index(index_path: Path = DEFAULT_INDEX_PATH) -> CategoryIndex:
    """Build complete category index (one-time operation)"""
    builder = CategoryIndexBuilder(index_path)
    builder.build_full_index()
    return builder.index


def update_index(days: int = 7, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """Update index with recent papers"""
    builder = CategoryIndexBuilder(index_path)
    return builder.update_index(days=days)


def get_category_dois(
    categories: List[str],
    index_path: Path = DEFAULT_INDEX_PATH
) -> List[str]:
    """Get DOIs for given categories"""
    builder = CategoryIndexBuilder(index_path)
    papers = builder.get_papers_by_category(categories)
    return [p["doi"] for p in papers]


def get_index_stats(index_path: Path = DEFAULT_INDEX_PATH) -> dict:
    """Get index statistics"""
    index = CategoryIndex.load(index_path)
    return index.get_stats()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="bioRxiv/medRxiv Category Index")
    parser.add_argument("--build", action="store_true", help="Build full index (2-3 hours)")
    parser.add_argument("--update", action="store_true", help="Update with recent papers")
    parser.add_argument("--days", type=int, default=7, help="Days to look back for update")
    parser.add_argument("--stats", action="store_true", help="Print index statistics")
    parser.add_argument("--category", type=str, help="Get papers for category")
    parser.add_argument("--output", type=str, help="Output file for DOIs")
    parser.add_argument("--index", type=str, default=str(DEFAULT_INDEX_PATH), help="Index file path")
    parser.add_argument("--sync", action="store_true", help="Sync local index to Wasabi")
    parser.add_argument("--download", action="store_true", help="Download index from Wasabi")
    
    args = parser.parse_args()
    index_path = Path(args.index)
    
    if args.build:
        build_index(index_path)
    
    elif args.update:
        count = update_index(days=args.days, index_path=index_path)
        print(f"Added {count} new papers")
    
    elif args.sync:
        # Manually sync local index to Wasabi
        if not index_path.exists():
            print(f"❌ Index not found: {index_path}")
        else:
            index = CategoryIndex.load(index_path, try_wasabi=False)
            index.save(index_path, sync_to_wasabi=True)
            print(f"✅ Synced {index.total_papers} papers to Wasabi")
    
    elif args.download:
        # Force download from Wasabi
        if index_path.exists():
            backup_path = index_path.with_suffix('.json.bak')
            import shutil
            shutil.copy(index_path, backup_path)
            print(f"📦 Backed up local index to {backup_path}")
            index_path.unlink()
        
        index = CategoryIndex.load(index_path, try_wasabi=True)
        if index.total_papers > 0:
            print(f"✅ Downloaded {index.total_papers} papers from Wasabi")
        else:
            print("❌ No index found in Wasabi")
    
    elif args.stats:
        builder = CategoryIndexBuilder(index_path)
        builder.print_stats()
    
    elif args.category:
        builder = CategoryIndexBuilder(index_path)
        papers = builder.get_papers_by_category([args.category])
        
        if args.output:
            builder.export_dois([args.category], Path(args.output))
        else:
            print(f"\nFound {len(papers)} papers in '{args.category}'")
            print("Sample DOIs:")
            for p in papers[:10]:
                print(f"  {p['doi']} - {p['title'][:60]}...")
    
    else:
        parser.print_help()
