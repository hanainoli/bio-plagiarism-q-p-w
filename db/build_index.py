#!/usr/bin/env python3
"""
UyarAI PostgreSQL Index Builder
================================
Build paper index in PostgreSQL from bioRxiv/medRxiv API.

Usage:
    python db/build_index.py --build          # Build full index
    python db/build_index.py --update 7       # Update last 7 days
    python db/build_index.py --stats          # Show statistics
    python db/build_index.py --category oncology  # Category stats
"""

import os
import time
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.models import (
    Paper, ProcessingStatus, BuildProgress,
    get_session, init_database, get_engine
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"
MEDRXIV_API = "https://api.biorxiv.org/details/medrxiv"

# Groups definition
GROUPS = {
    "core_molecular_biology": ["biochemistry", "molecular biology", "cell biology", "genetics", "genomics"],
    "cancer_oncology": ["cancer biology", "oncology", "hematology", "pathology"],
    "immunology_infectious": ["immunology", "allergy and immunology", "infectious diseases", "hiv/aids", "microbiology"],
    "neuroscience_psychiatry": ["neuroscience", "neurology", "psychiatry and clinical psychology", "pain medicine"],
    "cardiovascular_respiratory": ["cardiovascular medicine", "respiratory medicine", "physiology"],
    "computational_biology": ["bioinformatics", "systems biology", "synthetic biology", "bioengineering", "health informatics"],
    "genetics_genomic_medicine": ["genetic and genomic medicine", "developmental biology", "biophysics"],
    "internal_medicine": ["gastroenterology", "endocrinology", "nephrology", "rheumatology"],
    "surgery_specialties": ["surgery", "orthopedics", "ophthalmology", "otolaryngology", "urology", "dermatology", "transplantation", "dentistry and oral medicine"],
    "womens_childrens_health": ["obstetrics and gynecology", "pediatrics", "geriatric medicine"],
    "public_health": ["epidemiology", "public and global health", "occupational and environmental health", "nutrition", "clinical trials"],
    "pharmacology": ["pharmacology and toxicology", "pharmacology and therapeutics", "toxicology", "addiction medicine"],
    "emergency_critical_care": ["emergency medicine", "intensive care and critical care medicine", "anesthesia", "palliative medicine"],
    "health_systems": ["health economics", "health policy", "health systems and quality improvement", "medical education", "medical ethics", "primary care research", "nursing", "forensic medicine"],
    "ecology_evolution": ["ecology", "evolutionary biology", "animal behavior and cognition", "zoology", "plant biology", "paleontology", "scientific communication and education"],
    "other_specialties": ["radiology and imaging", "rehabilitation medicine and physical therapy", "sports medicine"],
}

# Category to group mapping
CATEGORY_TO_GROUP = {}
for group_id, categories in GROUPS.items():
    for cat in categories:
        CATEGORY_TO_GROUP[cat.lower()] = group_id


def get_group_for_category(category: str) -> str:
    return CATEGORY_TO_GROUP.get(category.lower(), "other_specialties")


# =============================================================================
# INDEX BUILDER
# =============================================================================

class PostgresIndexBuilder:
    """Build paper index in PostgreSQL"""
    
    def __init__(self):
        self.session = get_session()
    
    def _fetch_papers(self, server: str, start_date: str, end_date: str, cursor: int = 0) -> Dict:
        """Fetch papers from API"""
        base = BIORXIV_API if server == "biorxiv" else MEDRXIV_API
        url = f"{base}/{start_date}/{end_date}/{cursor}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"   ⚠️ API error: {e}")
            return {"collection": []}
    
    def _process_paper(self, paper: Dict, server: str) -> Dict:
        """Process API response to paper dict"""
        category = paper.get("category", "").lower()
        
        # Parse date
        date_str = paper.get("date", "")
        try:
            paper_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            paper_date = None
        
        return {
            "doi": paper.get("doi", ""),
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "abstract": paper.get("abstract", "")[:5000] if paper.get("abstract") else "",
            "category": category,
            "group_id": get_group_for_category(category),
            "server": server,
            "date": paper_date,
            "version": int(paper.get("version", 1) or 1),
            "license": paper.get("license", ""),
            "jatsxml": paper.get("jatsxml", ""),
        }
    
    def _get_progress(self, task: str = 'index_build') -> Optional[BuildProgress]:
        """Get build progress"""
        return self.session.query(BuildProgress).filter_by(task=task).first()
    
    def _save_progress(self, task: str, status: str, last_date: date, total_added: int):
        """Save build progress"""
        progress = self._get_progress(task)
        
        if progress:
            progress.status = status
            progress.last_date = last_date
            progress.total_added = total_added
            progress.updated_at = datetime.utcnow()
            if status == 'complete':
                progress.completed_at = datetime.utcnow()
        else:
            progress = BuildProgress(
                task=task,
                status=status,
                last_date=last_date,
                total_added=total_added,
                started_at=datetime.utcnow()
            )
            self.session.add(progress)
        
        self.session.commit()
    
    def _bulk_upsert(self, papers: List[Dict]):
        """Bulk upsert papers - simpler approach"""
        if not papers:
            return 0
        
        inserted = 0
        for paper in papers:
            try:
                # Check if exists
                existing = self.session.query(Paper).filter_by(doi=paper['doi']).first()
                
                if existing:
                    # Update
                    existing.title = paper.get('title', '')
                    existing.authors = paper.get('authors', '')
                    existing.abstract = paper.get('abstract', '')
                    existing.category = paper.get('category', '')
                    existing.group_id = paper.get('group_id', '')
                    existing.version = paper.get('version', 1)
                    existing.updated_at = datetime.utcnow()
                else:
                    # Insert new
                    new_paper = Paper(
                        doi=paper['doi'],
                        title=paper.get('title', ''),
                        authors=paper.get('authors', ''),
                        abstract=paper.get('abstract', ''),
                        category=paper.get('category', ''),
                        group_id=paper.get('group_id', ''),
                        server=paper.get('server', ''),
                        date=paper.get('date'),
                        version=paper.get('version', 1),
                        license=paper.get('license', ''),
                        jatsxml=paper.get('jatsxml', '')
                    )
                    self.session.add(new_paper)
                
                inserted += 1
                
            except Exception as e:
                print(f"   ⚠️ Error inserting {paper.get('doi', 'unknown')}: {e}")
                continue
        
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"   ⚠️ Commit error: {e}")
            return 0
        
        return inserted
    
    def build_full(self, start_year: int = 2013, resume: bool = True):
        """Build full index from bioRxiv/medRxiv API"""
        print("\n" + "="*60)
        print("🔨 BUILDING POSTGRESQL INDEX")
        print("="*60)
        
        # Check for resume
        progress = self._get_progress('index_build') if resume else None
        
        if progress and progress.status == 'in_progress':
            print(f"📂 Resuming from: {progress.last_date}")
            start_date = progress.last_date
            total_added = progress.total_added
        else:
            start_date = date(start_year, 1, 1)
            total_added = 0
            self._save_progress('index_build', 'in_progress', start_date, 0)
        
        today = date.today()
        current = start_date
        
        while current < today:
            # Month end
            if current.month == 12:
                month_end = date(current.year, 12, 31)
            else:
                month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)
            
            month_end = min(month_end, today)
            
            print(f"\n📅 {current.strftime('%Y-%m')}")
            
            for server in ["biorxiv", "medrxiv"]:
                cursor = 0
                count = 0
                batch = []
                
                while True:
                    data = self._fetch_papers(
                        server,
                        current.strftime("%Y-%m-%d"),
                        month_end.strftime("%Y-%m-%d"),
                        cursor
                    )
                    
                    papers = data.get("collection", [])
                    if not papers:
                        break
                    
                    for paper in papers:
                        doi = paper.get("doi")
                        if doi:
                            processed = self._process_paper(paper, server)
                            batch.append(processed)
                            count += 1
                    
                    # Bulk insert every 500 papers
                    if len(batch) >= 500:
                        self._bulk_upsert(batch)
                        batch = []
                    
                    messages = data.get("messages", [{}])
                    total = int(messages[0].get("total", 0)) if messages else 0
                    
                    cursor += len(papers)
                    if cursor >= total:
                        break
                    
                    time.sleep(0.3)
                
                # Insert remaining
                if batch:
                    self._bulk_upsert(batch)
                
                total_added += count
                
                if count > 0:
                    print(f"   {server}: +{count:,}")
            
            # Save progress
            self._save_progress('index_build', 'in_progress', month_end, total_added)
            
            # Next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        
        # Mark complete
        self._save_progress('index_build', 'complete', today, total_added)
        
        print(f"\n✅ Build complete! Total: {total_added:,} papers")
        
        return total_added
    
    def update(self, days: int = 7):
        """Update with recent papers"""
        print(f"\n📥 Updating index (last {days} days)...")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        print(f"   Date range: {start_date} to {end_date}")
        
        total_added = 0
        
        for server in ["biorxiv", "medrxiv"]:
            print(f"\n   📡 Fetching from {server}...")
            cursor = 0
            count = 0
            batch = []
            
            while True:
                print(f"      Fetching cursor={cursor}...", end=" ", flush=True)
                
                data = self._fetch_papers(
                    server,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    cursor
                )
                
                papers = data.get("collection", [])
                if not papers:
                    print("no more papers")
                    break
                
                messages = data.get("messages", [{}])
                total = int(messages[0].get("total", 0)) if messages else 0
                
                print(f"got {len(papers)} papers (total: {total})")
                
                for paper in papers:
                    doi = paper.get("doi")
                    if doi:
                        processed = self._process_paper(paper, server)
                        batch.append(processed)
                        count += 1
                
                if len(batch) >= 500:
                    print(f"      💾 Saving batch of {len(batch)} papers...")
                    self._bulk_upsert(batch)
                    batch = []
                
                cursor += len(papers)
                if cursor >= total:
                    break
                
                time.sleep(0.3)
            
            if batch:
                print(f"      💾 Saving final batch of {len(batch)} papers...")
                self._bulk_upsert(batch)
            
            total_added += count
            print(f"   ✅ {server}: +{count:,} papers")
        
        print(f"\n✅ Added/updated {total_added:,} papers total")
        
        return total_added
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        total = self.session.query(func.count(Paper.id)).scalar()
        
        by_category = dict(
            self.session.query(Paper.category, func.count(Paper.id))
            .group_by(Paper.category)
            .order_by(func.count(Paper.id).desc())
            .all()
        )
        
        by_group = dict(
            self.session.query(Paper.group_id, func.count(Paper.id))
            .group_by(Paper.group_id)
            .order_by(func.count(Paper.id).desc())
            .all()
        )
        
        by_server = dict(
            self.session.query(Paper.server, func.count(Paper.id))
            .group_by(Paper.server)
            .all()
        )
        
        # Processing stats
        downloaded = self.session.query(func.count(Paper.id)).filter(Paper.pdf_downloaded == True).scalar()
        extracted = self.session.query(func.count(Paper.id)).filter(Paper.content_extracted == True).scalar()
        embedded = self.session.query(func.count(Paper.id)).filter(Paper.text_embedded == True).scalar()
        
        return {
            "total_papers": total,
            "total_categories": len(by_category),
            "total_groups": len(by_group),
            "by_category": by_category,
            "by_group": by_group,
            "by_server": by_server,
            "processing": {
                "downloaded": downloaded,
                "extracted": extracted,
                "embedded": embedded
            }
        }
    
    def get_category_papers(self, category: str, limit: int = None) -> List[Paper]:
        """Get papers for a category"""
        query = self.session.query(Paper).filter(
            func.lower(Paper.category) == category.lower()
        )
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_group_papers(self, group_id: str, limit: int = None) -> List[Paper]:
        """Get papers for a group"""
        query = self.session.query(Paper).filter(Paper.group_id == group_id)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_unprocessed(self, step: str, category: str = None, limit: int = 100) -> List[Paper]:
        """Get papers that need processing"""
        query = self.session.query(Paper)
        
        if step == 'download':
            query = query.filter(Paper.pdf_downloaded == False)
        elif step == 'extract':
            query = query.filter(Paper.pdf_downloaded == True, Paper.content_extracted == False)
        elif step == 'embed':
            query = query.filter(Paper.content_extracted == True, Paper.text_embedded == False)
        
        if category:
            query = query.filter(func.lower(Paper.category) == category.lower())
        
        return query.limit(limit).all()
    
    def mark_processed(self, doi: str, step: str, **kwargs):
        """Mark paper as processed"""
        paper = self.session.query(Paper).filter_by(doi=doi).first()
        if paper:
            if step == 'download':
                paper.pdf_downloaded = True
            elif step == 'extract':
                paper.content_extracted = True
            elif step == 'embed':
                paper.text_embedded = True
                paper.figures_embedded = kwargs.get('figures', False)
                paper.tables_embedded = kwargs.get('tables', False)
            
            self.session.commit()
    
    def close(self):
        """Close session"""
        self.session.close()


# =============================================================================
# CLI
# =============================================================================

def print_stats(builder: PostgresIndexBuilder):
    """Print statistics"""
    stats = builder.get_stats()
    
    print(f"\n{'='*60}")
    print("📊 POSTGRESQL INDEX STATISTICS")
    print(f"{'='*60}")
    print(f"Total Papers:     {stats['total_papers']:,}")
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Total Groups:     {stats['total_groups']}")
    
    print(f"\n📦 By Server:")
    for server, count in stats['by_server'].items():
        print(f"   {server}: {count:,}")
    
    print(f"\n📁 By Group:")
    for group_id, count in stats['by_group'].items():
        print(f"   {group_id}: {count:,}")
    
    print(f"\n📂 Top 15 Categories:")
    for cat, count in list(stats['by_category'].items())[:15]:
        print(f"   {cat}: {count:,}")
    
    print(f"\n⚙️ Processing Status:")
    proc = stats['processing']
    print(f"   Downloaded: {proc['downloaded']:,}")
    print(f"   Extracted:  {proc['extracted']:,}")
    print(f"   Embedded:   {proc['embedded']:,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL Index Builder")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--build", action="store_true", help="Build full index")
    parser.add_argument("--update", type=int, metavar="DAYS", help="Update recent papers")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--category", type=str, help="Category stats")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume, start fresh")
    parser.add_argument("--start-year", type=int, default=2013, help="Start year")
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    
    elif args.build:
        init_database()  # Ensure tables exist
        builder = PostgresIndexBuilder()
        builder.build_full(start_year=args.start_year, resume=not args.no_resume)
        print_stats(builder)
        builder.close()
    
    elif args.update:
        builder = PostgresIndexBuilder()
        builder.update(days=args.update)
        print_stats(builder)
        builder.close()
    
    elif args.stats:
        builder = PostgresIndexBuilder()
        print_stats(builder)
        builder.close()
    
    elif args.category:
        builder = PostgresIndexBuilder()
        papers = builder.get_category_papers(args.category, limit=10)
        print(f"\n📂 {args.category}: {len(papers)} papers (showing 10)")
        for p in papers[:10]:
            print(f"   - {p.doi}: {p.title[:60]}...")
        builder.close()
    
    else:
        parser.print_help()
