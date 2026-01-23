"""
Progress Tracker
================
Track paper processing progress in PostgreSQL.

Usage:
    from db.progress import ProgressTracker
    
    tracker = ProgressTracker()
    
    # Check what needs processing
    papers = tracker.get_papers_to_download(category="oncology", limit=100)
    papers = tracker.get_papers_to_extract(category="oncology", limit=100)
    papers = tracker.get_papers_to_embed(category="oncology", limit=100)
    
    # Update status after processing
    tracker.mark_downloaded(doi, pdf_path, pdf_size)
    tracker.mark_extracted(doi, text_chunks=20, figures=5, tables=3)
    tracker.mark_embedded(doi, text_vectors=20, figure_vectors=5, table_vectors=3)
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import text, and_, or_
from sqlalchemy.orm import Session

from .models import get_session, Paper, ProcessingStatus

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track processing progress in PostgreSQL"""
    
    def __init__(self, session: Session = None):
        """Initialize with optional session"""
        self._session = session
        self._own_session = False
    
    @property
    def session(self) -> Session:
        """Get or create session"""
        if self._session is None:
            self._session = get_session()
            self._own_session = True
        return self._session
    
    def close(self):
        """Close session if we own it"""
        if self._own_session and self._session:
            self._session.close()
            self._session = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # GET PAPERS TO PROCESS
    # =========================================================================
    
    def get_papers_to_download(
        self,
        category: str = None,
        group_id: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get papers that need PDF download"""
        query = self.session.query(Paper).filter(Paper.pdf_downloaded == False)
        
        if category:
            query = query.filter(Paper.category == category)
        if group_id:
            query = query.filter(Paper.group_id == group_id)
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        return [{'doi': p.doi, 'title': p.title, 'category': p.category} for p in papers]
    
    def get_papers_to_extract(
        self,
        category: str = None,
        group_id: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get papers that have PDF but not extracted"""
        query = self.session.query(Paper).filter(
            and_(
                Paper.pdf_downloaded == True,
                Paper.content_extracted == False
            )
        )
        
        if category:
            query = query.filter(Paper.category == category)
        if group_id:
            query = query.filter(Paper.group_id == group_id)
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        return [{'doi': p.doi, 'title': p.title, 'category': p.category} for p in papers]
    
    def get_papers_to_embed(
        self,
        category: str = None,
        group_id: str = None,
        limit: int = 100,
        embed_type: str = 'all'  # 'all', 'text', 'figures', 'tables'
    ) -> List[Dict]:
        """
        Get papers that have content extracted but not embedded.
        
        Args:
            embed_type: What to embed - 'all', 'text', 'figures', 'tables'
        """
        base_filter = Paper.content_extracted == True
        
        if embed_type == 'all':
            # Papers missing ANY embedding
            embed_filter = or_(
                Paper.text_embedded == False,
                Paper.figures_embedded == False,
                Paper.tables_embedded == False
            )
        elif embed_type == 'text':
            embed_filter = Paper.text_embedded == False
        elif embed_type == 'figures':
            embed_filter = Paper.figures_embedded == False
        elif embed_type == 'tables':
            embed_filter = Paper.tables_embedded == False
        else:
            embed_filter = Paper.text_embedded == False
        
        query = self.session.query(Paper).filter(and_(base_filter, embed_filter))
        
        if category:
            query = query.filter(Paper.category == category)
        if group_id:
            query = query.filter(Paper.group_id == group_id)
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        return [{
            'doi': p.doi,
            'title': p.title,
            'category': p.category,
            'text_embedded': p.text_embedded,
            'figures_embedded': p.figures_embedded,
            'tables_embedded': p.tables_embedded
        } for p in papers]
    
    def get_papers_needing_figures(
        self,
        category: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get papers with text embedded but missing figure embeddings"""
        query = self.session.query(Paper).filter(
            and_(
                Paper.content_extracted == True,
                Paper.text_embedded == True,
                Paper.figures_embedded == False
            )
        )
        
        if category:
            query = query.filter(Paper.category == category)
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        return [{'doi': p.doi, 'title': p.title, 'category': p.category} for p in papers]
    
    def get_papers_needing_tables(
        self,
        category: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get papers with text embedded but missing table embeddings"""
        query = self.session.query(Paper).filter(
            and_(
                Paper.content_extracted == True,
                Paper.text_embedded == True,
                Paper.tables_embedded == False
            )
        )
        
        if category:
            query = query.filter(Paper.category == category)
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        return [{'doi': p.doi, 'title': p.title, 'category': p.category} for p in papers]
    
    # =========================================================================
    # CHECK STATUS
    # =========================================================================
    
    def is_downloaded(self, doi: str) -> bool:
        """Check if paper PDF is downloaded"""
        paper = self.session.query(Paper).filter(Paper.doi == doi).first()
        return paper.pdf_downloaded if paper else False
    
    def is_extracted(self, doi: str) -> bool:
        """Check if paper content is extracted"""
        paper = self.session.query(Paper).filter(Paper.doi == doi).first()
        return paper.content_extracted if paper else False
    
    def is_text_embedded(self, doi: str) -> bool:
        """Check if paper text is embedded"""
        paper = self.session.query(Paper).filter(Paper.doi == doi).first()
        return paper.text_embedded if paper else False
    
    def is_fully_embedded(self, doi: str) -> bool:
        """Check if paper is fully embedded (text + figures + tables)"""
        paper = self.session.query(Paper).filter(Paper.doi == doi).first()
        if not paper:
            return False
        return paper.text_embedded and paper.figures_embedded and paper.tables_embedded
    
    def get_status(self, doi: str) -> Optional[Dict]:
        """Get full processing status for a paper"""
        paper = self.session.query(Paper).filter(Paper.doi == doi).first()
        if not paper:
            return None
        
        status = {
            'doi': paper.doi,
            'title': paper.title,
            'pdf_downloaded': paper.pdf_downloaded,
            'content_extracted': paper.content_extracted,
            'text_embedded': paper.text_embedded,
            'figures_embedded': paper.figures_embedded,
            'tables_embedded': paper.tables_embedded,
        }
        
        # Get detailed processing info if available
        if paper.processing:
            ps = paper.processing
            status.update({
                'pdf_path': ps.pdf_path,
                'pdf_size': ps.pdf_size,
                'text_chunks': ps.text_chunks,
                'figures_count': ps.figures_count,
                'tables_count': ps.tables_count,
                'text_vectors': ps.text_vectors,
                'figure_vectors': ps.figure_vectors,
                'table_vectors': ps.table_vectors,
                'downloaded_at': ps.downloaded_at,
                'extracted_at': ps.extracted_at,
                'embedded_at': ps.embedded_at,
                'last_error': ps.last_error
            })
        
        return status
    
    # =========================================================================
    # UPDATE STATUS
    # =========================================================================
    
    def mark_downloaded(
        self,
        doi: str,
        pdf_path: str = None,
        pdf_size: int = None
    ) -> bool:
        """Mark paper as downloaded"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                logger.warning(f"Paper not found: {doi}")
                return False
            
            paper.pdf_downloaded = True
            paper.updated_at = datetime.utcnow()
            
            # Update or create processing status
            if not paper.processing:
                ps = ProcessingStatus(paper_id=paper.id, doi=doi)
                self.session.add(ps)
                paper.processing = ps
            
            paper.processing.pdf_path = pdf_path
            paper.processing.pdf_size = pdf_size
            paper.processing.downloaded_at = datetime.utcnow()
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark downloaded {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_extracted(
        self,
        doi: str,
        text_chunks: int = 0,
        figures_count: int = 0,
        tables_count: int = 0
    ) -> bool:
        """Mark paper as extracted"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                logger.warning(f"Paper not found: {doi}")
                return False
            
            paper.content_extracted = True
            paper.updated_at = datetime.utcnow()
            
            # Update processing status
            if not paper.processing:
                ps = ProcessingStatus(paper_id=paper.id, doi=doi)
                self.session.add(ps)
                paper.processing = ps
            
            paper.processing.text_chunks = text_chunks
            paper.processing.figures_count = figures_count
            paper.processing.tables_count = tables_count
            paper.processing.extracted_at = datetime.utcnow()
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark extracted {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_text_embedded(
        self,
        doi: str,
        text_vectors: int = 0
    ) -> bool:
        """Mark paper text as embedded"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                return False
            
            paper.text_embedded = True
            paper.updated_at = datetime.utcnow()
            
            if paper.processing:
                paper.processing.text_vectors = text_vectors
                paper.processing.embedded_at = datetime.utcnow()
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark text embedded {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_figures_embedded(
        self,
        doi: str,
        figure_vectors: int = 0
    ) -> bool:
        """Mark paper figures as embedded"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                return False
            
            paper.figures_embedded = True
            paper.updated_at = datetime.utcnow()
            
            if paper.processing:
                paper.processing.figure_vectors = figure_vectors
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark figures embedded {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_tables_embedded(
        self,
        doi: str,
        table_vectors: int = 0
    ) -> bool:
        """Mark paper tables as embedded"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                return False
            
            paper.tables_embedded = True
            paper.updated_at = datetime.utcnow()
            
            if paper.processing:
                paper.processing.table_vectors = table_vectors
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark tables embedded {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_fully_embedded(
        self,
        doi: str,
        text_vectors: int = 0,
        figure_vectors: int = 0,
        table_vectors: int = 0
    ) -> bool:
        """Mark paper as fully embedded (text + figures + tables)"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                return False
            
            paper.text_embedded = True
            paper.figures_embedded = True
            paper.tables_embedded = True
            paper.updated_at = datetime.utcnow()
            
            if not paper.processing:
                ps = ProcessingStatus(paper_id=paper.id, doi=doi)
                self.session.add(ps)
                paper.processing = ps
            
            paper.processing.text_vectors = text_vectors
            paper.processing.figure_vectors = figure_vectors
            paper.processing.table_vectors = table_vectors
            paper.processing.embedded_at = datetime.utcnow()
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark fully embedded {doi}: {e}")
            self.session.rollback()
            return False
    
    def mark_error(self, doi: str, error: str) -> bool:
        """Record an error for a paper"""
        try:
            paper = self.session.query(Paper).filter(Paper.doi == doi).first()
            if not paper:
                return False
            
            if not paper.processing:
                ps = ProcessingStatus(paper_id=paper.id, doi=doi)
                self.session.add(ps)
                paper.processing = ps
            
            paper.processing.last_error = error
            paper.processing.error_count = (paper.processing.error_count or 0) + 1
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark error {doi}: {e}")
            self.session.rollback()
            return False
    
    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================
    
    def mark_downloaded_batch(self, dois: List[str]) -> int:
        """Mark multiple papers as downloaded"""
        count = 0
        try:
            self.session.query(Paper).filter(Paper.doi.in_(dois)).update(
                {Paper.pdf_downloaded: True, Paper.updated_at: datetime.utcnow()},
                synchronize_session=False
            )
            self.session.commit()
            count = len(dois)
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            self.session.rollback()
        return count
    
    def mark_extracted_batch(self, dois: List[str]) -> int:
        """Mark multiple papers as extracted"""
        count = 0
        try:
            self.session.query(Paper).filter(Paper.doi.in_(dois)).update(
                {Paper.content_extracted: True, Paper.updated_at: datetime.utcnow()},
                synchronize_session=False
            )
            self.session.commit()
            count = len(dois)
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            self.session.rollback()
        return count
    
    def mark_embedded_batch(self, dois: List[str]) -> int:
        """Mark multiple papers as fully embedded"""
        count = 0
        try:
            self.session.query(Paper).filter(Paper.doi.in_(dois)).update(
                {
                    Paper.text_embedded: True,
                    Paper.figures_embedded: True,
                    Paper.tables_embedded: True,
                    Paper.updated_at: datetime.utcnow()
                },
                synchronize_session=False
            )
            self.session.commit()
            count = len(dois)
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            self.session.rollback()
        return count
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self, category: str = None) -> Dict:
        """Get processing statistics"""
        base_query = self.session.query(Paper)
        if category:
            base_query = base_query.filter(Paper.category == category)
        
        total = base_query.count()
        downloaded = base_query.filter(Paper.pdf_downloaded == True).count()
        extracted = base_query.filter(Paper.content_extracted == True).count()
        text_embedded = base_query.filter(Paper.text_embedded == True).count()
        figures_embedded = base_query.filter(Paper.figures_embedded == True).count()
        tables_embedded = base_query.filter(Paper.tables_embedded == True).count()
        fully_embedded = base_query.filter(
            and_(
                Paper.text_embedded == True,
                Paper.figures_embedded == True,
                Paper.tables_embedded == True
            )
        ).count()
        
        return {
            'total': total,
            'downloaded': downloaded,
            'extracted': extracted,
            'text_embedded': text_embedded,
            'figures_embedded': figures_embedded,
            'tables_embedded': tables_embedded,
            'fully_embedded': fully_embedded,
            'pending_download': total - downloaded,
            'pending_extract': downloaded - extracted,
            'pending_embed': extracted - fully_embedded
        }
    
    def print_stats(self, category: str = None):
        """Print processing statistics"""
        stats = self.get_stats(category)
        
        cat_str = f" ({category})" if category else ""
        print(f"\n📊 Processing Status{cat_str}")
        print("=" * 50)
        print(f"  Total papers:        {stats['total']:,}")
        print(f"  Downloaded:          {stats['downloaded']:,}")
        print(f"  Extracted:           {stats['extracted']:,}")
        print(f"  Text embedded:       {stats['text_embedded']:,}")
        print(f"  Figures embedded:    {stats['figures_embedded']:,}")
        print(f"  Tables embedded:     {stats['tables_embedded']:,}")
        print(f"  Fully embedded:      {stats['fully_embedded']:,}")
        print("-" * 50)
        print(f"  Pending download:    {stats['pending_download']:,}")
        print(f"  Pending extract:     {stats['pending_extract']:,}")
        print(f"  Pending embed:       {stats['pending_embed']:,}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Progress Tracker")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--status", type=str, help="Get status for DOI")
    parser.add_argument("--pending", type=str, choices=['download', 'extract', 'embed'],
                        help="List pending papers")
    parser.add_argument("--limit", type=int, default=10, help="Limit results")
    
    args = parser.parse_args()
    
    with ProgressTracker() as tracker:
        if args.stats:
            tracker.print_stats(args.category)
        
        elif args.status:
            status = tracker.get_status(args.status)
            if status:
                print(f"\n📄 Status for {args.status}")
                for k, v in status.items():
                    print(f"  {k}: {v}")
            else:
                print(f"Paper not found: {args.status}")
        
        elif args.pending:
            if args.pending == 'download':
                papers = tracker.get_papers_to_download(args.category, limit=args.limit)
                print(f"\n📥 Papers to download: {len(papers)}")
            elif args.pending == 'extract':
                papers = tracker.get_papers_to_extract(args.category, limit=args.limit)
                print(f"\n📄 Papers to extract: {len(papers)}")
            elif args.pending == 'embed':
                papers = tracker.get_papers_to_embed(args.category, limit=args.limit)
                print(f"\n🧠 Papers to embed: {len(papers)}")
            
            for p in papers[:10]:
                print(f"  - {p['doi']}")
        
        else:
            tracker.print_stats()
