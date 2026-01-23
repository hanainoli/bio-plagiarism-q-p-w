#!/usr/bin/env python3
"""
Paper Status Viewer
===================
View detailed processing status for papers.

Usage:
    # View single paper
    python paper_status.py --doi "10.1101/2024.01.001"
    
    # View all papers in category
    python paper_status.py --category oncology --limit 50
    
    # View summary stats
    python paper_status.py --summary
    
    # View papers missing embeddings
    python paper_status.py --missing embed --limit 20
    
    # Export to CSV
    python paper_status.py --category oncology --export status.csv
    
    # Check from Qdrant directly
    python paper_status.py --qdrant-check --doi "10.1101/2024.01.001"
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


def get_paper_status_from_postgres(doi: str) -> Optional[Dict]:
    """Get detailed status for a single paper from PostgreSQL"""
    try:
        from db.models import Paper, ProcessingStatus, get_session
        
        session = get_session()
        paper = session.query(Paper).filter(Paper.doi == doi).first()
        
        if not paper:
            session.close()
            return None
        
        status = {
            'doi': paper.doi,
            'title': paper.title[:80] + '...' if paper.title and len(paper.title) > 80 else paper.title,
            'category': paper.category,
            'server': paper.server,
            'date': str(paper.date) if paper.date else None,
            'pdf_downloaded': paper.pdf_downloaded,
            'content_extracted': paper.content_extracted,
            'text_embedded': paper.text_embedded,
            'figures_embedded': paper.figures_embedded,
            'tables_embedded': paper.tables_embedded,
            'fully_embedded': paper.text_embedded and paper.figures_embedded and paper.tables_embedded,
        }
        
        # Get processing details
        if paper.processing:
            ps = paper.processing
            status.update({
                'pdf_path': ps.pdf_path,
                'pdf_size_kb': round(ps.pdf_size / 1024, 1) if ps.pdf_size else 0,
                'text_chunks': ps.text_chunks or 0,
                'figures_count': ps.figures_count or 0,
                'tables_count': ps.tables_count or 0,
                'text_vectors': ps.text_vectors or 0,
                'figure_vectors': ps.figure_vectors or 0,
                'table_vectors': ps.table_vectors or 0,
                'downloaded_at': str(ps.downloaded_at) if ps.downloaded_at else None,
                'extracted_at': str(ps.extracted_at) if ps.extracted_at else None,
                'embedded_at': str(ps.embedded_at) if ps.embedded_at else None,
                'last_error': ps.last_error,
                'error_count': ps.error_count or 0
            })
        else:
            status.update({
                'pdf_path': None,
                'pdf_size_kb': 0,
                'text_chunks': 0,
                'figures_count': 0,
                'tables_count': 0,
                'text_vectors': 0,
                'figure_vectors': 0,
                'table_vectors': 0,
            })
        
        session.close()
        return status
        
    except Exception as e:
        print(f"PostgreSQL error: {e}")
        return None


def get_paper_status_from_qdrant(doi: str) -> Dict:
    """Get embedding counts from Qdrant directly"""
    try:
        from storage.qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        qdrant = QdrantClient()
        
        if not qdrant.client:
            return {'error': 'Qdrant not connected'}
        
        status = {
            'doi': doi,
            'abstracts': 0,
            'text_chunks': 0,
            'figures': 0,
            'tables': 0,
        }
        
        # Check each collection
        collections = {
            'bio_abstracts': 'abstracts',
            'bio_fulltext_chunks': 'text_chunks',
            'bio_figures': 'figures',
            'bio_tables': 'tables'
        }
        
        for col_name, key in collections.items():
            try:
                if qdrant.collection_exists(col_name):
                    # Count points with this DOI
                    result = qdrant.client.count(
                        collection_name=col_name,
                        count_filter=Filter(
                            must=[FieldCondition(key="doi", match=MatchValue(value=doi))]
                        )
                    )
                    status[key] = result.count
            except Exception as e:
                status[key] = f"Error: {e}"
        
        status['total_vectors'] = (
            (status['abstracts'] if isinstance(status['abstracts'], int) else 0) +
            (status['text_chunks'] if isinstance(status['text_chunks'], int) else 0) +
            (status['figures'] if isinstance(status['figures'], int) else 0) +
            (status['tables'] if isinstance(status['tables'], int) else 0)
        )
        
        return status
        
    except Exception as e:
        return {'error': str(e)}


def get_papers_list(
    category: str = None,
    status_filter: str = None,  # 'downloaded', 'extracted', 'embedded', 'missing_embed'
    limit: int = 50
) -> List[Dict]:
    """Get list of papers with status"""
    try:
        from db.models import Paper, ProcessingStatus, get_session
        from sqlalchemy import and_, or_
        
        session = get_session()
        query = session.query(Paper)
        
        # Filter by category
        if category:
            from sqlalchemy import func
            query = query.filter(func.lower(Paper.category).like(f"%{category.lower()}%"))
        
        # Filter by status
        if status_filter == 'downloaded':
            query = query.filter(Paper.pdf_downloaded == True)
        elif status_filter == 'not_downloaded':
            query = query.filter(Paper.pdf_downloaded == False)
        elif status_filter == 'extracted':
            query = query.filter(Paper.content_extracted == True)
        elif status_filter == 'not_extracted':
            query = query.filter(and_(
                Paper.pdf_downloaded == True,
                Paper.content_extracted == False
            ))
        elif status_filter == 'embedded':
            query = query.filter(and_(
                Paper.text_embedded == True,
                Paper.figures_embedded == True,
                Paper.tables_embedded == True
            ))
        elif status_filter == 'missing_embed':
            query = query.filter(and_(
                Paper.content_extracted == True,
                or_(
                    Paper.text_embedded == False,
                    Paper.figures_embedded == False,
                    Paper.tables_embedded == False
                )
            ))
        elif status_filter == 'missing_figures':
            query = query.filter(and_(
                Paper.content_extracted == True,
                Paper.text_embedded == True,
                Paper.figures_embedded == False
            ))
        elif status_filter == 'missing_tables':
            query = query.filter(and_(
                Paper.content_extracted == True,
                Paper.text_embedded == True,
                Paper.tables_embedded == False
            ))
        
        papers = query.order_by(Paper.date.desc()).limit(limit).all()
        
        results = []
        for p in papers:
            item = {
                'doi': p.doi,
                'title': p.title[:50] + '...' if p.title and len(p.title) > 50 else p.title,
                'category': p.category,
                'date': str(p.date) if p.date else '',
                'downloaded': '✓' if p.pdf_downloaded else '✗',
                'extracted': '✓' if p.content_extracted else '✗',
                'text_emb': '✓' if p.text_embedded else '✗',
                'fig_emb': '✓' if p.figures_embedded else '✗',
                'tbl_emb': '✓' if p.tables_embedded else '✗',
            }
            
            if p.processing:
                item.update({
                    'chunks': p.processing.text_chunks or 0,
                    'figures': p.processing.figures_count or 0,
                    'tables': p.processing.tables_count or 0,
                    'text_vec': p.processing.text_vectors or 0,
                    'fig_vec': p.processing.figure_vectors or 0,
                    'tbl_vec': p.processing.table_vectors or 0,
                })
            else:
                item.update({
                    'chunks': 0, 'figures': 0, 'tables': 0,
                    'text_vec': 0, 'fig_vec': 0, 'tbl_vec': 0
                })
            
            results.append(item)
        
        session.close()
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_summary_stats() -> Dict:
    """Get overall summary statistics"""
    try:
        from db.models import Paper, ProcessingStatus, get_session
        from sqlalchemy import func, and_
        
        session = get_session()
        
        # Overall counts
        total = session.query(Paper).count()
        downloaded = session.query(Paper).filter(Paper.pdf_downloaded == True).count()
        extracted = session.query(Paper).filter(Paper.content_extracted == True).count()
        text_embedded = session.query(Paper).filter(Paper.text_embedded == True).count()
        figures_embedded = session.query(Paper).filter(Paper.figures_embedded == True).count()
        tables_embedded = session.query(Paper).filter(Paper.tables_embedded == True).count()
        fully_embedded = session.query(Paper).filter(and_(
            Paper.text_embedded == True,
            Paper.figures_embedded == True,
            Paper.tables_embedded == True
        )).count()
        
        # Aggregate counts from processing_status
        agg = session.query(
            func.sum(ProcessingStatus.text_chunks),
            func.sum(ProcessingStatus.figures_count),
            func.sum(ProcessingStatus.tables_count),
            func.sum(ProcessingStatus.text_vectors),
            func.sum(ProcessingStatus.figure_vectors),
            func.sum(ProcessingStatus.table_vectors)
        ).first()
        
        # Category breakdown
        categories = session.query(
            Paper.category,
            func.count(Paper.id).label('total'),
            func.sum(func.cast(Paper.text_embedded, Integer)).label('embedded')
        ).group_by(Paper.category).order_by(func.count(Paper.id).desc()).limit(10).all()
        
        session.close()
        
        return {
            'total_papers': total,
            'downloaded': downloaded,
            'extracted': extracted,
            'text_embedded': text_embedded,
            'figures_embedded': figures_embedded,
            'tables_embedded': tables_embedded,
            'fully_embedded': fully_embedded,
            'pending_download': total - downloaded,
            'pending_extract': downloaded - extracted,
            'pending_embed': extracted - fully_embedded,
            'total_text_chunks': agg[0] or 0,
            'total_figures': agg[1] or 0,
            'total_tables': agg[2] or 0,
            'total_text_vectors': agg[3] or 0,
            'total_figure_vectors': agg[4] or 0,
            'total_table_vectors': agg[5] or 0,
        }
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_qdrant_stats() -> Dict:
    """Get stats directly from Qdrant"""
    try:
        from storage.qdrant_client import QdrantClient
        
        qdrant = QdrantClient()
        
        if not qdrant.client:
            return {'error': 'Qdrant not connected'}
        
        stats = {}
        collections = ['bio_abstracts', 'bio_fulltext_chunks', 'bio_figures', 'bio_tables']
        
        for col in collections:
            try:
                if qdrant.collection_exists(col):
                    info = qdrant.client.get_collection(col)
                    stats[col] = {
                        'vectors': info.points_count,
                        'status': info.status.name
                    }
                else:
                    stats[col] = {'vectors': 0, 'status': 'NOT_EXISTS'}
            except Exception as e:
                stats[col] = {'error': str(e)}
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}


def print_paper_status(doi: str):
    """Print detailed status for a single paper"""
    print(f"\n{'='*70}")
    print(f"PAPER STATUS: {doi}")
    print(f"{'='*70}")
    
    # PostgreSQL status
    pg_status = get_paper_status_from_postgres(doi)
    
    if pg_status:
        print(f"\n📄 Paper Info (PostgreSQL)")
        print(f"   Title:    {pg_status.get('title', 'N/A')}")
        print(f"   Category: {pg_status.get('category', 'N/A')}")
        print(f"   Server:   {pg_status.get('server', 'N/A')}")
        print(f"   Date:     {pg_status.get('date', 'N/A')}")
        
        print(f"\n📊 Processing Status")
        print(f"   PDF Downloaded:    {'✅' if pg_status.get('pdf_downloaded') else '❌'}")
        print(f"   Content Extracted: {'✅' if pg_status.get('content_extracted') else '❌'}")
        print(f"   Text Embedded:     {'✅' if pg_status.get('text_embedded') else '❌'}")
        print(f"   Figures Embedded:  {'✅' if pg_status.get('figures_embedded') else '❌'}")
        print(f"   Tables Embedded:   {'✅' if pg_status.get('tables_embedded') else '❌'}")
        print(f"   Fully Complete:    {'✅' if pg_status.get('fully_embedded') else '❌'}")
        
        print(f"\n📈 Metrics")
        print(f"   PDF Size:      {pg_status.get('pdf_size_kb', 0)} KB")
        print(f"   Text Chunks:   {pg_status.get('text_chunks', 0)}")
        print(f"   Figures:       {pg_status.get('figures_count', 0)}")
        print(f"   Tables:        {pg_status.get('tables_count', 0)}")
        
        print(f"\n🧠 Vectors Stored")
        print(f"   Text Vectors:   {pg_status.get('text_vectors', 0)}")
        print(f"   Figure Vectors: {pg_status.get('figure_vectors', 0)}")
        print(f"   Table Vectors:  {pg_status.get('table_vectors', 0)}")
        
        if pg_status.get('last_error'):
            print(f"\n⚠️ Last Error: {pg_status.get('last_error')}")
    else:
        print(f"\n❌ Paper not found in PostgreSQL database")
    
    # Qdrant status
    print(f"\n🔍 Qdrant Verification")
    qdrant_status = get_paper_status_from_qdrant(doi)
    
    if 'error' in qdrant_status:
        print(f"   Error: {qdrant_status['error']}")
    else:
        print(f"   Abstracts:    {qdrant_status.get('abstracts', 0)}")
        print(f"   Text Chunks:  {qdrant_status.get('text_chunks', 0)}")
        print(f"   Figures:      {qdrant_status.get('figures', 0)}")
        print(f"   Tables:       {qdrant_status.get('tables', 0)}")
        print(f"   Total:        {qdrant_status.get('total_vectors', 0)} vectors")


def print_papers_table(papers: List[Dict]):
    """Print papers in table format"""
    if not papers:
        print("No papers found.")
        return
    
    # Header
    print(f"\n{'DOI':<30} {'Down':^5} {'Ext':^4} {'Txt':^4} {'Fig':^4} {'Tbl':^4} │ {'Chunks':>6} {'Figs':>5} {'Tbls':>5} │ {'TxtV':>5} {'FigV':>5} {'TblV':>5}")
    print("─" * 120)
    
    for p in papers:
        short_doi = p['doi'][-28:] if len(p['doi']) > 28 else p['doi']
        print(f"{short_doi:<30} {p['downloaded']:^5} {p['extracted']:^4} {p['text_emb']:^4} {p['fig_emb']:^4} {p['tbl_emb']:^4} │ {p['chunks']:>6} {p['figures']:>5} {p['tables']:>5} │ {p['text_vec']:>5} {p['fig_vec']:>5} {p['tbl_vec']:>5}")
    
    print(f"\nTotal: {len(papers)} papers")


def print_summary():
    """Print overall summary"""
    print(f"\n{'='*70}")
    print(f"UYARAI PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    # PostgreSQL stats
    print(f"\n📊 PostgreSQL Status")
    stats = get_summary_stats()
    
    if stats:
        print(f"\n   Papers:")
        print(f"      Total:            {stats.get('total_papers', 0):,}")
        print(f"      Downloaded:       {stats.get('downloaded', 0):,}")
        print(f"      Extracted:        {stats.get('extracted', 0):,}")
        print(f"      Text Embedded:    {stats.get('text_embedded', 0):,}")
        print(f"      Figures Embedded: {stats.get('figures_embedded', 0):,}")
        print(f"      Tables Embedded:  {stats.get('tables_embedded', 0):,}")
        print(f"      Fully Complete:   {stats.get('fully_embedded', 0):,}")
        
        print(f"\n   Pending:")
        print(f"      Need Download:    {stats.get('pending_download', 0):,}")
        print(f"      Need Extract:     {stats.get('pending_extract', 0):,}")
        print(f"      Need Embed:       {stats.get('pending_embed', 0):,}")
        
        print(f"\n   Content Totals:")
        print(f"      Text Chunks:      {stats.get('total_text_chunks', 0):,}")
        print(f"      Figures:          {stats.get('total_figures', 0):,}")
        print(f"      Tables:           {stats.get('total_tables', 0):,}")
        
        print(f"\n   Vector Totals:")
        print(f"      Text Vectors:     {stats.get('total_text_vectors', 0):,}")
        print(f"      Figure Vectors:   {stats.get('total_figure_vectors', 0):,}")
        print(f"      Table Vectors:    {stats.get('total_table_vectors', 0):,}")
    
    # Qdrant stats
    print(f"\n🔍 Qdrant Collections")
    qdrant_stats = get_qdrant_stats()
    
    if 'error' in qdrant_stats:
        print(f"   Error: {qdrant_stats['error']}")
    else:
        for col, info in qdrant_stats.items():
            if 'error' in info:
                print(f"   {col}: Error - {info['error']}")
            else:
                print(f"   {col}: {info.get('vectors', 0):,} vectors ({info.get('status', 'UNKNOWN')})")


def export_to_csv(papers: List[Dict], filename: str):
    """Export papers list to CSV"""
    import csv
    
    if not papers:
        print("No papers to export.")
        return
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=papers[0].keys())
        writer.writeheader()
        writer.writerows(papers)
    
    print(f"Exported {len(papers)} papers to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Paper Status Viewer")
    
    # View options
    parser.add_argument("--doi", type=str, help="View status for specific DOI")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--limit", type=int, default=50, help="Limit results")
    parser.add_argument("--summary", action="store_true", help="Show summary stats")
    
    # Filter options
    parser.add_argument("--missing", type=str, 
                       choices=['download', 'extract', 'embed', 'figures', 'tables'],
                       help="Show papers missing specific step")
    parser.add_argument("--status", type=str,
                       choices=['downloaded', 'extracted', 'embedded', 'all'],
                       help="Filter by status")
    
    # Qdrant options
    parser.add_argument("--qdrant-check", action="store_true", 
                       help="Check Qdrant directly for DOI")
    parser.add_argument("--qdrant-stats", action="store_true",
                       help="Show Qdrant collection stats")
    
    # Export
    parser.add_argument("--export", type=str, metavar="FILE",
                       help="Export to CSV file")
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.summary:
        print_summary()
    
    elif args.doi:
        if args.qdrant_check:
            status = get_paper_status_from_qdrant(args.doi)
            print(f"\nQdrant status for {args.doi}:")
            for k, v in status.items():
                print(f"  {k}: {v}")
        else:
            print_paper_status(args.doi)
    
    elif args.qdrant_stats:
        print(f"\n🔍 Qdrant Collection Stats")
        stats = get_qdrant_stats()
        if 'error' in stats:
            print(f"  Error: {stats['error']}")
        else:
            for col, info in stats.items():
                if 'error' in info:
                    print(f"  {col}: Error - {info['error']}")
                else:
                    print(f"  {col}: {info.get('vectors', 0):,} vectors")
    
    elif args.missing:
        filter_map = {
            'download': 'not_downloaded',
            'extract': 'not_extracted',
            'embed': 'missing_embed',
            'figures': 'missing_figures',
            'tables': 'missing_tables'
        }
        papers = get_papers_list(
            category=args.category,
            status_filter=filter_map.get(args.missing),
            limit=args.limit
        )
        print(f"\n📋 Papers missing {args.missing}:")
        print_papers_table(papers)
        
        if args.export:
            export_to_csv(papers, args.export)
    
    elif args.category or args.status:
        papers = get_papers_list(
            category=args.category,
            status_filter=args.status,
            limit=args.limit
        )
        print_papers_table(papers)
        
        if args.export:
            export_to_csv(papers, args.export)
    
    else:
        # Default: show summary
        print_summary()


if __name__ == "__main__":
    # Fix import for Integer
    try:
        from sqlalchemy import Integer
    except:
        pass
    
    main()
