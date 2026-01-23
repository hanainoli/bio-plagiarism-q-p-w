"""
UyarAI Database Models
=======================
SQLAlchemy models for PostgreSQL.

Tables:
    - bio_papers: All bioRxiv/medRxiv papers
    - bio_processing_status: Track download/extract/embed status
    - bio_plagiarism_reports: Store check reports
"""

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Date, 
    DateTime, Boolean, Float, ForeignKey, Index, JSON, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


# =============================================================================
# MODELS
# =============================================================================

class Paper(Base):
    """Paper metadata from bioRxiv/medRxiv"""
    __tablename__ = 'bio_papers'
    
    id = Column(Integer, primary_key=True)
    doi = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(Text)
    authors = Column(Text)
    abstract = Column(Text)
    category = Column(String(100), index=True)
    group_id = Column(String(50), index=True)
    server = Column(String(20))  # biorxiv or medrxiv
    date = Column(Date, index=True)
    version = Column(Integer, default=1)
    license = Column(String(100))
    jatsxml = Column(String(500))
    
    # Status flags
    pdf_downloaded = Column(Boolean, default=False)
    content_extracted = Column(Boolean, default=False)
    text_embedded = Column(Boolean, default=False)
    figures_embedded = Column(Boolean, default=False)
    tables_embedded = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    processing = relationship("ProcessingStatus", back_populates="paper", uselist=False)
    
    def __repr__(self):
        return f"<Paper {self.doi}>"


class ProcessingStatus(Base):
    """Detailed processing status for each paper"""
    __tablename__ = 'bio_processing_status'
    
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('bio_papers.id'), unique=True)
    doi = Column(String(100), index=True)
    
    # Download
    pdf_path = Column(String(500))
    pdf_size = Column(Integer)
    downloaded_at = Column(DateTime)
    
    # Extraction
    text_chunks = Column(Integer, default=0)
    figures_count = Column(Integer, default=0)
    tables_count = Column(Integer, default=0)
    extracted_at = Column(DateTime)
    
    # Embedding
    text_vectors = Column(Integer, default=0)
    figure_vectors = Column(Integer, default=0)
    table_vectors = Column(Integer, default=0)
    embedded_at = Column(DateTime)
    
    # Errors
    last_error = Column(Text)
    error_count = Column(Integer, default=0)
    
    # Relationship
    paper = relationship("Paper", back_populates="processing")
    
    def __repr__(self):
        return f"<ProcessingStatus {self.doi}>"


class PlagiarismReport(Base):
    """Plagiarism check reports"""
    __tablename__ = 'bio_plagiarism_reports'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(String(50), unique=True, index=True)
    filename = Column(String(255))
    
    # Scores
    overall_score = Column(Float, default=0)
    text_score = Column(Float, default=0)
    image_score = Column(Float, default=0)
    risk_level = Column(String(20))
    
    # Counts
    text_matches = Column(Integer, default=0)
    image_matches = Column(Integer, default=0)
    matched_papers = Column(Integer, default=0)
    
    # Full report JSON
    report_data = Column(JSON)
    
    # Timestamps
    checked_at = Column(DateTime, default=datetime.utcnow)
    check_time_seconds = Column(Float)
    
    def __repr__(self):
        return f"<PlagiarismReport {self.report_id}>"


class BuildProgress(Base):
    """Track index build progress for resume"""
    __tablename__ = 'bio_build_progress'
    
    id = Column(Integer, primary_key=True)
    task = Column(String(50), unique=True)  # 'index_build', 'daily_update'
    status = Column(String(20))  # 'in_progress', 'complete', 'failed'
    last_date = Column(Date)
    total_added = Column(Integer, default=0)
    started_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<BuildProgress {self.task}>"


# =============================================================================
# INDEXES
# =============================================================================

# Composite indexes for common queries
Index('idx_papers_category_date', Paper.category, Paper.date)
Index('idx_papers_group_date', Paper.group_id, Paper.date)
Index('idx_papers_status', Paper.pdf_downloaded, Paper.content_extracted, Paper.text_embedded)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_database_url() -> str:
    """Get database URL from environment"""
    return os.getenv('DATABASE_URL', 'postgresql://localhost:5432/uyarai')


def get_engine(echo: bool = False):
    """Create database engine"""
    return create_engine(get_database_url(), echo=False)


def get_session():
    """Create database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize database tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Database tables created")


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("🗑️ All tables dropped")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Models")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--drop", action="store_true", help="Drop all tables")
    parser.add_argument("--check", action="store_true", help="Check connection")
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    
    elif args.drop:
        confirm = input("⚠️ This will DROP ALL TABLES. Type 'yes': ")
        if confirm == 'yes':
            drop_all_tables()
    
    elif args.check:
        try:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("✅ Database connection OK")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    else:
        parser.print_help()
