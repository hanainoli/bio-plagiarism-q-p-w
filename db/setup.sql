-- =============================================================================
-- UyarAI PostgreSQL Setup Script
-- =============================================================================
-- Run this script as postgres superuser to create database, user, and tables
--
-- Usage:
--   psql -U postgres -f db/setup.sql
--
-- Or run each section separately in pgAdmin/psql
-- =============================================================================


-- =============================================================================
-- 1. CREATE USER
-- =============================================================================
-- Change 'uyarai_password' to a secure password

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'uyarai_user') THEN
        CREATE USER uyarai_user WITH PASSWORD 'uyarai_password';
    END IF;
END
$$;

-- Grant privileges
ALTER USER uyarai_user CREATEDB;


-- =============================================================================
-- 2. CREATE DATABASE
-- =============================================================================

-- Check if database exists, create if not
SELECT 'CREATE DATABASE uyarai OWNER uyarai_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'uyarai')\gexec

-- Connect to the database
\c uyarai


-- =============================================================================
-- 3. CREATE TABLES
-- =============================================================================

-- Papers table (main table with 2.5M+ records)
CREATE TABLE IF NOT EXISTS bio_papers (
    id SERIAL PRIMARY KEY,
    doi VARCHAR(100) UNIQUE NOT NULL,
    title TEXT,
    authors TEXT,
    abstract TEXT,
    category VARCHAR(100),
    group_id VARCHAR(50),
    server VARCHAR(20),
    date DATE,
    version INTEGER DEFAULT 1,
    license VARCHAR(100),
    jatsxml VARCHAR(500),
    
    -- Status flags
    pdf_downloaded BOOLEAN DEFAULT FALSE,
    content_extracted BOOLEAN DEFAULT FALSE,
    text_embedded BOOLEAN DEFAULT FALSE,
    figures_embedded BOOLEAN DEFAULT FALSE,
    tables_embedded BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing status table
CREATE TABLE IF NOT EXISTS bio_processing_status (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER UNIQUE REFERENCES bio_papers(id),
    doi VARCHAR(100),
    
    -- Download
    pdf_path VARCHAR(500),
    pdf_size INTEGER,
    downloaded_at TIMESTAMP,
    
    -- Extraction
    text_chunks INTEGER DEFAULT 0,
    figures_count INTEGER DEFAULT 0,
    tables_count INTEGER DEFAULT 0,
    extracted_at TIMESTAMP,
    
    -- Embedding
    text_vectors INTEGER DEFAULT 0,
    figure_vectors INTEGER DEFAULT 0,
    table_vectors INTEGER DEFAULT 0,
    embedded_at TIMESTAMP,
    
    -- Errors
    last_error TEXT,
    error_count INTEGER DEFAULT 0
);

-- Plagiarism reports table
CREATE TABLE IF NOT EXISTS bio_plagiarism_reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(50) UNIQUE,
    filename VARCHAR(255),
    
    -- Scores
    overall_score FLOAT DEFAULT 0,
    text_score FLOAT DEFAULT 0,
    image_score FLOAT DEFAULT 0,
    risk_level VARCHAR(20),
    
    -- Counts
    text_matches INTEGER DEFAULT 0,
    image_matches INTEGER DEFAULT 0,
    matched_papers INTEGER DEFAULT 0,
    
    -- Full report
    report_data JSONB,
    
    -- Timestamps
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_time_seconds FLOAT
);

-- Build progress table (for resume capability)
CREATE TABLE IF NOT EXISTS bio_build_progress (
    id SERIAL PRIMARY KEY,
    task VARCHAR(50) UNIQUE,
    status VARCHAR(20),
    last_date DATE,
    total_added INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);


-- =============================================================================
-- 4. CREATE INDEXES
-- =============================================================================

-- Papers indexes
CREATE INDEX IF NOT EXISTS idx_bio_papers_doi ON bio_papers(doi);
CREATE INDEX IF NOT EXISTS idx_bio_papers_category ON bio_papers(category);
CREATE INDEX IF NOT EXISTS idx_bio_papers_group ON bio_papers(group_id);
CREATE INDEX IF NOT EXISTS idx_bio_papers_date ON bio_papers(date);
CREATE INDEX IF NOT EXISTS idx_bio_papers_server ON bio_papers(server);
CREATE INDEX IF NOT EXISTS idx_bio_papers_category_date ON bio_papers(category, date);
CREATE INDEX IF NOT EXISTS idx_bio_papers_group_date ON bio_papers(group_id, date);
CREATE INDEX IF NOT EXISTS idx_bio_papers_status ON bio_papers(pdf_downloaded, content_extracted, text_embedded);

-- Processing status indexes
CREATE INDEX IF NOT EXISTS idx_bio_processing_doi ON bio_processing_status(doi);
CREATE INDEX IF NOT EXISTS idx_bio_processing_paper_id ON bio_processing_status(paper_id);

-- Reports indexes
CREATE INDEX IF NOT EXISTS idx_bio_reports_report_id ON bio_plagiarism_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_bio_reports_checked_at ON bio_plagiarism_reports(checked_at);


-- =============================================================================
-- 5. GRANT PERMISSIONS
-- =============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO uyarai_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO uyarai_user;
GRANT USAGE ON SCHEMA public TO uyarai_user;

-- For future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO uyarai_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO uyarai_user;


-- =============================================================================
-- 6. VERIFY SETUP
-- =============================================================================

-- Show tables
\dt bio_*

-- Show indexes
\di bio_*

-- Done!
SELECT 'Setup complete! Tables created:' AS status;
SELECT tablename FROM pg_tables WHERE tablename LIKE 'bio_%';
