-- =============================================================================
-- UyarAI PostgreSQL Tables - RUN THIS AFTER create_db.sql
-- =============================================================================
-- Run in psql:
--   psql -U uyarai_user -d uyarai -f db/2_create_tables.sql
--
-- Or in pgAdmin: Connect to 'uyarai' database, then run this
-- =============================================================================

-- Papers table (2.5M+ records)
CREATE TABLE bio_papers (
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
    pdf_downloaded BOOLEAN DEFAULT FALSE,
    content_extracted BOOLEAN DEFAULT FALSE,
    text_embedded BOOLEAN DEFAULT FALSE,
    figures_embedded BOOLEAN DEFAULT FALSE,
    tables_embedded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing status
CREATE TABLE bio_processing_status (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER UNIQUE REFERENCES bio_papers(id),
    doi VARCHAR(100),
    pdf_path VARCHAR(500),
    pdf_size INTEGER,
    downloaded_at TIMESTAMP,
    text_chunks INTEGER DEFAULT 0,
    figures_count INTEGER DEFAULT 0,
    tables_count INTEGER DEFAULT 0,
    extracted_at TIMESTAMP,
    text_vectors INTEGER DEFAULT 0,
    figure_vectors INTEGER DEFAULT 0,
    table_vectors INTEGER DEFAULT 0,
    embedded_at TIMESTAMP,
    last_error TEXT,
    error_count INTEGER DEFAULT 0
);

-- Plagiarism reports
CREATE TABLE bio_plagiarism_reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(50) UNIQUE,
    filename VARCHAR(255),
    overall_score FLOAT DEFAULT 0,
    text_score FLOAT DEFAULT 0,
    image_score FLOAT DEFAULT 0,
    risk_level VARCHAR(20),
    text_matches INTEGER DEFAULT 0,
    image_matches INTEGER DEFAULT 0,
    matched_papers INTEGER DEFAULT 0,
    report_data JSONB,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    check_time_seconds FLOAT
);

-- Build progress (for resume)
CREATE TABLE bio_build_progress (
    id SERIAL PRIMARY KEY,
    task VARCHAR(50) UNIQUE,
    status VARCHAR(20),
    last_date DATE,
    total_added INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX idx_bio_papers_doi ON bio_papers(doi);
CREATE INDEX idx_bio_papers_category ON bio_papers(category);
CREATE INDEX idx_bio_papers_group ON bio_papers(group_id);
CREATE INDEX idx_bio_papers_date ON bio_papers(date);
CREATE INDEX idx_bio_papers_server ON bio_papers(server);
CREATE INDEX idx_bio_papers_status ON bio_papers(pdf_downloaded, content_extracted, text_embedded);
CREATE INDEX idx_bio_processing_doi ON bio_processing_status(doi);
CREATE INDEX idx_bio_reports_report_id ON bio_plagiarism_reports(report_id);

-- Verify
SELECT 'Tables created:' AS status;
SELECT tablename FROM pg_tables WHERE tablename LIKE 'bio_%';
