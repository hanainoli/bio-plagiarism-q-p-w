-- =============================================================================
-- UyarAI PostgreSQL Setup (Simple Version)
-- =============================================================================
-- Run these commands in pgAdmin or psql
-- =============================================================================


-- STEP 1: Create User (run as postgres superuser)
-- ================================================
CREATE USER plagiarism_system WITH PASSWORD 'nXAEhAl4FFCkID4Y3U9DXeaxfAZvPEC3';
ALTER USER plagiarism_system CREATEDB;


-- STEP 2: Create Database (run as postgres superuser)
-- ====================================================
CREATE DATABASE plagiarism_db_z8nz OWNER plagiarism_system;


-- STEP 3: Connect to uyarai database, then run below
-- ====================================================
-- In pgAdmin: Right-click uyarai database → Query Tool
-- In psql: \c uyarai


-- STEP 4: Create Tables
-- ======================

-- Papers table
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

-- Processing status table
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

-- Plagiarism reports table
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

-- Build progress table
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


-- STEP 5: Create Indexes
-- =======================
CREATE INDEX idx_bio_papers_doi ON bio_papers(doi);
CREATE INDEX idx_bio_papers_category ON bio_papers(category);
CREATE INDEX idx_bio_papers_group ON bio_papers(group_id);
CREATE INDEX idx_bio_papers_date ON bio_papers(date);
CREATE INDEX idx_bio_papers_server ON bio_papers(server);


-- STEP 6: Grant Permissions
-- ==========================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO plagiarism_system;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO plagiarism_system;


-- STEP 7: Verify
-- ===============
SELECT tablename FROM pg_tables WHERE tablename LIKE 'bio_%';
