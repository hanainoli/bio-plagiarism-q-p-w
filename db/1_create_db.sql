-- =============================================================================
-- UyarAI PostgreSQL Setup - RUN THIS DIRECTLY
-- =============================================================================
-- Run in psql as postgres superuser:
--   psql -U postgres -f db/create_db.sql
--
-- Or copy-paste into pgAdmin Query Tool (connected as postgres)
-- =============================================================================

-- 1. Create User
CREATE USER uyarai_user WITH PASSWORD 'uyarai_password';

-- 2. Create Database
CREATE DATABASE uyarai OWNER uyarai_user;

-- 3. Grant privileges
GRANT ALL PRIVILEGES ON DATABASE uyarai TO uyarai_user;

-- Done! Now run setup_tables.sql
