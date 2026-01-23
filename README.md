# UyarAI Plagiarism Detection

Biomedical plagiarism detection using bioRxiv/medRxiv papers.

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with PostgreSQL, Wasabi and Qdrant credentials

# 3. Initialize database
python db/models.py --init

# 4. Build index (2-3 hours, saves to PostgreSQL)
python db/build_index.py --build

# 5. Process papers
python run_system.py --download --categories "oncology" --max 100
python run_system.py --extract --categories "oncology" --max 100
python run_system.py --smart-embed --categories "oncology" --max 100

# 6. Start API
python api.py

# 7. Check plagiarism
curl -X POST http://localhost:8000/check -F "file=@paper.pdf"
```

## CLI Commands

### Database & Index (PostgreSQL)
```bash
python db/models.py --init              # Initialize database tables
python db/models.py --check             # Check database connection
python db/build_index.py --build        # Build full index (2-3 hours)
python db/build_index.py --update 7     # Update last 7 days
python db/build_index.py --stats        # Show statistics
python db/build_index.py --category oncology  # Category detail
```

### Process Papers
```bash
python run_system.py --download --categories "oncology" "genetics" --max 100
python run_system.py --extract --categories "oncology" "genetics" --max 100
python run_system.py --smart-embed --categories "oncology" "genetics" --max 100
```

### Check Plagiarism
```bash
python run_system.py --check paper.pdf --report report.json
```

### Reset & Sync
```bash
python reset.py --all                  # Delete everything
python reset.py --sync-from-wasabi     # Download from Wasabi
python reset.py --sync-to-wasabi       # Upload to Wasabi
```

## Stats & Debug

```bash
python stats.py                    # Full system status
python stats.py --index            # Index stats
python stats.py --qdrant           # Qdrant stats
python stats.py --groups           # Groups breakdown
python stats.py --category oncology # Category detail
python stats.py --downloads        # Download progress
python stats.py --check-config     # Verify configuration
```

## API Endpoints

```
GET  /metrics              - All metrics
GET  /metrics/index        - Index stats
GET  /metrics/qdrant       - Qdrant stats
GET  /metrics/groups       - Groups with counts
GET  /metrics/categories   - Categories with counts

POST /check                - Upload PDF and check
GET  /check/reports        - List reports
GET  /check/reports/{id}   - Get report
```

## Project Structure

```
uyarai/
├── run_system.py          # Main CLI
├── api.py                 # API server
├── stats.py               # Stats & debug
├── ingestion/             # Index & download
├── extraction/            # PDF extraction
├── detection/             # Plagiarism detection
├── storage/               # Wasabi & Qdrant
├── config/                # Settings
└── data/                  # Data directory
```

## 16 Groups

| # | Group | Sample Categories |
|---|-------|-------------------|
| 1 | Core Molecular Biology | biochemistry, genetics, genomics |
| 2 | Cancer & Oncology | cancer biology, oncology |
| 3 | Immunology & Infectious | immunology, microbiology |
| 4 | Neuroscience & Psychiatry | neuroscience, neurology |
| 5 | Cardiovascular & Respiratory | cardiovascular, respiratory |
| 6 | Computational Biology | bioinformatics, systems biology |
| 7 | Genetics & Genomic Medicine | genetic medicine |
| 8 | Internal Medicine | gastroenterology, nephrology |
| 9 | Surgery & Specialties | surgery, orthopedics |
| 10 | Women's & Children's Health | pediatrics, obstetrics |
| 11 | Public Health | epidemiology, nutrition |
| 12 | Pharmacology | pharmacology, toxicology |
| 13 | Emergency & Critical Care | emergency, ICU |
| 14 | Health Systems | health policy, education |
| 15 | Ecology & Evolution | ecology, zoology |
| 16 | Other Specialties | radiology, sports medicine |
