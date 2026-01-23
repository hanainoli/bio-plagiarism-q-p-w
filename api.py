#!/usr/bin/env python3
"""
UyarAI API Server
==================
Clean API with metrics and plagiarism check only.

Endpoints:
    GET  /                      - Health check
    GET  /metrics               - All metrics
    GET  /metrics/index         - Index stats
    GET  /metrics/qdrant        - Qdrant stats  
    GET  /metrics/groups        - Groups with paper counts
    GET  /metrics/categories    - Categories with paper counts
    
    POST /check                 - Upload PDF and check plagiarism
    GET  /check/reports         - List reports
    GET  /check/reports/{id}    - Get specific report

Usage:
    python api.py
    # Swagger: http://localhost:8000/docs
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# GROUPS DEFINITION
# =============================================================================

GROUPS = {
    "core_molecular_biology": {
        "name": "Core Molecular Biology",
        "priority": 1,
        "categories": ["biochemistry", "molecular biology", "cell biology", "genetics", "genomics"]
    },
    "cancer_oncology": {
        "name": "Cancer & Oncology",
        "priority": 2,
        "categories": ["cancer biology", "oncology", "hematology", "pathology"]
    },
    "immunology_infectious": {
        "name": "Immunology & Infectious Disease",
        "priority": 3,
        "categories": ["immunology", "allergy and immunology", "infectious diseases", "hiv/aids", "microbiology"]
    },
    "neuroscience_psychiatry": {
        "name": "Neuroscience & Psychiatry",
        "priority": 4,
        "categories": ["neuroscience", "neurology", "psychiatry and clinical psychology", "pain medicine"]
    },
    "cardiovascular_respiratory": {
        "name": "Cardiovascular & Respiratory",
        "priority": 5,
        "categories": ["cardiovascular medicine", "respiratory medicine", "physiology"]
    },
    "computational_biology": {
        "name": "Computational Biology",
        "priority": 6,
        "categories": ["bioinformatics", "systems biology", "synthetic biology", "bioengineering", "health informatics"]
    },
    "genetics_genomic_medicine": {
        "name": "Genetics & Genomic Medicine",
        "priority": 7,
        "categories": ["genetic and genomic medicine", "developmental biology", "biophysics"]
    },
    "internal_medicine": {
        "name": "Internal Medicine",
        "priority": 8,
        "categories": ["gastroenterology", "endocrinology", "nephrology", "rheumatology"]
    },
    "surgery_specialties": {
        "name": "Surgery & Specialties",
        "priority": 9,
        "categories": ["surgery", "orthopedics", "ophthalmology", "otolaryngology", "urology", "dermatology", "transplantation", "dentistry and oral medicine"]
    },
    "womens_childrens_health": {
        "name": "Women's & Children's Health",
        "priority": 10,
        "categories": ["obstetrics and gynecology", "pediatrics", "geriatric medicine"]
    },
    "public_health": {
        "name": "Public Health",
        "priority": 11,
        "categories": ["epidemiology", "public and global health", "occupational and environmental health", "nutrition", "clinical trials"]
    },
    "pharmacology": {
        "name": "Pharmacology",
        "priority": 12,
        "categories": ["pharmacology and toxicology", "pharmacology and therapeutics", "toxicology", "addiction medicine"]
    },
    "emergency_critical_care": {
        "name": "Emergency & Critical Care",
        "priority": 13,
        "categories": ["emergency medicine", "intensive care and critical care medicine", "anesthesia", "palliative medicine"]
    },
    "health_systems": {
        "name": "Health Systems",
        "priority": 14,
        "categories": ["health economics", "health policy", "health systems and quality improvement", "medical education", "medical ethics", "primary care research", "nursing", "forensic medicine"]
    },
    "ecology_evolution": {
        "name": "Ecology & Evolution",
        "priority": 15,
        "categories": ["ecology", "evolutionary biology", "animal behavior and cognition", "zoology", "plant biology", "paleontology", "scientific communication and education"]
    },
    "other_specialties": {
        "name": "Other Specialties",
        "priority": 16,
        "categories": ["radiology and imaging", "rehabilitation medicine and physical therapy", "sports medicine"]
    },
}

# Category → Group mapping
CATEGORY_TO_GROUP = {}
for group_id, info in GROUPS.items():
    for cat in info["categories"]:
        CATEGORY_TO_GROUP[cat.lower()] = group_id

# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="UyarAI Plagiarism Detection API",
    description="Biomedical plagiarism detection - Metrics & Check endpoints",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPORTS_DIR = Path("data/reports")
UPLOADS_DIR = Path("data/uploads")

# =============================================================================
# METRICS HELPERS
# =============================================================================

def get_index_metrics() -> Dict:
    """Get index statistics"""
    try:
        from ingestion.category_index import CategoryIndex, DEFAULT_INDEX_PATH
        
        if not DEFAULT_INDEX_PATH.exists():
            # Try Wasabi
            index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=True)
        else:
            index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=False)
        
        if index:
            stats = index.get_stats()
            return {
                "status": "available",
                "total_papers": stats.get("total_papers", 0),
                "total_categories": stats.get("total_categories", 0),
                "last_updated": stats.get("last_updated", ""),
                "by_category": stats.get("papers_by_category", {})
            }
    except Exception as e:
        return {"status": "error", "error": str(e), "total_papers": 0}
    
    return {"status": "not_found", "total_papers": 0}


def get_qdrant_metrics() -> Dict:
    """Get Qdrant statistics"""
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = ['bio_fulltext_chunks', 'bio_figures', 'bio_tables', 'bio_abstracts']
        metrics = {"status": "connected", "collections": {}, "total_vectors": 0}
        
        for col in collections:
            try:
                info = client.get_collection(col)
                count = info.points_count or 0
                metrics["collections"][col] = count
                metrics["total_vectors"] += count
            except:
                metrics["collections"][col] = 0
        
        return metrics
    except Exception as e:
        return {"status": "disconnected", "error": str(e), "total_vectors": 0}


def get_download_stats() -> Dict:
    """Get download statistics"""
    pdfs_dir = Path("data/pdfs")
    stats = {"total_downloaded": 0, "by_category": {}}
    
    if pdfs_dir.exists():
        for cat_dir in pdfs_dir.iterdir():
            if cat_dir.is_dir():
                count = len(list(cat_dir.glob("*.pdf")))
                stats["by_category"][cat_dir.name] = count
                stats["total_downloaded"] += count
    
    return stats


# =============================================================================
# ENDPOINTS: HEALTH
# =============================================================================

@app.get("/", tags=["Health"])
async def health():
    """Health check"""
    return {
        "status": "running",
        "name": "UyarAI Plagiarism Detection API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# ENDPOINTS: METRICS
# =============================================================================

@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """Get all system metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "index": get_index_metrics(),
        "qdrant": get_qdrant_metrics(),
        "downloads": get_download_stats()
    }


@app.get("/metrics/index", tags=["Metrics"])
async def metrics_index():
    """Get index metrics"""
    return get_index_metrics()


@app.get("/metrics/qdrant", tags=["Metrics"])
async def metrics_qdrant():
    """Get Qdrant metrics"""
    return get_qdrant_metrics()


@app.get("/metrics/groups", tags=["Metrics"])
async def metrics_groups():
    """Get all groups with paper counts"""
    index_stats = get_index_metrics()
    by_category = index_stats.get("by_category", {})
    
    groups_list = []
    for group_id in sorted(GROUPS.keys(), key=lambda x: GROUPS[x]["priority"]):
        info = GROUPS[group_id]
        paper_count = sum(by_category.get(cat.lower(), 0) for cat in info["categories"])
        
        groups_list.append({
            "id": group_id,
            "name": info["name"],
            "priority": info["priority"],
            "categories": info["categories"],
            "paper_count": paper_count
        })
    
    return {"total_groups": len(groups_list), "groups": groups_list}


@app.get("/metrics/categories", tags=["Metrics"])
async def metrics_categories():
    """Get all categories with paper counts"""
    index_stats = get_index_metrics()
    by_category = index_stats.get("by_category", {})
    
    categories_list = []
    for category, group_id in CATEGORY_TO_GROUP.items():
        categories_list.append({
            "category": category,
            "group_id": group_id,
            "group_name": GROUPS[group_id]["name"],
            "paper_count": by_category.get(category, 0)
        })
    
    categories_list.sort(key=lambda x: -x["paper_count"])
    
    return {"total_categories": len(categories_list), "categories": categories_list}


# =============================================================================
# ENDPOINTS: PLAGIARISM CHECK
# =============================================================================

@app.post("/check", tags=["Plagiarism Check"])
async def check_paper(file: UploadFile = File(..., description="PDF to check")):
    """Upload PDF and check for plagiarism"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    # Save file
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOADS_DIR / f"{timestamp}_{file.filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Run check
    start = time.time()
    
    try:
        from detection.world_class_detector import WorldClassPlagiarismDetector
        
        detector = WorldClassPlagiarismDetector()
        report = detector.check_document(str(file_path))
    except Exception as e:
        report = {"error": str(e), "overall_score": 0, "risk_level": "ERROR"}
    
    elapsed = time.time() - start
    
    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{timestamp}_report.json"
    
    report["filename"] = file.filename
    report["checked_at"] = datetime.now().isoformat()
    report["check_time"] = round(elapsed, 2)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    summary = report.get("summary", report)
    
    return {
        "report_id": timestamp,
        "filename": file.filename,
        "check_time_seconds": round(elapsed, 2),
        "overall_score": summary.get("overall_score", 0),
        "risk_level": summary.get("risk_level", "UNKNOWN"),
        "text_matches": summary.get("total_text_matches", 0),
        "image_matches": summary.get("total_image_matches", 0)
    }


@app.get("/check/reports", tags=["Plagiarism Check"])
async def list_reports(limit: int = Query(default=20, le=100)):
    """List recent reports"""
    if not REPORTS_DIR.exists():
        return {"total": 0, "reports": []}
    
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            summary = data.get("summary", data)
            reports.append({
                "report_id": f.stem.replace("_report", ""),
                "filename": data.get("filename"),
                "overall_score": summary.get("overall_score", 0),
                "risk_level": summary.get("risk_level"),
                "checked_at": data.get("checked_at")
            })
        except:
            pass
    
    return {"total": len(reports), "reports": reports}


@app.get("/check/reports/{report_id}", tags=["Plagiarism Check"])
async def get_report(report_id: str):
    """Get specific report"""
    for f in REPORTS_DIR.glob(f"*{report_id}*.json"):
        with open(f) as fp:
            return json.load(fp)
    
    raise HTTPException(status_code=404, detail="Report not found")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          UyarAI Plagiarism Detection API                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  METRICS:                                                ║
    ║    GET /metrics              - All metrics               ║
    ║    GET /metrics/index        - Index stats               ║
    ║    GET /metrics/qdrant       - Qdrant stats              ║
    ║    GET /metrics/groups       - Groups with counts        ║
    ║    GET /metrics/categories   - Categories with counts    ║
    ║                                                          ║
    ║  PLAGIARISM CHECK:                                       ║
    ║    POST /check               - Upload PDF and check      ║
    ║    GET  /check/reports       - List reports              ║
    ║    GET  /check/reports/{id}  - Get report                ║
    ║                                                          ║
    ║  Swagger: http://localhost:8000/docs                     ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
