#!/usr/bin/env python3
"""
UyarAI Stats & Debug CLI
=========================
Check system status, view stats, debug issues.

Usage:
    python stats.py                    # Full system status
    python stats.py --index            # Index stats only
    python stats.py --qdrant           # Qdrant stats only
    python stats.py --groups           # Groups breakdown
    python stats.py --category oncology # Single category detail
    python stats.py --downloads        # Download progress
    python stats.py --check-config     # Verify configuration
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# GROUPS
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

CATEGORY_TO_GROUP = {}
for gid, info in GROUPS.items():
    for cat in info["categories"]:
        CATEGORY_TO_GROUP[cat.lower()] = gid

# =============================================================================
# STATS FUNCTIONS
# =============================================================================

def check_config():
    """Check configuration"""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION CHECK")
    print("="*60)
    
    # Wasabi
    print("\n📦 Wasabi S3:")
    wasabi_vars = ['WASABI_ENDPOINT', 'WASABI_ACCESS_KEY', 'WASABI_SECRET_KEY', 'WASABI_BUCKET']
    for var in wasabi_vars:
        val = os.getenv(var)
        status = "✅" if val else "❌"
        display = val[:20] + "..." if val and len(val) > 20 else val or "NOT SET"
        print(f"   {status} {var}: {display}")
    
    # Qdrant
    print("\n🗄️ Qdrant:")
    qdrant_url = os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    print(f"   {'✅' if qdrant_url else '❌'} QDRANT_URL: {qdrant_url or 'NOT SET'}")
    print(f"   {'✅' if qdrant_key else '❌'} QDRANT_API_KEY: {'***' if qdrant_key else 'NOT SET'}")
    
    # Test connections
    print("\n🔌 Connection Tests:")
    
    # Wasabi
    try:
        from storage.wasabi_client import WasabiClient
        wasabi = WasabiClient()
        if wasabi.client:
            print("   ✅ Wasabi: Connected")
        else:
            print("   ❌ Wasabi: Not connected")
    except Exception as e:
        print(f"   ❌ Wasabi: {e}")
    
    # Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        collections = client.get_collections()
        print(f"   ✅ Qdrant: Connected ({len(collections.collections)} collections)")
    except Exception as e:
        print(f"   ❌ Qdrant: {e}")


def get_index_stats():
    """Get index statistics"""
    print("\n" + "="*60)
    print("📊 INDEX STATISTICS")
    print("="*60)
    
    try:
        from ingestion.category_index import CategoryIndex, DEFAULT_INDEX_PATH
        
        if not DEFAULT_INDEX_PATH.exists():
            print("\n   ⚠️ Local index not found, trying Wasabi...")
            index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=True)
        else:
            index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=False)
        
        if not index or not index.papers:
            print("\n   ❌ No index available")
            return
        
        stats = index.get_stats()
        
        print(f"\n   Total Papers:     {stats.get('total_papers', 0):,}")
        print(f"   Total Categories: {stats.get('total_categories', 0)}")
        print(f"   Last Updated:     {stats.get('last_updated', 'N/A')}")
        
        by_cat = stats.get("papers_by_category", {})
        
        print(f"\n   📂 Top 15 Categories:")
        for cat, count in list(by_cat.items())[:15]:
            print(f"      {cat}: {count:,}")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")


def get_qdrant_stats():
    """Get Qdrant statistics"""
    print("\n" + "="*60)
    print("🗄️ QDRANT STATISTICS")
    print("="*60)
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL') or os.getenv('QDRANT_HOST'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = ['bio_fulltext_chunks', 'bio_figures', 'bio_tables', 'bio_abstracts']
        total = 0
        
        print(f"\n   Collections:")
        for col in collections:
            try:
                info = client.get_collection(col)
                count = info.points_count or 0
                total += count
                print(f"      {col}: {count:,} vectors")
            except:
                print(f"      {col}: ❌ Not found")
        
        print(f"\n   Total Vectors: {total:,}")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")


def get_groups_stats():
    """Get groups breakdown"""
    print("\n" + "="*60)
    print("📁 GROUPS BREAKDOWN")
    print("="*60)
    
    try:
        from ingestion.category_index import CategoryIndex, DEFAULT_INDEX_PATH
        index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=True)
        
        if not index:
            print("\n   ❌ No index available")
            return
        
        stats = index.get_stats()
        by_cat = stats.get("papers_by_category", {})
        
        for gid in sorted(GROUPS.keys(), key=lambda x: GROUPS[x]["priority"]):
            info = GROUPS[gid]
            total = sum(by_cat.get(cat.lower(), 0) for cat in info["categories"])
            
            print(f"\n   {info['priority']:2d}. {info['name']}")
            print(f"       Total: {total:,} papers")
            print(f"       Categories:")
            for cat in info["categories"]:
                count = by_cat.get(cat.lower(), 0)
                if count > 0:
                    print(f"         - {cat}: {count:,}")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")


def get_category_detail(category: str):
    """Get detailed stats for a category"""
    print("\n" + "="*60)
    print(f"📂 CATEGORY: {category}")
    print("="*60)
    
    try:
        from ingestion.category_index import CategoryIndex, DEFAULT_INDEX_PATH
        index = CategoryIndex.load(DEFAULT_INDEX_PATH, try_wasabi=True)
        
        if not index:
            print("\n   ❌ No index available")
            return
        
        # Papers in index
        papers = [p for p in index.papers.values() if p.get("category", "").lower() == category.lower()]
        print(f"\n   Papers in Index: {len(papers):,}")
        
        # Group
        group_id = CATEGORY_TO_GROUP.get(category.lower())
        if group_id:
            print(f"   Group: {GROUPS[group_id]['name']}")
        
        # Downloaded PDFs
        pdfs_dir = Path("data/pdfs") / category.lower().replace(" ", "_")
        if pdfs_dir.exists():
            pdf_count = len(list(pdfs_dir.glob("*.pdf")))
            print(f"   Downloaded: {pdf_count:,}")
        else:
            print(f"   Downloaded: 0")
        
        # Extracted
        extracted_dir = Path("data/extracted")
        extracted = 0
        if extracted_dir.exists():
            for p in papers[:1000]:  # Sample
                doi = p.get("doi", "").replace("/", "_")
                if (extracted_dir / doi).exists():
                    extracted += 1
        print(f"   Extracted (sample): {extracted}")
        
        # Sample papers
        if papers:
            print(f"\n   Sample Papers (5):")
            for p in papers[:5]:
                print(f"      - {p.get('doi')}: {p.get('title', '')[:60]}...")
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")


def get_download_stats():
    """Get download progress"""
    print("\n" + "="*60)
    print("📥 DOWNLOAD PROGRESS")
    print("="*60)
    
    pdfs_dir = Path("data/pdfs")
    
    if not pdfs_dir.exists():
        print("\n   No downloads yet")
        return
    
    total = 0
    by_category = {}
    
    for cat_dir in sorted(pdfs_dir.iterdir()):
        if cat_dir.is_dir():
            count = len(list(cat_dir.glob("*.pdf")))
            by_category[cat_dir.name] = count
            total += count
    
    print(f"\n   Total Downloaded: {total:,}")
    
    if by_category:
        print(f"\n   By Category:")
        for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f"      {cat}: {count:,}")


def full_status():
    """Show full system status"""
    print("\n" + "="*60)
    print("🔍 UYARAI SYSTEM STATUS")
    print("="*60)
    print(f"   Timestamp: {datetime.now().isoformat()}")
    
    check_config()
    get_index_stats()
    get_qdrant_stats()
    get_download_stats()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UyarAI Stats & Debug")
    parser.add_argument("--index", action="store_true", help="Index stats")
    parser.add_argument("--qdrant", action="store_true", help="Qdrant stats")
    parser.add_argument("--groups", action="store_true", help="Groups breakdown")
    parser.add_argument("--category", type=str, help="Category detail")
    parser.add_argument("--downloads", action="store_true", help="Download progress")
    parser.add_argument("--check-config", action="store_true", help="Check configuration")
    
    args = parser.parse_args()
    
    if args.index:
        get_index_stats()
    elif args.qdrant:
        get_qdrant_stats()
    elif args.groups:
        get_groups_stats()
    elif args.category:
        get_category_detail(args.category)
    elif args.downloads:
        get_download_stats()
    elif args.check_config:
        check_config()
    else:
        full_status()
