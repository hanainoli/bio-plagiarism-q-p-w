#!/usr/bin/env python3
"""
Diagnostic script to check what's in Qdrant for a paper and compare with extraction.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def check_qdrant_content(doi: str):
    """Check what chunks exist in Qdrant for a DOI"""
    print("="*70)
    print(f"CHECKING QDRANT CONTENT FOR: {doi}")
    print("="*70)
    
    from storage.qdrant_client import QdrantClient
    qdrant = QdrantClient()
    
    # Search in text_embeddings collection
    print("\n[1] CHECKING text_embeddings COLLECTION...")
    try:
        # Scroll through all points with this DOI
        chunks = []
        offset = None
        
        while True:
            result = qdrant.client.scroll(
                collection_name="text_embeddings",
                scroll_filter={
                    "must": [
                        {"key": "doi", "match": {"value": doi}}
                    ]
                },
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, offset = result
            for point in points:
                chunks.append(point.payload)
            
            if offset is None:
                break
        
        print(f"    Found {len(chunks)} chunks in Qdrant")
        
        if chunks:
            # Sort by chunk_index if available
            chunks.sort(key=lambda x: x.get('chunk_index', 0))
            
            # Show first 3 chunks
            print("\n    FIRST 3 CHUNKS:")
            for i, chunk in enumerate(chunks[:3]):
                text = chunk.get('text', '')[:200]
                print(f"    --- Chunk {chunk.get('chunk_index', i)} ---")
                print(f"    {text}...")
                print()
            
            # Show last 3 chunks
            print("\n    LAST 3 CHUNKS:")
            for chunk in chunks[-3:]:
                text = chunk.get('text', '')[:200]
                print(f"    --- Chunk {chunk.get('chunk_index', '?')} ---")
                print(f"    {text}...")
                print()
            
            # Check for Discussion content
            print("\n    CHECKING FOR 'Discussion' KEYWORD:")
            discussion_chunks = []
            for chunk in chunks:
                text = chunk.get('text', '')
                if re.search(r'\bdiscussion\b', text, re.IGNORECASE):
                    discussion_chunks.append(chunk)
            
            if discussion_chunks:
                print(f"    ✓ Found {len(discussion_chunks)} chunks containing 'Discussion'")
                for chunk in discussion_chunks[:2]:
                    text = chunk.get('text', '')[:300]
                    print(f"      Chunk {chunk.get('chunk_index', '?')}: {text}...")
            else:
                print(f"    ✗ NO chunks contain 'Discussion' keyword!")
                
            # Check what sections are present
            print("\n    SECTION KEYWORDS FOUND:")
            keywords = ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion', 'reference']
            for kw in keywords:
                count = sum(1 for c in chunks if re.search(rf'\b{kw}\b', c.get('text', ''), re.IGNORECASE))
                status = "✓" if count > 0 else "✗"
                print(f"      {status} {kw.title()}: {count} chunks")
                
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()


def check_wasabi_extraction(doi: str):
    """Check the extraction JSON in Wasabi"""
    print("\n" + "="*70)
    print(f"CHECKING WASABI EXTRACTION FOR: {doi}")
    print("="*70)
    
    from storage.wasabi_client import WasabiClient
    import json
    
    wasabi = WasabiClient()
    
    # Try to find the extraction JSON
    possible_paths = [
        f"extractions/biorxiv/April_2019/{doi}.json",
        f"extractions/medrxiv/April_2019/{doi}.json",
    ]
    
    for path in possible_paths:
        try:
            print(f"\n    Trying: {path}")
            response = wasabi.client.get_object(Bucket=wasabi.bucket, Key=path)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            print(f"    ✓ Found extraction JSON!")
            
            full_text = data.get('full_text', '') or data.get('text', '')
            print(f"    Full text length: {len(full_text)} chars, {len(full_text.split())} words")
            
            # Check for sections
            print("\n    SECTION KEYWORDS IN FULL TEXT:")
            keywords = ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion', 'reference']
            for kw in keywords:
                matches = list(re.finditer(rf'\b{kw}\b', full_text, re.IGNORECASE))
                if matches:
                    first_pos = matches[0].start()
                    pct = (first_pos / len(full_text)) * 100 if full_text else 0
                    print(f"      ✓ {kw.title()}: {len(matches)} occurrences, first at {pct:.1f}%")
                else:
                    print(f"      ✗ {kw.title()}: NOT FOUND")
            
            # Show last 1000 chars
            print("\n    LAST 1000 CHARS OF FULL TEXT:")
            print("-"*60)
            print(full_text[-1000:] if len(full_text) > 1000 else full_text)
            print("-"*60)
            
            return data
            
        except Exception as e:
            print(f"    Not found: {e}")
    
    print("    ✗ Could not find extraction JSON!")
    return None


def check_query_extraction(pdf_path: str):
    """Check what gets extracted from a query PDF"""
    print("\n" + "="*70)
    print(f"CHECKING QUERY PDF EXTRACTION: {pdf_path}")
    print("="*70)
    
    from extraction.pdf_extractor import PDFExtractor, TextChunker
    
    extractor = PDFExtractor()
    extraction = extractor.extract(pdf_path)
    
    full_text = getattr(extraction, 'full_text', '') or getattr(extraction, 'text', '')
    print(f"\n    Raw text length: {len(full_text)} chars, {len(full_text.split())} words")
    
    # Check for sections in raw text
    print("\n    SECTION KEYWORDS IN RAW TEXT:")
    keywords = ['abstract', 'introduction', 'method', 'result', 'discussion', 'conclusion', 'reference']
    for kw in keywords:
        matches = list(re.finditer(rf'\b{kw}\b', full_text, re.IGNORECASE))
        if matches:
            first_pos = matches[0].start()
            pct = (first_pos / len(full_text)) * 100 if full_text else 0
            print(f"      ✓ {kw.title()}: {len(matches)} occurrences, first at {pct:.1f}%")
        else:
            print(f"      ✗ {kw.title()}: NOT FOUND")
    
    # Now apply the filter and chunk
    print("\n    APPLYING TEXT FILTER AND CHUNKING...")
    chunker = TextChunker(chunk_size=200, overlap=0.5, filter_sections=True)
    chunks = chunker.chunk_text(full_text, "body")
    
    print(f"    Created {len(chunks)} chunks after filtering")
    
    # Check for Discussion in chunks
    discussion_chunks = []
    for chunk in chunks:
        if re.search(r'\bdiscussion\b', chunk['text'], re.IGNORECASE):
            discussion_chunks.append(chunk)
    
    if discussion_chunks:
        print(f"    ✓ {len(discussion_chunks)} chunks contain 'Discussion'")
    else:
        print(f"    ✗ NO chunks contain 'Discussion' after filtering!")
    
    # Show last 3 chunks
    print("\n    LAST 3 CHUNKS AFTER FILTERING:")
    for chunk in chunks[-3:]:
        text = chunk['text'][:200]
        print(f"    --- Chunk {chunk['chunk_index']} ---")
        print(f"    {text}...")
        print()


if __name__ == "__main__":
    # The DOI from your test
    doi = "uuid_0ef37b7b-6c48-1014-9d11-d0c8d178dc67"
    
    print("\n" + "="*70)
    print("QDRANT & EXTRACTION DIAGNOSTIC")
    print("="*70)
    
    # Check Qdrant content
    check_qdrant_content(doi)
    
    # Check Wasabi extraction
    check_wasabi_extraction(doi)
    
    # If PDF path provided, check query extraction
    if len(sys.argv) > 1:
        check_query_extraction(sys.argv[1])
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
