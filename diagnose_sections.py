#!/usr/bin/env python3
"""
Diagnostic script to check what text sections are being extracted and filtered.
Run: python diagnose_sections.py <pdf_file>
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def diagnose_pdf(pdf_path: str):
    """Analyze what sections are extracted from a PDF"""
    
    print("="*70)
    print("SECTION EXTRACTION DIAGNOSTIC")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print()
    
    # Step 1: Extract raw text
    print("[1] EXTRACTING RAW TEXT...")
    try:
        from extraction.pdf_extractor import PDFExtractor
        extractor = PDFExtractor()
        extraction = extractor.extract(pdf_path)
        
        full_text = getattr(extraction, 'full_text', '') or getattr(extraction, 'text', '')
        print(f"    Raw text length: {len(full_text)} characters")
        print(f"    Raw text words: {len(full_text.split())} words")
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    # Step 2: Check for key sections in raw text
    print("\n[2] CHECKING FOR KEY SECTIONS IN RAW TEXT...")
    sections_to_find = [
        ('Abstract', r'(?i)\babstract\b'),
        ('Introduction', r'(?i)\bintroduction\b'),
        ('Methods', r'(?i)\b(methods?|materials?\s+and\s+methods?)\b'),
        ('Results', r'(?i)\bresults?\b'),
        ('Discussion', r'(?i)\bdiscussion\b'),
        ('Conclusion', r'(?i)\bconclusions?\b'),
        ('References', r'(?i)\breferences?\b'),
        ('Acknowledgements', r'(?i)\backnowledge?ments?\b'),
    ]
    
    import re
    for section_name, pattern in sections_to_find:
        matches = list(re.finditer(pattern, full_text))
        if matches:
            first_pos = matches[0].start()
            # Find position as percentage of document
            pct = (first_pos / len(full_text)) * 100
            print(f"    ✓ {section_name}: Found at position {first_pos} ({pct:.1f}% into doc), {len(matches)} occurrences")
        else:
            print(f"    ✗ {section_name}: NOT FOUND")
    
    # Step 3: Apply text filter
    print("\n[3] APPLYING TEXT FILTER...")
    try:
        from utils.text_filter import TextSectionFilter
        filter = TextSectionFilter()
        result = filter.filter_text(full_text)
        
        print(f"    Original length: {result.original_length} chars")
        print(f"    Filtered length: {result.filtered_length} chars")
        print(f"    Removed: {result.removal_percentage}%")
        print(f"    Removed sections: {result.removed_sections}")
        
        filtered_text = result.filtered_text
    except Exception as e:
        print(f"    ERROR: {e}")
        filtered_text = full_text
    
    # Step 4: Check what's in filtered text
    print("\n[4] CHECKING FILTERED TEXT FOR SECTIONS...")
    for section_name, pattern in sections_to_find:
        matches = list(re.finditer(pattern, filtered_text))
        if matches:
            first_pos = matches[0].start()
            pct = (first_pos / len(filtered_text)) * 100 if filtered_text else 0
            print(f"    ✓ {section_name}: Found at {pct:.1f}% into filtered text")
        else:
            print(f"    ✗ {section_name}: NOT FOUND in filtered text")
    
    # Step 5: Show last 2000 chars of filtered text
    print("\n[5] LAST 2000 CHARS OF FILTERED TEXT:")
    print("-"*70)
    print(filtered_text[-2000:] if len(filtered_text) > 2000 else filtered_text)
    print("-"*70)
    
    # Step 6: Chunk the text
    print("\n[6] CHUNKING TEXT...")
    try:
        from extraction.pdf_extractor import TextChunker
        chunker = TextChunker(chunk_size=200, overlap=0.5)
        chunks = chunker.chunk_text(full_text, "body")
        
        print(f"    Total chunks created: {len(chunks)}")
        
        # Check which chunks contain Discussion content
        discussion_chunks = []
        for i, chunk in enumerate(chunks):
            if re.search(r'(?i)\bdiscussion\b', chunk['text']):
                discussion_chunks.append(i)
        
        if discussion_chunks:
            print(f"    Chunks containing 'Discussion': {discussion_chunks}")
        else:
            print(f"    ✗ NO chunks contain the word 'Discussion'")
        
        # Show last 3 chunks
        print("\n    LAST 3 CHUNKS:")
        for chunk in chunks[-3:]:
            print(f"    --- Chunk {chunk['chunk_index']} ({len(chunk['text'].split())} words) ---")
            print(f"    {chunk['text'][:300]}...")
            print()
            
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Check the order of sections in original text
    print("\n[7] SECTION ORDER IN ORIGINAL TEXT:")
    section_positions = []
    for section_name, pattern in sections_to_find:
        match = re.search(pattern, full_text)
        if match:
            section_positions.append((match.start(), section_name))
    
    section_positions.sort(key=lambda x: x[0])
    for pos, name in section_positions:
        pct = (pos / len(full_text)) * 100
        print(f"    {pct:5.1f}% - {name}")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_sections.py <pdf_file>")
        sys.exit(1)
    
    diagnose_pdf(sys.argv[1])
