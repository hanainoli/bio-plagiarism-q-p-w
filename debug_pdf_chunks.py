#!/usr/bin/env python3
"""
Debug: See what text chunks are being extracted from the combined PDF.
"""

import os
import sys

from extraction.pdf_extractor import PDFExtractor

# Check for text chunker in different locations
try:
    from extraction.text_chunker import TextChunker
except ImportError:
    try:
        from detection.world_class_detector import TextChunker
    except ImportError:
        # Define a simple chunker
        class TextChunker:
            def __init__(self, chunk_size=500, overlap=50):
                self.chunk_size = chunk_size
                self.overlap = overlap
            
            def chunk_text(self, text, section="body"):
                chunks = []
                words = text.split()
                chunk_words = 100  # ~500 chars
                overlap_words = 10
                
                i = 0
                while i < len(words):
                    chunk_text = ' '.join(words[i:i+chunk_words])
                    chunks.append({'text': chunk_text, 'section': section})
                    i += chunk_words - overlap_words
                
                return chunks

# Extract from the test PDF
pdf_path = "combined_test_paper.pdf"

if not os.path.exists(pdf_path):
    print(f"PDF not found: {pdf_path}")
    print("Please run from the uyari_updated directory")
    sys.exit(1)

print("Extracting from PDF...")
extractor = PDFExtractor()
extraction = extractor.extract(pdf_path)

full_text = extraction.get('full_text', '')
print(f"\nFull text length: {len(full_text)} chars")

# Chunk the text
chunker = TextChunker()
chunks = chunker.chunk_text(full_text, "body")

print(f"\nTotal chunks: {len(chunks)}")
print("\n" + "="*70)
print("CHUNK ANALYSIS")
print("="*70)

for i, chunk in enumerate(chunks):
    text = chunk['text']
    
    # Check content type
    has_turtle = 'turtle' in text.lower() or 'crocodil' in text.lower()
    has_cggbp1 = 'cggbp1' in text.lower() or 'ctcf' in text.lower()
    
    content_type = "UNKNOWN"
    if has_turtle and has_cggbp1:
        content_type = "⚠️ MIXED (turtle + CGGBP1)"
    elif has_turtle:
        content_type = "🐢 Turtle only"
    elif has_cggbp1:
        content_type = "🧬 CGGBP1 only"
    
    print(f"\nChunk {i}: {content_type}")
    print(f"  Length: {len(text)} chars")
    print(f"  Preview: {text[:100]}...")
    
    if has_turtle and has_cggbp1:
        # This is a mixed chunk - show where each content is
        turtle_pos = min(text.lower().find('turtle'), text.lower().find('crocodil'))
        cggbp1_pos = min(text.lower().find('cggbp1'), text.lower().find('ctcf'))
        if turtle_pos == -1:
            turtle_pos = 9999
        if cggbp1_pos == -1:
            cggbp1_pos = 9999
        print(f"  Turtle content at pos: {turtle_pos}")
        print(f"  CGGBP1 content at pos: {cggbp1_pos}")
