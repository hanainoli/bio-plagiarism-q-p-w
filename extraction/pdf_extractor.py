"""
PDF Extraction Module.
Extracts text, figures, and tables from scientific paper PDFs.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
from io import BytesIO


def clean_text(text: str) -> str:
    """
    Clean text by removing invisible Unicode characters and normalizing whitespace.
    
    Removes:
    - Zero-width spaces (\u200b, \u200c, \u200d)
    - Zero-width non-joiner/joiner
    - Byte order marks (\ufeff)
    - Other invisible formatting characters
    """
    if not text:
        return text
    
    # Remove zero-width and invisible characters
    invisible_chars = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\ufeff',  # Byte order mark
        '\u00ad',  # Soft hyphen
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u180e',  # Mongolian vowel separator
    ]
    
    for char in invisible_chars:
        text = text.replace(char, '')
    
    # Normalize multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Remove excessive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text


@dataclass
class ExtractedFigure:
    """Extracted figure from PDF"""
    image: Image.Image
    page_number: int
    figure_index: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    caption: str = ""
    label: str = ""
    
    @property
    def image_data(self) -> Optional[bytes]:
        """Get image as bytes (for compatibility)"""
        if self.image is None:
            return None
        try:
            buffer = BytesIO()
            self.image.save(buffer, format='PNG')
            return buffer.getvalue()
        except Exception:
            return None


@dataclass
class ExtractedTable:
    """Extracted table from PDF"""
    text: str
    page_number: int
    table_index: int
    bbox: Tuple[float, float, float, float]
    caption: str = ""
    label: str = ""
    rows: List[List[str]] = field(default_factory=list)
    image: Optional[Image.Image] = None  # Table as image for visual comparison
    
    @property
    def image_data(self) -> Optional[bytes]:
        """Get table image as bytes"""
        if self.image is None:
            return None
        try:
            buffer = BytesIO()
            self.image.save(buffer, format='PNG')
            return buffer.getvalue()
        except Exception:
            return None


@dataclass
class ExtractedText:
    """Extracted text section"""
    text: str
    section: str
    page_start: int
    page_end: int


@dataclass
class PDFExtractionResult:
    """Complete PDF extraction result"""
    title: str
    authors: List[str]
    abstract: str
    sections: List[ExtractedText]
    figures: List[ExtractedFigure]
    tables: List[ExtractedTable]
    full_text: str
    page_count: int
    metadata: Dict


class PDFExtractor:
    """Extract content from scientific paper PDFs"""
    
    def __init__(self, min_figure_size: int = 100, dpi: int = 150):
        """
        Initialize PDF extractor.
        
        Args:
            min_figure_size: Minimum dimension for figure extraction
            dpi: DPI for image extraction
        """
        self.min_figure_size = min_figure_size
        self.dpi = dpi
    
    def extract(self, pdf_path: str) -> PDFExtractionResult:
        """
        Extract all content from a PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            PDFExtractionResult with all extracted content
        """
        # Try different extraction methods
        try:
            return self._extract_with_pymupdf(pdf_path)
        except ImportError:
            pass
        
        try:
            return self._extract_with_pdfplumber(pdf_path)
        except ImportError:
            pass
        
        raise ImportError("No PDF library available. Install pymupdf or pdfplumber.")
    
    def _extract_with_pymupdf(self, pdf_path: str) -> PDFExtractionResult:
        """Extract using PyMuPDF (fitz)"""
        import fitz
        
        doc = fitz.open(pdf_path)
        
        # Extract text
        full_text = ""
        page_texts = []
        
        for page in doc:
            text = page.get_text()
            text = clean_text(text)  # Clean invisible characters
            page_texts.append(text)
            full_text += text + "\n"
        
        # Final clean of full text
        full_text = clean_text(full_text)
        
        # Extract metadata
        metadata = doc.metadata or {}
        title = metadata.get('title', '')
        authors = self._parse_authors(metadata.get('author', ''))
        
        # If no title in metadata, try to extract from first page
        if not title:
            title = self._extract_title_from_text(page_texts[0] if page_texts else "")
        
        # Extract abstract
        abstract = self._extract_abstract(full_text)
        
        # Parse sections
        sections = self._parse_sections(full_text, page_texts)
        
        # Extract figures
        figures = self._extract_figures_pymupdf(doc)
        
        # Extract tables as images (better for plagiarism detection)
        tables = self._extract_tables_as_images_pymupdf(doc)
        
        # If no tables found, fall back to text-based detection
        if not tables:
            tables = self._extract_tables_from_text(full_text, page_texts)
        
        doc.close()
        
        return PDFExtractionResult(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            figures=figures,
            tables=tables,
            full_text=full_text,
            page_count=len(page_texts),
            metadata=metadata
        )
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> PDFExtractionResult:
        """Extract using pdfplumber"""
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            page_texts = []
            
            for page in pdf.pages:
                text = page.extract_text() or ""
                text = clean_text(text)  # Clean invisible characters
                page_texts.append(text)
                full_text += text + "\n"
            
            # Final clean of full text
            full_text = clean_text(full_text)
            
            # Extract metadata
            metadata = pdf.metadata or {}
            title = metadata.get('Title', '')
            authors = self._parse_authors(metadata.get('Author', ''))
            
            if not title:
                title = self._extract_title_from_text(page_texts[0] if page_texts else "")
            
            abstract = self._extract_abstract(full_text)
            sections = self._parse_sections(full_text, page_texts)
            
            # Extract figures (limited with pdfplumber)
            figures = self._extract_figures_pdfplumber(pdf)
            
            # Extract tables
            tables = self._extract_tables_pdfplumber(pdf)
            
            return PDFExtractionResult(
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                figures=figures,
                tables=tables,
                full_text=full_text,
                page_count=len(page_texts),
                metadata=metadata
            )
    
    # =========================================================================
    # TEXT EXTRACTION HELPERS
    # =========================================================================
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of names"""
        if not author_string:
            return []
        
        # Split by common separators
        separators = [';', ',', ' and ', '&']
        authors = [author_string]
        
        for sep in separators:
            new_authors = []
            for a in authors:
                new_authors.extend(a.split(sep))
            authors = new_authors
        
        # Clean up
        authors = [a.strip() for a in authors if a.strip()]
        
        return authors
    
    def _extract_title_from_text(self, first_page: str) -> str:
        """Extract title from first page text"""
        lines = first_page.strip().split('\n')
        
        # Title is usually in first few lines, often the longest
        candidates = []
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if len(line) > 20 and len(line) < 300:
                if not any(kw in line.lower() for kw in ['abstract', 'introduction', 'keywords', 'doi:', 'http']):
                    candidates.append((len(line), line))
        
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        return ""
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text"""
        # Look for abstract section
        patterns = [
            r'(?:^|\n)\s*Abstract\s*\n(.*?)(?=\n\s*(?:Introduction|Keywords|1\.|Background))',
            r'(?:^|\n)\s*ABSTRACT\s*\n(.*?)(?=\n\s*(?:INTRODUCTION|KEYWORDS|1\.))',
            r'(?:^|\n)\s*Summary\s*\n(.*?)(?=\n\s*(?:Introduction|Keywords|1\.))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:3000]  # Limit length
        
        return ""
    
    def _parse_sections(self, full_text: str, page_texts: List[str]) -> List[ExtractedText]:
        """Parse text into sections"""
        sections = []
        
        # Common section headers
        section_patterns = [
            r'(?:^|\n)\s*(Abstract)\s*\n',
            r'(?:^|\n)\s*(Introduction)\s*\n',
            r'(?:^|\n)\s*(Background)\s*\n',
            r'(?:^|\n)\s*(Methods?|Materials?\s+and\s+Methods?)\s*\n',
            r'(?:^|\n)\s*(Results?)\s*\n',
            r'(?:^|\n)\s*(Discussion)\s*\n',
            r'(?:^|\n)\s*(Conclusions?)\s*\n',
            r'(?:^|\n)\s*(References?)\s*\n',
            r'(?:^|\n)\s*(\d+\.?\s+\w+)\s*\n',  # Numbered sections
        ]
        
        # Find all section boundaries
        boundaries = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                boundaries.append((match.start(), match.group(1).strip()))
        
        # Sort by position
        boundaries.sort(key=lambda x: x[0])
        
        # Extract sections
        for i, (start, section_name) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            
            section_text = full_text[start:end].strip()
            
            # Remove section header from text
            section_text = re.sub(r'^' + re.escape(section_name) + r'\s*', '', section_text)
            
            sections.append(ExtractedText(
                text=section_text,
                section=section_name,
                page_start=self._find_page(start, page_texts, full_text),
                page_end=self._find_page(end, page_texts, full_text)
            ))
        
        return sections
    
    def _find_page(self, position: int, page_texts: List[str], full_text: str) -> int:
        """Find page number for a position in full text"""
        current_pos = 0
        for i, page_text in enumerate(page_texts):
            current_pos += len(page_text) + 1  # +1 for newline
            if current_pos >= position:
                return i + 1
        return len(page_texts)
    
    # =========================================================================
    # FIGURE EXTRACTION
    # =========================================================================
    
    def _extract_figures_pymupdf(self, doc) -> List[ExtractedFigure]:
        """Extract figures using PyMuPDF"""
        import fitz
        
        figures = []
        
        for page_num, page in enumerate(doc):
            images = page.get_images()
            
            for img_idx, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if base_image:
                        image_bytes = base_image["image"]
                        pil_image = Image.open(BytesIO(image_bytes))
                        
                        # Filter small images
                        if pil_image.size[0] >= self.min_figure_size and pil_image.size[1] >= self.min_figure_size:
                            # Get image position
                            img_rects = page.get_image_rects(xref)
                            bbox = img_rects[0] if img_rects else (0, 0, pil_image.size[0], pil_image.size[1])
                            
                            figures.append(ExtractedFigure(
                                image=pil_image,
                                page_number=page_num + 1,
                                figure_index=len([f for f in figures if f.page_number == page_num + 1]) + 1,
                                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                caption=self._find_caption_near(page.get_text(), bbox),
                                label=f"Figure {len(figures) + 1}"
                            ))
                
                except Exception as e:
                    print(f"Error extracting image: {e}")
                    continue
        
        return figures
    
    def _extract_figures_pdfplumber(self, pdf) -> List[ExtractedFigure]:
        """Extract figures using pdfplumber (limited)"""
        figures = []
        
        for page_num, page in enumerate(pdf.pages):
            # pdfplumber can render pages as images
            try:
                page_image = page.to_image(resolution=self.dpi)
                pil_image = page_image.original
                
                # This gets the whole page - would need more processing
                # to extract individual figures
                
            except Exception as e:
                print(f"Error rendering page: {e}")
        
        return figures
    
    def _find_caption_near(self, page_text: str, bbox: Tuple) -> str:
        """Find figure caption near the image position"""
        # Look for "Figure X" or "Fig. X" patterns
        patterns = [
            r'(?:Figure|Fig\.?)\s*\d+[.:]?\s*([^.]+\.)',
            r'(?:Figure|Fig\.?)\s*\d+\s*[-–]\s*([^.]+\.)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return ""
    
    # =========================================================================
    # TABLE EXTRACTION
    # =========================================================================
    
    def _extract_tables_as_images_pymupdf(self, doc) -> List[ExtractedTable]:
        """
        Extract REAL tables (with rows/columns) as images using PyMuPDF.
        Uses PyMuPDF's table detection to find actual tabular structures.
        """
        import fitz
        
        tables = []
        
        for page_num, page in enumerate(doc):
            try:
                # Use PyMuPDF's built-in table finder (requires PyMuPDF 1.23+)
                if hasattr(page, 'find_tables'):
                    table_finder = page.find_tables()
                    
                    for idx, table in enumerate(table_finder.tables):
                        try:
                            bbox = table.bbox
                            
                            # Validate it's a real table (has multiple rows/cols)
                            table_data = table.extract()
                            if not table_data or len(table_data) < 2:
                                continue  # Skip if less than 2 rows
                            
                            # Check if it has multiple columns
                            if table_data and len(table_data[0]) < 2:
                                continue  # Skip if less than 2 columns
                            
                            # Render table region as image
                            clip = fitz.Rect(bbox)
                            
                            # Add padding
                            clip.x0 = max(0, clip.x0 - 5)
                            clip.y0 = max(0, clip.y0 - 5)
                            clip.x1 = min(page.rect.width, clip.x1 + 5)
                            clip.y1 = min(page.rect.height, clip.y1 + 5)
                            
                            # Render at high resolution
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat, clip=clip)
                            
                            # Convert to PIL Image
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(BytesIO(img_data))
                            
                            # Extract text content
                            text_rows = []
                            for row in table_data:
                                row_text = "\t".join(str(cell) if cell else "" for cell in row)
                                text_rows.append(row_text)
                            table_text = "\n".join(text_rows)
                            
                            tables.append(ExtractedTable(
                                text=table_text,
                                page_number=page_num + 1,
                                table_index=len(tables) + 1,
                                bbox=tuple(bbox),
                                label=f"Table {len(tables) + 1}",
                                rows=[[str(c) if c else "" for c in row] for row in table_data],
                                image=pil_image
                            ))
                            
                        except Exception as e:
                            continue
                
            except Exception as e:
                # PyMuPDF version might not support find_tables
                pass
        
        # Fallback: try pdfplumber-style detection if no tables found
        if not tables:
            tables = self._detect_tables_by_structure(doc)
        
        return tables
    
    def _detect_tables_by_structure(self, doc) -> List[ExtractedTable]:
        """
        Detect tables by looking for grid-like structures.
        Fallback method if find_tables() is not available.
        """
        import fitz
        
        tables = []
        
        for page_num, page in enumerate(doc):
            # Look for horizontal and vertical lines that form a grid
            paths = page.get_drawings()
            
            h_lines = []
            v_lines = []
            
            for path in paths:
                for item in path.get("items", []):
                    if item[0] == "l":  # Line
                        p1, p2 = item[1], item[2]
                        
                        # Horizontal line (y coordinates similar)
                        if abs(p1.y - p2.y) < 2:
                            h_lines.append((p1, p2))
                        # Vertical line (x coordinates similar)
                        elif abs(p1.x - p2.x) < 2:
                            v_lines.append((p1, p2))
            
            # If we have enough lines to form a grid, it's likely a table
            if len(h_lines) >= 3 and len(v_lines) >= 2:
                # Calculate bounding box of the grid
                all_points = []
                for p1, p2 in h_lines + v_lines:
                    all_points.extend([p1, p2])
                
                if all_points:
                    x0 = min(p.x for p in all_points)
                    y0 = min(p.y for p in all_points)
                    x1 = max(p.x for p in all_points)
                    y1 = max(p.y for p in all_points)
                    
                    # Minimum size check
                    if (x1 - x0) > 100 and (y1 - y0) > 50:
                        try:
                            clip = fitz.Rect(x0 - 5, y0 - 5, x1 + 5, y1 + 5)
                            mat = fitz.Matrix(2.0, 2.0)
                            pix = page.get_pixmap(matrix=mat, clip=clip)
                            
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(BytesIO(img_data))
                            
                            # Get text from this region
                            text = page.get_text("text", clip=clip)
                            
                            tables.append(ExtractedTable(
                                text=text,
                                page_number=page_num + 1,
                                table_index=len(tables) + 1,
                                bbox=(x0, y0, x1, y1),
                                label=f"Table {len(tables) + 1}",
                                image=pil_image
                            ))
                        except Exception:
                            pass
        
        return tables
    
    def _extract_tables_from_text(self, full_text: str, page_texts: List[str]) -> List[ExtractedTable]:
        """
        Extract table references from text (metadata only, no images).
        Used as last resort when no actual tables can be detected.
        """
        tables = []
        
        # Look for table patterns - just for metadata
        pattern = r'(?:Table|TABLE)\s*(\d+)[.:]?\s*([^\n]+)'
        
        for match in re.finditer(pattern, full_text):
            table_num = match.group(1)
            caption = match.group(2).strip()
            
            tables.append(ExtractedTable(
                text=f"Table {table_num}: {caption}",
                page_number=self._find_page(match.start(), page_texts, full_text),
                table_index=int(table_num),
                bbox=(0, 0, 0, 0),
                caption=caption,
                label=f"Table {table_num}"
                # No image - this is just a text reference
            ))
        
        return tables
    
    def _extract_tables_pdfplumber(self, pdf) -> List[ExtractedTable]:
        """Extract tables using pdfplumber's table detection with images"""
        tables = []
        
        for page_num, page in enumerate(pdf.pages):
            try:
                # Find tables on this page
                page_tables = page.find_tables()
                
                for tbl_idx, table in enumerate(page_tables):
                    try:
                        # Extract table data
                        table_data = table.extract()
                        
                        if not table_data or len(table_data) < 2:
                            continue  # Need at least 2 rows
                        
                        if table_data and len(table_data[0]) < 2:
                            continue  # Need at least 2 columns
                        
                        # Get table bounding box
                        bbox = table.bbox
                        
                        # Crop and render table as image
                        try:
                            # Crop the page to the table region
                            cropped = page.crop(bbox)
                            table_image = cropped.to_image(resolution=150)
                            pil_image = table_image.original
                        except Exception:
                            pil_image = None
                        
                        # Convert to text
                        rows = []
                        for row in table_data:
                            clean_row = [str(cell) if cell else "" for cell in row]
                            rows.append(clean_row)
                        
                        text = '\n'.join(['\t'.join(row) for row in rows])
                        
                        tables.append(ExtractedTable(
                            text=text,
                            page_number=page_num + 1,
                            table_index=len(tables) + 1,
                            bbox=bbox,
                            rows=rows,
                            label=f"Table {len(tables) + 1}",
                            image=pil_image
                        ))
                    
                    except Exception as e:
                        continue
            
            except Exception as e:
                pass
        
        return tables


class TextChunker:
    """Chunk text with sliding window overlap"""
    
    def __init__(
        self,
        chunk_size: int = 200,
        overlap: float = 0.5,
        min_chunk_size: int = 50,
        filter_sections: bool = True
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target words per chunk
            overlap: Overlap percentage (0.5 = 50%)
            min_chunk_size: Minimum words for final chunk
            filter_sections: If True, remove references, authors, etc. before chunking
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.filter_sections = filter_sections
        
        # Initialize text filter
        self._text_filter = None
        if filter_sections:
            try:
                from utils.text_filter import TextSectionFilter
                self._text_filter = TextSectionFilter()
            except ImportError:
                pass
    
    def chunk_text(self, text: str, section: str = "body") -> List[Dict]:
        """
        Chunk text with sliding window.
        
        Args:
            text: Text to chunk
            section: Section name
        
        Returns:
            List of chunk dicts
        """
        # Filter out non-content sections if enabled
        original_length = len(text)
        if self.filter_sections and self._text_filter:
            result = self._text_filter.filter_text(text)
            text = result.filtered_text
        
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [{
                'text': text,
                'section': section,
                'chunk_index': 0,
                'start_word': 0,
                'end_word': len(words)
            }]
        
        chunks = []
        step = int(self.chunk_size * (1 - self.overlap))
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            
            if len(chunk_words) < self.min_chunk_size and chunks:
                # Append to previous chunk
                break
            
            chunks.append({
                'text': ' '.join(chunk_words),
                'section': section,
                'chunk_index': len(chunks),
                'start_word': i,
                'end_word': i + len(chunk_words)
            })
        
        return chunks
    
    def chunk_sections(self, sections: List[ExtractedText]) -> List[Dict]:
        """Chunk multiple sections, filtering excluded ones"""
        all_chunks = []
        
        # Section names to skip
        excluded_sections = [
            'references', 'bibliography', 'acknowledgements', 'acknowledgments',
            'funding', 'author_contributions', 'conflicts', 'data_availability',
            'supplementary', 'appendix'
        ]
        
        for section in sections:
            # Skip excluded section types
            section_lower = section.section.lower()
            if any(excl in section_lower for excl in excluded_sections):
                continue
            
            section_chunks = self.chunk_text(section.text, section.section)
            
            for chunk in section_chunks:
                chunk['chunk_index'] = len(all_chunks)
                all_chunks.append(chunk)
        
        return all_chunks


if __name__ == "__main__":
    # Test chunker
    chunker = TextChunker(chunk_size=100, overlap=0.5)
    
    sample_text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunker.chunk_text(sample_text, "test")
    
    print(f"Created {len(chunks)} chunks from 500 words")
    for i, chunk in enumerate(chunks[:3]):
        words = chunk['text'].split()
        print(f"  Chunk {i}: {len(words)} words, range {chunk['start_word']}-{chunk['end_word']}")
