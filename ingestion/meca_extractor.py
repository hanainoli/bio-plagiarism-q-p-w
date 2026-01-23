"""
MECA File Extractor
===================
Extracts content from MECA (Manuscript Exchange Common Approach) files.
MECA files are ZIP archives containing manuscript PDFs, metadata XML, and supplementary files.

MECA Structure:
    paper.meca (ZIP)
    ├── manifest.xml          (file list)
    ├── content/
    │   ├── manuscript.xml    (JATS XML with full text)
    │   └── manuscript.pdf    (PDF version)
    ├── transfer.xml          (metadata)
    └── supplements/          (supplementary files)

Usage:
    from ingestion.meca_extractor import MECAExtractor
    
    extractor = MECAExtractor()
    result = extractor.extract(Path("paper.meca"))
    
    print(result.title)
    print(result.abstract)
    for figure in result.figures:
        print(figure.label, figure.caption)
"""

import io
import re
import zipfile
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
import logging

# XML parsing
from lxml import etree

# PDF extraction
import fitz  # PyMuPDF
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Image processing
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# JATS XML namespaces
JATS_NS = {
    'xlink': 'http://www.w3.org/1999/xlink',
    'mml': 'http://www.w3.org/1998/Math/MathML',
}

# Minimum image dimensions (pixels)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_AREA = 10000  # 100x100

# Section header patterns
SECTION_PATTERNS = {
    'abstract': r'(?i)^abstract',
    'introduction': r'(?i)^introduction|^background',
    'methods': r'(?i)^methods?|^materials?\s+and\s+methods?|^experimental',
    'results': r'(?i)^results?',
    'discussion': r'(?i)^discussion',
    'conclusion': r'(?i)^conclusions?|^summary',
    'acknowledgements': r'(?i)^acknowledge?ments?',
    'funding': r'(?i)^funding|^financial|^grant',
    'references': r'(?i)^references?|^bibliography|^literature\s+cited',
    'conflicts': r'(?i)^conflict|^competing\s+interests?|^declaration',
    'author_contributions': r'(?i)^author\s+contrib|^contrib',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractedImage:
    """Image extracted from PDF"""
    image_id: str
    page_number: int
    image_index: int
    image_data: bytes
    image_format: str
    width: int
    height: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    label: str = ""
    caption: str = ""
    
    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "page_number": self.page_number,
            "image_index": self.image_index,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "bbox": list(self.bbox) if self.bbox else None,
            "label": self.label,
            "caption": self.caption,
            # Don't include binary data in dict
        }


@dataclass
class ExtractedTable:
    """Table extracted from PDF"""
    table_id: str
    page_number: int
    table_index: int
    label: str
    caption: str
    headers: List[List[str]]
    rows: List[List[str]]
    full_text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    
    @property
    def row_count(self) -> int:
        return len(self.rows)
    
    @property
    def col_count(self) -> int:
        if self.headers:
            return len(self.headers[0]) if self.headers[0] else 0
        if self.rows:
            return len(self.rows[0]) if self.rows[0] else 0
        return 0
    
    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "label": self.label,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "full_text": self.full_text,
            "row_count": self.row_count,
            "col_count": self.col_count,
        }


@dataclass
class ExtractedFormula:
    """Mathematical formula extracted from paper"""
    formula_id: str
    raw: str
    normalized: str
    formula_type: str  # "inline" or "display"
    context: str = ""
    
    def to_dict(self) -> dict:
        return {
            "formula_id": self.formula_id,
            "raw": self.raw,
            "normalized": self.normalized,
            "formula_type": self.formula_type,
            "context": self.context,
        }


@dataclass
class Author:
    """Paper author"""
    name: str
    given_names: str = ""
    surname: str = ""
    email: str = ""
    affiliation: str = ""
    orcid: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "given_names": self.given_names,
            "surname": self.surname,
            "email": self.email,
            "affiliation": self.affiliation,
            "orcid": self.orcid,
        }


@dataclass
class MECAExtractionResult:
    """Complete extraction result from a MECA file"""
    # Identification
    paper_id: str
    doi: str
    server: str  # "biorxiv" or "medrxiv"
    version: str = "1"
    
    # Metadata
    title: str = ""
    authors: List[Author] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    category: str = ""
    publication_date: str = ""
    posted_date: str = ""
    
    # Content
    full_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    word_count: int = 0
    
    # Figures
    figures: List[ExtractedImage] = field(default_factory=list)
    figure_count: int = 0
    
    # Tables
    tables: List[ExtractedTable] = field(default_factory=list)
    table_count: int = 0
    
    # Formulas
    formulas: List[ExtractedFormula] = field(default_factory=list)
    
    # Special sections
    acknowledgements: str = ""
    funding: str = ""
    conflicts: str = ""
    author_contributions: str = ""
    references_text: str = ""
    
    # Source info
    source_url: str = ""
    meca_file: str = ""
    pdf_extracted: bool = False
    xml_extracted: bool = False
    parse_success: bool = False
    extraction_errors: List[str] = field(default_factory=list)
    extracted_at: str = ""
    
    def to_dict(self, include_images: bool = False) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "server": self.server,
            "version": self.version,
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "abstract": self.abstract,
            "keywords": self.keywords,
            "category": self.category,
            "publication_date": self.publication_date,
            "posted_date": self.posted_date,
            "full_text": self.full_text,
            "sections": self.sections,
            "word_count": self.word_count,
            "figures": [f.to_dict() for f in self.figures],
            "figure_count": self.figure_count,
            "tables": [t.to_dict() for t in self.tables],
            "table_count": self.table_count,
            "formulas": [f.to_dict() for f in self.formulas],
            "acknowledgements": self.acknowledgements,
            "funding": self.funding,
            "conflicts": self.conflicts,
            "author_contributions": self.author_contributions,
            "references_text": self.references_text[:5000] if self.references_text else "",
            "source_url": self.source_url,
            "meca_file": self.meca_file,
            "pdf_extracted": self.pdf_extracted,
            "xml_extracted": self.xml_extracted,
            "parse_success": self.parse_success,
            "extraction_errors": self.extraction_errors,
            "extracted_at": self.extracted_at,
        }
        
        if include_images:
            result["figure_data"] = {
                f.image_id: f.image_data.hex() 
                for f in self.figures 
                if f.image_data
            }
        
        return result


# =============================================================================
# MECA EXTRACTOR
# =============================================================================

class MECAExtractor:
    """
    Extracts content from MECA (Manuscript Exchange Common Approach) files.
    
    MECA files are ZIP archives used by bioRxiv/medRxiv containing:
    - manuscript.xml: Full text in JATS XML format
    - manuscript.pdf: PDF version
    - Supplementary materials
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        extract_formulas: bool = True,
        min_image_area: int = MIN_IMAGE_AREA
    ):
        """
        Initialize extractor.
        
        Args:
            extract_images: Whether to extract figures from PDF
            extract_tables: Whether to extract tables
            extract_formulas: Whether to extract mathematical formulas
            min_image_area: Minimum image area to extract
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_formulas = extract_formulas
        self.min_image_area = min_image_area
    
    def extract(
        self,
        meca_path: Path = None,
        meca_bytes: bytes = None,
        filename: str = None
    ) -> MECAExtractionResult:
        """
        Extract content from a MECA file.
        
        Args:
            meca_path: Path to MECA file
            meca_bytes: MECA file as bytes (alternative to path)
            filename: Original filename (for identification)
            
        Returns:
            MECAExtractionResult with all extracted content
        """
        # Initialize result
        result = MECAExtractionResult(
            paper_id="",
            doi="",
            server="biorxiv",
            extracted_at=datetime.now().isoformat()
        )
        
        if meca_path:
            result.meca_file = str(meca_path)
            filename = meca_path.name
        
        try:
            # Open MECA archive
            if meca_path:
                zf = zipfile.ZipFile(meca_path, 'r')
            elif meca_bytes:
                zf = zipfile.ZipFile(io.BytesIO(meca_bytes), 'r')
            else:
                raise ValueError("Either meca_path or meca_bytes must be provided")
            
            with zf:
                # List contents
                file_list = zf.namelist()
                logger.debug(f"MECA contents: {file_list}")
                
                # Find and parse XML metadata
                xml_file = self._find_xml_file(file_list)
                if xml_file:
                    xml_bytes = zf.read(xml_file)
                    self._parse_xml(xml_bytes, result)
                    result.xml_extracted = True
                
                # Find and extract from PDF
                pdf_file = self._find_pdf_file(file_list)
                if pdf_file:
                    pdf_bytes = zf.read(pdf_file)
                    self._extract_from_pdf(pdf_bytes, result)
                    result.pdf_extracted = True
            
            # Generate paper ID if not set
            if not result.paper_id and result.doi:
                result.paper_id = self._generate_paper_id(result.doi, result.server)
            elif not result.paper_id and filename:
                result.paper_id = self._generate_paper_id_from_filename(filename)
            
            # Calculate stats
            result.word_count = len(result.full_text.split())
            result.figure_count = len(result.figures)
            result.table_count = len(result.tables)
            result.parse_success = True
            
        except Exception as e:
            logger.error(f"Error extracting MECA: {e}")
            result.extraction_errors.append(str(e))
            result.parse_success = False
        
        return result
    
    def extract_from_pdf(
        self,
        pdf_path: Path = None,
        pdf_bytes: bytes = None,
        doi: str = None,
        server: str = "biorxiv"
    ) -> MECAExtractionResult:
        """
        Extract content directly from a PDF file (no MECA wrapper).
        
        Args:
            pdf_path: Path to PDF file
            pdf_bytes: PDF file as bytes
            doi: DOI if known
            server: Server origin
            
        Returns:
            MECAExtractionResult
        """
        result = MECAExtractionResult(
            paper_id=self._generate_paper_id(doi, server) if doi else "",
            doi=doi or "",
            server=server,
            extracted_at=datetime.now().isoformat()
        )
        
        try:
            if pdf_path:
                pdf_bytes = pdf_path.read_bytes()
                result.meca_file = str(pdf_path)
            
            self._extract_from_pdf(pdf_bytes, result)
            result.pdf_extracted = True
            
            result.word_count = len(result.full_text.split())
            result.figure_count = len(result.figures)
            result.table_count = len(result.tables)
            result.parse_success = True
            
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            result.extraction_errors.append(str(e))
            result.parse_success = False
        
        return result
    
    def _find_xml_file(self, file_list: List[str]) -> Optional[str]:
        """Find the main XML file in MECA archive"""
        # Priority order for XML files
        patterns = [
            r'content/.*\.xml$',
            r'manuscript\.xml$',
            r'.*manuscript.*\.xml$',
            r'.*\.xml$'
        ]
        
        for pattern in patterns:
            for f in file_list:
                if re.search(pattern, f, re.IGNORECASE):
                    if 'manifest' not in f.lower() and 'transfer' not in f.lower():
                        return f
        return None
    
    def _find_pdf_file(self, file_list: List[str]) -> Optional[str]:
        """Find the main PDF file in MECA archive"""
        # Priority order for PDF files
        patterns = [
            r'content/.*\.pdf$',
            r'manuscript\.pdf$',
            r'.*manuscript.*\.pdf$',
            r'.*\.pdf$'
        ]
        
        for pattern in patterns:
            for f in file_list:
                if re.search(pattern, f, re.IGNORECASE):
                    if 'supplement' not in f.lower():
                        return f
        
        # Fall back to any PDF
        for f in file_list:
            if f.lower().endswith('.pdf'):
                return f
        
        return None
    
    def _parse_xml(self, xml_bytes: bytes, result: MECAExtractionResult):
        """Parse JATS XML metadata"""
        try:
            root = etree.fromstring(xml_bytes)
            
            # Extract DOI
            doi_elem = root.find('.//article-id[@pub-id-type="doi"]')
            if doi_elem is not None and doi_elem.text:
                result.doi = doi_elem.text.strip()
                
                # Determine server from DOI
                if 'medrxiv' in result.doi.lower():
                    result.server = "medrxiv"
                else:
                    result.server = "biorxiv"
            
            # Extract title
            title_elem = root.find('.//article-title')
            if title_elem is not None:
                result.title = self._get_text_content(title_elem)
            
            # Extract abstract
            abstract_elem = root.find('.//abstract')
            if abstract_elem is not None:
                result.abstract = self._get_text_content(abstract_elem)
            
            # Extract authors
            for contrib in root.findall('.//contrib[@contrib-type="author"]'):
                author = self._parse_author(contrib)
                if author:
                    result.authors.append(author)
            
            # Extract keywords
            for kwd in root.findall('.//kwd'):
                if kwd.text:
                    result.keywords.append(kwd.text.strip())
            
            # Extract dates
            pub_date = root.find('.//pub-date')
            if pub_date is not None:
                year = pub_date.findtext('year', '')
                month = pub_date.findtext('month', '01')
                day = pub_date.findtext('day', '01')
                if year:
                    result.publication_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            # Extract category/subject
            subj = root.find('.//subject')
            if subj is not None and subj.text:
                result.category = subj.text.strip().lower()
            
            # Extract full text from body
            body = root.find('.//body')
            if body is not None:
                result.full_text = self._get_text_content(body)
                self._extract_sections_from_xml(body, result)
            
            # Extract acknowledgements
            ack = root.find('.//ack')
            if ack is not None:
                result.acknowledgements = self._get_text_content(ack)
            
            # Extract funding
            funding = root.find('.//funding-group')
            if funding is not None:
                result.funding = self._get_text_content(funding)
            
            # Extract references
            ref_list = root.find('.//ref-list')
            if ref_list is not None:
                result.references_text = self._get_text_content(ref_list)
            
        except Exception as e:
            logger.warning(f"XML parsing error: {e}")
            result.extraction_errors.append(f"XML parsing: {e}")
    
    def _parse_author(self, contrib_elem) -> Optional[Author]:
        """Parse author from contrib element"""
        try:
            name_elem = contrib_elem.find('.//name')
            if name_elem is None:
                return None
            
            surname = name_elem.findtext('surname', '')
            given = name_elem.findtext('given-names', '')
            
            author = Author(
                name=f"{given} {surname}".strip(),
                given_names=given,
                surname=surname
            )
            
            # Email
            email_elem = contrib_elem.find('.//email')
            if email_elem is not None and email_elem.text:
                author.email = email_elem.text.strip()
            
            # ORCID
            for ext_id in contrib_elem.findall('.//contrib-id'):
                if ext_id.get('contrib-id-type') == 'orcid':
                    author.orcid = ext_id.text.strip() if ext_id.text else ""
            
            return author
            
        except Exception as e:
            logger.debug(f"Error parsing author: {e}")
            return None
    
    def _get_text_content(self, elem) -> str:
        """Get all text content from an element, handling mixed content"""
        if elem is None:
            return ""
        
        # Get all text including tail of children
        texts = []
        for text in elem.itertext():
            if text.strip():
                texts.append(text.strip())
        
        return ' '.join(texts)
    
    def _extract_sections_from_xml(self, body, result: MECAExtractionResult):
        """Extract named sections from XML body"""
        for sec in body.findall('.//sec'):
            title_elem = sec.find('title')
            if title_elem is not None and title_elem.text:
                section_title = title_elem.text.strip().lower()
                section_text = self._get_text_content(sec)
                
                # Categorize section
                for section_type, pattern in SECTION_PATTERNS.items():
                    if re.match(pattern, section_title, re.IGNORECASE):
                        result.sections[section_type] = section_text
                        break
                else:
                    # Store with original title if no match
                    result.sections[section_title] = section_text
    
    def _extract_from_pdf(self, pdf_bytes: bytes, result: MECAExtractionResult):
        """Extract content from PDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract text
            full_text_parts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                full_text_parts.append(text)
            
            result.full_text = '\n\n'.join(full_text_parts)
            
            # Extract sections from text
            self._extract_sections_from_text(result)
            
            # Extract images
            if self.extract_images:
                self._extract_images_from_pdf(doc, result)
            
            # Extract tables
            if self.extract_tables and HAS_PDFPLUMBER:
                self._extract_tables_from_pdf(pdf_bytes, result)
            
            # Extract formulas
            if self.extract_formulas:
                self._extract_formulas(result)
            
            # Extract special sections
            self._extract_special_sections(result)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            result.extraction_errors.append(f"PDF extraction: {e}")
    
    def _extract_images_from_pdf(self, doc, result: MECAExtractionResult):
        """Extract images from PDF document"""
        image_index = 0
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                try:
                    xref = img_info[0]
                    
                    # Extract image
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Get dimensions
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    
                    # Filter small images
                    if width * height < self.min_image_area:
                        continue
                    
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue
                    
                    # Convert to PNG for consistency
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1])
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Re-encode as PNG
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        image_ext = 'png'
                        width, height = img.size
                    except Exception:
                        pass
                    
                    # Generate image ID
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_id = f"fig_p{page_num + 1}_{image_index}_{image_hash}"
                    
                    extracted_image = ExtractedImage(
                        image_id=image_id,
                        page_number=page_num + 1,
                        image_index=image_index,
                        image_data=image_bytes,
                        image_format=image_ext,
                        width=width,
                        height=height,
                        label=f"Figure {image_index + 1}",
                    )
                    
                    result.figures.append(extracted_image)
                    image_index += 1
                    
                except Exception as e:
                    logger.debug(f"Error extracting image: {e}")
    
    def _extract_tables_from_pdf(self, pdf_bytes: bytes, result: MECAExtractionResult):
        """Extract tables using pdfplumber"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                table_index = 0
                
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_data in tables:
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        # First row as headers
                        headers = [table_data[0]] if table_data else []
                        rows = table_data[1:] if len(table_data) > 1 else []
                        
                        # Clean cells
                        headers = [[self._clean_cell(c) for c in row] for row in headers]
                        rows = [[self._clean_cell(c) for c in row] for row in rows]
                        
                        # Build full text representation
                        full_text_parts = []
                        if headers:
                            full_text_parts.append(' | '.join(headers[0]))
                        for row in rows:
                            full_text_parts.append(' | '.join(row))
                        full_text = '\n'.join(full_text_parts)
                        
                        # Generate table ID
                        table_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
                        table_id = f"tbl_p{page_num + 1}_{table_index}_{table_hash}"
                        
                        extracted_table = ExtractedTable(
                            table_id=table_id,
                            page_number=page_num + 1,
                            table_index=table_index,
                            label=f"Table {table_index + 1}",
                            caption="",
                            headers=headers,
                            rows=rows,
                            full_text=full_text,
                        )
                        
                        result.tables.append(extracted_table)
                        table_index += 1
                        
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
    
    def _clean_cell(self, cell) -> str:
        """Clean table cell value"""
        if cell is None:
            return ""
        return str(cell).strip().replace('\n', ' ')
    
    def _extract_formulas(self, result: MECAExtractionResult):
        """Extract mathematical formulas from text"""
        text = result.full_text
        
        # LaTeX display formulas: $$...$$
        display_pattern = r'\$\$(.+?)\$\$'
        for i, match in enumerate(re.finditer(display_pattern, text, re.DOTALL)):
            formula = ExtractedFormula(
                formula_id=f"formula_d{i}",
                raw=match.group(1).strip(),
                normalized=self._normalize_formula(match.group(1)),
                formula_type="display",
                context=text[max(0, match.start()-50):match.end()+50]
            )
            result.formulas.append(formula)
        
        # LaTeX inline formulas: $...$
        inline_pattern = r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'
        for i, match in enumerate(re.finditer(inline_pattern, text)):
            formula = ExtractedFormula(
                formula_id=f"formula_i{i}",
                raw=match.group(1).strip(),
                normalized=self._normalize_formula(match.group(1)),
                formula_type="inline",
                context=text[max(0, match.start()-30):match.end()+30]
            )
            result.formulas.append(formula)
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for comparison"""
        # Remove whitespace
        normalized = re.sub(r'\s+', '', formula)
        # Lowercase
        normalized = normalized.lower()
        return normalized
    
    def _extract_sections_from_text(self, result: MECAExtractionResult):
        """Extract sections from plain text"""
        text = result.full_text
        
        for section_type, pattern in SECTION_PATTERNS.items():
            # Find section header
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                start = match.end()
                
                # Find next section header
                next_match = None
                for other_pattern in SECTION_PATTERNS.values():
                    if other_pattern != pattern:
                        m = re.search(other_pattern, text[start:], re.MULTILINE | re.IGNORECASE)
                        if m and (next_match is None or m.start() < next_match.start()):
                            next_match = m
                
                if next_match:
                    section_text = text[start:start + next_match.start()].strip()
                else:
                    section_text = text[start:start + 5000].strip()  # Limit length
                
                if section_text and section_type not in result.sections:
                    result.sections[section_type] = section_text
    
    def _extract_special_sections(self, result: MECAExtractionResult):
        """Extract acknowledgements, funding, conflicts, etc."""
        text = result.full_text
        
        # Acknowledgements
        if not result.acknowledgements:
            ack_match = re.search(
                r'acknowledge?ments?\s*[:\n](.{50,2000}?)(?=funding|references|conflicts|$)',
                text, re.IGNORECASE | re.DOTALL
            )
            if ack_match:
                result.acknowledgements = ack_match.group(1).strip()
        
        # Funding
        if not result.funding:
            funding_match = re.search(
                r'(?:funding|financial|grant)[^:]*[:\n](.{30,1500}?)(?=acknowledge|references|conflicts|$)',
                text, re.IGNORECASE | re.DOTALL
            )
            if funding_match:
                result.funding = funding_match.group(1).strip()
        
        # Conflicts
        if not result.conflicts:
            conflicts_match = re.search(
                r'(?:conflict|competing\s+interests?|declaration)[^:]*[:\n](.{20,500}?)(?=acknowledge|funding|references|$)',
                text, re.IGNORECASE | re.DOTALL
            )
            if conflicts_match:
                result.conflicts = conflicts_match.group(1).strip()
        
        # Author contributions
        if not result.author_contributions:
            contrib_match = re.search(
                r'author\s+contrib[^:]*[:\n](.{50,1500}?)(?=acknowledge|funding|conflicts|references|$)',
                text, re.IGNORECASE | re.DOTALL
            )
            if contrib_match:
                result.author_contributions = contrib_match.group(1).strip()
    
    def _generate_paper_id(self, doi: str, server: str) -> str:
        """Generate paper ID from DOI"""
        if not doi:
            return ""
        
        # Sanitize DOI
        doi_suffix = doi.replace("10.1101/", "")
        sanitized = doi_suffix.replace(".", "_").replace("/", "_")
        
        return f"{server}_{sanitized}"
    
    def _generate_paper_id_from_filename(self, filename: str) -> str:
        """Generate paper ID from filename"""
        # Remove extension
        name = Path(filename).stem
        
        # Remove common prefixes
        name = re.sub(r'^10\.1101\.', '', name)
        
        # Sanitize
        sanitized = name.replace(".", "_").replace("/", "_")
        
        return f"paper_{sanitized}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_meca(meca_path: Path) -> MECAExtractionResult:
    """Extract content from a MECA file"""
    extractor = MECAExtractor()
    return extractor.extract(meca_path=meca_path)


def extract_pdf(pdf_path: Path, doi: str = None) -> MECAExtractionResult:
    """Extract content from a PDF file"""
    extractor = MECAExtractor()
    return extractor.extract_from_pdf(pdf_path=pdf_path, doi=doi)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MECA/PDF Content Extractor")
    parser.add_argument("file", type=str, help="MECA or PDF file to extract")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    parser.add_argument("--include-image-data", action="store_true", help="Include image data in JSON")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    
    extractor = MECAExtractor(
        extract_images=not args.no_images,
        extract_tables=not args.no_tables,
    )
    
    if file_path.suffix.lower() == '.meca':
        result = extractor.extract(meca_path=file_path)
    else:
        result = extractor.extract_from_pdf(pdf_path=file_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION RESULT")
    print(f"{'='*60}")
    print(f"Paper ID: {result.paper_id}")
    print(f"DOI: {result.doi}")
    print(f"Server: {result.server}")
    print(f"Title: {result.title[:80]}...")
    print(f"Authors: {len(result.authors)}")
    print(f"Abstract: {len(result.abstract)} chars")
    print(f"Full text: {result.word_count} words")
    print(f"Figures: {result.figure_count}")
    print(f"Tables: {result.table_count}")
    print(f"Formulas: {len(result.formulas)}")
    print(f"Parse success: {result.parse_success}")
    
    if result.extraction_errors:
        print(f"Errors: {result.extraction_errors}")
    
    # Save JSON
    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(include_images=args.include_image_data), f, indent=2)
        print(f"\nSaved to: {output_path}")
