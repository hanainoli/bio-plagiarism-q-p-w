#!/usr/bin/env python3
"""
Text Section Filter
===================
Filters out non-original-content sections from academic papers:
- References / Bibliography
- Author information and affiliations
- Author contributions
- Acknowledgements
- Funding statements
- Conflicts of interest
- Copyright notices

This ensures only the actual scientific content is embedded and compared,
reducing false positives from boilerplate text.

Usage:
    from utils.text_filter import TextSectionFilter
    
    filter = TextSectionFilter()
    filtered_text = filter.filter_text(full_text)
    filtered_chunks = filter.filter_chunks(chunks)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FilteredResult:
    """Result of text filtering"""
    filtered_text: str
    original_length: int
    filtered_length: int
    removed_sections: List[str]
    removal_percentage: float


class TextSectionFilter:
    """
    Filters non-content sections from academic papers.
    
    Sections removed:
    - References / Bibliography
    - Authors and Affiliations  
    - Author Contributions
    - Acknowledgements
    - Funding / Financial
    - Conflicts of Interest / Competing Interests
    - Copyright notices
    - Data Availability statements
    - Supplementary info references
    """
    
    # Patterns to detect section headers (case-insensitive)
    SECTION_START_PATTERNS = {
        'references': [
            r'^\s*references?\s*$',
            r'^\s*REFERENCES?\s*$',
            r'^\s*bibliography\s*$',
            r'^\s*literature\s+cited\s*$',
            r'^\s*works?\s+cited\s*$',
            r'^\s*cited\s+literature\s*$',
            r'^\s*\d+\.\s*references?\s*$',  # "1. References"
        ],
        'authors': [
            r'^\s*authors?:?\s*$',
            r'^\s*author\s+information\s*$',
            r'^\s*about\s+the\s+authors?\s*$',
        ],
        'affiliations': [
            r'^\s*affiliations?:?\s*$',
            r'^\s*author\s+affiliations?\s*$',
        ],
        'contributions': [
            r'^\s*author\s+contributions?:?\s*$',
            r'^\s*contributions?:?\s*$',
            r'^\s*author\s+roles?\s*$',
            r'^\s*credit\s+authorship\s*',
        ],
        'acknowledgements': [
            r'^\s*acknowledge?ments?:?\s*$',
            r'^\s*ACKNOWLEDGE?MENTS?:?\s*$',
        ],
        'funding': [
            r'^\s*funding\s*$',
            r'^\s*financial\s+support\s*$',
            r'^\s*grant\s+support\s*$',
            r'^\s*funding\s+sources?\s*$',
        ],
        'conflicts': [
            r'^\s*conflicts?\s+of\s+interests?\s*$',
            r'^\s*competing\s+interests?\s*$',
            r'^\s*declaration\s+of\s+interests?\s*$',
            r'^\s*disclosures?\s*$',
        ],
        'copyright': [
            r'^\s*copyright\s*$',
            r'^\s*©\s*\d{4}',
            r'^\s*all\s+rights\s+reserved\s*$',
        ],
        'data_availability': [
            r'^\s*data\s+availability\s*$',
            r'^\s*data\s+access\s*$',
            r'^\s*availability\s+of\s+data\s*$',
            r'^\s*code\s+availability\s*$',
        ],
        'supplementary': [
            r'^\s*supplementary\s+(?:material|information|data)\s*$',
            r'^\s*supporting\s+information\s*$',
        ],
        'ethics': [
            r'^\s*ethics\s+(?:statement|approval|declaration)\s*$',
            r'^\s*ethical\s+approval\s*$',
        ],
    }
    
    # Content section patterns (to know when to resume including text)
    CONTENT_SECTION_PATTERNS = [
        r'^\s*abstract\s*$',
        r'^\s*introduction\s*$',
        r'^\s*background\s*$',
        r'^\s*methods?\s*$',
        r'^\s*materials?\s+and\s+methods?\s*$',
        r'^\s*experimental\s*$',
        r'^\s*results?\s*$',
        r'^\s*discussion\s*$',
        r'^\s*conclusions?\s*$',
        r'^\s*summary\s*$',
        r'^\s*\d+\.\s+\w+',  # Numbered sections like "1. Introduction"
    ]
    
    # Inline patterns to remove (appear within paragraphs)
    INLINE_REMOVE_PATTERNS = [
        # Email patterns
        r'\S+@\S+\.\S+',
        # ORCID patterns
        r'orcid\.org/[\d\-]+',
        r'ORCID:?\s*[\d\-]+',
        # Copyright inline - STRONG patterns
        r'©\s*\d{4}[^.]*',
        r'All rights reserved\.?\s*No reuse allowed without permission\.?',
        r'certified by peer review\) is the author/funder[^.]*\.',
        r'who has granted (?:bioRxiv|medRxiv|arXiv) a license to display the preprint in perpetuity\.',
        r'The copyright holder for this preprint \(which was not[^)]*\)',
        r'this version posted [A-Za-z]+ \d+, \d{4}\.',
        # DOI patterns
        r'doi:\s*(?:https?://)?(?:doi\.org/)?10\.\d{4,}/[^\s]+',
        r'https://doi\.org/10\.\d{4,}/[^\s]+',
        # Preprint notices
        r'(?:This\s+)?(?:preprint|manuscript)\s+(?:was\s+)?(?:posted|submitted|uploaded)\s+(?:to|on)\s+(?:bioRxiv|medRxiv|arXiv)[^.]*\.?',
        r'NOTE:\s*This preprint reports new research[^.]*\.',
        r'not been certified by peer review and should not be used to guide clinical practice\.?',
        # Author contribution patterns
        r'Author contributions?:\s*[^.]+(?:\.[^.]+){0,5}\.',
        r'Study concept and design \([^)]+\);[^.]+\.',
        r'\([A-Z]{2,3}(?:,\s*[A-Z]{2,3})*\)\s*;',  # Like (MW, MJ, SM);
        # Affiliations patterns  
        r'Affiliations?:\s*\d+[^.]+(?:;\s*\d+[^.]+)*',
        r'\d+[A-Za-z\s]+(?:UMC|University|Institute|Hospital|Center|Centre)[^;.]*[;.]',
        # Corresponding author patterns
        r'[§*#]+\s*(?:Corresponding|Shared corresponding)\s+authors?:?[^.]*',
        r'Correspondence[^.]+should be addressed to[^.]+\.',
        # Funding patterns
        r'Sources? of funding:\s*[^.]+\.',
        r'This work was supported by[^.]+\.',
        r'funded by[^.]+\.',
        # Disclosure patterns
        r'Disclosures?:\s*[^.]+\.',
        r'No conflicts? of interest to declare\.?',
        r'The authors? declare[^.]+no (?:competing )?(?:conflicts? of )?interests?\.?',
        # Contact info
        r'Tel:\s*\+?[\d\s\-]+;?\s*Fax:\s*\+?[\d\s\-]+;?',
        r'Email:\s*\S+@\S+',
        # Word count
        r'Word [Cc]ount:\s*\d+',
        # Running head
        r'RUNNING HEAD:[^.]+',
        # Keywords line
        r'Keywords?:\s*[^.]+\.',
        # List of abbreviations header
        r'LIST OF ABBREVIATIONS',
        # These first/shared authors
        r'\*\s*These (?:first )?authors contributed equally',
        r'[§#]\s*(?:Shared|List of)[^.]+',
        # Working group
        r'on behalf of the [A-Z]+ working group[#*]?',
        r'List of working group members appears at the end of this manuscript',
    ]
    
    # Boilerplate phrases to remove
    BOILERPLATE_PHRASES = [
        r'the\s+authors?\s+declare\s+(?:that\s+)?(?:they\s+have\s+)?no\s+(?:competing\s+)?(?:conflicts?\s+of\s+)?interests?',
        r'all\s+authors?\s+(?:have\s+)?read\s+and\s+(?:approved|agreed)',
        r'this\s+work\s+was\s+supported\s+by',
        r'the\s+authors?\s+would\s+like\s+to\s+(?:thank|acknowledge)',
        r'we\s+(?:would\s+like\s+to\s+)?thank\s+[\w\s,]+\s+for\s+(?:their\s+)?(?:helpful|valuable)',
        r'the\s+funders?\s+had\s+no\s+role\s+in',
        r'data\s+(?:are|is)\s+available\s+(?:upon|on)\s+(?:reasonable\s+)?request',
        r'correspondence\s+(?:and\s+requests?\s+for\s+materials?\s+)?should\s+be\s+addressed\s+to',
        r'contributed\s+equally\s+to\s+this\s+work',
        r'these\s+authors?\s+contributed\s+equally',
        r'certified by peer review',
        r'granted (?:bioRxiv|medRxiv) a license',
        r'display the preprint in perpetuity',
        r'no reuse allowed without permission',
        r'all rights reserved',
    ]
    
    def __init__(self, aggressive: bool = False):
        """
        Initialize filter.
        
        Args:
            aggressive: If True, also removes acknowledgements and ethics sections
        """
        self.aggressive = aggressive
        
        # Compile patterns for efficiency
        self._section_patterns = {}
        for section_type, patterns in self.SECTION_START_PATTERNS.items():
            self._section_patterns[section_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) 
                for p in patterns
            ]
        
        self._content_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) 
            for p in self.CONTENT_SECTION_PATTERNS
        ]
        
        self._inline_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.INLINE_REMOVE_PATTERNS
        ]
        
        self._boilerplate_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.BOILERPLATE_PHRASES
        ]
    
    def filter_text(self, text: str) -> FilteredResult:
        """
        Filter out non-content sections from full text.
        
        Args:
            text: Full paper text
            
        Returns:
            FilteredResult with cleaned text and stats
        """
        if not text:
            return FilteredResult(
                filtered_text="",
                original_length=0,
                filtered_length=0,
                removed_sections=[],
                removal_percentage=0.0
            )
        
        original_length = len(text)
        removed_sections = []
        
        # STEP 1: Remove header block (title, authors, affiliations at start)
        text = self._remove_header_block(text)
        if len(text) < original_length:
            removed_sections.append('header_block')
        
        # STEP 2: Remove inline patterns first (copyright notices, emails, etc.)
        text = self._clean_inline(text)
        text = self._remove_boilerplate(text)
        
        # STEP 3: Split into lines for section detection
        lines = text.split('\n')
        filtered_lines = []
        
        # Track which sections we're in
        in_excluded_section = False
        current_excluded = None
        
        # These sections typically go to end of document
        # But we should STILL check for content sections after them
        # because some papers have unusual ordering
        terminal_sections = ['references', 'acknowledgements', 'funding', 'conflicts', 
                            'data_availability', 'supplementary', 'contributions']
        
        for line in lines:
            stripped = line.strip()
            
            # Check if this line starts an excluded section
            excluded_type = self._check_excluded_section(stripped)
            if excluded_type:
                in_excluded_section = True
                current_excluded = excluded_type
                if excluded_type not in removed_sections:
                    removed_sections.append(excluded_type)
                continue
            
            # ALWAYS check if this line starts a content section (resume including)
            # Even if we're in a "terminal" section - some papers have unusual ordering
            # e.g., References before Discussion, or content after Acknowledgements
            if in_excluded_section:
                if self._check_content_section(stripped):
                    in_excluded_section = False
                    current_excluded = None
                    # Include this line (the section header)
                    filtered_lines.append(line)
                    continue
            
            # Include line if not in excluded section
            if not in_excluded_section:
                if stripped:  # Don't add empty lines
                    filtered_lines.append(line)
        
        # Join back
        filtered_text = '\n'.join(filtered_lines)
        
        # Clean up extra whitespace
        filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
        filtered_text = filtered_text.strip()
        
        filtered_length = len(filtered_text)
        removal_percentage = ((original_length - filtered_length) / original_length * 100) if original_length > 0 else 0
        
        return FilteredResult(
            filtered_text=filtered_text,
            original_length=original_length,
            filtered_length=filtered_length,
            removed_sections=removed_sections,
            removal_percentage=round(removal_percentage, 2)
        )
    
    def _remove_header_block(self, text: str) -> str:
        """
        Remove the header block (title, authors, affiliations) from start of paper.
        This content appears before ABSTRACT and should not be embedded.
        """
        # Try to find where the abstract starts
        abstract_patterns = [
            r'\n\s*ABSTRACT\s*\n',
            r'\n\s*Abstract\s*\n',
            r'\n\s*ABSTRACT\s*$',
            r'\nAbstract\s*$',
            r'\n\s*ABSTRACT\s*[:\.]',
            r'\n\s*Objective:\s',  # Some papers start abstract with Objective
            r'\n\s*Background:\s',  # Or Background
            r'\n\s*Purpose:\s',  # Or Purpose
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                # Return text from abstract onwards
                return text[match.start():]
        
        # If no abstract found, try to remove obvious header patterns
        # Look for "Keywords:" or "Disclosures:" which often mark end of header
        end_header_patterns = [
            r'\n\s*Keywords?:',
            r'\n\s*Disclosures?:',
            r'\n\s*Sources? of [Ff]unding:',
            r'\n\s*INTRODUCTION\s*\n',
            r'\n\s*Introduction\s*\n',
        ]
        
        for pattern in end_header_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                return text[match.start():]
        
        return text
    
    def _check_excluded_section(self, line: str) -> Optional[str]:
        """Check if line starts an excluded section."""
        for section_type, patterns in self._section_patterns.items():
            # Only skip 'ethics' in non-aggressive mode (keep acknowledgements excluded always)
            if not self.aggressive and section_type in ['ethics']:
                continue
            
            for pattern in patterns:
                if pattern.search(line):
                    return section_type
        return None
    
    def _check_content_section(self, line: str) -> bool:
        """Check if line starts a content section."""
        for pattern in self._content_patterns:
            if pattern.search(line):
                return True
        return False
    
    def _clean_inline(self, text: str) -> str:
        """Remove inline patterns from text."""
        result = text
        for pattern in self._inline_patterns:
            result = pattern.sub('', result)
        return result
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate phrases from text."""
        result = text
        for pattern in self._boilerplate_patterns:
            result = pattern.sub('', result)
        return result
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Filter a list of text chunks, removing those from excluded sections.
        
        Args:
            chunks: List of chunk dicts with 'text' and optionally 'section' keys
            
        Returns:
            Filtered list of chunks
        """
        filtered = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            section = chunk.get('section', '').lower()
            
            # Skip chunks from excluded sections
            if self._is_excluded_section_name(section):
                continue
            
            # Filter the text content
            result = self.filter_text(text)
            
            # Only keep if enough content remains
            if len(result.filtered_text) >= 50:  # Minimum 50 chars
                filtered_chunk = chunk.copy()
                filtered_chunk['text'] = result.filtered_text
                filtered_chunk['original_length'] = result.original_length
                filtered.append(filtered_chunk)
        
        return filtered
    
    def _is_excluded_section_name(self, section_name: str) -> bool:
        """Check if a section name indicates excluded content."""
        excluded_keywords = [
            'reference', 'bibliography', 'author', 'affiliation',
            'contribution', 'acknowledge', 'funding', 'conflict',
            'competing', 'copyright', 'data_availability', 'supplementary',
            'appendix', 'ethics'
        ]
        
        section_lower = section_name.lower()
        return any(kw in section_lower for kw in excluded_keywords)
    
    def get_content_only(self, text: str) -> str:
        """
        Convenience method to get just the filtered text.
        
        Args:
            text: Full paper text
            
        Returns:
            Filtered text string
        """
        return self.filter_text(text).filtered_text


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def filter_paper_text(text: str, aggressive: bool = False) -> str:
    """
    Filter a paper's text to remove non-content sections.
    
    Args:
        text: Full paper text
        aggressive: If True, also removes acknowledgements
        
    Returns:
        Filtered text
    """
    filter = TextSectionFilter(aggressive=aggressive)
    return filter.get_content_only(text)


def filter_chunks_for_embedding(chunks: List[Dict]) -> List[Dict]:
    """
    Filter chunks before embedding to exclude non-content.
    
    Args:
        chunks: List of chunk dicts with 'text' key
        
    Returns:
        Filtered chunks
    """
    filter = TextSectionFilter()
    return filter.filter_chunks(chunks)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Test the filter with realistic medRxiv content
    test_text = """
    Histopathologist Features Predictive of Diagnostic Concordance at Expert Level
    Amongst a Large International Sample of Pathologists Diagnosing Barrett's Dysplasia
    Using Digital Pathology
    
    RUNNING HEAD: Quantitative model of Barrett's histopathology expert review
    
    Authors: Myrtle J. van der Wel1,2*, Helen G. Coleman3*, Jacques JGHM Bergman2
    
    Affiliations: 1Amsterdam UMC, University of Amsterdam, Dept. of Pathology, Amsterdam,
    The Netherlands; 2Amsterdam UMC, University of Amsterdam, Dept. of Gastroenterology
    
    * These first authors contributed equally
    § Shared corresponding authors
    
    Keywords: Barrett's oesophagus; Oesophageal neoplasms; Digital pathology
    
    Disclosures: No conflicts of interest to declare
    
    Author contributions: Study concept and design (MW, MJ, SM); acquisition of data (MW, MJ, SM);
    
    Sources of funding: Cancer Research UK and Dutch Cancer Society
    
    Corresponding authors: Marnix Jansen
    Email: test@example.com
    Tel: +31 20 5665648; Fax: +31 20 6917033
    
    All rights reserved. No reuse allowed without permission.
    certified by peer review) is the author/funder, who has granted medRxiv a license to display 
    the preprint in perpetuity. The copyright holder for this preprint (which was not
    this version posted June 25, 2019. ; https://doi.org/10.1101/19000174 doi: medRxiv preprint
    
    Word Count: 4344
    
    LIST OF ABBREVIATIONS
    BO; Barrett's oesophagus
    BMI; body mass index
    
    ABSTRACT
    
    Objective: Guidelines recommend expert pathology review of Barrett's oesophagus (BO)
    biopsies that reveal dysplasia, but there are no evidence-based standards to corroborate
    expert reviewer status. We investigated BO concordance rates and pathologist features
    predictive of diagnostic discordance amongst a large international cohort of gastrointestinal
    pathologists to develop a quantitative model of BO expert review.
    
    Design: Pathologists (n=55) from over 20 countries assessed 55 digitised BO biopsies from
    across the diagnostic spectrum, before and after viewing matched p53 immunohistochemistry.
    
    Results: We recorded over 6,000 individual case diagnoses. Of 2,805 H&E diagnoses, we
    found excellent concordance (>70%) for non-dysplastic Barrett's oesophagus (NDBO) and
    high-grade dysplasia (HGD), and intermediate concordance for low-grade dysplasia.
    
    Conclusion: We have developed an evidence-based quantitative model of BO histopathology
    diagnosis at expert consensus level that will inform guideline development.
    
    INTRODUCTION
    
    Barrett's oesophagus (BO) is a premalignant condition, which predisposes to esophageal
    adenocarcinoma (OAC), with a reported annual conversion rate of 0.1 - 0.2%. BO is defined
    histopathologically as the replacement of normal stratified squamous epithelial lining.
    
    The implementation of formal surveillance strategies and widespread adoption of endoscopic
    treatment techniques have led to a surge in diagnostic pathology workload.
    
    METHODS
    
    Sixty-five gastrointestinal pathologists worldwide were approached to join this study through
    either professional gastrointestinal pathology working groups or direct professional contacts.
    
    All rights reserved. No reuse allowed without permission.
    certified by peer review) is the author/funder, who has granted medRxiv a license to display 
    the preprint in perpetuity.
    
    RESULTS
    
    This study is based on assessments of digitised slides to investigate diagnostic concordance
    of BO biopsies amongst a large and heterogeneous sample of gastrointestinal pathologists.
    
    DISCUSSION
    
    We have carried out the largest investigation of diagnostic concordance of BO biopsy
    reporting amongst gastrointestinal pathologists to date.
    
    ACKNOWLEDGEMENTS
    
    We thank Dr. Smith for helpful discussions.
    The authors would like to thank the funding agencies.
    
    FUNDING
    
    This work was supported by NIH grant R01-12345.
    The funders had no role in study design.
    
    CONFLICTS OF INTEREST
    
    The authors declare that they have no competing interests.
    
    REFERENCES
    
    1. Smith J, et al. Nature 2020;123:456-789.
    2. Jones A, et al. Science 2021;234:567-890.
    3. Brown B, et al. Cell 2022;345:678-901.
    """
    
    filter = TextSectionFilter()
    result = filter.filter_text(test_text)
    
    print("="*60)
    print("TEXT SECTION FILTER TEST")
    print("="*60)
    print(f"\nOriginal length: {result.original_length} chars")
    print(f"Filtered length: {result.filtered_length} chars")
    print(f"Removed: {result.removal_percentage}%")
    print(f"Removed sections: {result.removed_sections}")
    print(f"\n{'='*60}")
    print("FILTERED TEXT (should NOT contain authors, affiliations, copyright, references):")
    print("="*60)
    print(result.filtered_text)
