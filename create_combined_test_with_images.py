#!/usr/bin/env python3
"""
Create a combined test PDF with text AND images from two source papers.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import inch
import os

# Paper A content (Lamin B1 - chromatin dynamics)
PAPER_A_TITLE = "Chromatin-lamin B1 interaction promotes genomic compartmentalization"
PAPER_A_DOI = "uuid_0bde5539-6c0d-1014-8c2e-b50b932a95b0"
PAPER_A_TEXT = """
Chromatin-lamin B1 interaction promotes genomic compartmentalization and constrains chromatin dynamics

Lei Chang, Mengfan Li, Shipeng Shao, Boxin Xue, Yingping Hou, Yiwen Zhang, Ruifeng Li, Cheng Li, Yujie Sun

State Key Laboratory of Membrane Biology, School of Life Sciences, and Biomedical Pioneering Innovation Center (BIOPIC), Peking University, Beijing, China, 100871

Abstract

The eukaryotic genome is folded into higher-order conformation accompanied with constrained dynamics for coordinated genome functions. However, the molecular machinery underlying these hierarchically organized chromatin architecture and dynamics remains poorly understood. Here by combining imaging and Hi-C sequencing, we studied the role of lamin B1 in chromatin architecture and dynamics. We found that lamin B1 depletion leads to chromatin redistribution and decompaction. Consequently, the inter-chromosomal interactions and overlap between chromosome territories are increased.

Moreover, Hi-C data revealed that lamin B1 is required for the integrity and segregation of chromatin compartments but not for the topologically associating domains (TADs). We further proved that depletion of lamin B1 leads to increased chromatin dynamics, owing to chromatin decompaction and redistribution toward nuclear interior.

Introduction

Chromatin in the interphase nucleus of eukaryotic cells are highly compartmentalized and structured. Owing to technological breakthroughs in imaging and sequencing, chromatin higher-order structure has been increasingly studied over the last decade. Hierarchical chromatin architecture is composed of loops, TADs, active and inactive A/B compartments, and chromosome territories, in increasing scales.

Nuclear lamina consists of many protein complexes, and lamins are the main components of nuclear lamina in most mammalian cells and can be classified into A- and B-type lamins. Lamin B1 mainly localizes at the nuclear periphery, while A-type lamins are also found in the nucleoplasm.

Results

Lamin B1 depletion leads to chromatin redistribution and decompaction. To explore the potential role of lamin B1 in nuclear chromatin organization, we first investigated the subnuclear distribution of lamin B1 using super-resolution (STORM) imaging.

Discussion

Along with chromatin decompaction and relocation, the motion of genomic loci became more active in lamin B1-depleted cells. In control cells, mobility of the same chromosomal locus was found to be correlated with its subnuclear location.
"""

# Paper B content (H3.3 histone - Xenopus development)
PAPER_B_TITLE = "H3.3 is essential for gastrulation of Xenopus laevis"
PAPER_B_DOI = "uuid_0ba6a324-6c1f-1014-b700-b93a8e0baef4"
PAPER_B_TEXT = """
H3.3 is essential for gastrulation of Xenopus laevis

Figure 1: H3.3 is essential for gastrulation of Xenopus laevis. The two well-characterized forms of non-centromeric H3 variants correspond to the replicative histones H3.1 and H3.2 and the non-replicative histone H3.3. In human, the replicative H3 form is comprised of H3.1 and H3.2 that differ by only one residue at position 96.

Figure 2: eH3.3 AIG single mutants rescue depletion of H3.3 during Xenopus laevis early development. Dedicated histone chaperones recognize histone variants by the H3.2 SVM and H3.3 AIG motifs. The morpholino against H3.3 induces gastrulation defects that are rescued by all eH3.3 with single mutations of the AIG motif.

Figure 3: eH3.3 AIG triple mutant rescues depletion of H3.3 during Xenopus laevis early development. In contrast, a triple mutant of H3.3 for the AIG motif, eH3.3 SVM, rescues the phenotype while carrying the same SVM motif than H3.2.

Figure 4: Swapping the AIG motif to the SVM motif leads to changes in chaperone interactions and histone modes of incorporation into chromatin in vivo. HIRA only recognizes the AIG motif while p60 recognizes solely the SVM motif.

Figure 5: H3.3S31 is critical Xenopus laevis early development and is phosphorylated in this model organism. Compared with eH3.3 WT, eH3.3 S31A mutant form cannot rescue the phenotype.

Figure 6: H3.3S31 negative charge is essential to rescue depletion of H3.3 during Xenopus laevis early development. In contrast to eH3.3 S31A, the use of an exogenous phosphomimic mutant, eH3.3 S31D, rescues the defects during gastrulation.

Figure 7: Graphical abstract. Defects associated with H3.3 depletion can be rescued with H3 histone variants carrying a potential negatively charge residue regardless of the mode of incorporation.
"""

def get_image_size(img_path, max_width=5*inch, max_height=4*inch):
    """Calculate appropriate image size maintaining aspect ratio."""
    from PIL import Image as PILImage
    try:
        with PILImage.open(img_path) as img:
            w, h = img.size
            ratio = min(max_width/w, max_height/h)
            return w*ratio, h*ratio
    except:
        return max_width, max_height

def create_combined_pdf():
    output_path = "/home/claude/uyari_final/uyari_updated/test_papers/multi_source_test_with_images.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           leftMargin=54, rightMargin=54,
                           topMargin=54, bottomMargin=54)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        spaceAfter=15,
        spaceBefore=5
    )
    
    story = []
    img_dir = "test_papers/images"
    
    # Title page
    story.append(Paragraph("MULTI-SOURCE PLAGIARISM TEST DOCUMENT", title_style))
    story.append(Paragraph("(WITH TEXT AND IMAGES)", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("This document combines content from TWO different source papers:", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Paper A:</b> {PAPER_A_TITLE}", body_style))
    story.append(Paragraph(f"<b>DOI:</b> {PAPER_A_DOI}", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Paper B:</b> {PAPER_B_TITLE}", body_style))
    story.append(Paragraph(f"<b>DOI:</b> {PAPER_B_DOI}", body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Expected Result:</b> Plagiarism detection should identify TEXT and IMAGE matches from BOTH sources.", body_style))
    story.append(PageBreak())
    
    # ==================== PAPER A SECTION ====================
    story.append(Paragraph("=" * 50, body_style))
    story.append(Paragraph("SECTION 1: CONTENT FROM PAPER A", title_style))
    story.append(Paragraph(f"Source: {PAPER_A_DOI}", body_style))
    story.append(Paragraph("=" * 50, body_style))
    story.append(Spacer(1, 20))
    
    # Paper A text
    paragraphs_a = [p.strip() for p in PAPER_A_TEXT.split('\n\n') if p.strip()]
    for para in paragraphs_a:
        para = para.replace('\n', ' ').strip()
        if para:
            story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # ==================== PAPER B SECTION ====================
    story.append(Paragraph("=" * 50, body_style))
    story.append(Paragraph("SECTION 2: CONTENT FROM PAPER B", title_style))
    story.append(Paragraph(f"Source: {PAPER_B_DOI}", body_style))
    story.append(Paragraph("=" * 50, body_style))
    story.append(Spacer(1, 20))
    
    # Paper B text
    paragraphs_b = [p.strip() for p in PAPER_B_TEXT.split('\n\n') if p.strip()]
    for para in paragraphs_b:
        para = para.replace('\n', ' ').strip()
        if para:
            story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # ==================== FIGURES SECTION ====================
    story.append(Paragraph("=" * 50, body_style))
    story.append(Paragraph("FIGURES FROM PAPER B", title_style))
    story.append(Paragraph(f"Source: {PAPER_B_DOI}", body_style))
    story.append(Paragraph("=" * 50, body_style))
    story.append(Spacer(1, 20))
    
    # Add all images
    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(img_dir, img_file)
        try:
            w, h = get_image_size(img_path, max_width=5.5*inch, max_height=4*inch)
            img = Image(img_path, width=w, height=h)
            story.append(img)
            story.append(Paragraph(f"Figure {i+1}: {img_file}", caption_style))
            story.append(Spacer(1, 15))
            
            # Page break after every 2 images
            if (i + 1) % 2 == 0 and i < len(image_files) - 1:
                story.append(PageBreak())
        except Exception as e:
            print(f"Error adding {img_file}: {e}")
            story.append(Paragraph(f"[Figure {i+1}: {img_file} - Error loading]", caption_style))
    
    # Build PDF
    doc.build(story)
    print(f"Created: {output_path}")
    print(f"Total images: {len(image_files)}")
    return output_path

if __name__ == "__main__":
    create_combined_pdf()
