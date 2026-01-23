#!/usr/bin/env python3
"""
Create a combined test PDF from two source papers for multi-source plagiarism testing.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle

# Paper A content (Lamin B1 - chromatin dynamics)
PAPER_A_TITLE = "Chromatin-lamin B1 interaction promotes genomic compartmentalization"
PAPER_A_DOI = "uuid_0bde5539-6c0d-1014-8c2e-b50b932a95b0"
PAPER_A_TEXT = """
Chromatin-lamin B1 interaction promotes genomic compartmentalization and constrains chromatin dynamics

Lei Chang, Mengfan Li, Shipeng Shao, Boxin Xue, Yingping Hou, Yiwen Zhang, Ruifeng Li, Cheng Li, Yujie Sun

State Key Laboratory of Membrane Biology, School of Life Sciences, and Biomedical Pioneering Innovation Center (BIOPIC), Peking University, Beijing, China, 100871

Abstract

The eukaryotic genome is folded into higher-order conformation accompanied with constrained dynamics for coordinated genome functions. However, the molecular machinery underlying these hierarchically organized chromatin architecture and dynamics remains poorly understood. Here by combining imaging and Hi-C sequencing, we studied the role of lamin B1 in chromatin architecture and dynamics. We found that lamin B1 depletion leads to chromatin redistribution and decompaction. Consequently, the inter-chromosomal interactions and overlap between chromosome territories are increased.

Moreover, Hi-C data revealed that lamin B1 is required for the integrity and segregation of chromatin compartments but not for the topologically associating domains (TADs). We further proved that depletion of lamin B1 leads to increased chromatin dynamics, owing to chromatin decompaction and redistribution toward nuclear interior. Taken together, our data suggest that chromatin-lamin B1 interactions promote chromosomal territory segregation and genomic compartmentalization, and confine chromatin dynamics, supporting its crucial role in chromatin higher-order structure and dynamics.

Introduction

Chromatin in the interphase nucleus of eukaryotic cells are highly compartmentalized and structured. Owing to technological breakthroughs in imaging and sequencing, chromatin higher-order structure has been increasingly studied over the last decade. Hierarchical chromatin architecture is composed of loops, TADs, active and inactive A/B compartments, and chromosome territories, in increasing scales. A number of architectural proteins and molecular machineries governing chromatin organization and dynamics have been identified. For instance, CTCF and cohesin are partly responsible for the formation and maintenance of chromatin loops and TADs. Nevertheless, CTCF does not impact higher-order genomic compartmentalization and cohesin even limits segregation of A/B compartments. Until now, the mechanisms that underlie the insulation and distribution of A/B compartments and chromosome territories remain elusive.

Microscopy and chromosome conformation capture techniques provide complementary insights into chromatin higher-order structure and sub-nuclear chromatin spatial distribution. Genomic regions that belong to A-compartments identified by Hi-C are gene rich, enriched with euchromatin histone markers and transcriptionally active. Microscopy reveals that transcriptionally active euchromatic loci prefer to localize in nuclear interior. On the contrary, B-compartments identified by Hi-C are gene poor, enriched with heterochromatin markers, and frequently associated with the nuclear lamina. These findings are concordant with imaging results that transcriptionally inactive heterochromatin is mainly found near nuclear periphery and nucleoli.

Nuclear lamina consists of many protein complexes, and lamins are the main components of nuclear lamina in most mammalian cells and can be classified into A- and B-type lamins. Lamin A and C are the most common A-type lamins and are splice variants of the same gene, while B-type lamins, B1 and B2, are the products of two different genes. Lamin B1 mainly localizes at the nuclear periphery, while A-type lamins are also found in the nucleoplasm. DamID of lamin B1 has revealed many nuclear lamina-associated genomic regions named lamina-associated domains (LADs). Typically, a mammalian genome contains 1100-1400 LADs and 71% of the genome has conserved relationship with lamina across different species.

Results

Lamin B1 depletion leads to chromatin redistribution and decompaction

To explore the potential role of lamin B1 in nuclear chromatin organization, we first investigated the subnuclear distribution of lamin B1 using super-resolution (STORM) imaging. Lamin B1 was found to be almost exclusively located at the nuclear periphery, in contrast to A-type lamins which were located at both nuclear periphery and nucleoplasm. It was previously reported that lamin B1 interacts with chromatin directly or via adaptor proteins. We then created a LMNB1 (lamin B1 encoding gene)-knockout MDA-MB-231 breast cancer cell line using the CRISPR-Cas9 genome editing tool.

We reasoned that if the anchorage of chromatin to nuclear periphery is mediated by the interaction with lamin B1, the loss of lamin B1 can lead to changes in distribution and compaction of chromatin in the nucleus. To investigate the effect of lamin B1 on chromatin spatial localization and compaction at the single chromosome level, we performed chromosome painting for chromosomes 2 and 18 using FISH probes.

In lamin B1-KO cells, chromosome 2 became significantly more centrally located while the position of chromosome 18 remained at the nuclear interior. In addition, compared with wild type cells, the volume of both chromosomes were significantly increased in lamin B1-KO cells. This expansion of chromosome territories upon lamin B1 depletion is not due to nuclear volume expansion. These findings indicate that the nuclear location and volume of individual chromosomes are affected in lamin B1-KO cells.

Lamin B1 depletion reduces the segregation of chromosome territories and A/B compartments

Changes in location and volume of chromosomes may affect the territories between chromosomes. Indeed, along with the redistribution and decompaction of chromatin, more than 50% of lamin B1-KO cells showed overlap between the territories of chromosomes 2 and 18, compared with 15.1% in wild type cells. This large-scale reorganization of chromosome territories promoted us to investigate the role of lamin B1 in genome architecture using in situ Hi-C assay, which provides information about multiscale chromatin interaction maps including chromosome compartments and TADs.

We first focused on inter-chromosomal interactions. In agreement with the FISH results, Hi-C data showed higher inter-chromosomal interaction frequency between chromosomes 2 and 18 in lamin B1-KO cells, although the interaction frequency between different chromosomes is much less than that within the same chromosome as reported in previous studies. The inter-chromosomal interaction ratio of all chromosomes also showed a significant increase in lamin B1-KO cells.

Lamin B1 is not required for TAD insulation

Within A/B compartments, chromatin is further packaged in the form of TADs, which are considered as the basic structural units of chromatin and are largely conserved between cell types and across species. We calculated insulation scores for each 40 kb bin of the Hi-C normalized matrix, and the local minima of insulation scores indicated TAD boundaries. The contact maps and insulation scores of an example region on chromosome 10 showed similar TAD patterns in WT and lamin B1-KO cells.

Discussion

Along with chromatin decompaction and relocation, the motion of genomic loci became more active in lamin B1-depleted cells. In control cells, mobility of the same chromosomal locus was found to be correlated with its subnuclear location in that genomic sites close to lamina were generally less mobile than those localized in nuclear interior. These two lines of evidence suggest that chromatin dynamics are dependent on both chromatin compaction and loci location.
"""

# Paper B content (H3.3 histone - Xenopus development)
PAPER_B_TITLE = "H3.3 is essential for gastrulation of Xenopus laevis"
PAPER_B_DOI = "uuid_0ba6a324-6c1f-1014-b700-b93a8e0baef4"
PAPER_B_TEXT = """
H3.3 is essential for gastrulation of Xenopus laevis

Figure Legends

Figure 1: H3.3 is essential for gastrulation of Xenopus laevis. A) Best-studied non-centromeric H3 histone variants in Homo sapiens and Xenopus laevis. The two well-characterized forms of non-centromeric H3 variants correspond to the replicative histones H3.1 and H3.2 and the non-replicative histone H3.3, depicted in purple and green respectively. In human, the replicative H3 form is comprised of H3.1 and H3.2 that differ by only one residue at position 96, a cysteine and a serine respectively. The non-replicative form H3.3 shares more than 96% of identity with replicative forms, with five and four residue differences with H3.1 and H3.2 respectively. Additionally, Xenopus laevis embryos possess only one replicative histone variant, H3.2. Finally, histone sequences are conserved between Homo sapiens and Xenopus laevis.

B) Logo for H3 variant sequence differences of Homo sapiens, Mus musculus, Drosophila melanogaster, Xenopus laevis, and Arabidopsis thaliana after performing multiple sequence alignment using MUSCLE. We display this alignment using WebLogo3. Specific histone variants are highlighted in green and purple. Histone variants show an extreme conservation between species.

C) Developmental assay to monitor H3.3 functions. Morpholino and eH3.3 mRNA are injected at the 2-cell stage and associated defects can be observed at the gastrulation stage if there is no rescue. Scale bar corresponds to 500µm.

Figure 2: eH3.3 AIG single mutants rescue depletion of H3.3 during Xenopus laevis early development. A) Highlights of the histone chaperone recognition motif residues of H3 variants. Dedicated histone chaperones recognize histone variants by the H3.2 SVM and H3.3 AIG motifs. Although both motifs are structurally similar, the main difference appears for the residue 90 that is critical for histone chaperone binding. Crystal structure adapted from PDB ID codes: 5B0Z (Suzuki et al. 2016) and 3AV2 (Tachiwana et al. 2011).

B) Rescue assays with H3.3 AIG single mutants. Injections are performed at 2-cell stage. The morpholino against H3.3 induces gastrulation defects that are rescued by all eH3.3 with single mutations of the AIG motif. Scale bar corresponds to 500µm. Quantification of properly developed embryos after injections of the different H3.3 AIG mutant forms shows no difference of rescue efficiency after H3.3 depletion. Each experiment has been reproduced at least 3 times with a minimum of 30 embryos.

Figure 3: eH3.3 AIG triple mutant rescues depletion of H3.3 during Xenopus laevis early development. Rescue assays with eH3.3 triple AIG mutant. Injections are performed at 2-cell stage. The morpholino against H3.3 induces gastrulation defects that are not rescued by eH3.2. In contrast, a triple mutant of H3.3 for the AIG motif, eH3.3 SVM, rescues the phenotype while carrying the same SVM motif than H3.2. Scale bar corresponds to 500µm. Quantification of properly developed embryos after injection of the eH3.3 triple AIG mutant form shows similar rescue efficiency than eH3.3 WT after H3.3 depletion.

Figure 4: Swapping the AIG motif to the SVM motif leads to changes in chaperone interactions and histone modes of incorporation into chromatin in vivo. A) Immunoprecipitation of eH3 variant forms in interphase extract. Recombinant proteins are produced in rabbit reticulocyte lysates and pulled down by their HA-tag after incubation. HIRA only recognizes the AIG motif while p60 recognizes solely the SVM motif. B) Incorporation of eH3 variant forms into sperm chromatin in interphase extract. Purified nuclei remodeled in the interphase extracts supplemented with indicated eH3.3 in the presence or absence of aphidicolin were analyzed by WB with indicated antibodies.

Figure 5: H3.3S31 is critical Xenopus laevis early development and is phosphorylated in this model organism. A) Rescue assays with eH3.3 S31A mutant. Injections are performed at 2-cell stage. Compared with eH3.3 WT, eH3.3 S31A mutant form cannot rescue the phenotype. Scale bar corresponds to 500µm. B) 3D-distribution and timing of H3.3S31 phosphorylation in A6 cell line. H3.3S31p follows the same dynamics than H3S10p, but does not localize to the same physical places. Scale bar represents 10 µm.

C) Characterization of H3.3S31 phosphorylation in Xenopus cell-free extract. Non-remodeled sperm nuclei and nuclei purified after remodeling in interphase or mitotic extracts, H3 PTMs are analyzed by WB. H3.3S31p and H3S10p are enriched exclusively in sperm nuclei purified from mitotic extracts. The marks are prevalently found in the chromatin and not in H3.3 soluble form.

Figure 6: H3.3S31 negative charge is essential to rescue depletion of H3.3 during Xenopus laevis early development. A) Rescue assays with eH3.3 S31D mutant. Injections are performed at 2-cell stage. In contrast to eH3.3 S31A, the use of an exogenous phosphomimic mutant, eH3.3 S31D, rescues, as the eH3.3 WT, the defects during gastrulation. Scale bar corresponds to 500µm. Quantification of properly developed embryos after injection of the H3.3 S31D mutant form shows similar rescue efficiency than eH3.3 WT after H3.3 depletion.

B) Immunoprecipitation of eH3 S31 mutant forms in interphase extract. Recombinant proteins are produced in rabbit reticulocyte lysates and pulled down by their HA-tag after incubation. Mutations of the residue 31 do not alter histone chaperone interactions.

C) Incorporation of eH3 S31 mutant forms into sperm chromatin in interphase extract. Purified nuclei remodeled in the interphase extracts supplemented with indicated eH3.3 in the presence or absence of aphidicolin were analyzed by WB with indicated antibodies. Mutations of the residue 31 do not change the mode of incorporation of these forms.

Figure 7: Graphical abstract. Defects associated with H3.3 depletion can be rescued with H3 histone variants carrying a potential negatively charge residue regardless of the mode of incorporation.

Supplemental figure 1: Histone chaperone sequence conservation and morpholino strategy and titration. A) Logo for UBN1, DAXX and p60 partial sequences of Homo sapiens, Mus musculus, Drosophila melanogaster, Xenopus laevis after performing multiple sequence alignment with MUSCLE. Complete histone-binding domain HRD (131-171) of UBN1 is presented, while only a part (from 300 to 339) of the DAXX histone-binding domain HBD (178-389) is displayed.

B) Sequences of the morpholino against endogenous H3.3 and its targeted sequence. The corresponding sequences matching the targeted sequence are highlighted in green for the first 10 codons of Xenopus laevis H3F3B and eH3 sequences.

Supplemental figure 2: Single mutant forms of eH3.3 AIG motif are all expressed and incorporated in a similar fashion in vivo. A) Western blot to control eH3.3 AIG single mutant expression in embryos. Every eH3.3 AIG single mutants are expressed at similar levels in whole embryo extracts at stage 12.

Supplemental figure 3: Triple mutant form of eH3.3 AIG motif is efficiently expressed and incorporated in vivo. A) Western blot to control eH3.3 AIG triple mutant expression in embryos. eH3.3 AIG triple mutant expression compares with the other eH3 conditions in whole embryo extracts at stage 12.

Supplemental figure 4: p60 histone chaperone interaction in vivo and mode of incorporation of eH3.3 AIG triple mutant in mitotic extract in vitro. A) Immunoprecipitation of eH3 variant forms from stage 12 embryos. Only the SVM motif is recognized by p60 in vivo.

Supplemental figure 5: H3.3S31 phosphorylation in human has the same dynamics that in Xenopus laevis. A) Highlights of the histone tail specific residue of H3 variants. H3.3 tail possesses a serine at position 31 that can be phosphorylated, while the alanine of H3.2 cannot.

Supplemental figure 6: Mutant forms of H3.3S31 are all expressed and incorporated in a similar fashion. A) Western blot to control eH3.3S31 mutant expression in embryos. eH3.3S31 mutant expression compares with the other eH3 conditions in whole embryo extracts at stage 12.
"""

def create_combined_pdf():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    
    output_path = "/home/claude/uyari_final/uyari_updated/test_papers/multi_source_test.pdf"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                           leftMargin=72, rightMargin=72,
                           topMargin=72, bottomMargin=72)
    
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
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=15
    )
    
    story = []
    
    # Title page
    story.append(Paragraph("MULTI-SOURCE PLAGIARISM TEST DOCUMENT", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("This document combines content from TWO different source papers:", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Paper A:</b> {PAPER_A_TITLE}", body_style))
    story.append(Paragraph(f"<b>DOI:</b> {PAPER_A_DOI}", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Paper B:</b> {PAPER_B_TITLE}", body_style))
    story.append(Paragraph(f"<b>DOI:</b> {PAPER_B_DOI}", body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Expected Result:</b> Plagiarism detection should identify matches from BOTH sources.", body_style))
    story.append(PageBreak())
    
    # Paper A content
    story.append(Paragraph("=" * 50, body_style))
    story.append(Paragraph("SECTION 1: CONTENT FROM PAPER A", title_style))
    story.append(Paragraph(f"Source: {PAPER_A_DOI}", body_style))
    story.append(Paragraph("=" * 50, body_style))
    story.append(Spacer(1, 20))
    
    # Split Paper A into paragraphs
    paragraphs_a = [p.strip() for p in PAPER_A_TEXT.split('\n\n') if p.strip()]
    for para in paragraphs_a:
        # Clean up the text
        para = para.replace('\n', ' ').strip()
        if para:
            story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # Paper B content
    story.append(Paragraph("=" * 50, body_style))
    story.append(Paragraph("SECTION 2: CONTENT FROM PAPER B", title_style))
    story.append(Paragraph(f"Source: {PAPER_B_DOI}", body_style))
    story.append(Paragraph("=" * 50, body_style))
    story.append(Spacer(1, 20))
    
    # Split Paper B into paragraphs
    paragraphs_b = [p.strip() for p in PAPER_B_TEXT.split('\n\n') if p.strip()]
    for para in paragraphs_b:
        para = para.replace('\n', ' ').strip()
        if para:
            story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 8))
    
    # Build PDF
    doc.build(story)
    print(f"Created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_combined_pdf()
