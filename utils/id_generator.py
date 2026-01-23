"""
ID generation utilities for bioRxiv/medRxiv paper naming convention.
Ensures consistent, unique, and traceable identifiers across the system.
"""

from typing import Tuple


def generate_paper_id(doi: str, server: str) -> str:
    """
    Generate a unique paper ID from DOI and server.
    
    Args:
        doi: Full DOI like "10.1101/2024.01.15.575685"
        server: "biorxiv" or "medrxiv"
    
    Returns:
        Sanitized paper ID like "biorxiv_2024_01_15_575685"
    
    Examples:
        >>> generate_paper_id("10.1101/2024.01.15.575685", "biorxiv")
        'biorxiv_2024_01_15_575685'
        >>> generate_paper_id("10.1101/007a0c1a-6c21-1014-bff5-8dacd8be8f0a", "biorxiv")
        'biorxiv_007a0c1a-6c21-1014-bff5-8dacd8be8f0a'
    """
    # Extract suffix after "10.1101/"
    if doi.startswith("10.1101/"):
        doi_suffix = doi[8:]
    else:
        doi_suffix = doi
    
    # Sanitize: replace . and / with _
    sanitized = doi_suffix.replace(".", "_").replace("/", "_")
    
    # Combine with server
    paper_id = f"{server}_{sanitized}"
    
    return paper_id


def generate_figure_id(paper_id: str, page_number: int, figure_index: int) -> str:
    """
    Generate unique figure ID.
    
    Args:
        paper_id: The paper ID like "biorxiv_2024_01_15_575685"
        page_number: Page where figure appears
        figure_index: Index of figure on that page (1-based)
    
    Returns:
        Figure ID like "biorxiv_2024_01_15_575685_fig_p5_1"
    """
    return f"{paper_id}_fig_p{page_number}_{figure_index}"


def generate_subfigure_id(figure_id: str, label: str) -> str:
    """
    Generate unique sub-figure ID.
    
    Args:
        figure_id: Parent figure ID
        label: Sub-figure label (A, B, C, etc.)
    
    Returns:
        Sub-figure ID like "biorxiv_2024_01_15_575685_fig_p5_1_a"
    """
    return f"{figure_id}_{label.lower()}"


def generate_table_id(paper_id: str, page_number: int, table_index: int) -> str:
    """
    Generate unique table ID.
    
    Args:
        paper_id: The paper ID
        page_number: Page where table appears
        table_index: Index of table on that page (1-based)
    
    Returns:
        Table ID like "biorxiv_2024_01_15_575685_tbl_p4_1"
    """
    return f"{paper_id}_tbl_p{page_number}_{table_index}"


def generate_chunk_id(paper_id: str, chunk_index: int) -> str:
    """
    Generate unique chunk ID.
    
    Args:
        paper_id: The paper ID
        chunk_index: Index of the chunk (0-based)
    
    Returns:
        Chunk ID like "biorxiv_2024_01_15_575685_chunk_0"
    """
    return f"{paper_id}_chunk_{chunk_index}"


def parse_doi_parts(doi: str) -> Tuple[str, str, str, str]:
    """
    Parse DOI into components for path building.
    
    Args:
        doi: Full DOI like "10.1101/2024.01.15.575685"
    
    Returns:
        Tuple of (year, month, day, manuscript_id) or (None, None, None, suffix) for legacy
    """
    doi_suffix = doi.replace("10.1101/", "")
    parts = doi_suffix.split(".")
    
    if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 4:
        # New format: 2024.01.15.575685
        return parts[0], parts[1], parts[2], parts[3]
    else:
        # Old format: UUID
        return None, None, None, doi_suffix


def build_wasabi_base_path(doi: str, server: str) -> str:
    """
    Build Wasabi base path from DOI.
    
    Args:
        doi: Full DOI like "10.1101/2024.01.15.575685"
        server: "biorxiv" or "medrxiv"
    
    Returns:
        Path like "biorxiv/2024/01/15/575685"
    """
    year, month, day, manuscript_id = parse_doi_parts(doi)
    
    if year:
        return f"{server}/{year}/{month}/{day}/{manuscript_id}"
    else:
        return f"{server}/legacy/{manuscript_id}"


def build_wasabi_figure_url(
    doi: str, 
    server: str, 
    page: int, 
    index: int,
    bucket: str = "uyar-plagiarism"
) -> str:
    """
    Build full Wasabi S3 URL for a figure.
    
    Args:
        doi: Paper DOI
        server: Server name
        page: Page number
        index: Figure index on page
        bucket: S3 bucket name
    
    Returns:
        S3 URL like "s3://uyar-plagiarism/biorxiv/2024/01/15/575685/figures/fig_p5_1.png"
    """
    base_path = build_wasabi_base_path(doi, server)
    return f"s3://{bucket}/{base_path}/figures/fig_p{page}_{index}.png"


def build_wasabi_https_url(
    doi: str, 
    server: str, 
    page: int, 
    index: int,
    bucket: str = "uyar-plagiarism"
) -> str:
    """
    Build HTTPS URL for direct browser access.
    
    Args:
        doi: Paper DOI
        server: Server name
        page: Page number
        index: Figure index on page
        bucket: S3 bucket name
    
    Returns:
        HTTPS URL for direct access
    """
    base_path = build_wasabi_base_path(doi, server)
    return f"https://s3.wasabisys.com/{bucket}/{base_path}/figures/fig_p{page}_{index}.png"


def paper_id_to_doi(paper_id: str) -> Tuple[str, str]:
    """
    Reconstruct DOI from paper ID.
    
    Args:
        paper_id: Paper ID like "biorxiv_2024_01_15_575685"
    
    Returns:
        Tuple of (doi, server)
    """
    parts = paper_id.split("_", 1)
    server = parts[0]
    
    if len(parts) > 1:
        suffix = parts[1]
        
        # Check if it's the new date format
        suffix_parts = suffix.split("_")
        if (len(suffix_parts) >= 4 and 
            suffix_parts[0].isdigit() and 
            len(suffix_parts[0]) == 4):
            # Reconstruct date format DOI
            doi = f"10.1101/{suffix_parts[0]}.{suffix_parts[1]}.{suffix_parts[2]}.{suffix_parts[3]}"
        else:
            # Legacy format
            doi = f"10.1101/{suffix.replace('_', '.')}"
    else:
        doi = ""
    
    return doi, server


# Validation functions
def validate_paper_id(paper_id: str) -> bool:
    """Validate paper ID format"""
    if not paper_id:
        return False
    
    parts = paper_id.split("_", 1)
    if len(parts) < 2:
        return False
    
    server = parts[0]
    if server not in ["biorxiv", "medrxiv", "pmc"]:
        return False
    
    return True


def validate_figure_id(figure_id: str) -> bool:
    """Validate figure ID format"""
    if not figure_id:
        return False
    
    if "_fig_p" not in figure_id:
        return False
    
    return True


def validate_table_id(table_id: str) -> bool:
    """Validate table ID format"""
    if not table_id:
        return False
    
    if "_tbl_p" not in table_id:
        return False
    
    return True


if __name__ == "__main__":
    # Test the functions
    doi = "10.1101/2024.01.15.575685"
    server = "biorxiv"
    
    print(f"DOI: {doi}")
    print(f"Server: {server}")
    print()
    
    paper_id = generate_paper_id(doi, server)
    print(f"Paper ID: {paper_id}")
    
    figure_id = generate_figure_id(paper_id, 5, 1)
    print(f"Figure ID: {figure_id}")
    
    subfig_id = generate_subfigure_id(figure_id, "A")
    print(f"Sub-figure ID: {subfig_id}")
    
    table_id = generate_table_id(paper_id, 4, 1)
    print(f"Table ID: {table_id}")
    
    chunk_id = generate_chunk_id(paper_id, 0)
    print(f"Chunk ID: {chunk_id}")
    
    print()
    print(f"Wasabi base path: {build_wasabi_base_path(doi, server)}")
    print(f"Wasabi figure URL: {build_wasabi_figure_url(doi, server, 5, 1)}")
    print(f"Wasabi HTTPS URL: {build_wasabi_https_url(doi, server, 5, 1)}")
    
    print()
    reconstructed_doi, reconstructed_server = paper_id_to_doi(paper_id)
    print(f"Reconstructed DOI: {reconstructed_doi}")
    print(f"Reconstructed Server: {reconstructed_server}")
