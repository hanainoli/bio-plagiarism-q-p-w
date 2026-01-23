"""
Utility functions for plagiarism detection system.
"""

from .id_generator import (
    generate_paper_id,
    generate_figure_id,
    generate_subfigure_id,
    generate_table_id,
    generate_chunk_id,
    parse_doi_parts,
    build_wasabi_base_path,
    build_wasabi_figure_url,
    build_wasabi_https_url,
    paper_id_to_doi,
    validate_paper_id,
    validate_figure_id,
    validate_table_id,
)

__all__ = [
    'generate_paper_id',
    'generate_figure_id',
    'generate_subfigure_id',
    'generate_table_id',
    'generate_chunk_id',
    'parse_doi_parts',
    'build_wasabi_base_path',
    'build_wasabi_figure_url',
    'build_wasabi_https_url',
    'paper_id_to_doi',
    'validate_paper_id',
    'validate_figure_id',
    'validate_table_id',
]
