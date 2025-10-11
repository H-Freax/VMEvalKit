"""
VMEvalKit Data Version Logging Module

Tracks all versions of the VMEvalKit dataset package for reproducibility 
and version control.

Usage:
    from data.data_logging import log_new_version, get_version_info
    
    # Log a new version
    log_new_version(
        version="1.1",
        description="Added new chess puzzles",
        s3_uri="s3://vmevalkit/20251015/data",
        size_mb=185.2,
        file_count=1420,
        questions_count=1350,
        outputs_count=70
    )
    
    # Get version information
    v1_info = get_version_info("1.0")
    all_versions = list_all_versions()
"""

from .version_tracker import (
    log_new_version,
    get_version_info,
    list_all_versions,
    get_latest_version,
    get_version_summary,
    print_version_summary
)

__all__ = [
    'log_new_version',
    'get_version_info', 
    'list_all_versions',
    'get_latest_version',
    'get_version_summary',
    'print_version_summary'
]

__version__ = "1.0.0"
