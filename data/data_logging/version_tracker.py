#!/usr/bin/env python3
"""
VMEvalKit Data Version Tracker

Utilities for logging and querying data package versions.
Tracks all dataset versions uploaded to S3 with comprehensive metadata.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Paths
DATA_LOGGING_DIR = Path(__file__).parent
VERSION_LOG_PATH = DATA_LOGGING_DIR / "version_log.json"
VERSIONS_DIR = DATA_LOGGING_DIR / "versions"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except:
        return 'unknown'


def load_version_log() -> Dict[str, Any]:
    """Load the master version log."""
    if VERSION_LOG_PATH.exists():
        with open(VERSION_LOG_PATH) as f:
            return json.load(f)
    return {
        'description': 'VMEvalKit Dataset Version Log',
        'created': datetime.now().isoformat() + 'Z',
        'last_updated': datetime.now().isoformat() + 'Z',
        'versions': []
    }


def save_version_log(log: Dict[str, Any]) -> None:
    """Save the master version log."""
    log['last_updated'] = datetime.now().isoformat() + 'Z'
    with open(VERSION_LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


def log_new_version(
    version: str,
    description: str,
    s3_uri: str,
    size_mb: float,
    file_count: int,
    questions_count: int,
    outputs_count: int,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a new data version.
    
    Args:
        version: Version string (e.g., "1.1", "2.0")
        description: Human-readable description of changes
        s3_uri: Full S3 URI to the data package
        size_mb: Total package size in MB
        file_count: Total number of files
        questions_count: Number of question/image files
        outputs_count: Number of output/result files
        additional_metadata: Optional additional information
    """
    # Load current log
    log = load_version_log()
    
    # Create version entry
    date_str = datetime.now().strftime('%Y%m%d')
    version_entry = {
        'version': version,
        'date': date_str,
        's3_uri': s3_uri,
        'size_mb': size_mb,
        'file_count': file_count,
        'questions_count': questions_count,
        'outputs_count': outputs_count,
        'description': description,
        'git_commit': get_git_commit(),
        'upload_timestamp': datetime.now().isoformat() + 'Z'
    }
    
    # Add to log
    log['versions'].append(version_entry)
    save_version_log(log)
    
    # Create detailed version file
    detailed_log = {
        **version_entry,
        'release_name': f'VMEvalKit v{version}',
        's3_location': {
            'uri': s3_uri,
            'bucket': s3_uri.split('/')[2] if s3_uri.startswith('s3://') else 'unknown',
            'prefix': '/'.join(s3_uri.split('/')[3:]) if s3_uri.startswith('s3://') else 'unknown'
        },
        'technical_details': {
            'git_commit': get_git_commit(),
            'upload_timestamp': datetime.now().isoformat() + 'Z',
            'data_structure_version': 'unified_v1',
            'sync_script_version': 's3_sync.py v1.0'
        }
    }
    
    if additional_metadata:
        detailed_log.update(additional_metadata)
    
    # Save detailed log
    detailed_path = VERSIONS_DIR / f"v{version}_{date_str}.json"
    with open(detailed_path, 'w') as f:
        json.dump(detailed_log, f, indent=2)
    
    print(f"✅ Logged version {version} to {s3_uri}")
    print(f"📊 {file_count} files, {size_mb:.2f} MB")


def get_version_info(version: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific version."""
    log = load_version_log()
    for v in log['versions']:
        if v['version'] == version:
            return v
    return None


def list_all_versions() -> List[Dict[str, Any]]:
    """List all logged versions."""
    log = load_version_log()
    return log['versions']


def get_latest_version() -> Optional[Dict[str, Any]]:
    """Get the most recent version."""
    versions = list_all_versions()
    if not versions:
        return None
    return max(versions, key=lambda v: v['date'])


def get_version_summary() -> Dict[str, Any]:
    """Get a summary of all versions."""
    versions = list_all_versions()
    if not versions:
        return {'total_versions': 0}
    
    total_size = sum(v['size_mb'] for v in versions)
    total_files = sum(v['file_count'] for v in versions)
    
    return {
        'total_versions': len(versions),
        'latest_version': get_latest_version()['version'] if versions else None,
        'total_cumulative_size_mb': total_size,
        'total_cumulative_files': total_files,
        'first_version_date': min(v['date'] for v in versions),
        'latest_version_date': max(v['date'] for v in versions)
    }


def print_version_summary() -> None:
    """Print a formatted summary of all versions."""
    print("📊 VMEvalKit Data Version Summary")
    print("=" * 50)
    
    summary = get_version_summary()
    if summary['total_versions'] == 0:
        print("No versions logged.")
        return
    
    print(f"Total Versions: {summary['total_versions']}")
    print(f"Latest Version: v{summary['latest_version']}")
    print(f"Version Range: {summary['first_version_date']} → {summary['latest_version_date']}")
    print()
    
    print("Version Details:")
    for version in list_all_versions():
        print(f"  v{version['version']} ({version['date']})")
        print(f"    📦 {version['file_count']} files, {version['size_mb']:.2f} MB")
        print(f"    📍 {version['s3_uri']}")
        print(f"    📝 {version['description']}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "summary":
            print_version_summary()
        elif sys.argv[1] == "list":
            versions = list_all_versions()
            for v in versions:
                print(f"v{v['version']} - {v['description']}")
        elif sys.argv[1] == "latest":
            latest = get_latest_version()
            if latest:
                print(f"v{latest['version']} ({latest['date']}) - {latest['s3_uri']}")
            else:
                print("No versions found.")
        else:
            print("Usage: python version_tracker.py [summary|list|latest]")
    else:
        print_version_summary()
