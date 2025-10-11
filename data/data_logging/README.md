# VMEvalKit Data Version Logging

This directory tracks all versions of the VMEvalKit dataset package for reproducibility and version control.

## Structure

```
data_logging/
├── README.md                 # This file
├── version_log.json          # Master log of all data versions
└── versions/
    ├── v1.0_20251010.json    # Detailed version logs
    ├── v1.1_YYYYMMDD.json    # Future versions...
    └── ...
```

## Version Log Format

Each version entry contains:
- **version**: Semantic version (e.g., "1.0", "1.1", "2.0")
- **date**: Upload date (YYYYMMDD format)
- **s3_uri**: Full S3 path to the data package
- **size_mb**: Total package size in megabytes
- **file_count**: Total number of files in the package
- **questions_count**: Number of question/image files
- **outputs_count**: Number of output/video files
- **description**: Human-readable description of changes
- **git_commit**: Git commit hash when version was created
- **upload_timestamp**: ISO timestamp of S3 upload

## Usage

### Logging a New Version
```python
from data_logging.version_tracker import log_new_version

log_new_version(
    version="1.1",
    description="Added 100 new chess puzzles and rotation task variants",
    s3_uri="s3://vmevalkit/20251015/data"
)
```

### Querying Versions
```python
from data_logging.version_tracker import get_version_info, list_all_versions

# Get specific version
v1_info = get_version_info("1.0")

# List all versions
all_versions = list_all_versions()
```

## Data Package Structure

Each data package version contains:
- **questions/**: All dataset files, images, and task definitions
- **outputs/**: Model-generated videos and experiment results  
- **VERSION.md**: Human-readable version information
- **s3_sync.py**: Upload script used for that version

## Reproducibility

This logging system ensures:
1. **Version Traceability**: Every dataset version is tracked with metadata
2. **S3 Persistence**: All versions remain accessible on S3
3. **Change Documentation**: Clear descriptions of what changed between versions
4. **Size Tracking**: Monitor dataset growth over time
5. **Git Integration**: Link data versions to code versions
