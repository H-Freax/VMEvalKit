# VMEvalKit Experiments

Date-specific experiment scripts for reproducible video generation model evaluation.

## Current Experiment: 2025-10-14

**File:** `experiment_2025-10-14.py`

### Overview
Comprehensive evaluation of 6 state-of-the-art text+image→video models across ALL human-approved VMEvalKit tasks:

| Model | Provider | Description |
|-------|----------|-------------|
| `luma-ray-2` | Luma Dream Machine | Latest high-quality model |
| `veo-3.0-generate` | Google Veo 3.0 | Advanced capabilities preview |
| `veo-3.1-720p` | Google Veo 3.1 (via WaveSpeed) | 720p with audio generation |
| `runway-gen4-turbo` | Runway ML | Fast high-quality generation |
| `openai-sora-2` | OpenAI Sora | High-quality video generation |
| `wavespeed-wan-2.2-i2v-720p` | WaveSpeed WAN 2.2 | High-resolution I2V |

### Features
- **Parallel Execution**: 6 concurrent workers (one per model)
- **Structured Output**: Each inference creates self-contained folder
- **Human Curation**: Only processes tasks with existing folders 
- **Progress Tracking**: Real-time statistics and intermediate saves

### Quick Start

```bash
# Ensure venv is activated
source venv/bin/activate

# Set required API keys
export LUMA_API_KEY="your_key"
export GOOGLE_PROJECT_ID="your_project" 
export WAVESPEED_API_KEY="your_key"
export RUNWAYML_API_SECRET="your_key"
export OPENAI_API_KEY="your_key"

# Run experiment
python examples/experiment_2025-10-14.py
```

### Output Structure
```
data/outputs/pilot_experiment/
├── logs/
│   ├── logs_final.json          # Detailed results
│   ├── statistics.json          # Performance stats
│   └── SUMMARY.txt             # Human-readable report
├── {model}_{task_id}_{timestamp}/
│   ├── video/                   # Generated video
│   ├── question/               # Input images + prompt
│   └── metadata.json          # Complete inference data
```

### Task Domains
- **Chess**: Strategic move prediction
- **Maze**: Path-finding navigation  
- **Raven**: Abstract pattern reasoning
- **Rotation**: Spatial transformation

### Requirements
- All API keys configured
- VMEvalKit installed with dependencies
- Human-approved tasks in `data/questions/`

### Reproducibility
Each experiment is date-stamped to preserve exact code used for future reference and result reproduction.
