# VMEvalKit Quick Reference Card

## 🎯 The One Rule
**ALWAYS use InferenceRunner - NEVER use API clients directly!**

## 🐍 Python Usage

```python
from vmevalkit import InferenceRunner

# Single video generation
runner = InferenceRunner()
result = runner.run(
    model_name="luma-dream-machine",  # ← Just change this!
    image_path="maze.png",
    text_prompt="Solve this maze"
)

# From dataset task
result = runner.run_from_task(
    model_name="luma-dream-machine",
    task_data=task
)
```

## 💻 CLI Usage

```bash
# Single inference
vmevalkit inference luma-dream-machine \
    --image maze.png \
    --prompt "Solve this maze"

# Batch on dataset
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks.json \
    --max-tasks 5
```

## 🚫 Never Do This

```python
# ❌ WRONG
from vmevalkit.api_clients import LumaClient

# ❌ WRONG  
from vmevalkit.models.luma import LumaModel

# ❌ WRONG
model = ModelRegistry.load_model(...)
video = model.generate(...)  # No logging!
```

## ✅ Always Do This

```python
# ✅ CORRECT
from vmevalkit import InferenceRunner
runner = InferenceRunner()
result = runner.run(...)
```

## 📁 Output Location
- Videos: `outputs/luma_<id>.mp4`
- Logs: `outputs/inference_runs.json`
- Batch results: `outputs/batch_results/`

## 🔄 Switch Models
Just change the model name:
- `"luma-dream-machine"`
- `"google-veo-001"`
- `"runway-gen3"`

---
*See TEAM_INSTRUCTIONS.md for full details*
