# VMEvalKit Team Instructions

## 🚨 IMPORTANT: How to Use VMEvalKit Properly

This document outlines the **correct** way to use VMEvalKit for video generation inference. Please read this before writing any code.

---

## ✅ The Golden Rule

**ALWAYS use the inference module - NEVER call API clients directly!**

```python
# ✅ CORRECT
from vmevalkit import InferenceRunner

# ❌ WRONG - Never do this!
from vmevalkit.api_clients import LumaClient
```

---

## 🏗️ Architecture Overview

VMEvalKit follows a clean, layered architecture:

```
┌─────────────────────────────────────────┐
│          CLI (vmevalkit)                │  ← Command line interface
├─────────────────────────────────────────┤
│     InferenceRunner / BatchRunner       │  ← ALWAYS USE THIS LAYER
├─────────────────────────────────────────┤
│         ModelRegistry                   │  ← Handles model loading
├─────────────────────────────────────────┤
│      BaseVideoModel (Abstract)          │  ← Defines interface
├─────────────────────────────────────────┤
│   LumaModel, VeoModel, RunwayModel     │  ← Model implementations
├─────────────────────────────────────────┤
│   LumaClient, VeoClient, etc.          │  ← Low-level API clients
└─────────────────────────────────────────┘
```

**Your code should ONLY interact with the InferenceRunner layer!**

---

## 📋 Quick Reference

### Python API

```python
from vmevalkit import InferenceRunner, BatchInferenceRunner

# Single inference
runner = InferenceRunner(output_dir="./outputs")
result = runner.run(
    model_name="luma-dream-machine",  # Just change this to switch models!
    image_path="maze.png",
    text_prompt="Solve this maze",
    duration=5.0
)

# From task file
result = runner.run_from_task(
    model_name="luma-dream-machine",
    task_data={"prompt": "...", "first_image_path": "..."}
)

# Batch processing
batch_runner = BatchInferenceRunner()
results = batch_runner.run_dataset(
    model_name="luma-dream-machine",
    dataset_path="data/tasks.json"
)
```

### Command Line

```bash
# Single inference
vmevalkit inference luma-dream-machine \
    --image maze.png \
    --prompt "Solve this maze"

# Batch inference
vmevalkit batch luma-dream-machine \
    --dataset data/tasks.json
```

---

## ❌ What NOT to Do

### 1. Don't Use API Clients Directly

```python
# ❌ WRONG - Breaks abstraction
from vmevalkit.api_clients import LumaClient
client = LumaClient(api_key="...")
generation_id = client.generate_from_image(...)
```

### 2. Don't Import Model Classes Directly

```python
# ❌ WRONG - Use ModelRegistry instead
from vmevalkit.models.luma import LumaModel
model = LumaModel()
```

### 3. Don't Skip the Runner

```python
# ❌ WRONG - Loses logging, error handling, etc.
model = ModelRegistry.load_model("luma-dream-machine")
video = model.generate(...)  # Where's the logging? Error handling?
```

---

## 🎯 Why Use the Inference Module?

1. **Unified Interface**: Same code works for ALL models
   ```python
   # Switch models with just one parameter!
   runner.run(model_name="luma-dream-machine", ...)
   runner.run(model_name="google-veo-001", ...)  # Same interface!
   ```

2. **Automatic Logging**: Every run is tracked in `outputs/inference_runs.json`

3. **Consistent Error Handling**: Failures are handled gracefully

4. **Built-in Features**:
   - Batch processing
   - Model comparison
   - Progress tracking
   - Result organization

5. **CLI Integration**: Can be used from command line

6. **Future-Proof**: New models automatically work with existing code

---

## 📚 Common Use Cases

### 1. Testing a Single Maze
```python
from vmevalkit import InferenceRunner

runner = InferenceRunner()
result = runner.run(
    model_name="luma-dream-machine",
    image_path="data/generated_mazes/irregular_0001_first.png",
    text_prompt="Navigate through the maze from start to finish"
)

if result["status"] == "success":
    print(f"Video saved: {result['video_path']}")
```

### 2. Running on Dataset
```python
from vmevalkit import BatchInferenceRunner

batch_runner = BatchInferenceRunner()
results = batch_runner.run_dataset(
    model_name="luma-dream-machine",
    dataset_path="data/maze_tasks/irregular_tasks.json",
    max_tasks=5  # Start small for testing
)
```

### 3. Comparing Models
```python
comparison = batch_runner.run_models_comparison(
    model_names=["luma-dream-machine", "google-veo-001"],
    dataset_path="data/maze_tasks/irregular_tasks.json"
)
```

### 4. Using CLI for Quick Tests
```bash
# Test single task from dataset
vmevalkit inference luma-dream-machine \
    --task-file data/maze_tasks/irregular_tasks.json \
    --task-id irregular_0001
```

---

## 🔧 Environment Setup

1. Create `.env` file:
   ```bash
   LUMA_API_KEY=your_key_here
   AWS_ACCESS_KEY_ID=your_aws_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret
   S3_BUCKET=your-bucket
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. You're ready to use the inference module!

---

## 📖 Additional Resources

- **Usage Guide**: See `USAGE.md` for comprehensive examples
- **Inference Module**: See `vmevalkit/inference/README.md`
- **Examples**: Check `examples/` directory

---

## 🚀 Quick Start Template

Save this as your starting point:

```python
#!/usr/bin/env python3
"""
Template for using VMEvalKit inference.
"""

from vmevalkit import InferenceRunner

def main():
    # Initialize runner
    runner = InferenceRunner(output_dir="./outputs")
    
    # Run inference
    result = runner.run(
        model_name="luma-dream-machine",  # Change model here
        image_path="path/to/image.png",
        text_prompt="Your prompt here",
        duration=5.0
    )
    
    # Check result
    if result["status"] == "success":
        print(f"✅ Success! Video: {result['video_path']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
```

---

## ⚠️ Exception: Testing API Clients

The ONLY time you should use API clients directly is when:
1. Testing the API client implementation itself
2. Debugging low-level API issues
3. Implementing new model support

These should be in files clearly marked as tests (e.g., `test_luma_client.py`).

---

## 💡 Remember

- **InferenceRunner** is your friend - it handles all the complexity
- **Model names** are just strings - easy to switch
- **Everything is logged** - check `outputs/inference_runs.json`
- **Use the CLI** for quick experiments
- **Read the docs** - `USAGE.md` has many examples

---

*Last updated: January 2025*
*Questions? Check the codebase documentation or ask the team lead.*
