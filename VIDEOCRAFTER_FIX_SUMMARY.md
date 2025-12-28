# VideoCrafter Integration Fix

## Problem Summary

The previous `videocrafter_inference.py` was a **non-functional scaffold** that pretended to do video generation but actually just:
- Created a temporary Python script
- Ran it via subprocess
- Inside that script, **duplicated the input image N times** (no diffusion!)
- Saved it as a video (basically a slideshow)

### Red Flags in the Old Implementation

1. **No actual diffusion**: The core generation loop was literally `for i in range(num_frames): frames.append(np.array(image))`
2. **Subprocess hack**: Created temp scripts and ran them via `subprocess.run()`, reloading the model every time
3. **Ignored text prompts**: Had placeholder comments like "simplified version - actual VideoCrafter may need more sophisticated text encoding" but never actually did it
4. **Model loading without using it**: Created a `DDIMSampler(model)` but never called any sampling methods
5. **Hard-coded paths**: Filesystem-specific paths baked into generated code

## Solution: Proper Integration

### What Changed

**Complete rewrite** following the pattern from VideoCrafter's own code:

1. **Load model once at initialization** (in `__init__`)
   - No subprocess spawning
   - No repeated model loading
   - Model stays on GPU in eval mode

2. **Real diffusion-based video generation**
   - Uses `load_image_batch()` for proper image preprocessing
   - Gets text embeddings: `model.get_learned_conditioning([prompt])`
   - Gets image embeddings: `model.get_image_embeds(cond_images)`
   - Concatenates for i2v conditioning: `torch.cat([text_emb, img_emb], dim=1)`
   - Runs actual DDIM sampling: `batch_ddim_sampling(...)`
   - Decodes latents to video frames via the model's VAE

3. **Proper imports from VideoCrafter submodule**
   - Imports `funcs.py` for helper functions (load_model_checkpoint, load_image_batch, batch_ddim_sampling)
   - Uses `utils.utils.instantiate_from_config` for model instantiation
   - Follows the exact pattern from `predict.py` and `scripts/gradio/i2v_test.py`

### Key Methods

#### `_load_model()`
Loads VideoCrafter model once:
- Reads config from `configs/inference_i2v_512_v1.0.yaml`
- Loads checkpoint from `weights/videocrafter/base_512_v2/model.ckpt`
- Disables gradient checkpointing for faster inference
- Moves to GPU and sets eval mode

#### `_run_videocrafter_inference()`
Real inference pipeline:
```python
# 1. Preprocess image
cond_images = load_image_batch([image_path], (height, width))

# 2. Get conditioning
text_emb = model.get_learned_conditioning([text_prompt])
img_emb = model.get_image_embeds(cond_images)
imtext_cond = torch.cat([text_emb, img_emb], dim=1)

# 3. Run diffusion sampling
batch_samples = batch_ddim_sampling(
    model, cond, noise_shape,
    ddim_steps=50, cfg_scale=12.0, ...
)

# 4. Decode and save video
```

### Verification

The old code had this tell-tale sign:
```python
# For now, we'll create a basic video by duplicating the input image
# Real implementation would use model.sample() or similar
frames = []
for i in range(num_frames):
    frames.append(np.array(image))
```

The new code does:
```python
# Run DDIM sampling (actual diffusion inference!)
batch_samples = batch_ddim_sampling(
    self.model, cond, noise_shape,
    n_samples=1, ddim_steps=ddim_steps,
    ddim_eta=ddim_eta, cfg_scale=cfg_scale, **kwargs
)
# batch_samples shape: [batch, samples, c, t, h, w]
```

## Testing

To test the new implementation:

```bash
# 1. Ensure model is set up
bash setup/models/videocrafter2-512/setup.sh

# 2. Run a test generation
python -m vmevalkit.run --models videocrafter2-512 --experiment test_real_inference
```

Expected behavior:
- Model loads **once** (not per-generation)
- Actual diffusion inference runs (~30-60 seconds on GPU)
- Generated video shows **motion**, not repeated frames
- Console shows DDIM sampling progress

## References

The implementation now matches these official VideoCrafter files:
- `submodules/VideoCrafter/predict.py` (lines 75-155)
- `submodules/VideoCrafter/scripts/gradio/i2v_test.py` (lines 31-68)
- `submodules/VideoCrafter/scripts/evaluation/inference.py` (lines 109-125)

## Impact

✅ **Before**: Placeholder code that created slideshows  
✅ **After**: Real diffusion-based video generation  

✅ **Before**: Subprocess overhead, model reloading every call  
✅ **After**: Load once, reuse model, proper GPU utilization  

✅ **Before**: Text prompts ignored  
✅ **After**: Full text+image conditioning  

✅ **Before**: No actual VideoCrafter inference  
✅ **After**: Calls the real DDIM sampling pipeline  

---

**Bottom line**: This is now a real VideoCrafter integration, not a scaffold pretending to be one.

