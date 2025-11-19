# Object Subtraction Task Documentation

## Overview

The Object Subtraction Task evaluates video generation models' ability to demonstrate **selective attention**, **inhibitory control**, and **causal consistency** by generating videos that show the removal of specific objects while keeping others stationary.

This task introduces a scalable, cognitively layered reasoning benchmark where models must:
- Understand selective removal instructions
- Remove specific objects based on explicit or abstract rules
- Keep non-target objects unchanged (do not do anything to other objects)
- Generate physically plausible transitions

## Task Description

### Core Challenge

Models must:
1. **Parse Instructions**: Understand which objects to remove based on the prompt
2. **Selective Removal**: Remove only the specified objects
3. **Spatial Invariance**: Do not do anything to other objects (keep them unchanged)
4. **Generate Transition**: Create smooth video showing objects being removed

### Visual Format

- **Canvas Size**: 256x256 pixels (default)
- **Objects**: 5-8 objects per scene (default)
- **Object Types**: Colored shapes (cubes, spheres, pyramids, cones)
- **Colors**: Red, green, blue, yellow, orange, purple
- **Shapes**: Cube (square), sphere (circle), pyramid (triangle), cone (trapezoid)

## Cognitive Levels

### Level 1: Explicit Specificity ✅ (Implemented)

**Task Type:**  
Remove objects defined by **explicit visual attributes** (e.g., color, shape, size).

**Prompt Examples:**
- "Remove all red objects from the scene. Do not do anything to other objects."
- "Remove all pyramid objects from the scene. Do not do anything to other objects."
- "Remove the largest object. Do not do anything to other objects." (when only one largest)
- "Remove all largest objects. Do not do anything to other objects." (when multiple largest)
- "Remove the smallest object. Do not do anything to other objects." (when only one smallest)
- "Remove all smallest objects. Do not do anything to other objects." (when multiple smallest)

**Example Scene:**
- **First Frame**: White background with 5-7 colored shapes (red cubes, green spheres, blue pyramids)
- **Final Frame**: Only non-red objects remain, identical positions

**Rule Structure:**
```python
{
  "level": "L1",
  "rule_type": "color",  # or "shape" or "size"
  "remove_color": "red",  # or "remove_shape": "cube" or "size_type": "largest"/"smallest"
  "target_object_ids": [0, 1]  # Explicit object IDs to remove
}
```

**Visual Distinction (Size-based rules):**  
When using size-based rules (largest/smallest), the target object is **visually obvious**:
- Largest object is at least 12 pixels larger than the second largest
- Smallest object is at least 12 pixels smaller than the second smallest
- Objects are actively adjusted during generation to ensure clear visual distinction
- If size differences cannot be made clear enough, the system falls back to color or shape rules

**Cognitive Focus:**  
Visual recognition · Simple selection · Static invariance

### Level 2: Enumerated Selection ✅ (Implemented)

**Task Type:**  
Remove multiple **explicitly listed** objects by color and shape.

**Prompt Examples:**
- "Remove the red cube, the green sphere, and the blue pyramid from the scene. Do not do anything to other objects."
- "Remove the orange pyramid and the red cone from the scene. Do not do anything to other objects."

### Level 3: Relational Reference ✅ (Implemented)

**Task Type:**  
Remove objects using **spatial or numeric relations** instead of explicit labels.

**Prompt Examples:**
- "Remove the 2 leftmost objects. Do not do anything to other objects."
- "Remove the object closest to a corner. Do not do anything to other objects."
- "Remove the 2 topmost objects. Do not do anything to other objects."
- "Remove all objects in the upper half of the image. Do not do anything to other objects."

### Level 4: Conceptual Abstraction ✅ (Implemented)

**Task Type:**  
Remove objects based on **semantic or conceptual properties** (outlier detection).

**Prompt Examples:**
- "Remove the object that looks different from the others. Do not do anything to other objects."

## Data Structure

### ObjectSubtractionTaskPair

Each task consists of:
```python
{
    "id": "object_subtraction_l1_0001",
    "prompt": "Remove all red objects...",
    "first_image_path": "path/to/first_frame.png",
    "final_image_path": "path/to/final_frame.png",
    "task_category": "ObjectSubtraction",
    "level": "L1",
    "object_subtraction_data": {
        "objects": [...],  # All objects with id, color, shape, x, y, size, area
        "rule": {...},     # Rule definition
        "remove_object_ids": [0, 1],
        "keep_object_ids": [2, 3, 4],
        "num_objects": 5,
        "num_removed": 2,
        "num_kept": 3
    },
    "difficulty": "easy",
    "created_at": "2025-01-XX..."
}
```

## Implementation Details

### Object Generation

- **Collision Detection**: Ensures objects don't overlap
- **Grid Fallback**: If random placement fails, uses grid-based layout
- **Deterministic**: Uses seeds for reproducibility
- **Size Enhancement (Level 1)**: For size-based rules, objects are actively adjusted to ensure:
  - Largest object is at least 12 pixels larger than the second largest
  - Smallest object is at least 12 pixels smaller than the second smallest
  - Overall size range is at least 15 pixels
  - This ensures the target object is visually obvious in the scene

### Rule Generation

**Level 1:**
- **Color-based**: Selects a color and finds all objects with that color
- **Shape-based**: Selects a shape and finds all objects with that shape
- **Size-based**: Selects largest or smallest objects with **very clear visual distinction**
  - Minimum 12 pixels difference between largest and second largest (or smallest and second smallest)
  - Objects are actively adjusted during generation to ensure obvious size differences
  - If size differences are not clear enough, falls back to color or shape rules
- **Uniqueness**: Each rule explicitly lists `target_object_ids` for unambiguous removal

**Level 2:**
- **Enumerated Selection**: Removes 2-3 explicitly listed objects by color and shape combination

**Level 3:**
- **Spatial Relations**: Removes objects based on spatial positions (leftmost, rightmost, topmost, bottommost, corners, quadrants, distance from center)

**Level 4:**
- **Outlier Detection**: Removes the object that looks different from others (based on color+shape majority)

### Image Rendering

- **Library**: matplotlib
- **Shapes**:
  - Cube: Rectangle
  - Sphere: Circle
  - Pyramid: Equilateral triangle
  - Cone: Trapezoid
- **Colors**: Standard color mapping (red, green, blue, yellow, orange, purple)

## Usage

### Generate Dataset

```python
from vmevalkit.tasks.object_subtraction_task import create_dataset

# Generate 50 tasks (Level 1 only)
dataset = create_dataset(num_samples=50, levels=["L1"])

# Generate tasks for multiple levels
dataset = create_dataset(num_samples=100, levels=["L1", "L2", "L3", "L4"])

# Generate tasks for all levels with deterministic seed
dataset = create_dataset(num_samples=100, levels=["L1", "L2", "L3", "L4"], random_seed=42)
```

### Command Line

```bash
# Generate questions using the standard VMEvalKit script
python examples/create_questions.py --task object_subtraction --pairs-per-domain 50
```

## Evaluation Metrics (Planned)

| Metric                  | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| final_object_match      | IoU overlap between generated and target final frame |
| removed_object_count    | Correct number of removed items                      |
| kept_object_stability   | Avg. displacement ≤ 3 px                             |
| motion_continuity       | Smooth optical flow (no teleportation)               |
| rule_accuracy          | Removed items satisfy logical rule in metadata       |

## Integration with VMEvalKit

### Domain Registry

Registered in `vmevalkit/utils/constant.py`:
```python
'object_subtraction': {
    'name': 'Object Subtraction',
    'description': 'Selective object removal with multi-level cognitive reasoning',
    'module': 'vmevalkit.tasks.object_subtraction_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

### Output Structure

Each generated question follows the standard VMEvalKit format:
```
data/questions/object_subtraction_task/{task_id}/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

## Future Extensions

- [x] Implement Level 2 (Enumerated Selection)
- [x] Implement Level 3 (Relational Reference)
- [x] Implement Level 4 (Conceptual Abstraction)
- [ ] Add evaluation metrics
- [ ] Support for more object types and colors
- [ ] Configurable canvas sizes
- [ ] Animation styles (fade out, slide out, etc.)

## References

- [GitHub Issue #54](https://github.com/hokindeng/VMEvalKit/issues/54) - Original task proposal



