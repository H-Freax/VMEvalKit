# Paper Assets - Mixed Model Score=5 Examples

This directory contains one exemplary video output (score=5) per task type from the **best-performing models**.

## Models Used

- **Sora**: Chess, Maze, Rotation
- **Veo 3.1**: Raven, Sudoku

## Directory Structure

```
paper/assets/
├── chess_task/
│   ├── chess_example_12frames.png        # 12-frame temporal decomposition
│   ├── chess_example_12frames.eps        # Vector format for paper
│   ├── first_frame.png                   # Original input image
│   ├── final_frame.png                   # Target/solution image  
│   ├── prompt.txt                        # Text prompt used
│   └── question_metadata.json            # Task metadata
├── maze_task/
│   └── ... (same structure)
├── raven_task/
│   └── ... (same structure)
├── rotation_task/
│   └── ... (same structure)
└── sudoku_task/
    └── ... (same structure)
```

## Examples Used

| Task | Model | Question ID | Score |
|------|-------|-------------|-------|
| Chess | **Sora** | chess_0002 | 5/5 |
| Maze | **Sora** | maze_0003 | 5/5 |
| Raven | **Veo 3.1** | raven_0014 | 5/5 |
| Rotation | **Sora** | rotation_0014 | 5/5 |
| Sudoku | **Veo 3.1** | sudoku_0011 | 5/5 |

## Frame Decomposition Details

Each 12-frame sequence shows:
- **12 evenly-spaced frames** from 8-second videos
- **Correct aspect ratio** preserved (16:9 for 1280x720, 1920x1080)
- **Frame labels** show actual frame indices (not sequential)
- **Publication-ready** at 300 DPI (PNG) and vector (EPS)
- Frame dimensions: 64 inches wide × 4 inches tall
