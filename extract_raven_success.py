#!/usr/bin/env python3
"""
Extract all successful raven task outputs with 12-frame EPS visualizations
"""

import json
import os
import shutil
from pathlib import Path
import subprocess

# Configuration
EVAL_DIR = Path("/Users/access/VMEvalKit/data/evaluations/gpt4o-eval/pilot_experiment")
OUTPUT_DIR = Path("/Users/access/VMEvalKit/data/outputs/pilot_experiment")
QUESTIONS_DIR = Path("/Users/access/VMEvalKit/data/questions/raven_task")
TEMP_DIR = Path("/tmp/raven_success_outputs")

# Success threshold (GPT4O scores are typically 1-5, with 5 being best)
SUCCESS_THRESHOLD = 4

def find_successful_raven_tasks():
    """Find all successful raven tasks across all models"""
    successful_tasks = []
    
    # Get all model directories
    model_dirs = [d for d in EVAL_DIR.iterdir() if d.is_dir() and d.name != "GPT4OEvaluator_all_models.json"]
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        raven_dir = model_dir / "raven_task"
        
        if not raven_dir.exists():
            continue
        
        # Check each raven task
        for task_dir in sorted(raven_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            
            task_id = task_dir.name
            eval_file = task_dir / "GPT4OEvaluator.json"
            
            if not eval_file.exists():
                continue
            
            # Read evaluation
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            score = eval_data.get('result', {}).get('solution_correctness_score', 0)
            
            if score >= SUCCESS_THRESHOLD:
                successful_tasks.append({
                    'model': model_name,
                    'task_id': task_id,
                    'score': score,
                    'explanation': eval_data.get('result', {}).get('explanation', '')
                })
                print(f"✓ Found success: {model_name} - {task_id} (score: {score})")
    
    return successful_tasks

def create_12frame_visualization(video_path, output_eps, output_png):
    """Create 12-frame grid visualization as EPS and PNG"""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages
    import cv2
    import numpy as np
    
    # Read video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract 12 evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, 12, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) != 12:
        print(f"Warning: Could only extract {len(frames)} frames")
        return False
    
    # Create 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('12-Frame Video Sequence - Raven\'s Matrices Task', fontsize=16)
    
    for idx, (ax, frame) in enumerate(zip(axes.flat, frames)):
        ax.imshow(frame)
        ax.set_title(f'Frame {idx + 1}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save as EPS and PNG
    plt.savefig(output_eps, format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def organize_successful_outputs(successful_tasks):
    """Organize all successful outputs into temp directory"""
    
    # Create temp directory
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    
    # Create summary file
    summary = {
        'total_successful': len(successful_tasks),
        'by_model': {},
        'tasks': []
    }
    
    for task_info in successful_tasks:
        model = task_info['model']
        task_id = task_info['task_id']
        score = task_info['score']
        
        # Count by model
        if model not in summary['by_model']:
            summary['by_model'][model] = 0
        summary['by_model'][model] += 1
        
        # Create output directory for this task
        task_output_dir = TEMP_DIR / model / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original question folder
        question_src = QUESTIONS_DIR / task_id
        question_dst = task_output_dir / "original_question"
        if question_src.exists():
            shutil.copytree(question_src, question_dst)
            print(f"  Copied question: {task_id}")
        
        # Find and copy output folder
        output_model_dir = OUTPUT_DIR / model / "raven_task" / task_id
        
        if output_model_dir.exists():
            # Find the actual output folder (has timestamp in name)
            output_folders = list(output_model_dir.iterdir())
            if output_folders:
                output_src = output_folders[0]
                output_dst = task_output_dir / "model_output"
                shutil.copytree(output_src, output_dst)
                
                # Find video file
                video_dir = output_src / "video"
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4"))
                    if video_files:
                        video_path = video_files[0]
                        
                        # Create 12-frame visualization
                        eps_path = task_output_dir / f"{model}_{task_id}_12frames.eps"
                        png_path = task_output_dir / f"{model}_{task_id}_12frames.png"
                        
                        print(f"  Creating 12-frame visualization for {model}/{task_id}...")
                        try:
                            create_12frame_visualization(video_path, eps_path, png_path)
                            print(f"    ✓ Created: {eps_path.name}")
                        except Exception as e:
                            print(f"    ✗ Error: {e}")
        
        # Add to summary
        summary['tasks'].append({
            'model': model,
            'task_id': task_id,
            'score': score,
            'explanation': task_info['explanation']
        })
    
    # Save summary
    with open(TEMP_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create README
    readme_content = f"""# Successful Raven's Matrices Task Outputs

Total successful raven tasks: {len(successful_tasks)}

## By Model:
"""
    for model, count in sorted(summary['by_model'].items()):
        readme_content += f"- {model}: {count} tasks\n"
    
    readme_content += f"""
## Structure:
```
{TEMP_DIR.name}/
├── summary.json                    # JSON summary of all successful tasks
├── README.md                       # This file
└── <model_name>/
    └── <task_id>/
        ├── original_question/      # Original question data (first_frame, final_frame, prompt, etc.)
        ├── model_output/           # Model's generated output (video, metadata, etc.)
        ├── <model>_<task>_12frames.eps  # 12-frame visualization (EPS format)
        └── <model>_<task>_12frames.png  # 12-frame visualization (PNG format)
```

## Files Generated:
Each successful task includes:
1. **original_question/** - The original Raven's Progressive Matrices puzzle setup
2. **model_output/** - The model's video generation output
3. **12-frame visualizations** - Grid showing 12 evenly-spaced frames from the video (both EPS and PNG)

## Success Criteria:
Tasks with GPT-4O evaluation score >= {SUCCESS_THRESHOLD} (out of 5)
"""
    
    with open(TEMP_DIR / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\n{'='*60}")
    print(f"✓ All outputs organized in: {TEMP_DIR}")
    print(f"✓ Total successful tasks: {len(successful_tasks)}")
    print(f"{'='*60}")

def main():
    print("Scanning for successful raven tasks...")
    print("=" * 60)
    
    successful_tasks = find_successful_raven_tasks()
    
    print("\n" + "=" * 60)
    print(f"Found {len(successful_tasks)} successful raven tasks")
    print("=" * 60)
    
    if successful_tasks:
        print("\nOrganizing outputs...")
        organize_successful_outputs(successful_tasks)
    else:
        print("No successful tasks found.")

if __name__ == "__main__":
    main()

