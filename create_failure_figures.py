#!/usr/bin/env python3
"""
Create failure case visualizations for each task
Shows: First Frame | Expected Final | Actual Final + 12-frame progression
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import cv2
import numpy as np

# Configuration
EVAL_DIR = Path("/Users/access/VMEvalKit/data/evaluations/gpt4o-eval/pilot_experiment")
OUTPUT_DIR = Path("/Users/access/VMEvalKit/data/outputs/pilot_experiment")
QUESTIONS_DIR = Path("/Users/access/VMEvalKit/data/questions")
FIGURES_DIR = Path("/tmp/failure_figures")

TASKS = ["chess_task", "maze_task", "rotation_task", "sudoku_task", "raven_task"]
MODELS = ["openai-sora-2", "veo-3.1-720p", "veo-3.0-generate", "luma-ray-2", "runway-gen4-turbo", "wavespeed-wan-2.2-i2v-720p"]
FAILURE_THRESHOLD = 3  # Score <= 3 is considered failure

def find_failure_cases():
    """Find one failure case per model per task"""
    failure_cases = {}
    
    for task in TASKS:
        print(f"\nScanning {task}...")
        failure_cases[task] = {}
        
        for model in MODELS:
            eval_task_dir = EVAL_DIR / model / task
            if not eval_task_dir.exists():
                continue
            
            # Find first failure case
            for task_instance_dir in sorted(eval_task_dir.iterdir()):
                if not task_instance_dir.is_dir():
                    continue
                
                eval_file = task_instance_dir / "GPT4OEvaluator.json"
                if not eval_file.exists():
                    continue
                
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                score = eval_data.get('result', {}).get('solution_correctness_score', 0)
                
                if score <= FAILURE_THRESHOLD:
                    task_id = task_instance_dir.name
                    failure_cases[task][model] = {
                        'task_id': task_id,
                        'score': score,
                        'explanation': eval_data.get('result', {}).get('explanation', ''),
                    }
                    print(f"  ✗ {model}: {task_id} (score: {score})")
                    break
        
        if not failure_cases[task]:
            print(f"  No failures found for {task}")
    
    return failure_cases

def load_video_frames(video_path, num_frames=12):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames if len(frames) == num_frames else None

def load_image(image_path):
    """Load and return image"""
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_prompt(prompt_file):
    """Read prompt text from file"""
    if not prompt_file.exists():
        return "Prompt not available"
    with open(prompt_file, 'r') as f:
        return f.read().strip()

def create_failure_figure(task_name, failure_cases, output_path):
    """Create a figure showing failure cases for one task"""
    
    # Filter models that have failures
    models_with_failures = [(model, data) for model, data in failure_cases.items() if data]
    
    if not models_with_failures:
        print(f"No failures to visualize for {task_name}")
        return
    
    num_models = len(models_with_failures)
    
    # Create figure: each model gets 2 rows (1 for frames, 1 for progression)
    fig = plt.figure(figsize=(20, 5 * num_models))
    
    task_display_name = task_name.replace('_task', '').replace('_', ' ').title()
    fig.suptitle(f'Failure Cases: {task_display_name}', fontsize=20, fontweight='bold', y=0.995)
    
    # Create grid: 2 rows per model, 15 columns (3 for main frames, 12 for progression)
    gs = fig.add_gridspec(num_models * 2, 15, hspace=0.4, wspace=0.3,
                          left=0.03, right=0.97, top=0.96, bottom=0.02)
    
    for model_idx, (model_name, failure_data) in enumerate(models_with_failures):
        task_id = failure_data['task_id']
        score = failure_data['score']
        explanation = failure_data['explanation']
        
        row_offset = model_idx * 2
        
        # Load data
        question_dir = QUESTIONS_DIR / task_name / task_id
        output_model_dir = OUTPUT_DIR / model_name / task_name / task_id
        
        first_frame_path = question_dir / "first_frame.png"
        final_frame_path = question_dir / "final_frame.png"
        prompt_path = question_dir / "prompt.txt"
        
        first_frame = load_image(first_frame_path)
        expected_final = load_image(final_frame_path)
        prompt = read_prompt(prompt_path)
        
        # Find video
        video_frames = None
        actual_final = None
        if output_model_dir.exists():
            output_folders = list(output_model_dir.iterdir())
            if output_folders:
                video_dir = output_folders[0] / "video"
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4"))
                    if video_files:
                        video_frames = load_video_frames(video_files[0])
                        if video_frames:
                            actual_final = video_frames[-1]
        
        # Row 1: First Frame | Expected Final | Actual Final
        # First Frame
        ax1 = fig.add_subplot(gs[row_offset, 0:4])
        if first_frame is not None:
            ax1.imshow(first_frame)
        ax1.set_title('First Frame', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Expected Final
        ax2 = fig.add_subplot(gs[row_offset, 5:9])
        if expected_final is not None:
            ax2.imshow(expected_final)
        ax2.set_title('Expected Final', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Actual Final (from video)
        ax3 = fig.add_subplot(gs[row_offset, 10:14])
        if actual_final is not None:
            ax3.imshow(actual_final)
            ax3.add_patch(Rectangle((0, 0), actual_final.shape[1], actual_final.shape[0],
                                   fill=False, edgecolor='red', linewidth=3))
        ax3.set_title('Actual Final (Failed)', fontsize=12, fontweight='bold', color='red')
        ax3.axis('off')
        
        # Row 2: 12-frame progression
        if video_frames:
            for i, frame in enumerate(video_frames):
                col = i
                if col >= 12:
                    break
                ax = fig.add_subplot(gs[row_offset + 1, col])
                ax.imshow(frame)
                ax.set_title(f'F{i+1}', fontsize=8)
                ax.axis('off')
        
        # Add model info and explanation
        model_display = model_name.replace('-', ' ').replace('_', ' ')
        info_text = f"{model_display}\nTask: {task_id} | Score: {score}/5\n{explanation[:150]}..."
        
        fig.text(0.015, 1 - (model_idx + 0.5) / num_models, info_text,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: {output_path}")

def main():
    print("Finding failure cases...")
    print("=" * 60)
    
    failure_cases = find_failure_cases()
    
    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Creating failure figures...")
    print("=" * 60)
    
    for task in TASKS:
        if failure_cases[task]:
            output_file = FIGURES_DIR / f"{task}_failures.png"
            create_failure_figure(task, failure_cases[task], output_file)
            
            # Also create EPS version
            output_eps = FIGURES_DIR / f"{task}_failures.eps"
            create_failure_figure(task, failure_cases[task], output_eps)
    
    print("\n" + "=" * 60)
    print(f"✓ All failure figures saved to: {FIGURES_DIR}")
    print("=" * 60)
    
    # Create summary
    summary = {
        'total_figures': len([t for t in TASKS if failure_cases[t]]),
        'by_task': {}
    }
    
    for task in TASKS:
        summary['by_task'][task] = {
            'num_failures': len(failure_cases[task]),
            'models': list(failure_cases[task].keys())
        }
    
    with open(FIGURES_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    for task in TASKS:
        num_failures = len(failure_cases[task])
        print(f"  {task}: {num_failures} failure cases visualized")

if __name__ == "__main__":
    main()

