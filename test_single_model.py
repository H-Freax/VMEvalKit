#!/usr/bin/env python3
"""
Simple Single Model Test - Isolate and test one model at a time.

This script allows you to test individual models with a single task to:
1. Isolate problematic models
2. Verify working models  
3. Debug API issues
4. Generate videos to the output folder

Usage:
    python test_single_model.py --model luma-ray-2 --task maze_0000
    python test_single_model.py --model luma-ray-2 --task chess_0000
    python test_single_model.py --model wavespeed-wan-2.2-i2v-720p --task sudoku_0000
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from vmevalkit.runner.inference import InferenceRunner, AVAILABLE_MODELS


def find_task_data(task_id: str, questions_dir: Path = Path("data/questions")) -> dict:
    """
    Find and load task data from the questions directory.
    
    Args:
        task_id: Task identifier (e.g., "maze_0000", "chess_0001")
        questions_dir: Directory containing task folders
        
    Returns:
        Dictionary with task information
    """
    print(f"🔍 Looking for task: {task_id}")
    
    # Extract domain from task_id (e.g., "maze_0000" -> "maze")
    domain = task_id.split('_')[0]
    task_dir = questions_dir / f"{domain}_task" / task_id
    
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    # Check for required files
    prompt_file = task_dir / "prompt.txt"
    first_image = task_dir / "first_frame.png"
    final_image = task_dir / "final_frame.png"
    metadata_file = task_dir / "question_metadata.json"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt.txt not found in {task_dir}")
    if not first_image.exists():
        raise FileNotFoundError(f"first_frame.png not found in {task_dir}")
    
    # Load prompt
    prompt = prompt_file.read_text().strip()
    
    # Load metadata if available
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    task_data = {
        "id": task_id,
        "domain": domain,
        "prompt": prompt,
        "first_image_path": str(first_image.absolute()),
        "final_image_path": str(final_image.absolute()) if final_image.exists() else None,
        **metadata  # Add any additional metadata
    }
    
    print(f"   ✅ Found task: {task_id}")
    print(f"   📁 Domain: {domain}")
    print(f"   📝 Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    print(f"   🖼️ First image: {first_image}")
    print(f"   🎯 Final image: {'Yes' if task_data['final_image_path'] else 'No'}")
    
    return task_data


def test_single_model(model_name: str, task_data: dict, output_dir: str = "data/outputs") -> dict:
    """
    Test a single model with a single task.
    
    Args:
        model_name: Model to test
        task_data: Task information
        output_dir: Output directory for results
        
    Returns:
        Test result dictionary
    """
    print(f"\n{'='*60}")
    print(f"🎬 SINGLE MODEL TEST")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Task: {task_data['id']}")
    print(f"Output: {output_dir}")
    
    # Check if model exists
    if model_name not in AVAILABLE_MODELS:
        print(f"\n❌ Model '{model_name}' not found!")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())[:10]}..." if len(AVAILABLE_MODELS) > 10 else f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return {"success": False, "error": f"Model '{model_name}' not found"}
    
    model_info = AVAILABLE_MODELS[model_name]
    print(f"Family: {model_info.get('family', 'Unknown')}")
    print(f"Description: {model_info.get('description', 'No description')}")
    
    # Create runner
    runner = InferenceRunner(output_dir=output_dir)
    
    # Generate unique run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"test_{model_name}_{task_data['id']}_{timestamp}"
    
    print(f"\n🚀 Starting inference...")
    print(f"Run ID: {run_id}")
    
    start_time = datetime.now()
    
    try:
        # Run inference
        result = runner.run(
            model_name=model_name,
            image_path=task_data["first_image_path"],
            text_prompt=task_data["prompt"],
            run_id=run_id,
            question_data=task_data
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        if result.get("success"):
            print(f"✅ SUCCESS! ({duration:.1f}s)")
            print(f"   📁 Output folder: {result.get('inference_dir')}")
            print(f"   🎥 Video file: {result.get('video_path')}")
            print(f"   📊 Generation ID: {result.get('generation_id', 'N/A')}")
        else:
            print(f"❌ FAILED! ({duration:.1f}s)")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print(f"   📁 Error details: {result.get('inference_dir')}/metadata.json")
        print(f"{'='*60}")
        
        return {
            "success": result.get("success", False),
            "model": model_name,
            "task": task_data["id"],
            "duration": duration,
            "inference_dir": result.get("inference_dir"),
            "video_path": result.get("video_path"),
            "error": result.get("error"),
            "result": result
        }
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"❌ EXCEPTION! ({duration:.1f}s)")
        print(f"   Error: {str(e)}")
        print(f"{'='*60}")
        
        return {
            "success": False,
            "model": model_name,
            "task": task_data["id"],
            "duration": duration,
            "error": str(e),
            "exception": True
        }


def list_available_models():
    """List all available models organized by family."""
    from vmevalkit.runner.inference import MODEL_FAMILIES
    
    print("🎯 AVAILABLE MODELS")
    print("="*60)
    
    total = 0
    for family_name, models in MODEL_FAMILIES.items():
        print(f"\n📦 {family_name} ({len(models)} models):")
        for model_name, config in models.items():
            desc = config.get('description', 'No description')
            print(f"   • {model_name}: {desc}")
            total += 1
    
    print(f"\n📊 Total: {total} models across {len(MODEL_FAMILIES)} families")
    return MODEL_FAMILIES


def list_available_tasks(questions_dir: Path = Path("data/questions")):
    """List available tasks by domain."""
    print("🎯 AVAILABLE TASKS")
    print("="*60)
    
    if not questions_dir.exists():
        print(f"❌ Questions directory not found: {questions_dir}")
        return
    
    total_tasks = 0
    for domain_dir in sorted(questions_dir.glob("*_task")):
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name.replace("_task", "")
        tasks = list(domain_dir.glob(f"{domain}_*"))
        task_count = len([t for t in tasks if t.is_dir()])
        
        print(f"\n📁 {domain.upper()} ({task_count} tasks):")
        for task_dir in sorted(tasks):
            if task_dir.is_dir():
                task_id = task_dir.name
                prompt_file = task_dir / "prompt.txt"
                if prompt_file.exists():
                    prompt = prompt_file.read_text().strip()[:60] + "..." if len(prompt_file.read_text().strip()) > 60 else prompt_file.read_text().strip()
                    print(f"   • {task_id}: {prompt}")
                else:
                    print(f"   • {task_id}: (no prompt)")
        total_tasks += task_count
    
    print(f"\n📊 Total: {total_tasks} tasks across {len(list(questions_dir.glob('*_task')))} domains")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test a single model with a single task",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--model", "-m", type=str, help="Model name to test")
    parser.add_argument("--task", "-t", type=str, help="Task ID to use (e.g., maze_0000, chess_0001)")
    parser.add_argument("--output", "-o", type=str, default="data/outputs", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--list-tasks", action="store_true", help="List all available tasks") 
    
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_models:
        list_available_models()
        return
    
    if args.list_tasks:
        list_available_tasks()
        return
    
    # Validate required arguments
    if not args.model or not args.task:
        print("❌ Both --model and --task are required")
        print("\nExamples:")
        print("  python test_single_model.py --model luma-ray-2 --task maze_0000")
        print("  python test_single_model.py --model wavespeed-wan-2.2-i2v-720p --task chess_0001")
        print("\nUse --list-models and --list-tasks to see available options")
        sys.exit(1)
    
    try:
        # Find task data
        task_data = find_task_data(args.task)
        
        # Test model
        result = test_single_model(args.model, task_data, args.output)
        
        # Summary
        print(f"\n🎯 TEST SUMMARY")
        print(f"Model: {args.model}")
        print(f"Task: {args.task}")
        print(f"Result: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        
        if result['success']:
            print(f"\n📁 Your video is ready at: {result['video_path']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
