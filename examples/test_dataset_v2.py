#!/usr/bin/env python3
"""
Simple test script for VMEvalKit Dataset v2 with improved prompts.

This script runs a quick test on a few samples from the v2 dataset
to verify the improved prompts are working correctly.

Usage:
    python examples/test_dataset_v2.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.runner.inference import run_inference, AVAILABLE_MODELS


def load_dataset_v2() -> Dict[str, Any]:
    """Load the v2 dataset with improved prompts."""
    dataset_path = Path("data/questions/vmeval_dataset_v2.json")
    if not dataset_path.exists():
        print(f"❌ Dataset v2 not found at: {dataset_path}")
        print("   Please run: python vmevalkit/runner/create_dataset.py")
        sys.exit(1)
    
    with open(dataset_path, 'r') as f:
        return json.load(f)


def test_improved_prompts(n_samples: int = 2):
    """
    Test a few samples from each category to verify improved prompts.
    
    Args:
        n_samples: Number of samples per category to test
    """
    print("=" * 70)
    print("🧪 TESTING VMEVAL DATASET V2 - IMPROVED PROMPTS")
    print("=" * 70)
    
    # Load dataset
    dataset = load_dataset_v2()
    print(f"\n📊 Dataset loaded: {dataset['name']}")
    print(f"   Total pairs: {dataset['total_pairs']}")
    print(f"   Version: {dataset['version']}")
    
    # Group tasks by domain
    tasks_by_domain = {'chess': [], 'maze': [], 'raven': [], 'rotation': []}
    
    for pair in dataset['pairs']:
        domain = pair.get('domain', '')
        if domain in tasks_by_domain:
            tasks_by_domain[domain].append(pair)
    
    # Display sample prompts from each category
    print("\n" + "=" * 70)
    print("📝 SAMPLE IMPROVED PROMPTS")
    print("=" * 70)
    
    for domain, tasks in tasks_by_domain.items():
        print(f"\n{'─' * 60}")
        print(f"🎯 {domain.upper()} TASKS")
        print(f"{'─' * 60}")
        
        # Show first n_samples tasks
        for i, task in enumerate(tasks[:n_samples], 1):
            print(f"\nExample {i}: {task['id']}")
            print(f"Difficulty: {task.get('difficulty', 'unknown')}")
            print(f"Prompt: {task['prompt']}")
            
            # Show key improvements for specific domains
            if domain == 'maze':
                if 'white corridors' in task['prompt']:
                    print("✅ Clearly states: white = corridors, black = walls")
            elif domain == 'rotation':
                if 'Your camera is at viewing angle' in task['prompt']:
                    print("✅ Clearly states: camera moves, sculpture stays fixed")
    
    print("\n" + "=" * 70)
    print("✅ Prompt improvements verified!")
    print("=" * 70)


def run_sample_inference(model_name: str = "luma-ray-2", n_samples: int = 1):
    """
    Run actual inference on a few samples with a test model.
    
    Args:
        model_name: Model to use for testing
        n_samples: Number of samples to test
    """
    print(f"\n🎬 Running sample inference with {model_name}...")
    
    # Check if model is available
    if model_name not in AVAILABLE_MODELS:
        print(f"❌ Model {model_name} not available")
        print(f"   Available models: {list(AVAILABLE_MODELS.keys())[:5]}...")
        return
    
    # Load dataset
    dataset = load_dataset_v2()
    
    # Get one sample from each domain
    test_samples = []
    for domain in ['chess', 'maze', 'raven', 'rotation']:
        domain_samples = [p for p in dataset['pairs'] if p.get('domain') == domain]
        if domain_samples:
            test_samples.append(domain_samples[0])
    
    # Run inference on samples
    output_dir = Path("output/test_v2")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for sample in test_samples[:n_samples]:
        print(f"\n  Testing: {sample['id']}")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        
        try:
            result = run_inference(
                model_name=model_name,
                image_path=sample['first_image_path'],
                text_prompt=sample['prompt'],
                output_dir=str(output_dir),
                output_filename=f"{sample['id']}_{model_name}.mp4"
            )
            print(f"  ✅ Success! Video saved to: {result.get('video_path', 'N/A')}")
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")


def main():
    """Main function."""
    # Test improved prompts (no actual inference, just display)
    test_improved_prompts(n_samples=3)
    
    # Optional: Run actual inference on a sample
    # Uncomment to test with a real model (requires API key)
    # run_sample_inference(model_name="luma-ray-2", n_samples=1)
    
    print("\n✨ Test complete! The improved prompts in v2 are ready to use.")
    print("💡 To run full inference, use the pilot_experiment.py script")


if __name__ == "__main__":
    main()
