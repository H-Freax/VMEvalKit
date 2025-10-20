#!/usr/bin/env python3
"""
VMEvalKit Evaluation Runner

This script provides easy access to VMEvalKit's evaluation methods:
- Human evaluation with Gradio interface
- GPT-4O automatic evaluation
- Custom evaluation examples

Usage:
    python run_evaluation.py human [--annotator NAME] [--port PORT] [--share]
    python run_evaluation.py gpt4o
    python run_evaluation.py custom
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_human_evaluation(annotator_name="John Doe", port=7860, share=False):
    """Example of running human evaluation."""
    print("\n=== Human Evaluation Example ===")
    
    # Create evaluator
    evaluator = HumanEvaluator(
        experiment_name="pilot_experiment",
        annotator_name=annotator_name
    )
    
    # Launch interface (this will block until closed)
    print(f"Launching human evaluation interface for annotator: {annotator_name}")
    print(f"Open http://localhost:{port} in your browser")
    evaluator.launch_interface(port=port, share=share)


def example_gpt4o_evaluation():
    """Example of running GPT-4O evaluation."""
    print("\n=== GPT-4O Evaluation Example ===")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Create evaluator
    evaluator = GPT4OEvaluator(
        experiment_name="pilot_experiment",
        max_frames=8,
        temperature=0.1
    )
    
    # Example 1: Evaluate a specific model
    print("\nEvaluating specific model: luma-ray-2")
    try:
        results = evaluator.evaluate_model("luma-ray-2")
        
        # Count evaluations
        total_tasks = 0
        evaluated_tasks = 0
        for task_type, tasks in results["evaluations"].items():
            for task_id, result in tasks.items():
                total_tasks += 1
                if "error" not in result:
                    evaluated_tasks += 1
        
        print(f"Evaluation complete. {evaluated_tasks}/{total_tasks} tasks evaluated.")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Evaluate all models
    print("\nEvaluating all models...")
    all_results = evaluator.evaluate_all_models()
    
    # Print basic counts for each model
    for model_name, results in all_results.items():
        if "evaluations" in results:
            total_tasks = 0
            evaluated_tasks = 0
            for task_type, tasks in results["evaluations"].items():
                for task_id, result in tasks.items():
                    total_tasks += 1
                    if "error" not in result:
                        evaluated_tasks += 1
            
            print(f"\n{model_name}:")
            print(f"  - Tasks evaluated: {evaluated_tasks}/{total_tasks}")


def example_custom_evaluation():
    """Example of using the evaluation API programmatically."""
    print("\n=== Custom Evaluation Example ===")
    
    from vmevalkit.eval import BaseEvaluator
    
    class SimpleEvaluator(BaseEvaluator):
        """A simple custom evaluator for demonstration."""
        
        def evaluate_single(self, model_name, task_type, task_id, video_path, question_data):
            # Simple evaluation logic
            import random
            
            # Random score for demo
            score = random.randint(1, 5)
            
            return {
                "solution_correctness_score": score,
                "explanation": f"Demo evaluation: solution scored {score}/5",
                "status": "completed"
            }
    
    # Use the custom evaluator
    evaluator = SimpleEvaluator(experiment_name="pilot_experiment")
    
    # Evaluate a model
    print("Running custom evaluation...")
    results = evaluator.evaluate_model("luma-ray-2")
    
    # Count results
    total_tasks = 0
    evaluated_tasks = 0
    for task_type, tasks in results["evaluations"].items():
        for task_id, result in tasks.items():
            total_tasks += 1
            if "error" not in result:
                evaluated_tasks += 1
    
    print(f"Custom evaluation complete: {evaluated_tasks}/{total_tasks} tasks evaluated")


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VMEvalKit Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run human evaluation
  python run_evaluation.py human
  
  # Run human evaluation with custom annotator
  python run_evaluation.py human --annotator "Jane Smith"
  
  # Run GPT-4O evaluation
  python run_evaluation.py gpt4o
  
  # Demonstrate custom evaluator
  python run_evaluation.py custom
        """
    )
    
    parser.add_argument(
        'method',
        choices=['human', 'gpt4o', 'custom'],
        help='Evaluation method to use'
    )
    
    # Human evaluation specific arguments
    parser.add_argument(
        '--annotator',
        type=str,
        default='John Doe',
        help='Annotator name for human evaluation (default: John Doe)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for Gradio interface (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link for the interface'
    )
    
    args = parser.parse_args()
    
    # Check if pilot_experiment exists
    if not Path("data/outputs/pilot_experiment").exists():
        print("Error: pilot_experiment not found. Please run inference first.")
        return
    
    # Run the selected evaluation method
    if args.method == "human":
        example_human_evaluation(
            annotator_name=args.annotator,
            port=args.port,
            share=args.share
        )
    elif args.method == "gpt4o":
        example_gpt4o_evaluation()
    elif args.method == "custom":
        example_custom_evaluation()


if __name__ == "__main__":
    main()
