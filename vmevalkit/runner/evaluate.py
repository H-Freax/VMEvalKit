"""Evaluation runner for VMEvalKit.

This script runs various evaluation methods on generated videos.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def run_human_evaluation(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/evaluations",
    annotator_name: str = "anonymous",
    port: int = 7860,
    share: bool = False,
    models: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None
):
    """Run human evaluation interface.
    
    Args:
        experiment_name: Name of the experiment to evaluate
        output_dir: Directory to save evaluation results
        annotator_name: Name of the human annotator
        port: Port to run Gradio interface on
        share: Whether to create a public share link
        models: List of models to filter to (None for all)
        task_types: List of task types to filter to (None for all)
    """
    logger.info(f"Starting human evaluation for experiment: {experiment_name}")
    logger.info(f"Annotator: {annotator_name}")
    
    evaluator = HumanEvaluator(
        output_dir=output_dir,
        experiment_name=experiment_name,
        annotator_name=annotator_name,
        models_filter=models,
        task_types_filter=task_types
    )
    
    # Launch the Gradio interface
    logger.info(f"Launching interface on port {port}")
    evaluator.launch_interface(share=share, port=port)


def run_gpt4o_evaluation(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/evaluations",
    models: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    max_frames: int = 8,
    temperature: float = 0.1
):
    """Run GPT-4O automatic evaluation.
    
    Args:
        experiment_name: Name of the experiment to evaluate
        output_dir: Directory to save evaluation results
        models: List of model names to evaluate (None for all)
        task_types: List of task types to evaluate (None for all)
        task_ids: List of task IDs to evaluate (None for all)
        max_frames: Maximum frames to extract per video
        temperature: Temperature for GPT-4O responses
    """
    logger.info(f"Starting GPT-4O evaluation for experiment: {experiment_name}")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)
    
    evaluator = GPT4OEvaluator(
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_frames=max_frames,
        temperature=temperature
    )
    
    # Use selective evaluation if any filters are provided
    if models or task_types or task_ids:
        logger.info("Running selective evaluation with filters:")
        if models:
            logger.info(f"  Models: {models}")
        if task_types:
            logger.info(f"  Task types: {task_types}")
        if task_ids:
            logger.info(f"  Task IDs: {task_ids}")
            
        results = evaluator.evaluate_selective(
            models=models,
            task_types=task_types,
            task_ids=task_ids
        )
        logger.info("Completed selective evaluation")
    else:
        # Evaluate all models and tasks
        logger.info("Evaluating all models and tasks in experiment")
        results = evaluator.evaluate_all_models()
        logger.info("Completed evaluation for all models")
        
        # Print basic counts
        for model_name, model_results in results.items():
            if "evaluations" in model_results:
                total_tasks = 0
                evaluated_tasks = 0
                for task_type, tasks in model_results["evaluations"].items():
                    for task_id, result in tasks.items():
                        total_tasks += 1
                        if "error" not in result:
                            evaluated_tasks += 1
                
                logger.info(f"\n{model_name}:")
                logger.info(f"  Total tasks: {total_tasks}")
                logger.info(f"  Evaluated: {evaluated_tasks}")


def main():
    """Main entry point for evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluations on VMEvalKit experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run human evaluation
  python -m vmevalkit.runner.evaluate human --annotator "John Doe" --port 7860
  
  # Run GPT-4O evaluation on all models
  python -m vmevalkit.runner.evaluate gpt4o
  
  # Run GPT-4O evaluation on specific models
  python -m vmevalkit.runner.evaluate gpt4o --models luma-ray-2 openai-sora-2
  
  # Run evaluation on a different experiment
  python -m vmevalkit.runner.evaluate gpt4o --experiment my_experiment
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='method', help='Evaluation method')
    
    # Human evaluation subcommand
    human_parser = subparsers.add_parser('human', help='Run human evaluation interface')
    human_parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='pilot_experiment',
        help='Name of the experiment to evaluate'
    )
    human_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/evaluations',
        help='Directory to save evaluation results'
    )
    human_parser.add_argument(
        '--annotator', '-a',
        type=str,
        default='anonymous',
        help='Name of the human annotator'
    )
    human_parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Port to run Gradio interface on'
    )
    human_parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link for the interface'
    )
    human_parser.add_argument(
        '--task-types', '-t',
        type=str,
        nargs='+',
        help='Filter to specific task types (e.g., chess_task maze_task)'
    )
    human_parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        help='Filter to specific models'
    )
    
    # GPT-4O evaluation subcommand
    gpt4o_parser = subparsers.add_parser('gpt4o', help='Run GPT-4O automatic evaluation')
    gpt4o_parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='pilot_experiment',
        help='Name of the experiment to evaluate'
    )
    gpt4o_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/evaluations',
        help='Directory to save evaluation results'
    )
    gpt4o_parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        help='Specific models to evaluate (default: all models)'
    )
    gpt4o_parser.add_argument(
        '--task-types', '-t',
        type=str,
        nargs='+',
        help='Specific task types to evaluate (e.g., chess_task maze_task)'
    )
    gpt4o_parser.add_argument(
        '--task-ids',
        type=str,
        nargs='+',
        help='Specific task IDs to evaluate (e.g., chess_0001 maze_0005)'
    )
    gpt4o_parser.add_argument(
        '--max-frames',
        type=int,
        default=8,
        help='Maximum frames to extract per video'
    )
    gpt4o_parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for GPT-4O responses'
    )
    
    args = parser.parse_args()
    
    if not args.method:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate evaluation method
    if args.method == 'human':
        run_human_evaluation(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            annotator_name=args.annotator,
            port=args.port,
            share=args.share,
            models=args.models,
            task_types=args.task_types
        )
    elif args.method == 'gpt4o':
        run_gpt4o_evaluation(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            models=args.models,
            task_types=args.task_types,
            task_ids=args.task_ids,
            max_frames=args.max_frames,
            temperature=args.temperature
        )
    else:
        logger.error(f"Unknown evaluation method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
