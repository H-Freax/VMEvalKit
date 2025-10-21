#!/usr/bin/env python3
"""
VMEvalKit Analysis Tool

Analyzes evaluation results to show model performance by domain and overall rankings.
Only scores 4 and 5 are considered "correct" (successful).

Usage:
    python analysis/plot.py --eval-folder data/evaluations/human-eval/
    python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/ --output results.png
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_evaluation_data(eval_folder: Path) -> list:
    """Load all evaluation JSON files from the specified folder."""
    evaluations = []
    
    for json_file in eval_folder.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract relevant information
            if "metadata" in data and "result" in data:
                eval_data = {
                    "model_name": data["metadata"].get("model_name", "unknown"),
                    "task_type": data["metadata"].get("task_type", "unknown"),
                    "task_id": data["metadata"].get("task_id", "unknown"),
                    "score": data["result"].get("solution_correctness_score", 0),
                    "evaluator": data["metadata"].get("evaluator", "unknown"),
                    "annotator": data["metadata"].get("annotator", "unknown")
                }
                evaluations.append(eval_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    return evaluations

def calculate_domain_performance(evaluations: list) -> pd.DataFrame:
    """Calculate performance by model and domain (task_type)."""
    results = []
    
    # Group by model and domain
    grouped = defaultdict(lambda: defaultdict(list))
    for eval_data in evaluations:
        model = eval_data["model_name"]
        domain = eval_data["task_type"].replace("_task", "")  # Remove "_task" suffix
        score = eval_data["score"]
        grouped[model][domain].append(score)
    
    # Calculate performance metrics
    for model, domains in grouped.items():
        for domain, scores in domains.items():
            total_tasks = len(scores)
            if total_tasks > 0:
                # Count scores 4 and 5 as correct
                correct_tasks = sum(1 for score in scores if score >= 4)
                success_rate = (correct_tasks / total_tasks) * 100
                avg_score = np.mean(scores)
                
                results.append({
                    "model": model,
                    "domain": domain,
                    "total_tasks": total_tasks,
                    "correct_tasks": correct_tasks,
                    "success_rate": success_rate,
                    "average_score": avg_score,
                    "scores": scores
                })
    
    return pd.DataFrame(results)

def calculate_overall_performance(evaluations: list) -> pd.DataFrame:
    """Calculate overall performance ranking for all models."""
    results = []
    
    # Group by model
    grouped = defaultdict(list)
    for eval_data in evaluations:
        model = eval_data["model_name"]
        score = eval_data["score"]
        grouped[model].append(score)
    
    # Calculate overall metrics
    for model, scores in grouped.items():
        total_tasks = len(scores)
        if total_tasks > 0:
            correct_tasks = sum(1 for score in scores if score >= 4)
            success_rate = (correct_tasks / total_tasks) * 100
            avg_score = np.mean(scores)
            
            results.append({
                "model": model,
                "total_tasks": total_tasks,
                "correct_tasks": correct_tasks,
                "success_rate": success_rate,
                "average_score": avg_score
            })
    
    # Sort by success rate
    df = pd.DataFrame(results)
    df = df.sort_values("success_rate", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    return df

def create_visualizations(domain_df: pd.DataFrame, overall_df: pd.DataFrame, output_path: str = None):
    """Create comprehensive visualizations."""
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Heatmap of success rates by model and domain
    plt.subplot(2, 3, 1)
    pivot_success = domain_df.pivot(index="model", columns="domain", values="success_rate")
    sns.heatmap(pivot_success, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Success Rate (%)'})
    plt.title("Success Rate by Model and Domain\n(Scores 4-5 = Correct)", fontsize=14, fontweight='bold')
    plt.xlabel("Domain", fontweight='bold')
    plt.ylabel("Model", fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Overall ranking bar chart
    plt.subplot(2, 3, 2)
    colors = plt.cm.RdYlGn(overall_df["success_rate"] / 100)
    bars = plt.barh(range(len(overall_df)), overall_df["success_rate"], color=colors)
    plt.yticks(range(len(overall_df)), overall_df["model"])
    plt.xlabel("Success Rate (%)", fontweight='bold')
    plt.title("Overall Model Ranking\n(All Domains Combined)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, overall_df["success_rate"])):
        plt.text(rate + 1, i, f'{rate:.1f}%', va='center', ha='left', fontweight='bold')
    
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    
    # 3. Domain-specific performance comparison
    plt.subplot(2, 3, 3)
    domain_means = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=True)
    colors_domain = plt.cm.viridis(np.linspace(0, 1, len(domain_means)))
    bars = plt.barh(range(len(domain_means)), domain_means.values, color=colors_domain)
    plt.yticks(range(len(domain_means)), domain_means.index)
    plt.xlabel("Average Success Rate (%)", fontweight='bold')
    plt.title("Domain Difficulty Ranking\n(Average Across All Models)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, rate in enumerate(domain_means.values):
        plt.text(rate + 1, i, f'{rate:.1f}%', va='center', ha='left', fontweight='bold')
    
    plt.xlim(0, max(domain_means) * 1.2)
    plt.grid(axis='x', alpha=0.3)
    
    # 4. Score distribution
    plt.subplot(2, 3, 4)
    all_scores = []
    for _, row in domain_df.iterrows():
        all_scores.extend(row["scores"])
    
    score_counts = pd.Series(all_scores).value_counts().sort_index()
    colors_scores = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(score_counts)]
    
    bars = plt.bar(score_counts.index, score_counts.values, color=colors_scores, alpha=0.7)
    plt.xlabel("Score", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.title("Score Distribution\n(All Models & Domains)", fontsize=14, fontweight='bold')
    plt.xticks(range(1, 6))
    plt.grid(axis='y', alpha=0.3)
    
    # Add success/failure line
    plt.axvline(x=3.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(2.5, max(score_counts) * 0.8, 'Failure', ha='center', fontweight='bold', color='red')
    plt.text(4.5, max(score_counts) * 0.8, 'Success', ha='center', fontweight='bold', color='green')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Model performance by domain (detailed)
    plt.subplot(2, 3, 5)
    
    # Create grouped bar chart
    models = domain_df["model"].unique()
    domains = domain_df["domain"].unique()
    
    x = np.arange(len(domains))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = domain_df[domain_df["model"] == model]
        rates = [model_data[model_data["domain"] == d]["success_rate"].iloc[0] 
                if len(model_data[model_data["domain"] == d]) > 0 else 0 
                for d in domains]
        
        plt.bar(x + i * width, rates, width, label=model, alpha=0.8)
    
    plt.xlabel("Domain", fontweight='bold')
    plt.ylabel("Success Rate (%)", fontweight='bold')
    plt.title("Model Performance by Domain\n(Detailed Comparison)", fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(models) - 1) / 2, domains, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    
    # 6. Average score heatmap
    plt.subplot(2, 3, 6)
    pivot_avg = domain_df.pivot(index="model", columns="domain", values="average_score")
    sns.heatmap(pivot_avg, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                vmin=1, vmax=5, cbar_kws={'label': 'Average Score'})
    plt.title("Average Score by Model and Domain\n(1-5 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Domain", fontweight='bold')
    plt.ylabel("Model", fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_path}")
    else:
        plt.show()

def print_detailed_results(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Print detailed text results."""
    
    print("\n" + "="*80)
    print("🎯 VMEVALKIT EVALUATION ANALYSIS RESULTS")
    print("="*80)
    
    # Overall ranking
    print("\n📈 OVERALL MODEL RANKING (All Domains Combined)")
    print("-" * 50)
    for _, row in overall_df.iterrows():
        print(f"{row['rank']:2d}. {row['model']:<25} | "
              f"Success: {row['success_rate']:5.1f}% ({row['correct_tasks']:3d}/{row['total_tasks']:3d}) | "
              f"Avg Score: {row['average_score']:.2f}")
    
    # Domain performance
    print(f"\n🎲 PERFORMANCE BY DOMAIN (Scores 4-5 = Correct)")
    print("-" * 50)
    
    for domain in sorted(domain_df["domain"].unique()):
        print(f"\n📊 {domain.upper()} TASKS:")
        domain_data = domain_df[domain_df["domain"] == domain].sort_values("success_rate", ascending=False)
        
        for _, row in domain_data.iterrows():
            print(f"  • {row['model']:<25} | "
                  f"{row['success_rate']:5.1f}% ({row['correct_tasks']:2d}/{row['total_tasks']:2d}) | "
                  f"Avg: {row['average_score']:.2f}")
    
    # Domain difficulty ranking
    print(f"\n🏆 DOMAIN DIFFICULTY RANKING (Easiest to Hardest)")
    print("-" * 50)
    domain_difficulty = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=False)
    
    for rank, (domain, avg_rate) in enumerate(domain_difficulty.items(), 1):
        difficulty = "🟢 Easy" if avg_rate > 70 else "🟡 Medium" if avg_rate > 40 else "🔴 Hard"
        print(f"{rank}. {domain.upper():<10} | {avg_rate:5.1f}% average success | {difficulty}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("-" * 50)
    total_evaluations = sum(overall_df["total_tasks"])
    total_correct = sum(overall_df["correct_tasks"])
    overall_success = (total_correct / total_evaluations) * 100 if total_evaluations > 0 else 0
    
    print(f"Total Evaluations: {total_evaluations:,}")
    print(f"Total Correct (4-5): {total_correct:,}")
    print(f"Overall Success Rate: {overall_success:.1f}%")
    print(f"Models Evaluated: {len(overall_df)}")
    print(f"Domains Covered: {len(domain_df['domain'].unique())}")
    
    best_model = overall_df.iloc[0]["model"]
    best_rate = overall_df.iloc[0]["success_rate"]
    print(f"🏆 Best Performing Model: {best_model} ({best_rate:.1f}%)")
    
    hardest_domain = domain_difficulty.index[-1]
    hardest_rate = domain_difficulty.iloc[-1]
    print(f"🔴 Most Challenging Domain: {hardest_domain} ({hardest_rate:.1f}% avg)")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VMEvalKit evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze and save to analysis/figures/ (default behavior)
  python analysis/plot.py --eval-folder data/evaluations/human-eval/
  
  # Show plot interactively instead of saving
  python analysis/plot.py --eval-folder data/evaluations/human-eval/ --show-plot
  
  # Save with custom filename
  python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/ --output my_results.png
  
  # Only print text results, no plots
  python analysis/plot.py --eval-folder data/evaluations/human-eval/ --no-plot
        """
    )
    
    parser.add_argument("--eval-folder", required=True, type=str,
                      help="Path to evaluation folder (e.g., data/evaluations/human-eval/)")
    parser.add_argument("--output", type=str, default=None,
                      help="Output path for visualization (optional, auto-generates filename in analysis/figures/)")
    parser.add_argument("--no-plot", action="store_true",
                      help="Skip showing/creating plots, only print text results")
    parser.add_argument("--show-plot", action="store_true",
                      help="Show plot interactively (default saves to analysis/figures/)")
    
    args = parser.parse_args()
    
    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        print(f"❌ Error: Evaluation folder not found: {eval_folder}")
        return
    
    # Load and analyze data
    print(f"📂 Loading evaluations from: {eval_folder}")
    evaluations = load_evaluation_data(eval_folder)
    
    if not evaluations:
        print(f"❌ No evaluation files found in {eval_folder}")
        return
    
    print(f"✅ Loaded {len(evaluations)} evaluations")
    
    # Calculate performance metrics
    domain_df = calculate_domain_performance(evaluations)
    overall_df = calculate_overall_performance(evaluations)
    
    # Print detailed results
    print_detailed_results(domain_df, overall_df)
    
    # Create visualizations
    if not args.no_plot:
        print(f"\n📊 Creating visualizations...")
        
        # Generate default output path if none specified
        output_path = args.output
        if not output_path and not args.show_plot:
            # Extract evaluation type from folder path
            eval_type = "unknown"
            if "human-eval" in str(eval_folder):
                eval_type = "human-eval"
            elif "gpt4o-eval" in str(eval_folder):
                eval_type = "gpt4o-eval"
            elif "custom-eval" in str(eval_folder):
                eval_type = "custom-eval"
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analysis/figures/vmevalkit_{eval_type}_{timestamp}.png"
        
        # Use None for show_plot to display interactively
        final_output_path = None if args.show_plot else output_path
        create_visualizations(domain_df, overall_df, final_output_path)

if __name__ == "__main__":
    main()