#!/usr/bin/env python3
"""
VideoThinkBench Dataset Downloader

This script downloads the complete VideoThinkBench dataset from HuggingFace
and converts it to the VMEvalKit folder structure format.

Source: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
License: MIT (VideoThinkBench is licensed under MIT)

Author: VMEvalKit Team
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_videothinkbench_dataset(output_dir: str = "data/questions") -> Dict[str, Any]:
    """
    Download complete VideoThinkBench dataset from HuggingFace.
    
    The dataset contains 5 subsets:
    - ARC_AGI_2: Abstract reasoning tasks (1k rows)
    - Eyeballing_Puzzles: Spatial reasoning and visual estimation (1.05k rows)  
    - Mazes: Path-finding and navigation (150 rows)
    - Text_Centric_Tasks: Mathematical and multimodal reasoning (1.45k rows)
    - Visual_Puzzles: Pattern recognition and visual logic (496 rows)
    
    Total: ~4,149 tasks
    
    Args:
        output_dir: Base directory for output (default: data/questions)
        
    Returns:
        Dictionary with download statistics and metadata
    """
    
    from datasets import load_dataset
    from PIL import Image
    
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # VideoThinkBench subsets configuration
    subsets = {
        'ARC_AGI_2': {
            'name': 'ARC AGI 2',
            'description': 'Abstract reasoning tasks requiring few-shot learning',
            'domain_folder': 'arc_agi_2_task'
        },
        'Eyeballing_Puzzles': {
            'name': 'Eyeballing Puzzles',
            'description': 'Spatial reasoning tasks requiring visual estimation',
            'domain_folder': 'eyeballing_puzzles_task'
        },
        'Mazes': {
            'name': 'Mazes',
            'description': 'Path-finding and navigation challenges',
            'domain_folder': 'mazes_task'
        },
        'Text_Centric_Tasks': {
            'name': 'Text Centric Tasks',
            'description': 'Mathematical reasoning and multimodal understanding',
            'domain_folder': 'text_centric_tasks_task'
        },
        'Visual_Puzzles': {
            'name': 'Visual Puzzles',
            'description': 'Pattern recognition and visual logic problems',
            'domain_folder': 'visual_puzzles_task'
        }
    }
    
    print("=" * 80)
    print("📥 VIDEOTHINKBENCH DATASET DOWNLOADER")
    print("=" * 80)
    print(f"🔗 Source: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench")
    print(f"📜 License: MIT")
    print(f"📁 Output: {base_path.absolute()}")
    print(f"📊 Total subsets: {len(subsets)}")
    print()
    
    all_tasks = []
    subset_stats = {}
    
    for subset_name, subset_info in subsets.items():
        print(f"📦 Processing: {subset_info['name']} ({subset_name})")
        print(f"   Description: {subset_info['description']}")
        
        # Load dataset from HuggingFace
        print(f"   ⬇️  Downloading from HuggingFace...")
        dataset = load_dataset(
            'OpenMOSS-Team/VideoThinkBench',
            subset_name,
            split='test'
        )
        
        print(f"   📊 Found {len(dataset)} tasks")
        
        # Create domain folder
        domain_folder = base_path / subset_info['domain_folder']
        domain_folder.mkdir(parents=True, exist_ok=True)
        
        downloaded_count = 0
        skipped_count = 0
        
        for idx, item in enumerate(dataset):
            # Extract task information
            task_id = item.get('id', f"{subset_name.lower()}_{idx:04d}")
            prompt = item.get('prompt', '')
            first_image = item.get('image')
            solution_image = item.get('solution_image')
            task_type = item.get('task', subset_name.lower())
            
            # Validate required fields
            if not prompt or first_image is None:
                print(f"      ⚠️  Skipping {task_id}: Missing required fields")
                skipped_count += 1
                continue
            
            # Create task folder
            task_folder = domain_folder / task_id
            task_folder.mkdir(parents=True, exist_ok=True)
            
            # Process and save first image
            if not isinstance(first_image, Image.Image):
                first_image = Image.fromarray(first_image) if hasattr(first_image, 'shape') else Image.open(first_image)
            if first_image.mode != "RGB":
                first_image = first_image.convert("RGB")
            
            first_image_path = task_folder / "first_frame.png"
            first_image.save(first_image_path, format="PNG")
            
            # Process and save solution image (if available)
            final_image_rel_path = None
            if solution_image is not None:
                if not isinstance(solution_image, Image.Image):
                    solution_image = Image.fromarray(solution_image) if hasattr(solution_image, 'shape') else Image.open(solution_image)
                if solution_image.mode != "RGB":
                    solution_image = solution_image.convert("RGB")
                
                solution_image_path = task_folder / "final_frame.png"
                solution_image.save(solution_image_path, format="PNG")
                final_image_rel_path = str(Path(subset_info['domain_folder']) / task_id / "final_frame.png")
            
            # Save prompt
            prompt_file = task_folder / "prompt.txt"
            prompt_file.write_text(prompt)
            
            # Create metadata
            task_metadata = {
                "id": task_id,
                "domain": subset_name.lower(),
                "task_type": task_type,
                "prompt": prompt,
                "first_image_path": str(Path(subset_info['domain_folder']) / task_id / "first_frame.png"),
                "final_image_path": final_image_rel_path,
                "created_at": datetime.now().isoformat() + 'Z',
                "source": "OpenMOSS-Team/VideoThinkBench",
                "source_license": "MIT",
                "subset": subset_name,
                "description": subset_info['description']
            }
            
            # Save metadata
            metadata_file = task_folder / "question_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(task_metadata, f, indent=2, default=str)
            
            all_tasks.append(task_metadata)
            downloaded_count += 1
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"      📥 Downloaded {idx + 1}/{len(dataset)} tasks...")
        
        subset_stats[subset_name] = {
            'name': subset_info['name'],
            'description': subset_info['description'],
            'downloaded': downloaded_count,
            'skipped': skipped_count,
            'total': len(dataset)
        }
        
        print(f"   ✅ Completed: {downloaded_count} downloaded, {skipped_count} skipped")
        print()
    
    # Create master dataset JSON
    dataset_metadata = {
        "name": "videothinkbench_dataset",
        "description": "VideoThinkBench - Comprehensive benchmark for video generation models' reasoning capabilities",
        "source": "https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench",
        "license": "MIT",
        "created_at": datetime.now().isoformat() + 'Z',
        "total_tasks": len(all_tasks),
        "subsets": subset_stats,
        "citation": """@article{tong2025thinkingwithvideo,
    title={Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm},
    author={Jingqi Tong and Yurong Mou and Hangcheng Li and Mingzhe Li and Yongzhuo Yang and Ming Zhang and Qiguang Chen and Tianyi Liang and Xiaomeng Hu and Yining Zheng and Xinchi Chen and Jun Zhao and Xuanjing Huang and Xipeng Qiu},
    journal={arXiv preprint arXiv:2511.04570},
    year={2025}
}""",
        "tasks": all_tasks
    }
    
    # Save master dataset JSON
    json_path = base_path / "videothinkbench_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2, default=str)
    
    # Print summary
    print("=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    total_downloaded = sum(s['downloaded'] for s in subset_stats.values())
    total_skipped = sum(s['skipped'] for s in subset_stats.values())
    
    for subset_name, stats in subset_stats.items():
        print(f"✓ {stats['name']:25} {stats['downloaded']:4d} tasks")
    
    print(f"\n{'Total Downloaded:':27} {total_downloaded:4d} tasks")
    if total_skipped > 0:
        print(f"{'Total Skipped:':27} {total_skipped:4d} tasks")
    
    print(f"\n📁 Data saved to: {base_path.absolute()}")
    print(f"📄 Master JSON: {json_path.absolute()}")
    print("\n✅ VideoThinkBench download complete!")
    print("=" * 80)
    
    return dataset_metadata


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download VideoThinkBench dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (data/questions)
  python download_videothinkbench.py
  
  # Download to custom location
  python download_videothinkbench.py --output-dir /path/to/output
  
Citation:
  If you use VideoThinkBench, please cite:
  
  @article{tong2025thinkingwithvideo,
      title={Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm},
      author={Jingqi Tong and Yurong Mou and Hangcheng Li and Mingzhe Li and Yongzhuo Yang and Ming Zhang and Qiguang Chen and Tianyi Liang and Xiaomeng Hu and Yining Zheng and Xinchi Chen and Jun Zhao and Xuanjing Huang and Xipeng Qiu},
      journal={arXiv preprint arXiv:2511.04570},
      year={2025}
  }

License:
  VideoThinkBench is licensed under MIT License.
  See: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/questions',
        help='Output directory for downloaded data (default: data/questions)'
    )
    
    args = parser.parse_args()
    
    # Download dataset
    download_videothinkbench_dataset(output_dir=args.output_dir)


if __name__ == "__main__":
    main()

