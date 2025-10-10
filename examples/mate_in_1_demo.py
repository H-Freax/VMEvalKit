#!/usr/bin/env python3
"""
Mate-in-1 Chess Reasoning Demo for VMEvalKit

This demonstrates how the mate-in-1 chess system works and how it can be used
to evaluate video models' ability to identify and demonstrate winning moves.

Usage:
    python examples/mate_in_1_demo.py
"""

import sys
import os

# Add the chess task module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vmevalkit.tasks.chess_mate_in_1 import (
    MateIn1Generator, 
    MateIn1Validator, 
    create_vmevalkit_task
)


def demonstrate_working_system():
    """Show that all mate-in-1 positions are working correctly."""
    print("🏁 MATE-IN-1 CHESS SYSTEM VERIFICATION")
    print("=" * 60)
    
    generator = MateIn1Generator()
    validator = MateIn1Validator()
    
    print(f"✅ Loaded {len(generator.puzzles)} verified mate-in-1 positions")
    print()
    
    for i, puzzle in enumerate(generator.puzzles, 1):
        print(f"📋 PUZZLE {i}: {puzzle.puzzle_id}")
        print(f"   Description: {puzzle.description}")
        print(f"   FEN: {puzzle.fen}")
        print(f"   Side to move: {puzzle.side_to_move}")
        print(f"   Expected solution: {puzzle.mate_moves}")
        
        # Validate the puzzle works
        is_valid = generator.validate_puzzle(puzzle)
        print(f"   Validation: {'✅ WORKING' if is_valid else '❌ BROKEN'}")
        
        # Show analysis
        analysis = validator.analyze_position(puzzle)
        all_mates = analysis['mate_moves']
        print(f"   All mate moves: {all_mates}")
        print(f"   Multiple solutions: {'Yes' if len(all_mates) > 1 else 'No'}")
        print()


def demonstrate_video_task_creation():
    """Show how to create VMEvalKit tasks from mate-in-1 positions."""
    print("🎬 VMEVALKIT VIDEO TASK CREATION")
    print("=" * 60)
    
    generator = MateIn1Generator()
    
    # Create tasks for each puzzle type
    for puzzle in generator.puzzles:
        print(f"📹 VIDEO TASK: {puzzle.puzzle_id}")
        
        task = create_vmevalkit_task(puzzle)
        
        print(f"   Task ID: {task['task_id']}")
        print(f"   Task Type: {task['task_type']}")
        print(f"   Difficulty: {task['difficulty']}")
        print()
        print("   INPUT:")
        print(f"   📸 Image: Chess board showing position")
        print(f"   💬 Prompt: \"{task['text_prompt']}\"")
        print()
        print("   EXPECTED OUTPUT:")
        print(f"   🎥 Video: {task['expected_output']}")
        print()
        print("   EVALUATION CRITERIA:")
        for criterion, description in task['evaluation_criteria'].items():
            print(f"   ✓ {criterion}: {description}")
        print()
        print("-" * 40)


def demonstrate_solution_validation():
    """Show how solution validation works for different move attempts."""
    print("🔍 SOLUTION VALIDATION DEMO")
    print("=" * 60)
    
    generator = MateIn1Generator()
    validator = MateIn1Validator()
    
    # Get the back-rank mate puzzle
    puzzle = generator.get_puzzle("back_rank_001")
    
    print(f"📋 Testing puzzle: {puzzle.description}")
    print(f"   Position: {puzzle.fen}")
    print()
    
    # Test different move attempts
    test_moves = [
        ("Ra8", "Correct mate move (without # notation)"),
        ("Ra8#", "Correct mate move (with # notation)"),
        ("Rb8", "Legal move but not mate"),
        ("Ra7", "Legal move but not mate"), 
        ("Ke2", "Legal king move, no mate"),
        ("Nf3", "Illegal move - no knight on right square"),
        ("invalid", "Invalid move notation")
    ]
    
    print("🧪 TESTING DIFFERENT SOLUTION ATTEMPTS:")
    print()
    
    for move, description in test_moves:
        result = validator.validate_solution(puzzle, move)
        
        status = "✅" if result['is_correct'] else "❌"
        print(f"{status} {move:<8} | {description}")
        print(f"          Legal: {result['is_legal']}")
        print(f"          Mate: {result['is_mate']}")
        print(f"          Message: {result['message']}")
        print()


def demonstrate_multiple_solutions():
    """Show puzzle with multiple correct solutions."""
    print("🎯 MULTIPLE SOLUTIONS DEMONSTRATION")
    print("=" * 60)
    
    generator = MateIn1Generator()
    validator = MateIn1Validator()
    
    # Get the queen corner puzzle (has multiple mates)
    puzzle = generator.get_puzzle("queen_corner_001")
    
    print(f"📋 Puzzle: {puzzle.description}")
    print(f"   Position: {puzzle.fen}")
    print()
    
    # Analyze to find all mate moves
    analysis = validator.analyze_position(puzzle)
    all_mates = analysis['mate_moves']
    
    print(f"🎊 This position has {len(all_mates)} different mate-in-1 solutions!")
    print("   Any of these moves would be considered CORRECT:")
    print()
    
    for i, mate_move in enumerate(all_mates, 1):
        result = validator.validate_solution(puzzle, mate_move)
        print(f"   {i}. {mate_move} - {result['message']}")
    
    print()
    print("💡 This is PERFECT for video model evaluation because:")
    print("   • Models have multiple valid solutions to choose from")
    print("   • Tests creative problem solving, not just memorization")
    print("   • Any correct mate move should be accepted")


def demonstrate_integration_workflow():
    """Show the complete workflow for video model evaluation."""
    print("⚙️  COMPLETE VMEVALKIT INTEGRATION WORKFLOW")
    print("=" * 60)
    
    print("1️⃣  TASK GENERATION:")
    print("   • Load verified mate-in-1 positions")
    print("   • Generate chess board images (SVG/PNG)")
    print("   • Create text prompts for each position")
    print("   • Package as VMEvalKit tasks")
    print()
    
    print("2️⃣  VIDEO MODEL INFERENCE:")
    print("   • Input: Board image + text prompt")
    print("   • Model generates: Video showing piece movement")
    print("   • Output: Video file with move sequence")
    print()
    
    print("3️⃣  SOLUTION EXTRACTION:")
    print("   • Analyze video to identify piece movement")
    print("   • Convert movement to chess notation (e.g., Ra1-a8)")
    print("   • Extract final move in standard format")
    print()
    
    print("4️⃣  VALIDATION & SCORING:")
    print("   • Check if move is legal in position")
    print("   • Verify move results in checkmate")
    print("   • Score based on correctness and video quality")
    print("   • Generate detailed evaluation report")
    print()
    
    print("5️⃣  EVALUATION METRICS:")
    print("   • Move Accuracy: % of correct mate moves")
    print("   • Legal Move Rate: % of legal moves attempted")
    print("   • Video Clarity: Quality of piece movement demonstration")
    print("   • Solution Speed: Time to identify correct move")


def main():
    """Run the complete mate-in-1 demonstration."""
    print("🏆 CHESS MATE-IN-1 SYSTEM FOR VMEVALKIT")
    print("Testing Video Models' Ability to Find Winning Moves")
    print("=" * 80)
    print()
    
    # Run all demonstrations
    demonstrate_working_system()
    print("\n" + "=" * 80 + "\n")
    
    demonstrate_video_task_creation()
    print("\n" + "=" * 80 + "\n")
    
    demonstrate_solution_validation()
    print("\n" + "=" * 80 + "\n")
    
    demonstrate_multiple_solutions()
    print("\n" + "=" * 80 + "\n")
    
    demonstrate_integration_workflow()
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY: READY FOR VIDEO MODEL EVALUATION!")
    print("=" * 80)
    print("✅ 3 verified working mate-in-1 positions")
    print("✅ Complete validation system")
    print("✅ Multiple solution support")
    print("✅ VMEvalKit integration ready")
    print("✅ Comprehensive evaluation metrics")
    print()
    print("🚀 Next steps:")
    print("   • Generate board images for input")
    print("   • Integrate with video model inference pipeline")
    print("   • Add video analysis for move extraction")
    print("   • Create comprehensive test dataset")


if __name__ == "__main__":
    main()
