"""
Task definitions and generators for VMEvalKit.

This package contains modules for generating different types of reasoning tasks
that can be used to evaluate video generation models.
"""

from .maze_reasoning import (
    MazeTaskPair,
    MazeDataset,
    KnowWhatTaskGenerator,
    IrregularTaskGenerator,
    create_knowwhat_dataset,
    create_irregular_dataset,
    create_combined_dataset
)

__all__ = [
    "MazeTaskPair",
    "MazeDataset",
    "KnowWhatTaskGenerator",
    "IrregularTaskGenerator",
    "create_knowwhat_dataset",
    "create_irregular_dataset", 
    "create_combined_dataset"
]
