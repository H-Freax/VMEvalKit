"""Centralized prompts for Sliding Puzzle Task."""

# Single prompt with dynamic move count
PROMPTS = [
    "Complete this sliding puzzle by moving the numbered tiles to their correct positions. "
    "Only {num_moves} move{plural} {is_are} needed. Show the tile(s) sliding into place.",
]

DEFAULT_PROMPT_INDEX = 0

