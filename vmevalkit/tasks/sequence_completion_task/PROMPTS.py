"""
Prompts for Sequence Completion Reasoning Tasks

This file centralizes all prompts used for sequence completion reasoning tasks.
Each type has a prompt template that describes the sequence pattern.
"""

# Type 1: Arithmetic Sequence
TYPE_1_PROMPT = "This is a sequence. Observe its pattern, then fill in the number in the last cell to complete the arithmetic sequence. Sequence: {sequence_str}"

# Type 2: Geometric Sequence
TYPE_2_PROMPT = "This is a sequence. Observe its pattern, then fill in the number in the last cell to complete the geometric sequence. Sequence: {sequence_str}"

# Type 3: Power Sequence
TYPE_3_PROMPT = "This is a sequence. Observe its pattern, then fill in the number in the last cell to complete the square sequence. Sequence: {sequence_str}"

# Type 4: Fibonacci Sequence
TYPE_4_PROMPT = "This is a sequence. Observe its pattern, then fill in the number in the last cell to complete the Fibonacci sequence. Sequence: {sequence_str}"

# Type 5: Shape Cycle
TYPE_5_PROMPT = "This is a sequence. Observe its pattern, then fill in the shape in the last cell to complete the shape cycle. Sequence: {sequence_str}"

# Type 6: Color Cycle
TYPE_6_PROMPT = "This is a sequence. Observe its pattern, then fill in the color in the last cell to complete the color cycle. Sequence: {sequence_str}"

# Type 7: Position Cycle
TYPE_7_PROMPT = "This is a sequence. Observe its pattern, then fill in the position in the last cell to complete the position cycle. Sequence: {sequence_str}"

# Type 8: Mixed Sequence
TYPE_8_PROMPT = "This is a sequence. Observe its pattern, then fill in the element in the last cell to complete the mixed sequence. Sequence: {sequence_str}"

# All Type prompts organized by type index (1-8)
TYPE_PROMPTS = {
    1: TYPE_1_PROMPT,
    2: TYPE_2_PROMPT,
    3: TYPE_3_PROMPT,
    4: TYPE_4_PROMPT,
    5: TYPE_5_PROMPT,
    6: TYPE_6_PROMPT,
    7: TYPE_7_PROMPT,
    8: TYPE_8_PROMPT
}

