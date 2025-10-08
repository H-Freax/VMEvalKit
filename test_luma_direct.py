#!/usr/bin/env python3
"""
Direct test of Luma API with our maze images.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from vmevalkit.api_clients.luma_client import LumaDreamMachine

# Test with different settings
client = LumaDreamMachine(
    enhance_prompt=False,  # Disable prompt enhancement
    model="ray-2",
    verbose=True  # Show detailed progress updates (set to False for minimal output)
)

# Try with a public test image from the internet first
# Using a nature image that should pass moderation
result = client.generate(
    image="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1280&h=720&fit=crop",
    text_prompt="The mountains come alive with movement and clouds swirl dramatically.",
    duration=5.0,
    resolution=(1280, 720)
)

print(f"\nSuccess! Video saved to: {result}")
