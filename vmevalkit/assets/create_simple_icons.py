#!/usr/bin/env python3
"""
Create SIMPLE visual icons with clean and minimal styling.
No gradients or fancy effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_simple_green_dot(size=64):
    """Create a simple solid green circle."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple solid green circle - no gradients, no effects
    circle = patches.Circle((0.5, 0.5), 0.4, 
                          facecolor='#22c55e', edgecolor='none')  # Simple green
    ax.add_patch(circle)
    
    return fig


def create_simple_flag(size=64):
    """Create a simple red flag."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple brown pole
    pole = patches.Rectangle((0.15, 0.1), 0.08, 0.8, 
                           facecolor='#a3723b', edgecolor='none')  # Brown pole
    ax.add_patch(pole)
    
    # Simple red triangular flag
    flag_points = np.array([[0.23, 0.75], [0.75, 0.6], [0.23, 0.45]])
    flag = patches.Polygon(flag_points, 
                         facecolor='#ef4444', edgecolor='none')  # Simple red
    ax.add_patch(flag)
    
    return fig


def create_simple_trophy(size=64):
    """Create a simple golden trophy."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1) 
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple trophy base
    base = patches.Rectangle((0.35, 0.1), 0.3, 0.15, 
                           facecolor='#f59e0b', edgecolor='none')  # Golden
    ax.add_patch(base)
    
    # Simple trophy cup
    cup = patches.Rectangle((0.3, 0.25), 0.4, 0.4, 
                          facecolor='#f59e0b', edgecolor='none')  # Golden
    ax.add_patch(cup)
    
    return fig


def create_simple_star(size=64):
    """Create a simple blue star."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_alpha(0)  # Transparent background
    
    # Simple 5-pointed star
    angles = np.linspace(0, 2*np.pi, 11)  # 10 points + back to start
    outer_radius = 0.35
    inner_radius = 0.15
    
    star_points = []
    for i, angle in enumerate(angles[:-1]):  # Skip last point (same as first)
        if i % 2 == 0:  # Outer points
            x = 0.5 + outer_radius * np.cos(angle - np.pi/2)
            y = 0.5 + outer_radius * np.sin(angle - np.pi/2)
        else:  # Inner points
            x = 0.5 + inner_radius * np.cos(angle - np.pi/2)
            y = 0.5 + inner_radius * np.sin(angle - np.pi/2)
        star_points.append([x, y])
    
    star = patches.Polygon(star_points, 
                         facecolor='#3b82f6', edgecolor='none')  # Simple blue
    ax.add_patch(star)
    
    return fig


def save_simple_icons(output_dir):
    """Generate and save simple icons."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    icons = {
        'green_dot.png': create_simple_green_dot(),
        'flag.png': create_simple_flag(), 
        'trophy.png': create_simple_trophy(),
        'star.png': create_simple_star()
    }
    
    for filename, fig in icons.items():
        filepath = output_path / filename
        fig.savefig(filepath, transparent=True, bbox_inches='tight', 
                   pad_inches=0, dpi=64, facecolor='none')
        plt.close(fig)
        print(f"✓ Created SIMPLE {filepath}")
    
    print(f"\n🎨 All SIMPLE icons created in {output_path}")
    print("Clean, minimal style icons generated.")


if __name__ == "__main__":
    save_simple_icons("vmevalkit/assets/icons")
