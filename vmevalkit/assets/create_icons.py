#!/usr/bin/env python3
"""
Create visual icons for maze rendering (green dot, flag, trophy, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_green_dot_icon(size=64):
    """Create a green circle icon for start position."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create green circle with white border
    circle = patches.Circle((0.5, 0.5), 0.35, 
                          facecolor='#4caf50', edgecolor='white', linewidth=3)
    ax.add_patch(circle)
    
    # Add inner highlight for 3D effect
    highlight = patches.Circle((0.45, 0.55), 0.15, 
                             facecolor='#66bb6a', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def create_flag_icon(size=64):
    """Create a red flag icon for end position."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Flag pole (gray)
    pole = patches.Rectangle((0.45, 0.1), 0.1, 0.8, 
                           facecolor='#757575', edgecolor='#424242', linewidth=1)
    ax.add_patch(pole)
    
    # Flag cloth (red with white border)
    flag_points = np.array([[0.55, 0.8], [0.9, 0.7], [0.9, 0.5], [0.55, 0.6]])
    flag = patches.Polygon(flag_points, 
                         facecolor='#f44336', edgecolor='white', linewidth=2)
    ax.add_patch(flag)
    
    # Flag highlight
    highlight_points = np.array([[0.55, 0.75], [0.75, 0.7], [0.75, 0.55], [0.55, 0.6]])
    highlight = patches.Polygon(highlight_points, 
                              facecolor='#ef5350', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def create_trophy_icon(size=64):
    """Create a golden trophy icon for success."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1) 
    ax.axis('off')
    
    # Trophy base
    base = patches.Rectangle((0.3, 0.1), 0.4, 0.15, 
                           facecolor='#ffc107', edgecolor='#ff8f00', linewidth=2)
    ax.add_patch(base)
    
    # Trophy cup
    cup = patches.FancyBboxPatch((0.35, 0.25), 0.3, 0.4, 
                               boxstyle="round,pad=0.02",
                               facecolor='#ffb300', edgecolor='#ff8f00', linewidth=2)
    ax.add_patch(cup)
    
    # Trophy handles
    left_handle = patches.Arc((0.25, 0.45), 0.2, 0.3, angle=0, theta1=270, theta2=90, 
                            color='#ff8f00', linewidth=3)
    right_handle = patches.Arc((0.75, 0.45), 0.2, 0.3, angle=0, theta1=90, theta2=270,
                             color='#ff8f00', linewidth=3)
    ax.add_patch(left_handle)
    ax.add_patch(right_handle)
    
    # Trophy highlight
    highlight = patches.Ellipse((0.45, 0.5), 0.15, 0.2, 
                              facecolor='#fff176', alpha=0.6)
    ax.add_patch(highlight)
    
    return fig


def create_star_icon(size=64):
    """Create a blue star icon for moving element."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create 5-pointed star
    angles = np.linspace(0, 2*np.pi, 11)  # 10 points + back to start
    outer_radius = 0.4
    inner_radius = 0.2
    
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
                         facecolor='#2196f3', edgecolor='white', linewidth=2)
    ax.add_patch(star)
    
    # Star highlight
    highlight = patches.Circle((0.45, 0.55), 0.1, 
                             facecolor='#64b5f6', alpha=0.7)
    ax.add_patch(highlight)
    
    return fig


def save_icons(output_dir):
    """Generate and save all icon assets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    icons = {
        'green_dot.png': create_green_dot_icon(),
        'flag.png': create_flag_icon(), 
        'trophy.png': create_trophy_icon(),
        'star.png': create_star_icon()
    }
    
    for filename, fig in icons.items():
        filepath = output_path / filename
        fig.savefig(filepath, transparent=True, bbox_inches='tight', 
                   pad_inches=0, dpi=64, facecolor='none')
        plt.close(fig)
        print(f"✓ Created {filepath}")
    
    print(f"\n🎨 All icons created in {output_path}")


if __name__ == "__main__":
    save_icons("vmevalkit/assets/icons")
