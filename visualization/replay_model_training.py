#!/usr/bin/env python3
"""
Script to create a movie from training PNG files.

This script takes a directory containing PNG files that start with "training_"
and creates an MP4 movie showing the training progress over time.
"""

import os
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import image
import numpy as np


def get_training_files(directory):
    """
    Get all PNG files starting with 'training_' from the specified directory.
    
    Args:
        directory (str): Path to directory containing training PNG files
        
    Returns:
        list: Sorted list of PNG file paths
    """
    pattern = os.path.join(directory, "training_*.png")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No PNG files starting with 'training_' found in {directory}")
    
    # Sort files to ensure proper sequence
    files.sort()
    return files


def create_training_movie(directory, output_file="training_replay.mp4", fps=10, duration=None):
    """
    Create a movie from training PNG files.
    
    Args:
        directory (str): Directory containing training PNG files
        output_file (str): Output MP4 filename
        fps (int): Frames per second for the movie
        duration (float, optional): Total movie duration in seconds. If None, uses all frames.
    """
    # Get training files
    training_files = get_training_files(directory)
    print(f"Found {len(training_files)} training PNG files")
    
    # Load first image to get dimensions
    first_img = image.imread(training_files[0])
    height, width = first_img.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width/100, height/100))  # Scale figure size
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create image object
    img_plot = ax.imshow(first_img)
    # ax.set_title("Training Progress", fontsize=14, pad=20)
    
    def animate(frame_idx):
        """Animation function to update image for each frame."""
        if frame_idx < len(training_files):
            img = image.imread(training_files[frame_idx])
            img_plot.set_array(img)
            
            # Update title with frame info
            filename = os.path.basename(training_files[frame_idx])
            ax.set_title(f"Iteration {frame_idx + 1}/{len(training_files)}", 
                        fontsize=16, pad=2)
        
        return [img_plot]
    
    # Calculate number of frames based on duration if specified
    if duration is not None:
        total_frames = int(duration * fps)
        # Repeat frames if needed to fill duration
        if total_frames > len(training_files):
            # Pad with last frame
            training_files.extend([training_files[-1]] * (total_frames - len(training_files)))
        elif total_frames < len(training_files):
            # Sample frames to fit duration
            indices = np.linspace(0, len(training_files) - 1, total_frames, dtype=int)
            training_files = [training_files[i] for i in indices]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(training_files), 
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save animation
    output_file = Path(directory) / "training_replay.mp4"
    print(f"Creating movie with {len(training_files)} frames at {fps} FPS...")
    print(f"Saving to: {output_file}")
    
    # Use ffmpeg writer for MP4 output
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Training Replay Script'), bitrate=1800)
    
    anim.save(output_file, writer=writer, dpi=100)
    print(f"Movie saved successfully: {output_file}")
    
    plt.close(fig)


def main():
    """Main function to parse arguments and create the movie."""
    parser = argparse.ArgumentParser(
        description="Create a movie from training PNG files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replay_model_training.py /path/to/training/directory
  python replay_model_training.py /path/to/training/directory --fps 15
  python replay_model_training.py /path/to/training/directory --fps 20 --duration 30
  python replay_model_training.py /path/to/training/directory --output my_training.mp4
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing training PNG files'
    )
    
    parser.add_argument(
        '--fps', '-f',
        type=int,
        default=10,
        help='Frames per second for the movie (default: 10)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        help='Total movie duration in seconds (optional, will use all frames if not specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='training_replay.mp4',
        help='Output MP4 filename (default: training_replay.mp4)'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return 1
    
    try:
        create_training_movie(
            directory=args.directory,
            output_file=args.output,
            fps=args.fps,
            duration=args.duration
        )
        return 0
    except Exception as e:
        print(f"Error creating movie: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
