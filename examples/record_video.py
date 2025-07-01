#!/usr/bin/env python3

"""
Video Recording Example

This example demonstrates how to use rgb_array mode to record a soccer game
as a video file. This is useful for creating replays, demonstrations, or
visual analysis of AI performance.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env


def record_game_video(output_path="game_recording.mp4", duration=30, fps=30):
    """
    Record a soccer game as a video file using rgb_array mode.
    
    Args:
        output_path: Path to save the video file
        duration: Game duration in seconds
        fps: Frames per second for recording
    """
    
    print(f"Recording {duration}s soccer game to {output_path}")
    print("Note: This runs headless (no window shown)")
    
    # Create environment with rgb_array mode
    env_instance = env(
        render_mode="rgb_array",  # Returns image data instead of showing window
        num_players_per_team=2,
        game_duration_seconds=duration,
        randomize_starting_positions=True,
    )
    
    # Reset environment
    observations, infos = env_instance.reset()
    
    # Collect frames
    frames = []
    
    try:
        while env_instance.agents:
            # Random actions for demonstration
            actions = {
                agent: np.random.uniform(-0.5, 0.5, 4) 
                for agent in env_instance.agents
            }
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env_instance.step(actions)
            
            # Get frame as RGB array
            frame = env_instance.render()  # Returns numpy array when mode="rgb_array"
            if frame is not None:
                frames.append(frame)
            
            # Check if game ended
            if all(terminations.values()) or all(truncations.values()):
                break
                
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        env_instance.close()
    
    print(f"Recorded {len(frames)} frames")
    
    # Save video (requires imageio or opencv-python)
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"✅ Video saved to: {output_path}")
    except ImportError:
        print("❌ imageio not installed. Install with: pip install imageio[ffmpeg]")
        print("Frames collected but not saved to video file")
        return frames
    
    return frames


def record_screenshots(num_screenshots=10):
    """
    Capture individual screenshots using rgb_array mode.
    """
    
    print(f"Capturing {num_screenshots} screenshots")
    
    # Create environment
    env_instance = env(
        render_mode="rgb_array",
        num_players_per_team=3,
        game_duration_seconds=60,
        randomize_starting_positions=True,
    )
    
    observations, infos = env_instance.reset()
    screenshots = []
    
    try:
        for i in range(num_screenshots):
            # Random actions
            actions = {
                agent: np.random.uniform(-0.3, 0.3, 4) 
                for agent in env_instance.agents
            }
            
            # Step a few times between screenshots
            for _ in range(10):
                if env_instance.agents:
                    observations, rewards, terminations, truncations, infos = env_instance.step(actions)
            
            # Capture screenshot
            frame = env_instance.render()
            if frame is not None:
                screenshots.append(frame)
                print(f"Screenshot {i+1}/{num_screenshots} captured")
            
            if all(terminations.values()) or all(truncations.values()):
                break
                
    finally:
        env_instance.close()
    
    return screenshots


if __name__ == "__main__":
    print("=== Soccer Environment Video Recording Demo ===")
    print()
    
    # Example 1: Record a full game
    frames = record_game_video("demo_game.mp4", duration=15, fps=30)
    
    print()
    
    # Example 2: Capture screenshots
    screenshots = record_screenshots(5)
    print(f"Captured {len(screenshots)} screenshots")
    
    print()
    print("🎥 rgb_array mode is perfect for:")
    print("  - Video recording") 
    print("  - Screenshot capture")
    print("  - Headless environments")
    print("  - Visual observations for AI")
    print("  - Batch processing") 