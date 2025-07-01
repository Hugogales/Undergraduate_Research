#!/usr/bin/env python3

"""
Simple Soccer Game Example

This example demonstrates basic usage of the soccer environment
with human controls for a quick 1v1 game.
"""

import sys
import time
from pathlib import Path

# ======== STANDARD IMPORT METHOD: sys.path manipulation ========
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env


def play_quick_game(num_players_per_team=1, game_duration_seconds=60.0, target_fps=42):
    """Play a quick soccer game with human controls."""
    
    print(f"Starting {num_players_per_team}v{num_players_per_team} soccer game")
    print(f"Game duration: {game_duration_seconds} seconds")
    print(f"Target FPS: {target_fps}")
    print("Starting positions: Original tactical positions")
    print("Controls: WASD for Team 0, Arrow keys for Team 1")
    
    # Create environment
    env_instance = env(
        render_mode="human", 
        num_players_per_team=num_players_per_team,
        game_duration_seconds=game_duration_seconds,
        randomize_starting_positions=False  # Use original tactical positions
    )
    
    # Reset and run game
    observations, infos = env_instance.reset()
    
    # FPS control
    frame_time = 1.0 / target_fps  # Time per frame in seconds
    
    try:
        while env_instance.agents:
            frame_start = time.time()
            
            # Dummy actions - human input will override these
            actions = {agent: [0.0, 0.0, 0.0, 0.0] for agent in env_instance.agents}
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env_instance.step(actions)
            
            # Check if game ended
            if all(terminations.values()) or all(truncations.values()):
                break
                
            # Check for quit
            if any(info.get('quit', False) for info in infos.values()):
                break
            
            # FPS limiting - sleep to maintain target FPS
            frame_end = time.time()
            frame_duration = frame_end - frame_start
            sleep_time = frame_time - frame_duration
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("Game interrupted")
    finally:
        env_instance.close()
        
    print("Game finished!")


if __name__ == "__main__":
    play_quick_game() 