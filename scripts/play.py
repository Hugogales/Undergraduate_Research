#!/usr/bin/env python3

"""
Human Soccer Game

This script allows humans to play soccer using keyboard controls.
- Team 0: WASD keys (W=up, A=left, S=down, D=right)
- Team 1: Arrow keys (↑↓←→)
- ESC: Quit game

The environment uses the Parallel API for natural simultaneous gameplay.
"""

import sys
import argparse
import time
from pathlib import Path

# ======== STANDARD IMPORT METHOD: sys.path manipulation ========
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import constants and environment
from soccer_env.envs.soccer_env import env


def main():
    parser = argparse.ArgumentParser(description="Play soccer with human controls")
    parser.add_argument('--time', type=float, default=120.0, help='Game duration in seconds (default: 120)')
    parser.add_argument('--players', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       help='Number of players per team (default: 2)')
    parser.add_argument('--random', action='store_true', 
                       help='Start players in random positions instead of original tactical positions')
    parser.add_argument('--game-mode', type=str, default='default',
                       help='Game mode preset (default, air_hockey, arcade, realistic, giant_players, pinball, slow_motion, speed_demon, tiny_players, bouncy_castle)')
    parser.add_argument('--fps', type=int, default=42, 
                       help='Target FPS for human play (default: 42)')
    
    args = parser.parse_args()
    
    # Parse arguments
    game_duration = args.time
    num_players_per_team = args.players
    randomize_positions = args.random  # Default False = original positions
    game_mode = args.game_mode
    target_fps = args.fps
    
    print(f"Starting {num_players_per_team}v{num_players_per_team} soccer game")
    print(f"Game duration: {game_duration} seconds")
    print(f"Target FPS: {target_fps}")
    print(f"Starting positions: {'Random' if randomize_positions else 'Original tactical positions'}")
    print("Controls:")
    print("  Team 0 (Blue): WASD keys")
    print("  Team 1 (Red): Arrow keys") 
    print("  ESC: Quit game")
    print()

    # Initialize environment with simplified parameters
    env_instance = env(
        render_mode="human",  # Human play needs visible window and input handling
        num_players_per_team=num_players_per_team,
        game_duration_seconds=game_duration,
        randomize_starting_positions=randomize_positions,
        game_mode=game_mode,  # Apply game mode preset
    )
    
    # Reset environment
    observations, infos = env_instance.reset()
    
    # FPS control and monitoring
    frame_time = 1.0 / target_fps  # Time per frame in seconds
    frame_count = 0
    fps_timer = time.time()
    actual_fps = 0
    
    print(f"Game starting... (Target FPS: {target_fps})")
    
    # Main game loop
    try:
        while env_instance.agents:  # Continue while there are active agents
            frame_start = time.time()
            
            # In parallel environments, all agents act simultaneously
            # For human play, the renderer will handle input and provide actions
            # We need to provide dummy actions that will be overridden by human input
            actions = {}
            for agent in env_instance.agents:
                actions[agent] = [0.0, 0.0, 0.0, 0.0]  # Dummy action, will be overridden
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env_instance.step(actions)

            print(f"🔍 Step - Rewards: {rewards}")
            
            # Check if game ended (time up or user quit)
            if all(terminations.values()) or all(truncations.values()):
                print("Game completed - time's up!")
                break
                
            # Check for quit in info
            if any(info.get('quit', False) for info in infos.values()):
                print("Game quit by user")
                break
            
            # FPS monitoring and limiting
            frame_count += 1
            frame_end = time.time()
            frame_duration = frame_end - frame_start
            
            # Calculate actual FPS every 60 frames
            if frame_count % 60 == 0:
                current_time = time.time()
                time_elapsed = current_time - fps_timer
                actual_fps = 60 / time_elapsed if time_elapsed > 0 else 0
                fps_timer = current_time
                
                # Display current game info
                if infos:
                    sample_info = next(iter(infos.values()))
                    remaining_time = sample_info.get('time_remaining', 0)
                    team_0_score = sample_info.get('team_0_score', 0)
                    team_1_score = sample_info.get('team_1_score', 0)
                    print(f"⚽ Score: {team_0_score}-{team_1_score} | Time: {remaining_time} | FPS: {actual_fps:.1f}")
            
            # Sleep to maintain target FPS
            sleep_time = frame_time - frame_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env_instance.close()
        
    print("Game finished!")


if __name__ == "__main__":
    main() 