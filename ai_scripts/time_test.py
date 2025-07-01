#!/usr/bin/env python3
"""
Game Timing Test Script

This script precisely measures the game's internal timing logic
to verify if the timing issues have been resolved.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env


def test_game_timing():
    """Test game timing with different configurations."""
    
    test_configs = [
        {"duration": 10.0, "players": 1, "name": "1v1 - 10 seconds"},
        {"duration": 15.0, "players": 2, "name": "2v2 - 15 seconds"},
        {"duration": 20.0, "players": 3, "name": "3v3 - 20 seconds"},
    ]
    
    for config in test_configs:
        print(f"\n=== Testing {config['name']} ===")
        print(f"Expected duration: {config['duration']} seconds")
        
        # Create environment (headless - no rendering for speed)
        env_instance = env(
            render_mode=None,  # Headless
            num_players_per_team=config['players'],
            game_duration_seconds=config['duration'],
            randomize_starting_positions=True,
        )
        
        # Reset environment
        observations, infos = env_instance.reset()
        
        # Track game timing
        start_time = time.time()
        step_count = 0
        initial_time_remaining = None
        final_time_remaining = None
        
        try:
            while env_instance.agents:
                # Dummy actions for all agents
                actions = {agent: [0.0, 0.0, 0.0, 0.0] for agent in env_instance.agents}
                
                # Step environment
                observations, rewards, terminations, truncations, infos = env_instance.step(actions)
                step_count += 1
                
                # Track game time from info
                if infos and step_count == 1:
                    sample_info = next(iter(infos.values()))
                    initial_time_remaining = sample_info.get('time_remaining', 0)
                    print(f"Initial time remaining: {initial_time_remaining} steps")
                
                if infos:
                    sample_info = next(iter(infos.values()))
                    final_time_remaining = sample_info.get('time_remaining', 0)
                
                # Check if game ended
                if all(terminations.values()) or all(truncations.values()):
                    break
                    
        except Exception as e:
            print(f"Error during game: {e}")
        finally:
            # Stop timing
            end_time = time.time()
            wall_clock_duration = end_time - start_time
            
            env_instance.close()
            
            # Report results
            print(f"Wall-clock duration: {wall_clock_duration:.2f} seconds")
            print(f"Steps executed: {step_count}")
            print(f"Expected max steps: {env_instance.max_episode_steps}")
            print(f"Final time remaining: {final_time_remaining} steps")
            
            # Calculate expected game time progression
            target_fps = 42  # From constants
            expected_game_time_used = step_count / target_fps
            actual_game_duration = config['duration']
            
            print(f"Game time used: {expected_game_time_used:.2f} seconds")
            print(f"Expected game duration: {actual_game_duration} seconds")
            
            # Check step count accuracy
            if step_count == env_instance.max_episode_steps:
                print("✅ STEP COUNT OK")
            else:
                print(f"❌ STEP COUNT ISSUE: Expected {env_instance.max_episode_steps}, got {step_count}")
            
            # Check if final time remaining is 0 (game ended due to time)
            if final_time_remaining == 0:
                print("✅ GAME TIME OK - Ended due to time")
            else:
                print(f"❌ GAME TIME ISSUE - Still had {final_time_remaining} time remaining")


if __name__ == "__main__":
    test_game_timing() 