#!/usr/bin/env python3
"""
Debug script to figure out why ball metrics aren't working in RLLib training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.train_rllib import SingleAgentSoccerWrapper

def debug_ball_metrics():
    """Debug ball metrics step by step."""
    print("🔧 Debugging Ball Metrics...")
    
    # Create environment config
    env_config = {
        "num_players": 1,
        "game_duration": 5,  # Very short test
        "render_training": False,
        "random_positions": True,
        "stage": 1
    }
    
    # Create wrapped environment
    env = SingleAgentSoccerWrapper(env_config)
    
    print("✅ Environment created")
    
    # Reset environment
    obs, info = env.reset()
    print("✅ Environment reset")
    print(f"Reset info: {info}")
    
    # Take some steps and debug what happens
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Info keys: {list(info.keys())}")
        print(f"  Ball Hits (internal): {getattr(env, 'episode_ball_hits', 'NOT FOUND')}")
        print(f"  Ball Dist (internal): {getattr(env, 'episode_ball_distance', 'NOT FOUND')}")
        print(f"  Previous Ball Pos: {getattr(env, 'previous_ball_position', 'NOT FOUND')}")
        
        # Check if ball metrics are in info
        if 'ball_hits' in info:
            print(f"  Info ball_hits: {info['ball_hits']}")
        if 'ball_distance' in info:
            print(f"  Info ball_distance: {info['ball_distance']}")
        
        # Try to access game state directly
        try:
            game_state = env.env.game_engine.get_game_state()
            if game_state and hasattr(game_state, 'ball'):
                ball_pos = game_state.ball.position
                ball_vel = game_state.ball.velocity
                ball_speed = np.sqrt(ball_vel[0]**2 + ball_vel[1]**2)
                print(f"  Ball pos: {ball_pos}")
                print(f"  Ball vel: {ball_vel}")  
                print(f"  Ball speed: {ball_speed:.2f}")
            else:
                print(f"  Game state access failed")
        except Exception as e:
            print(f"  Game state error: {e}")
        
        if terminated or truncated:
            print(f"  Episode ended!")
            break
    
    print(f"\n✅ Debug completed!")

if __name__ == "__main__":
    debug_ball_metrics() 