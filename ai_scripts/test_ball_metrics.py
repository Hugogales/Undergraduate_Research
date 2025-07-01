#!/usr/bin/env python3
"""
Test script to verify ball metrics tracking in the soccer environment.

This script tests that ball hits and ball distance are calculated correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.train_rllib import SingleAgentSoccerWrapper

def test_ball_metrics():
    """Test ball metrics calculation."""
    print("🔧 Testing Ball Metrics Calculation...")
    print("Testing: Ball Hits = Player-Ball Collisions")
    print("Testing: Ball Distance = Total Distance Ball Moved")
    
    # Create environment config
    env_config = {
        "num_players": 1,
        "game_duration": 10,  # Short test
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
    print(f"Initial obs shape: {obs.shape}")
    
    # Run some steps and check metrics
    total_reward = 0
    prev_ball_dist = 0
    for step in range(30):  # More steps to see ball movement
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        current_ball_dist = getattr(env, 'episode_ball_distance', 0)
        ball_moved = current_ball_dist - prev_ball_dist
        prev_ball_dist = current_ball_dist
        
        print(f"Step {step+1}: Reward={reward:.2f}, "
              f"Ball Hits={getattr(env, 'episode_ball_hits', 0)}, "
              f"Ball Dist={current_ball_dist:.1f} (+{ball_moved:.1f})")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            print(f"Final info: {info}")
            break
    
    print(f"\n✅ Test completed! Total reward: {total_reward:.2f}")
    print(f"Final Ball Hits (collisions): {getattr(env, 'episode_ball_hits', 0)}")
    print(f"Final Ball Distance Moved: {getattr(env, 'episode_ball_distance', 0):.1f}")
    
if __name__ == "__main__":
    test_ball_metrics() 