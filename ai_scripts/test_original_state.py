#!/usr/bin/env python3
"""
Test script to verify the updated state parser matches the original format.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from soccer_env.envs.soccer_env import SoccerEnv
from soccer_env.utils.state_parser import StateParser


def test_original_state_format():
    """Test that the state parser produces the original format."""
    print("🧪 Testing Original State Format")
    print("=" * 50)
    
    # Create 1v1 environment
    env = SoccerEnv(
        num_players_per_team=1,
        game_duration_seconds=30.0,
        randomize_starting_positions=True
    )
    
    # Reset environment
    observations, info = env.reset()
    print(f"✅ Environment created with {len(observations)} agents")
    
    # Test state parser directly
    state_parser = StateParser()
    game_state = env.game_engine.get_game_state()
    
    # Get observation for agent 0
    obs = state_parser.get_agent_observation(game_state, 0)
    
    print(f"\n📊 State Vector Analysis:")
    print(f"Observation size: {len(obs)}")
    print(f"Expected size for 1v1: {state_parser.get_observation_size(2)}")
    
    # Break down the observation according to original format
    idx = 0
    print(f"\n📊 Full observation: {obs}")
    
    # 1. Teammates (0 for 1v1 since no teammates excluding self)
    teammates_count = 0  # 1 player per team - 1 (self) = 0 
    teammates_end = idx + teammates_count * 2
    teammates = obs[idx:teammates_end]
    print(f"\n1. Teammates vectors: {teammates} (size: {len(teammates)})")
    idx = teammates_end
    
    # 2. Opponents (1 opponent in 1v1)
    opponents_count = 1
    opponents_end = idx + opponents_count * 2
    opponents = obs[idx:opponents_end]
    print(f"2. Opponents vectors: {opponents} (size: {len(opponents)})")
    idx = opponents_end
    
    # 3. Ball position
    ball_pos_end = idx + 2
    ball_pos = obs[idx:ball_pos_end]
    print(f"3. Ball position: {ball_pos} (size: {len(ball_pos)})")
    idx = ball_pos_end
    
    # 4. Ball velocity
    ball_vel_end = idx + 2
    ball_vel = obs[idx:ball_vel_end]
    print(f"4. Ball velocity: {ball_vel} (size: {len(ball_vel)})")
    idx = ball_vel_end
    
    # 5. Ball angle (added feature)
    if idx < len(obs):
        ball_angle = obs[idx]
        print(f"5. Ball angle: {ball_angle:.3f} (normalized atan2)")
        idx += 1
    
    # 6. Goal vectors (opponent goal first, then own goal)
    if idx + 4 <= len(obs):
        goal_vectors_end = idx + 4
        goal_vectors = obs[idx:goal_vectors_end]
        print(f"6. Goal vectors: {goal_vectors} (size: {len(goal_vectors)})")
        print(f"   - Opponent goal: [{goal_vectors[0]:.3f}, {goal_vectors[1]:.3f}]")
        print(f"   - Own goal: [{goal_vectors[2]:.3f}, {goal_vectors[3]:.3f}]")
        idx = goal_vectors_end
    
    # 7. Raycasts (wall distances)
    if idx + 4 <= len(obs):
        raycasts_end = idx + 4
        raycasts = obs[idx:raycasts_end]
        print(f"7. Wall distances: {raycasts} (size: {len(raycasts)})")
        print(f"   - North: {raycasts[0]:.3f}")
        print(f"   - South: {raycasts[1]:.3f}")
        print(f"   - East: {raycasts[2]:.3f}")
        print(f"   - West: {raycasts[3]:.3f}")
        idx = raycasts_end
    
    print(f"\n✅ Total indices used: {idx} / {len(obs)}")
    
    if idx < len(obs):
        remaining = obs[idx:]
        print(f"🔍 Unused data: {remaining}")
    
    # Test different team sizes
    print(f"\n🔄 Testing Different Team Sizes:")
    for team_size in [1, 2]:
        expected_size = state_parser.get_observation_size(team_size * 2)
        print(f"   {team_size}v{team_size}: Expected observation size = {expected_size}")
        
        # Create environment and test
        test_env = SoccerEnv(num_players_per_team=team_size, game_duration_seconds=10.0)
        test_obs, _ = test_env.reset()
        actual_size = len(test_obs[list(test_obs.keys())[0]])
        print(f"   {team_size}v{team_size}: Actual observation size = {actual_size}")
        
        if actual_size == expected_size:
            print(f"   ✅ {team_size}v{team_size} size matches!")
        else:
            print(f"   ❌ {team_size}v{team_size} size mismatch!")
    
    print(f"\n🎯 State Format Summary:")
    print(f"   - Teammates: Relative positions of own team (excluding self)")
    print(f"   - Opponents: Relative positions of enemy team")
    print(f"   - Ball: Relative position + velocity")
    print(f"   - Ball angle: atan2 direction (your favorite feature!)")
    print(f"   - Goals: Vectors to opponent goal and own goal")
    print(f"   - Walls: Raycast distances (north, south, east, west)")
    
    return True


if __name__ == "__main__":
    test_original_state_format() 