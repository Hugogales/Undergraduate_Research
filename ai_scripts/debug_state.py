#!/usr/bin/env python3
"""
Debug script to step through state building process.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from soccer_env.envs.soccer_env import SoccerEnv


def debug_observation_building():
    """Debug the observation building step by step."""
    print("🔍 Debugging Observation Building")
    print("=" * 40)
    
    # Create 1v1 environment
    env = SoccerEnv(num_players_per_team=1, game_duration_seconds=10.0)
    observations, info = env.reset()
    
    print(f"Number of agents: {len(observations)}")
    print(f"Agent names: {list(observations.keys())}")
    
    for agent_name, obs in observations.items():
        print(f"\n🎯 Agent: {agent_name}")
        print(f"Observation size: {len(obs)}")
        print(f"Full observation: {obs}")
        break  # Just debug the first agent


if __name__ == "__main__":
    debug_observation_building() 