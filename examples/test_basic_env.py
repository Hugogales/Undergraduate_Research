#!/usr/bin/env python3
"""
Basic Environment Test

This script tests the core functionality of the soccer environment
and demonstrates basic usage patterns.
"""

import sys
from pathlib import Path

# ======== STANDARD IMPORT METHOD: sys.path manipulation ========
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env


def test_environment():
    """Test basic environment functionality."""
    print("=== Testing Soccer Environment ===\n")
    
    # Test environment creation
    print("1. Creating environment...")
    env_instance = env(render_mode=None, num_players_per_team=2)
    print(f"✓ Environment created with {len(env_instance.possible_agents)} possible agents")
    
    # Test environment reset
    print("\n2. Resetting environment...")
    observations, infos = env_instance.reset()
    print(f"✓ Environment reset - {len(observations)} active agents")
    
    # Test observation and action spaces
    print("\n3. Checking spaces...")
    for agent in env_instance.agents:
        obs_shape = env_instance.observation_space(agent).shape
        action_shape = env_instance.action_space(agent).shape if hasattr(env_instance.action_space(agent), 'shape') else env_instance.action_space(agent).n
        print(f"  Agent {agent}: obs_shape={obs_shape}, action_space={action_shape}")
    
    # Test a few steps
    print("\n4. Running 10 test steps...")
    for step in range(10):
        # Generate random actions
        actions = {}
        for agent in env_instance.agents:
            actions[agent] = env_instance.action_space(agent).sample()
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env_instance.step(actions)
        
        # Print step info
        total_reward = sum(rewards.values())
        active_agents = len([a for a in env_instance.agents])
        print(f"  Step {step+1}: {active_agents} agents, total_reward={total_reward:.3f}")
        
        # Check if episode ended
        if all(terminations.values()) or all(truncations.values()):
            print("  Episode ended early")
            break
    
    # Test closing
    print("\n5. Closing environment...")
    env_instance.close()
    print("✓ Environment closed successfully")
    
    return True


def test_different_team_sizes():
    """Test environment with different team sizes."""
    print("\n=== Testing Different Team Sizes ===\n")
    
    for team_size in [1, 2, 3]:
        print(f"Testing {team_size}v{team_size}...")
        
        env_instance = env(
            render_mode=None, 
            num_players_per_team=team_size,
            game_duration_seconds=10.0
        )
        
        observations, infos = env_instance.reset()
        expected_agents = team_size * 2
        
        print(f"  Expected agents: {expected_agents}, Actual: {len(observations)}")
        assert len(observations) == expected_agents, f"Agent count mismatch for {team_size}v{team_size}"
        
        env_instance.close()
        print(f"✓ {team_size}v{team_size} test passed")
    
    return True


def test_timing():
    """Test that timing is independent of team size."""
    print("\n=== Testing Timing Independence ===\n")
    
    game_duration = 10.0  # seconds
    
    for team_size in [1, 2, 3]:
        env_instance = env(
            render_mode=None,
            num_players_per_team=team_size,
            game_duration_seconds=game_duration
        )
        
        max_steps = env_instance.max_episode_steps
        expected_steps = int(game_duration * 42)  # Assuming 42 FPS
        
        print(f"{team_size}v{team_size}: max_steps={max_steps}, expected≈{expected_steps}")
        
        # Check that max_steps is reasonable and similar across team sizes
        assert abs(max_steps - expected_steps) < 50, f"Timing calculation seems off for {team_size}v{team_size}"
        
        env_instance.close()
    
    print("✓ Timing is independent of team size")
    return True


def run_all_tests():
    """Run all tests."""
    print("Soccer Environment Test Suite")
    print("=" * 40)
    
    try:
        # Test basic functionality
        if not test_environment():
            return False
        
        # Test different team sizes
        if not test_different_team_sizes():
            return False
        
        # Test timing
        if not test_timing():
            return False
        
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The soccer environment is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 