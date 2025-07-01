#!/usr/bin/env python3

"""
Basic Environment Tests

Tests the core functionality of the soccer environment.
"""

import sys
import unittest
from pathlib import Path

# ======== STANDARD IMPORT METHOD: sys.path manipulation ========
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env, raw_env


class TestSoccerEnvironment(unittest.TestCase):
    """Test suite for the soccer environment."""

    def test_env_creation(self):
        """Test basic environment creation."""
        test_env = env(render_mode=None, num_players_per_team=1)
        self.assertIsNotNone(test_env)
        self.assertEqual(len(test_env.possible_agents), 2)  # 1v1 = 2 agents
        test_env.close()

    def test_raw_env_creation(self):
        """Test raw environment creation."""
        test_env = raw_env(render_mode=None, num_players_per_team=1)
        self.assertIsNotNone(test_env)
        self.assertEqual(len(test_env.possible_agents), 2)  # 1v1 = 2 agents
        test_env.close()

    def test_env_reset(self):
        """Test environment reset functionality."""
        test_env = env(render_mode=None, num_players_per_team=1)
        
        observations, infos = test_env.reset()
        
        # Check that we get observations for all agents
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(infos), 2)
        
        # Check observation shapes
        for agent, obs in observations.items():
            expected_shape = test_env.observation_space(agent).shape
            self.assertEqual(obs.shape, expected_shape)
        
        test_env.close()

    def test_env_step(self):
        """Test environment step functionality."""
        test_env = env(render_mode=None, num_players_per_team=1)
        
        observations, infos = test_env.reset()
        
        # Create random actions for all agents
        actions = {}
        for agent in test_env.agents:
            actions[agent] = test_env.action_space(agent).sample()
        
        # Step the environment
        obs, rewards, terminations, truncations, infos = test_env.step(actions)
        
        # Check return types and shapes
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(rewards, dict)
        self.assertIsInstance(terminations, dict)
        self.assertIsInstance(truncations, dict)
        self.assertIsInstance(infos, dict)
        
        # Check that all agents are included
        for agent in test_env.agents:
            self.assertIn(agent, obs)
            self.assertIn(agent, rewards)
            self.assertIn(agent, terminations)
            self.assertIn(agent, truncations)
            self.assertIn(agent, infos)
        
        test_env.close()

    def test_multi_step_episode(self):
        """Test running multiple steps in an episode."""
        test_env = env(render_mode=None, num_players_per_team=1, game_duration_seconds=5.0)
        
        observations, infos = test_env.reset()
        
        steps_taken = 0
        max_steps = 100  # Safety limit
        
        while test_env.agents and steps_taken < max_steps:
            # Generate random actions
            actions = {}
            for agent in test_env.agents:
                actions[agent] = test_env.action_space(agent).sample()
            
            # Step environment
            obs, rewards, terminations, truncations, infos = test_env.step(actions)
            steps_taken += 1
            
            # Check if episode is done
            if all(terminations.values()) or all(truncations.values()):
                break
        
        self.assertGreater(steps_taken, 0)
        test_env.close()

    def test_variable_team_sizes(self):
        """Test different team sizes."""
        for team_size in [1, 2, 3]:
            with self.subTest(team_size=team_size):
                test_env = env(render_mode=None, num_players_per_team=team_size)
                
                expected_agents = team_size * 2
                self.assertEqual(len(test_env.possible_agents), expected_agents)
                
                observations, infos = test_env.reset()
                self.assertEqual(len(observations), expected_agents)
                
                test_env.close()


def run_all_tests():
    """Run all environment tests."""
    print("Running Soccer Environment Tests...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSoccerEnvironment)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 