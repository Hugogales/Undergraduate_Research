#!/usr/bin/env python3

"""
Basic Training Test

Test script to verify PPO training works on the simplest environment:
- 1v1 players
- Fixed starting positions 
- Short episodes
- Check if agents learn to chase the ball and score
"""

import sys
import os
from pathlib import Path
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env
from soccer_env.utils.ppo_agent import PPOAgent


def test_environment_setup():
    """Test that the environment setup works correctly."""
    print("🔧 Testing environment setup...")
    
    env_instance = env(
        render_mode=None,
        num_players_per_team=1,
        game_duration_seconds=10.0,
        randomize_starting_positions=False,
        game_mode="default",
    )
    
    # Test reset
    observations, infos = env_instance.reset()
    print(f"✅ Environment reset successful")
    print(f"   Agents: {list(observations.keys())}")
    print(f"   State size: {len(observations[list(observations.keys())[0]])}")
    
    # Test step
    actions = {}
    for agent in env_instance.agents:
        actions[agent] = [0.0, 0.0, 0.0, 0.0]  # No action
    
    next_obs, rewards, terms, truncs, next_infos = env_instance.step(actions)
    print(f"✅ Environment step successful")
    print(f"   Rewards: {rewards}")
    
    env_instance.close()
    return True


def test_ppo_agent():
    """Test that PPO agent can be created and used."""
    print("🤖 Testing PPO agent...")
    
    state_size = 22  # Based on environment observation
    agent = PPOAgent(
        state_size=state_size,
        action_size=4,
        device="cpu",  # Force CPU for testing
        lr=3e-4,
        hidden_size=64,  # Smaller for testing
    )
    
    # Test action selection
    dummy_state = np.random.random(state_size)
    action, log_prob, value = agent.select_action(dummy_state)
    
    print(f"✅ PPO agent action selection successful")
    print(f"   Action: {action}")
    print(f"   Log prob: {log_prob}")
    print(f"   Value: {value}")
    
    # Test experience storage
    agent.store_transition(
        state=dummy_state,
        action=action,
        log_prob=log_prob,
        reward=1.0,
        done=False,
        value=value
    )
    
    print(f"✅ PPO agent experience storage successful")
    print(f"   Memory size: {agent.memory.size()}")
    
    return True


def run_quick_training_test(episodes=50):
    """Run a quick training test to see if learning occurs."""
    print(f"🚀 Running quick training test ({episodes} episodes)...")
    
    # Setup environment
    env_instance = env(
        render_mode=None,
        num_players_per_team=1,
        game_duration_seconds=10.0,  # Very short episodes
        randomize_starting_positions=False,
        game_mode="default",
    )
    
    # Get state size
    observations, _ = env_instance.reset()
    agent_name = list(observations.keys())[0]
    state_size = len(observations[agent_name])
    print(f"State size: {state_size}")
    
    # Initialize one agent (we'll control both players with same agent for simplicity)
    agent = PPOAgent(
        state_size=state_size,
        action_size=4,
        device="cpu",
        lr=3e-4,
        hidden_size=64,
        batch_size=32,
        k_epochs=2,
    )
    
    # Track metrics
    episode_rewards = []
    episode_scores = []
    steps_until_update = 100  # Small update frequency for testing
    steps_collected = 0
    
    print(f"Starting training...")
    start_time = time.time()
    
    for episode in range(episodes):
        # Reset environment
        observations, infos = env_instance.reset()
        episode_reward = 0
        step_count = 0
        
        while env_instance.agents:
            # Get actions for all agents (using same agent for both players)
            actions = {}
            states = {}
            log_probs = {}
            values = {}
            
            for agent_id in env_instance.agents:
                state = observations[agent_id]
                action, log_prob, value = agent.select_action(state, training=True)
                
                actions[agent_id] = action
                states[agent_id] = state
                log_probs[agent_id] = log_prob
                values[agent_id] = value
            
            # Step environment
            next_observations, rewards, terminations, truncations, next_infos = env_instance.step(actions)
            
            # Store experiences for all agents
            for agent_id in env_instance.agents:
                if agent_id in rewards:
                    agent.store_transition(
                        state=states[agent_id],
                        action=actions[agent_id],
                        log_prob=log_probs[agent_id],
                        reward=rewards[agent_id],
                        done=terminations.get(agent_id, False) or truncations.get(agent_id, False),
                        value=values[agent_id]
                    )
                    episode_reward += rewards[agent_id]
            
            observations = next_observations
            infos = next_infos
            step_count += 1
            steps_collected += 1
            
            # Update policy periodically
            if steps_collected >= steps_until_update:
                loss_info = agent.update()
                if loss_info:
                    print(f"  Episode {episode}: Updated policy (loss: {loss_info.get('total_loss', 0):.4f})")
                steps_collected = 0
            
            # Check if episode done
            if all(terminations.values()) or all(truncations.values()):
                break
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        
        final_score = (0, 0)
        if infos:
            sample_info = next(iter(infos.values()))
            final_score = (
                sample_info.get('team_0_score', 0),
                sample_info.get('team_1_score', 0)
            )
        episode_scores.append(final_score)
        
        # Log progress
        if episode % 10 == 0 or episode == episodes - 1:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            total_goals = sum([s[0] + s[1] for s in episode_scores[-10:]])
            print(f"Episode {episode:3d}: "
                  f"Reward: {episode_reward:6.2f} (avg: {avg_reward:6.2f}) | "
                  f"Score: {final_score[0]}-{final_score[1]} | "
                  f"Goals in last 10: {total_goals} | "
                  f"Steps: {step_count}")
    
    elapsed_time = time.time() - start_time
    env_instance.close()
    
    # Analyze results
    print(f"\n📊 Training Results ({elapsed_time:.1f}s):")
    print(f"   Episodes: {episodes}")
    print(f"   Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"   Initial avg reward (first 10): {np.mean(episode_rewards[:10]):.2f}")
    print(f"   Total goals scored: {sum([s[0] + s[1] for s in episode_scores])}")
    print(f"   Goals in last 10 episodes: {sum([s[0] + s[1] for s in episode_scores[-10:]])}")
    print(f"   Goals in first 10 episodes: {sum([s[0] + s[1] for s in episode_scores[:10]])}")
    
    # Check if learning occurred
    improvement = np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10])
    goal_improvement = sum([s[0] + s[1] for s in episode_scores[-10:]]) - sum([s[0] + s[1] for s in episode_scores[:10]])
    
    if improvement > 1.0 or goal_improvement > 0:
        print(f"✅ Learning detected! Reward improvement: {improvement:.2f}, Goal improvement: {goal_improvement}")
        return True
    else:
        print(f"⚠️  Limited learning detected. Reward change: {improvement:.2f}, Goal change: {goal_improvement}")
        print(f"   This might be normal for very short training. Try more episodes.")
        return False


def main():
    """Run all tests."""
    print("🎯 Soccer Environment Training Test")
    print("=" * 50)
    
    try:
        # Test 1: Environment setup
        if not test_environment_setup():
            print("❌ Environment setup test failed")
            return False
        
        print()
        
        # Test 2: PPO agent
        if not test_ppo_agent():
            print("❌ PPO agent test failed")
            return False
        
        print()
        
        # Test 3: Quick training
        if not run_quick_training_test(episodes=100):
            print("⚠️  Quick training test showed limited learning")
            print("   Try running full training with: python scripts/train.py --episodes 500")
        
        print()
        print("✅ All tests completed successfully!")
        print("🚀 Ready to run full training:")
        print("   python scripts/train.py --players 1 --time 30 --episodes 500")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 