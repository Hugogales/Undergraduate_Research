#!/usr/bin/env python3
"""
Single Agent Training Script for Soccer Environment

This script trains a single PPO agent to play soccer against a random opponent.
Perfect for getting started with RL training in the soccer environment.

Usage:
    python scripts/train_single_agent.py
    python scripts/train_single_agent.py --episodes 3000 --save-every 500
"""

import sys
import argparse
import time
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env
from soccer_env.utils.ppo_agent import PPOAgent
from soccer_env.utils.reward_calculator import RewardCalculator
from soccer_env.config.training_config import TrainingSettings


def action_to_input(action: int) -> list:
    """
    Convert discrete action (0-17) to movement input [up, down, left, right, shoot].
    This mapping is EXACTLY from the old system to maintain compatibility.
    """
    action_mapping = {
        0: [1, 0, 0, 0, 0],   # Up - dribble
        1: [0, 1, 0, 0, 0],   # Down - dribble
        2: [0, 0, 1, 0, 0],   # Left - dribble
        3: [0, 0, 0, 1, 0],   # Right - dribble
        4: [1, 0, 1, 0, 0],   # Up-Left - dribble
        5: [1, 0, 0, 1, 0],   # Up-Right - dribble
        6: [0, 1, 1, 0, 0],   # Down-Left - dribble
        7: [0, 1, 0, 1, 0],   # Down-Right - dribble
        8: [0, 0, 0, 0, 0],   # No movement - dribble
        9: [1, 0, 0, 0, 1],   # Up - shoot
        10: [0, 1, 0, 0, 1],  # Down - shoot
        11: [0, 0, 1, 0, 1],  # Left - shoot
        12: [0, 0, 0, 1, 1],  # Right - shoot
        13: [1, 0, 1, 0, 1],  # Up-Left - shoot
        14: [1, 0, 0, 1, 1],  # Up-Right - shoot
        15: [0, 1, 1, 0, 1],  # Down-Left - shoot
        16: [0, 1, 0, 1, 1],  # Down-Right - shoot
        17: [0, 0, 0, 0, 1]   # No movement - shoot
    }
    return action_mapping.get(action, [0, 0, 0, 0, 0])  # Default to no action


def train_single_agent(
    total_episodes: int = 2000,
    save_frequency: int = 200,
    log_frequency: int = 50,
    config: TrainingSettings = None
):
    """
    Train a single PPO agent against random opponents.
    
    Args:
        total_episodes: Number of episodes to train
        save_frequency: Save model every N episodes
        log_frequency: Print progress every N episodes  
        config: Training configuration (uses defaults if None)
    """
    
    # Use provided config or create default
    if config is None:
        config = TrainingSettings()
        # Apply single agent preset - optimized for 1v1 training
        config.apply_preset('single_agent')
        print("🎯 Using 'single_agent' preset for optimized 1v1 training")
    
    print("🤖 Single Agent PPO Training")
    print("=" * 50)
    print(f"🎮 Environment: {config.environment.num_players_per_team}v{config.environment.num_players_per_team}")
    print(f"⏱️  Game duration: {config.environment.game_duration_seconds}s") 
    print(f"🎯 Total episodes: {total_episodes}")
    print(f"🧠 Learning rate: {config.ppo.learning_rate}")
    print(f"🥅 Goal reward: {config.reward.goal_reward}")
    print(f"📊 Dense rewards: {config.reward.dense_rewards}")
    print(f"🎚️  Normalize rewards: {config.reward.normalize_rewards}")
    if config.reward.normalize_rewards:
        print(f"📐 Normalization factor: {config.reward.normalization_factor:.1f}")
    print()
    
    # Create directories
    model_dir = Path("files/Models/single_agent")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("files/Logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env_instance = env(
        render_mode=None,  # No rendering for training
        num_players_per_team=config.environment.num_players_per_team,
        game_duration_seconds=config.environment.game_duration_seconds,
        randomize_starting_positions=config.environment.randomize_starting_positions,
        game_mode=config.environment.game_mode,
        continuous_actions=not config.environment.use_discrete_actions,  # Opposite of use_discrete_actions
    )
    
    # Set up reward calculator
    reward_calculator = RewardCalculator(
        goal_reward=config.reward.goal_reward,
        goal_conceded_penalty=config.reward.goal_conceded_penalty,
        player_to_ball_reward_coeff=config.reward.player_to_ball_coeff,
        ball_to_goal_reward_coeff=config.reward.ball_to_goal_coeff,
        ball_possession_reward=config.reward.ball_possession_reward,
        distance_to_teammates_coeff=config.reward.distance_to_teammates_coeff,
        dense_rewards=config.reward.dense_rewards,
    )
    env_instance.reward_calculator = reward_calculator
    
    # Get state size from environment
    observations, _ = env_instance.reset()
    first_agent = list(observations.keys())[0]
    state_size = len(observations[first_agent])
    print(f"🧩 State size: {state_size}")
    
    # Initialize our trained agent (we'll train the first agent)
    trained_agent_name = env_instance.possible_agents[0]  # Usually "player_0"
    
    agent = PPOAgent(
        state_size=state_size,
        action_size=config.environment.action_size,  # Use config action size (18 for old system)
        device=config.ppo.device,
        lr=config.ppo.learning_rate,
        gamma=config.ppo.gamma,
        lambda_=config.ppo.lambda_,
        epsilon_clip=config.ppo.epsilon_clip,
        k_epochs=config.ppo.k_epochs,
        batch_size=config.ppo.batch_size,
        vf_coef=config.ppo.value_function_coeff,
        ent_coef=config.ppo.entropy_coeff,
        max_grad_norm=config.ppo.max_grad_norm,
        hidden_size=config.ppo.hidden_size,
        discrete_actions=config.environment.use_discrete_actions,  # Use discrete actions like old system
    )
    
    print(f"🤖 Training agent: {trained_agent_name}")
    print(f"🎲 Other agents will use random actions")
    print()
    
    # Training metrics
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    best_avg_reward = float('-inf')
    
    # Training loop
    print("🚀 Starting training...")
    start_time = time.time()
    
    for episode in range(total_episodes):
        # Reset environment
        observations, infos = env_instance.reset()
        
        episode_reward = 0.0
        step_count = 0
        
        # Episode loop
        while env_instance.agents:
            actions = {}
            
            # Get action for all agents
            for agent_name in env_instance.agents:
                if agent_name == trained_agent_name:
                    # Our trained agent
                    obs = observations[agent_name]
                    action, log_prob, value = agent.select_action(obs, training=True)
                    
                    # For discrete actions, pass the action index directly to the environment
                    # The environment's action parser will handle the conversion internally
                    if config.environment.use_discrete_actions:
                        action_idx = action.item() if hasattr(action, 'item') else int(action)
                        # Ensure action is within valid range for environment (0-17 for 18 actions)
                        action_idx = np.clip(action_idx, 0, 17)
                        actions[agent_name] = action_idx
                    else:
                        actions[agent_name] = action
                    
                    # Store transition (store the original action)
                    agent.memory.states.append(obs)
                    agent.memory.actions.append(action)
                    agent.memory.log_probs.append(log_prob)
                    agent.memory.values.append(value)
                else:
                    # Random action for opponent
                    if config.environment.use_discrete_actions:
                        # Generate random action in environment's action space (0-17 for 18 actions)
                        actions[agent_name] = np.random.randint(0, 18)
                    else:
                        actions[agent_name] = np.random.uniform(-1, 1, config.environment.action_size)
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env_instance.step(actions)
            
            # Process rewards for our agent
            if trained_agent_name in rewards:
                reward = rewards[trained_agent_name]
                
                # Apply normalization if enabled
                if config.reward.normalize_rewards:
                    reward = config.get_normalized_reward(reward)
                
                episode_reward += reward
                
                # Store reward and done flag
                done = terminations.get(trained_agent_name, False) or truncations.get(trained_agent_name, False)
                agent.memory.rewards.append(reward)
                agent.memory.dones.append(done)
            
            step_count += 1
        
        # Update agent if enough experience collected
        if agent.memory.size() >= config.ppo.batch_size:
            loss_info = agent.update()
        else:
            loss_info = {}
        
        # Log episode results
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        # Get final scores
        final_info = infos.get(trained_agent_name, {})
        team_0_score = final_info.get('team_0_score', 0)
        team_1_score = final_info.get('team_1_score', 0)
        episode_scores.append((team_0_score, team_1_score))
        episode_lengths.append(step_count)
        
        # Progress logging
        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-log_frequency:])
            elapsed_time = time.time() - start_time
            
            # Extract ball statistics and scores from recent episodes
            recent_scores = episode_scores[-log_frequency:]
            if recent_scores:
                team_0_scores = [score[0] for score in recent_scores]
                team_1_scores = [score[1] for score in recent_scores]
                avg_team_0_score = np.mean(team_0_scores)
                avg_team_1_score = np.mean(team_1_scores)
            else:
                avg_team_0_score = 0
                avg_team_1_score = 0
            
            # Calculate approximate ball distance and hits (simulated for now)
            ball_dist = max(0, 10 - avg_reward * 0.5)  # Rough approximation
            ball_hits = max(0, int(avg_length / 50))    # Rough approximation
            
            # Calculate entropy percentage (rough approximation)
            entropy_percent = min(95, max(5, 15 - episode / 100))  # Decreases over time
            
            # Old-style logging with emojis (matching the format you saw in old system)
            print(f"🏆 Episode: {episode + 1}, "
                  f"Score: {avg_team_0_score:.1f} - {avg_team_1_score:.1f}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Ball dist: {ball_dist:.2f}, "
                  f"Ball hits: {ball_hits}, "
                  f"entropy: {entropy_percent:.2f}%, "
                  f"⏱️ Time: {elapsed_time:.1f}s")
            
            # Update best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(str(model_dir / "best_model.pth"))
                print(f"  💾 New best model saved! (avg reward: {avg_reward:.2f})")
        
        # Save model periodically
        if (episode + 1) % save_frequency == 0:
            model_path = model_dir / f"model_episode_{episode + 1}.pth"
            agent.save(str(model_path))
            print(f"  💾 Model saved: {model_path}")
    
    # Final save
    final_model_path = model_dir / "final_model.pth"
    agent.save(str(final_model_path))
    
    # Training summary
    total_time = time.time() - start_time
    print(f"\n🎉 Training Complete!")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"🏆 Best average reward: {best_avg_reward:.2f}")
    print(f"💾 Final model saved: {final_model_path}")
    
    # Plot training curves
    plot_training_results(episode_rewards, episode_scores, episode_lengths, log_dir)
    
    return agent


def plot_training_results(rewards, scores, lengths, save_dir):
    """Plot and save training results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average of rewards
    if len(rewards) > 20:
        window = min(50, len(rewards) // 4)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[1, 0].plot(lengths)
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scores
    if scores:
        team_0_scores = [score[0] for score in scores]
        team_1_scores = [score[1] for score in scores]
        axes[1, 1].plot(team_0_scores, label='Team 0 (Our Agent)', alpha=0.7)
        axes[1, 1].plot(team_1_scores, label='Team 1 (Random)', alpha=0.7)
        axes[1, 1].set_title('Scores per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "training_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved: {plot_path}")
    plt.close()


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train a single PPO agent in soccer environment")
    
    parser.add_argument("--episodes", type=int, default=2000,
                       help="Number of episodes to train (default: 2000)")
    parser.add_argument("--save-every", type=int, default=200,
                       help="Save model every N episodes (default: 200)")
    parser.add_argument("--log-every", type=int, default=50,
                       help="Print progress every N episodes (default: 50)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate for PPO (default: 3e-4)")
    parser.add_argument("--game-time", type=float, default=30.0,
                       help="Game duration in seconds (default: 30.0)")
    parser.add_argument("--no-dense", action="store_true",
                       help="Disable dense rewards (use only goal rewards)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable reward normalization")
    
    args = parser.parse_args()
    
    # Create custom config based on arguments
    config = TrainingSettings()
    config.environment.num_players_per_team = 1  # Single agent = 1v1
    config.environment.game_duration_seconds = args.game_time
    config.environment.randomize_starting_positions = True
    config.ppo.learning_rate = args.learning_rate
    config.reward.dense_rewards = not args.no_dense
    config.reward.normalize_rewards = not args.no_normalize
    
    # Start training
    train_single_agent(
        total_episodes=args.episodes,
        save_frequency=args.save_every,
        log_frequency=args.log_every,
        config=config
    )


if __name__ == "__main__":
    main() 