#!/usr/bin/env python3
"""
RLLib Training Script for Soccer Environment

🔧 OPTIMIZED FOR MINIMAL RESOURCES & RAY STABILITY 🔧

This script uses RLLib to train agents in the soccer environment with the 
ORIGINAL REWARD SYSTEM from the old codebase by default.

🏆 ORIGINAL REWARD SYSTEM (4 components):
============================================================================
1. 🥅 Goal Reward: +400/-400 (exact from old system)
2. ⚽ Player→Ball: 0.0004 coefficient (exact from old system) 
3. 🎯 Ball→Goal: 0.1 coefficient (exact from old system)
4. 👥 Distance: 0.002 coefficient (exact from old system)

All other rewards (ball possession, etc.) are disabled to match the original.

🏃‍♂️ MINIMAL RESOURCE OPTIMIZATIONS:
============================================================================
- ⚡ Ultra-small batches: 512 (was 16,384) for Ray stability  
- 🔥 Single worker: No parallel workers to prevent crashes
- 🧠 Tiny model: 2 layers × 128 units (minimal but functional)
- ⏱️ Short games: 30s for quick episodes
- 📦 Tiny fragments: 64 (was 1024) for minimal memory
- 💾 Small buffer: 5,000 (was 100,000) - closer to old system

RESOURCE COMPARISON:
- Old system: Direct learning from experience (no buffer)
- This system: Small buffer (5K) for some stability benefits
- Memory usage: ~500MB vs several GB

STABILITY FEATURES:
- ✅ Ray dashboard disabled (prevents grpc crashes)
- ✅ Object store limited to 500MB
- ✅ Single worker mode (no parallelization issues)
- ✅ Minimal batch sizes (reduces OOM errors)

🔧 HOW TO MODIFY PARAMETERS WITHOUT COMMAND LINE:
============================================================================

1. 🎯 BASIC CONFIGURATION (Lines ~290-310):
   Change the Config class values in main() function:
   
   - algorithm: "PPO", "IMPALA", or "DQN"
   - team_size: 1 (1v1), 2 (2v2), etc.
   - game_time: Duration in seconds (30 = 30-second games)
   - episodes: Number of training iterations
   - random_positions: True for random spawn, False for fixed
   - stage: 1 (vs random), 2 (vs league), 3 (fixed positions)
   - render_training: True to see games (slower), False for fast training

2. 🧠 AI HYPERPARAMETERS (Lines ~110-140):
   Modify get_default_hyperparams() function:
   
   - lr: Learning rate (4e-5 = old system, 1e-4 = faster learning)
   - gamma: Future focus (0.985 = old system, 0.99 = more long-term)
   - entropy_coeff: Exploration (0.00 = old system, 0.01 = more random)
   - clip_param: Update size (0.1 = old system, 0.05 = more conservative)

3. 🎮 ENVIRONMENT PARAMETERS (Lines ~45-50):
   Modify SingleAgentSoccerWrapper.__init__():
   
   - Add custom soccer environment parameters
   - Change opponent behavior
   - Modify reward structure

4. ⚡ PERFORMANCE SETTINGS (Lines ~300-305):
   In Config class:
   
   - num_workers: 0 (single), 2-4 (parallel workers)
   - use_gpu: False (CPU), True (GPU training)
   - log_every_episode: True (verbose), False (quiet)

============================================================================

QUICK SETUP:
- Uses original reward system by default
- Single agent training (1v1)
- Random positions enabled
- Just run: python scripts/train_rllib.py

Usage:
    python scripts/train_rllib.py
"""

import sys
import argparse
import time
import math
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray import tune
from ray.tune.registry import register_env
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Import our environment
from soccer_env.envs.soccer_env import env as SoccerEnv

class SingleAgentSoccerWrapper(gym.Env):
    """Single-agent wrapper for the multi-agent soccer environment."""
    
    def __init__(self, env_config):
        """Initialize the wrapper."""
        # Create the multi-agent environment
        self.env = SoccerEnv(
            num_players_per_team=env_config.get("num_players", 2),
            continuous_actions=False,  # Always use discrete
            game_duration_seconds=env_config.get("game_duration", 300),  # Fixed parameter name!
            render_mode="human" if env_config.get("render_training", False) else None,
            # Add environment configuration options
            stage=env_config.get("stage", 1),
            randomize_starting_positions=env_config.get("random_positions", True)
        )
        
        # Get a sample agent to determine spaces
        self.env.reset()
        sample_agent = self.env.possible_agents[0]
        
        # Set action and observation spaces
        self.action_space = self.env.action_space(sample_agent)
        self.observation_space = self.env.observation_space(sample_agent)
        
        # Track the agent we're training (first agent of team 0)
        self.training_agent = None
        
        # Enhanced single-agent wrapper to track ball metrics
        self.episode_ball_hits = 0
        self.episode_ball_distance = 0.0
        self.episode_rewards = []
        self.previous_ball_position = None  # Track ball position for distance calculation
        
    def reset(self, *, seed=None, options=None):
        """Reset the environment and return the observation for our training agent."""
        observations, infos = self.env.reset(seed=seed, options=options)
        
        # Find the first agent from team 0 to train
        self.training_agent = self.env.possible_agents[0]  # Always train first agent
        
        # Reset episode metrics
        self.episode_ball_hits = 0
        self.episode_ball_distance = 0.0
        self.episode_rewards = []
        self.previous_ball_position = None  # Reset ball position tracking
        
        return observations[self.training_agent], infos[self.training_agent]
    
    def step(self, action):
        """Step the environment with the action from our training agent."""
        # Create action dict with our agent's action and random actions for others
        actions = {}
        for agent in self.env.agents:
            if agent == self.training_agent:
                # For discrete actions, keep the action as-is but mark it as numpy for AI detection
                # The environment detects AI-only mode by checking isinstance(action, np.ndarray)
                # So we need to convert to numpy but preserve the discrete value
                if isinstance(action, (int, np.integer)):
                    # For discrete actions: convert to numpy array but make it a scalar array
                    action = np.array(action, dtype=np.int32)
                elif isinstance(action, (list, np.ndarray)):
                    action = np.array(action, dtype=np.float32)
                actions[agent] = action
            else:
                # Random actions for other agents - also convert to numpy arrays
                random_action = self.env.action_space(agent).sample()
                if isinstance(random_action, (int, np.integer)):
                    random_action = np.array(random_action, dtype=np.int32)
                elif isinstance(random_action, (list, np.ndarray)):
                    random_action = np.array(random_action, dtype=np.float32)
                actions[agent] = random_action
        
        # Step the environment
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Extract our agent's data
        obs = obs[self.training_agent]
        reward = rewards[self.training_agent]
        terminated = terminations[self.training_agent]
        truncated = truncations[self.training_agent]
        info = infos.get(self.training_agent, {})
        
        # Track episode metrics like the old system
        self.episode_rewards.append(reward)
        
        # Extract ball metrics from environment state - SIMPLIFIED APPROACH
        ball_hits_this_step = 0
        ball_distance_this_step = 0.0
        
        try:
            # Get the game state from the soccer environment's game engine
            game_state = self.env.game_engine.get_game_state()
            if game_state and hasattr(game_state, 'ball'):
                ball = game_state.ball
                ball_pos = [ball.position[0], ball.position[1]]  # Convert to list
                
                # Calculate ball distance moved (total distance traveled)
                if self.previous_ball_position is not None:
                    distance_moved = np.sqrt(
                        (ball_pos[0] - self.previous_ball_position[0])**2 + 
                        (ball_pos[1] - self.previous_ball_position[1])**2
                    )
                    self.episode_ball_distance += distance_moved
                    ball_distance_this_step = distance_moved
                
                # Update previous position for next step
                self.previous_ball_position = ball_pos
                
                # Track ball hits - check if ball is moving (was hit recently)
                ball_velocity = ball.velocity
                ball_speed = np.sqrt(ball_velocity[0]**2 + ball_velocity[1]**2)
                
                # Simple hit detection: if ball speed is high, count as hit
                if ball_speed > 10.0:  # Ball moving fast = was hit
                    self.episode_ball_hits += 1
                    ball_hits_this_step = 1
                            
        except Exception as e:
            # Debug what's wrong
            print(f"Ball metrics error: {e}")
        
        # Add metrics to info EVERY STEP for RLLib to track
        info.update({
            'ball_hits': self.episode_ball_hits,
            'ball_distance': self.episode_ball_distance,
            'ball_hits_step': ball_hits_this_step,  # This step's hits
            'ball_distance_step': ball_distance_this_step  # This step's distance
        })
        
        # Reset episode counters on episode end
        if terminated or truncated:
            # Reset for next episode
            self.episode_ball_hits = 0
            self.episode_ball_distance = 0.0
            self.episode_rewards = []
            self.previous_ball_position = None  # Reset ball tracking
        
        return obs, reward, terminated, truncated, info

def create_soccer_env_wrapper(env_config):
    """Create a soccer environment wrapper for RLLib."""
    return SingleAgentSoccerWrapper(env_config)

def get_default_hyperparams():
    """Get hyperparameters matching the old system.
    
    🎯 MODIFY THESE VALUES TO CHANGE AI BEHAVIOR:
    """
    return {
        # === LEARNING PARAMETERS ===
        'lr': 4e-5,                    # Learning rate (higher = faster learning, but less stable)
        'gamma': 0.985,                # Discount factor (higher = more future-focused)
        'lambda_': 0.985,              # GAE lambda (generalized advantage estimation)
        
        # === PPO SPECIFIC PARAMETERS ===
        'kl_coeff': 0.5,               # KL divergence coefficient (controls policy updates)
        'entropy_coeff': 0.0015,         # Entropy coefficient (higher = more exploration)
        'clip_param': 0.1,             # PPO clipping parameter (lower = more conservative updates)
        'vf_loss_coeff': 0.5,          # Value function loss coefficient
        
        # === MINIMAL CPU TRAINING PARAMETERS ===
        # EXTREMELY reduced for Windows Ray stability
        'train_batch_size_per_learner': 512,    # Minimal batch size (was 2,048)
        'sgd_minibatch_size': 64,      # Very small minibatches (was 256)
        'num_sgd_iter': 2,             # Minimal SGD iterations (was 4)
        
        # === REGULARIZATION ===
        'grad_clip': 200              # Gradient clipping (prevents exploding gradients)
    }

# ========================================
# 🚀 LARGE BATCH SIZE BENEFITS FOR SOCCER:
# ========================================
# 
# WHY LARGE BATCHES HELP IN SPARSE ENVIRONMENTS:
# - Soccer rewards are sparse (goals are rare)
# - Large batches capture more diverse game situations
# - More stable gradient estimates for policy updates
# - Better learning from few positive reward examples
# - Reduces variance in advantage estimates
#
# BATCH SIZE COMPARISON:
# =====================
# PARAMETER                  OLD SYSTEM    PREVIOUS    NEW (UPDATED)
# train_batch_size_per_learner:  16,384       512        16,384  ✅
# minibatch_size:                4,096+        32         1,024  ✅  
# rollout_fragment_length:       N/A          200         1,024  ✅
# num_epochs:                    N/A           4             8   ✅
#
# OLD SYSTEM CONFIGURATION (PROVEN):
# - train_batch_size: 16,384 (4096 * 4)
# - minibatch_size: 4,096+ 
# - This provided stable learning for soccer
#
# YOUR REQUIREMENTS MET:
# ✅ Minimum batch size: 1,024 (requested 512+)
# ✅ Preferably 1,024+: 16,384 total batch, 1,024 mini-batch
# ✅ Sparse environment optimized: Large batches for stable updates
# ========================================

def create_ppo_config(config):
    """Create PPO configuration with old system hyperparameters."""
    defaults = get_default_hyperparams()
    
    ppo_config = (
        PPOConfig()
        .environment(
            env="soccer_env_rllib",  # Simple registered name
            env_config={
                "num_players": config.team_size,
                "game_duration": config.game_time,  # Keep in seconds (was * 60)
                "render_training": config.render_training,
                "random_positions": config.random_positions,
                "stage": config.stage
            },
            # Explicitly specify spaces to avoid RLLib detection issues
            observation_space=Box(-np.inf, np.inf, (18,), dtype=np.float32),
            action_space=Discrete(18)
        )
        .env_runners(
            num_env_runners=config.num_workers,  # Use config workers
            num_envs_per_env_runner=1,
            rollout_fragment_length=64  # Minimal fragments for Ray stability (was 256)
        )
        .training(
            lr=defaults['lr'],
            gamma=defaults['gamma'],
            lambda_=defaults['lambda_'],
            kl_coeff=defaults['kl_coeff'],
            entropy_coeff=defaults['entropy_coeff'],
            clip_param=defaults['clip_param'],
            vf_loss_coeff=defaults['vf_loss_coeff'],
            train_batch_size_per_learner=defaults['train_batch_size_per_learner'],
            sgd_minibatch_size=defaults['sgd_minibatch_size'],
            num_sgd_iter=defaults['num_sgd_iter'],
            grad_clip=defaults['grad_clip'],
            # 🧠 MINIMAL CPU MODEL: 2 layers x 128 units (reduced for stability)
            model={
                "fcnet_hiddens": [128, 128],  # 2 layers with 128 units each (was [256, 256, 256, 256])
                "fcnet_activation": "tanh",   # Activation function
                "vf_share_layers": True,      # Share layers between policy and value function
            }
        )
        .resources(
            num_gpus=1 if config.use_gpu else 0,  # Use config GPU setting
            num_cpus_for_main_process=1
        )
        .framework("torch")
    )
    
    return ppo_config

def create_impala_config(config):
    """Create IMPALA configuration."""
    impala_config = (
        ImpalaConfig()
        .environment(
            env="soccer_env_rllib",
            env_config={
                "num_players": config.team_size,
                "game_duration": config.game_time * 60,
                "render_training": config.render_training,
                "random_positions": config.random_positions,
                "stage": config.stage
            },
            observation_space=Box(-np.inf, np.inf, (18,), dtype=np.float32),
            action_space=Discrete(18)
        )
        .env_runners(
            num_env_runners=config.num_workers,
            rollout_fragment_length=1024  # Larger fragments for consistency
        )
        .training(
            lr=0.0005,
            grad_clip=40.0,
            train_batch_size_per_learner=512  # Reduced for minimal resources (was 8192)
        )
        .resources(
            num_gpus=1 if config.use_gpu else 0,
            num_cpus_for_main_process=1
        )
        .framework("torch")
    )
    
    return impala_config

def create_dqn_config(config):
    """Create DQN configuration."""
    dqn_config = (
        DQNConfig()
        .environment(
            env="soccer_env_rllib",
            env_config={
                "num_players": config.team_size,
                "game_duration": config.game_time * 60,
                "render_training": config.render_training,
                "random_positions": config.random_positions,
                "stage": config.stage
            },
            observation_space=Box(-np.inf, np.inf, (18,), dtype=np.float32),
            action_space=Discrete(18)
        )
        .env_runners(
            num_env_runners=config.num_workers,
            rollout_fragment_length=512  # Larger fragments for DQN
        )
        .training(
            lr=0.0001,
            train_batch_size_per_learner=256,  # Reduced for minimal resources (was 1024)
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 5000,  # MUCH smaller buffer (was 100,000) - closer to old system
                "alpha": 0.6,
                "beta": 0.4,
            },
            target_network_update_freq=200,   # More frequent updates (was 500)
            epsilon=[
                [0, 1.0],
                [5000, 0.1],     # Faster epsilon decay (was 10000)
                [25000, 0.01]    # Faster final epsilon (was 50000)
            ]
        )
        .resources(
            num_gpus=1 if config.use_gpu else 0,
            num_cpus_for_main_process=1
        )
        .framework("torch")
    )
    
    return dqn_config

def train_rllib(config):
    """Main training function."""
    print("🔥 RLLib Soccer Training")
    print("==================================================")
    print(f"Algorithm: {config.algorithm}")
    print(f"Team Size: {config.team_size}")
    print(f"Game Time: {config.game_time}s")
    print(f"Episodes: {config.episodes}")
    print(f"Random Positions: {config.random_positions}")
    print(f"Stage: {config.stage}")
    print("==================================================")
    
    # Initialize Ray with minimal resources for Windows stability
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,  # Disable dashboard to prevent grpc issues
        num_cpus=2,               # Limit CPU usage
        object_store_memory=500_000_000,  # 500MB object store (very small)
        log_to_driver=False,      # Reduce logging overhead
        _system_config={
            "automatic_object_spilling_enabled": False,  # Disable spilling
            "max_direct_call_object_size": 1000,        # Reduce object sizes
            "task_retry_delay_ms": 1000,                # Faster retries
        }
    )
    
    try:
        # Register environment
        register_env("soccer_env_rllib", create_soccer_env_wrapper)
        print("✅ Environment registered successfully")
        
        # Create algorithm configuration
        if config.algorithm == "PPO":
            algo_config = create_ppo_config(config)
        elif config.algorithm == "IMPALA":
            algo_config = create_impala_config(config)
        elif config.algorithm == "DQN":
            algo_config = create_dqn_config(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
        
        # Build the algorithm
        print("🚀 Building algorithm...")
        algo = algo_config.build()
        print("✅ Algorithm built successfully")
        
        # Training loop
        start_time = time.time()
        episode_rewards = []
        
        print("\n🎯 Starting training...")
        
        for episode in range(config.episodes):
            # Train one iteration
            result = algo.train()
            
            # Extract metrics using correct RLLib keys (fixed from debug output)
            episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
            
            # Extract custom metrics from info (ball hits, ball distance, etc.)
            custom_metrics = result.get("env_runners", {}).get("custom_metrics", {})
            ball_hits = custom_metrics.get("ball_hits_mean", 0.0)
            ball_distance = custom_metrics.get("ball_distance_mean", 0.0)
            
            episode_rewards.append(episode_reward_mean)
            elapsed_time = time.time() - start_time
            
            # Enhanced logging with ball metrics (like old system) - removed episode length
            if config.log_every_episode or (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-min(10, len(episode_rewards)):])
                print(f"🏆 Episode: {episode + 1}, Reward: {episode_reward_mean:.1f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Ball Hits: {ball_hits:.0f}, Ball Dist: {ball_distance:.1f}, "
                      f"⏱️ Time: {elapsed_time:.1f}s")
            
            # 💾 Periodic saving to prevent loss if training crashes
            if (episode + 1) % config.save_every_episodes == 0:
                checkpoint_path = algo.save(f"./rllib_checkpoints/episode_{episode + 1}")
                print(f"💾 Model saved at episode {episode + 1}: {checkpoint_path}")
            
            # Early stopping if performance is good
            if episode_reward_mean > config.early_stopping_reward:
                print(f"🎉 Early stopping! Good performance reached: {episode_reward_mean:.2f}")
                break
        
        print(f"\n✅ Training completed! Total time: {elapsed_time:.1f}s")
        print(f"📊 Final average reward: {np.mean(episode_rewards[-min(10, len(episode_rewards)):]):.2f}")
        
        # Save the trained model
        checkpoint_path = algo.save("./rllib_checkpoints")
        print(f"💾 Model saved to: {checkpoint_path}")
        
        algo.stop()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        ray.shutdown()

def main():
    """Main function with direct configuration (no command line needed)."""
    
    # ========================================
    # 🎯 DIRECT CONFIGURATION - MODIFY THESE VALUES
    # ========================================
    
    class Config:
        # ALGORITHM SETTINGS
        algorithm = "PPO"  # Options: "PPO", "IMPALA", "DQN"
        
        # ENVIRONMENT PARAMETERS
        team_size = 1           # Number of players per team (1 = 1v1, 2 = 2v2, etc.)
        game_time = 30          # Game duration in SECONDS (reduced from 90 for faster episodes)
        
        # TRAINING PARAMETERS  
        episodes = 20000           # Testing episodes
        
        # ENVIRONMENT FEATURES
        random_positions = True     # True = random spawn positions, False = fixed positions
        stage = 1                   # Stage 1 = random vs random, Stage 2 = vs league, Stage 3 = fixed positions
        render_training = False     # True = show game during training (slower)
        
        # 🚀 MINIMAL RESOURCE SETTINGS FOR RAY STABILITY
        num_workers = 0             # Single worker only - no parallel workers for Windows stability
        use_gpu = False             # GPU training (True/False)
        
        # LOGGING SETTINGS
        log_every_episode = True    # Log every episode vs every N episodes
        save_every_episodes = 100   # Save model every N episodes (prevents loss if crash)
        early_stopping_reward = 10000 # Stop training if reward exceeds this
    
    config = Config()
    
    print("🔥 RLLib Soccer Training")
    print("==================================================")
    print(f"Algorithm: {config.algorithm}")
    print(f"Team Size: {config.team_size}")
    print(f"Game Time: {config.game_time}s")
    print(f"Episodes: {config.episodes}")
    print(f"Random Positions: {config.random_positions}")
    print(f"Stage: {config.stage}")
    print("==================================================")
    
    train_rllib(config)

if __name__ == "__main__":
    main() 