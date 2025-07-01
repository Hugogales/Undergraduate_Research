"""
Training Configuration for Soccer Environment

This file contains all configurable parameters for training.
Users can easily modify these values to experiment with different settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """PPO Algorithm Hyperparameters - Matching Old System"""
    # Learning parameters (from old AIHyperparameters)
    learning_rate: float = 4e-5             # Old: 4e-5
    gamma: float = 0.985                    # Old: 0.985 (discount rate)
    lambda_: float = 0.985                  # Old: 0.985 (GAE lambda, was called 'lam')
    epsilon_clip: float = 0.15              # Old: 0.15
    
    # Training parameters (from old AIHyperparameters)
    k_epochs: int = 25                      # Old: 25 (was called 'K_epochs')
    batch_size: int = 16384                 # Old: 4096 * 4 = 16384
    update_frequency: int = 16384           # Steps before policy update
    
    # Loss coefficients (from old AIHyperparameters)
    value_function_coeff: float = 1.0       # Old: 1 (was called 'c_value')
    entropy_coeff: float = 0.0015           # Old: 0.0015 (was called 'c_entropy')
    max_grad_norm: float = 2000.0           # Old: 2000
    
    # Network architecture
    hidden_size: int = 256                  # Reasonable default (not specified in old system)
    
    # Device
    device: str = "auto"                    # "auto", "cpu", or "cuda"


@dataclass
class RewardConfig:
    """Reward Function Parameters - Matching Old System Exactly"""
    # Goal rewards (from old AIHyperparameters)
    goal_reward: float = 400.0              # Old: 400 (GOAL_REWARD)
    goal_conceded_penalty: float = -400.0   # Old: -400 (negative of GOAL_REWARD)
    
    # Dense reward coefficients (from old AIHyperparameters)
    player_to_ball_coeff: float = 0.0004    # Old: 0.0004 (PLAYER_TO_BALL_REWARD_COEFF)
    ball_to_goal_coeff: float = 0.1         # Old: 0.1 (BALL_TO_GOAL_REWARD_COEFF)
    ball_possession_reward: float = 0.0     # Not in old system, keep minimal
    distance_to_teammates_coeff: float = 0.002 # Old: 0.002 (DISTANCE_REWARD_COEFF)
    
    # Settings
    dense_rewards: bool = True              # Use dense rewards (True) or only goal rewards (False)
    normalize_distances: bool = True        # Normalize distance-based rewards
    
    # Reward Normalization (like the old system)
    normalize_rewards: bool = True          # Normalize all rewards to keep them closer to -1 to 1
    normalization_factor: float = 100.0    # Divider for reward normalization (auto-calculated if None)
    auto_calculate_normalization: bool = True # Auto-calculate normalization based on game params


@dataclass
class EnvironmentConfig:
    """Environment Settings - Matching Old System"""
    # Basic setup (from old EnvironmentHyperparameters)
    num_players_per_team: int = 3           # Old: NUMBER_OF_PLAYERS = 3
    game_duration_seconds: float = 120.0    # Old: GAME_DURATION = 120
    randomize_starting_positions: bool = False # Old: RANDOMIZE_PLAYERS = False
    game_mode: str = "default"              # Game mode preset
    fps: int = 42                           # Old: FPS = 42
    
    # Action space (from old AIHyperparameters) 
    use_discrete_actions: bool = True       # Old system used discrete actions
    action_size: int = 18                   # Old: ACTION_SIZE = 18 (now matches exactly)
    continuous_actions: bool = False        # Old system used discrete actions
    
    # Episode termination
    terminate_on_goal: bool = True          # End episode when goal is scored (recommended)
    max_episode_steps: Optional[int] = None # Max steps (calculated from duration)
    
    # Rendering
    render_during_training: bool = False    # Old: RENDER = False (for training)
    render_frequency: int = 100             # Render every N episodes if enabled


@dataclass
class TrainingConfig:
    """Main Training Configuration"""
    # Training parameters
    total_episodes: int = 1000              # Total episodes to train
    save_frequency: int = 100               # Save model every N episodes
    log_frequency: int = 10                 # Log progress every N episodes
    eval_frequency: int = 50                # Evaluate model every N episodes
    
    # Paths
    model_save_dir: str = "files/Models"    # Model save directory
    log_dir: str = "files/Logs"            # Log directory
    
    # Early stopping
    early_stopping_patience: int = 200     # Stop if no improvement for N episodes
    target_reward_threshold: float = 50.0   # Stop if average reward exceeds this
    
    # Evaluation
    eval_episodes: int = 25                 # Episodes for evaluation
    
    # Multi-agent settings
    shared_agent: bool = True              # Use same agent for all players
    separate_agents: bool = False            # Train separate agents


@dataclass
class CurriculumConfig:
    """Curriculum Learning Settings"""
    # Stage progression
    enable_curriculum: bool = False         # Enable curriculum learning
    
    # Stage 1: Fixed positions, random opponent
    stage1_episodes: int = 300
    stage1_randomize_positions: bool = False
    stage1_opponent_type: str = "random"
    
    # Stage 2: Random positions, league opponent
    stage2_episodes: int = 500
    stage2_randomize_positions: bool = True
    stage2_opponent_type: str = "league"
    
    # Stage 3: Tournament mode
    stage3_episodes: int = 200
    stage3_randomize_positions: bool = False
    stage3_opponent_type: str = "League"


class TrainingSettings:
    """
    Complete Training Settings - Based on Old System Defaults
    
    QUICK TUNING GUIDE:
    ===================
    
    🎯 For Faster Learning:
    - Increase goal_reward (400 → 600)
    - Decrease game_duration_seconds (120 → 60)
    - Increase learning_rate (4e-5 → 1e-4)
    
    ⚽ For Better Ball-Chasing:
    - Increase player_to_ball_coeff (0.0004 → 0.001)
    - Increase ball_possession_reward (0.0 → 0.2)
    
    🥅 For Better Goal-Scoring:
    - Increase ball_to_goal_coeff (0.1 → 0.2)
    - Increase ball_possession_reward (0.0 → 0.5)
    
    🤖 For More Exploration:
    - Increase entropy_coeff (0.0015 → 0.005)
    - Decrease epsilon_clip (0.15 → 0.1)
    
    📊 For Stabler Training:
    - Increase batch_size (16384 → 32768)
    - Decrease learning_rate (4e-5 → 2e-5)
    - Increase k_epochs (25 → 40)
    
    🔄 Old System vs New:
    - Default config now matches the tested old system exactly
    - Use apply_preset('fast_learning') for quicker training
    - Use apply_preset('single_agent') for 1v1 training
    """
    
    def __init__(self):
        # Initialize all config sections
        self.ppo = PPOConfig()
        self.reward = RewardConfig()
        self.environment = EnvironmentConfig()
        self.training = TrainingConfig()
        self.curriculum = CurriculumConfig()
        
        # Apply old system preset as default (this is now the base)
        # No need to apply preset since defaults are already set to old system values
        
        # Calculate reward normalization factor if needed
        self._update_reward_normalization()
    
    def _update_reward_normalization(self):
        """Update reward normalization factor based on environment settings."""
        if self.reward.auto_calculate_normalization and self.reward.normalize_rewards:
            # Calculate normalization factor like the old system:
            # normalization = (num_players * 2 * fps * game_duration) / 100
            total_players = self.environment.num_players_per_team * 2
            total_steps = self.environment.fps * self.environment.game_duration_seconds
            self.reward.normalization_factor = (total_players * total_steps) / 100.0
            
            print(f"🔧 Auto-calculated reward normalization factor: {self.reward.normalization_factor:.1f}")
            print(f"   This means rewards will be divided by {self.reward.normalization_factor:.1f} to keep them in a reasonable range")
    
    def get_normalized_reward(self, raw_reward: float) -> float:
        """Apply reward normalization if enabled."""
        if self.reward.normalize_rewards:
            return raw_reward / self.reward.normalization_factor
        return raw_reward
    
    def get_quick_presets(self) -> dict:
        """Get predefined configuration presets for common scenarios."""
        return {
            "fast_learning": {
                "description": "Fast learning with higher rewards and shorter episodes",
                "changes": {
                    "reward.goal_reward": 600.0,
                    "reward.player_to_ball_coeff": 0.001,
                    "environment.game_duration_seconds": 60.0,
                    "ppo.learning_rate": 1e-4,
                }
            },
            "single_agent": {
                "description": "Optimized for single agent training (1v1)",
                "changes": {
                    "environment.num_players_per_team": 1,
                    "environment.game_duration_seconds": 30.0,
                    "environment.randomize_starting_positions": True,
                    "reward.goal_reward": 200.0,
                    "reward.player_to_ball_coeff": 0.001,
                    "ppo.learning_rate": 3e-4,
                }
            },
            "stable_training": {
                "description": "More stable training with conservative settings",
                "changes": {
                    "ppo.learning_rate": 2e-5,
                    "ppo.batch_size": 32768,
                    "ppo.k_epochs": 40,
                    "ppo.epsilon_clip": 0.1,
                }
            },
            "ball_chaser": {
                "description": "Encourage aggressive ball-chasing behavior",
                "changes": {
                    "reward.player_to_ball_coeff": 0.002,
                    "reward.ball_possession_reward": 0.5,
                }
            },
            "goal_scorer": {
                "description": "Focus on goal-scoring behavior",
                "changes": {
                    "reward.ball_to_goal_coeff": 0.3,
                    "reward.goal_reward": 600.0,
                    "reward.ball_possession_reward": 0.8,
                }
            },
            "exploration": {
                "description": "Encourage exploration and diverse strategies",
                "changes": {
                    "ppo.entropy_coeff": 0.05,
                    "ppo.epsilon_clip": 0.3,
                    "reward.distance_to_teammates_coeff": 0.005,
                }
            },
            "original_defaults": {
                "description": "The exact old system parameters (same as default now)",
                "changes": {
                    # These are now the defaults, but listing for reference
                    "reward.goal_reward": 400.0,
                    "reward.goal_conceded_penalty": -400.0,
                    "reward.player_to_ball_coeff": 0.0004,
                    "reward.ball_to_goal_coeff": 0.1,
                    "reward.distance_to_teammates_coeff": 0.002,
                    "ppo.learning_rate": 4e-5,
                    "ppo.gamma": 0.985,
                    "ppo.lambda_": 0.985,
                    "ppo.epsilon_clip": 0.15,
                    "ppo.k_epochs": 25,
                    "ppo.batch_size": 16384,
                    "environment.num_players_per_team": 3,
                    "environment.game_duration_seconds": 120.0,
                    "environment.fps": 42,
                }
            }
        }
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        presets = self.get_quick_presets()
        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        preset = presets[preset_name]
        print(f"🎯 Applying preset: {preset_name}")
        print(f"   Description: {preset['description']}")
        
        for key, value in preset["changes"].items():
            # Parse nested key (e.g., "reward.goal_reward")
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            print(f"   Set {key} = {value}")
        
        # Recalculate normalization after changes
        self._update_reward_normalization()


# Create default instance
default_config = TrainingSettings()


# Quick access functions for users
def get_reward_config() -> RewardConfig:
    """Get reward configuration for easy modification."""
    return default_config.reward


def get_ppo_config() -> PPOConfig:
    """Get PPO configuration for easy modification."""
    return default_config.ppo


def get_training_config() -> TrainingConfig:
    """Get training configuration for easy modification.""" 
    return default_config.training


def print_config_summary():
    """Print a summary of current configuration."""
    config = default_config
    
    print("🔧 Current Training Configuration")
    print("=" * 50)
    print(f"PPO Settings:")
    print(f"  Learning Rate: {config.ppo.learning_rate}")
    print(f"  Batch Size: {config.ppo.batch_size}")
    print(f"  Update Frequency: {config.ppo.update_frequency}")
    print()
    print(f"Reward Settings:")
    print(f"  Goal Reward: {config.reward.goal_reward}")
    print(f"  Ball Chasing: {config.reward.player_to_ball_coeff}")
    print(f"  Goal Direction: {config.reward.ball_to_goal_coeff}")
    print(f"  Ball Possession: {config.reward.ball_possession_reward}")
    print(f"  Normalize Rewards: {config.reward.normalize_rewards}")
    if config.reward.normalize_rewards:
        print(f"  Normalization Factor: {config.reward.normalization_factor:.1f}")
    print()
    print(f"Environment:")
    print(f"  Team Size: {config.environment.num_players_per_team}v{config.environment.num_players_per_team}")
    print(f"  Game Duration: {config.environment.game_duration_seconds}s")
    print(f"  Random Positions: {config.environment.randomize_starting_positions}")
    print(f"  FPS: {config.environment.fps}")
    print()
    print(f"Training:")
    print(f"  Total Episodes: {config.training.total_episodes}")
    print(f"  Save Every: {config.training.save_frequency} episodes")
    print("=" * 50) 