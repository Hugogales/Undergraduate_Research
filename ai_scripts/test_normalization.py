#!/usr/bin/env python3
"""
Test script to demonstrate reward normalization feature.
This shows how the new normalization system works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.config.training_config import TrainingSettings


def test_normalization():
    """Test the reward normalization feature."""
    
    print("🎯 Testing Reward Normalization System")
    print("=" * 50)
    
    # Test 1: Default configuration
    print("\n1️⃣ DEFAULT CONFIGURATION:")
    config = TrainingSettings()
    print(f"   Team size: {config.environment.num_players_per_team}v{config.environment.num_players_per_team}")
    print(f"   Game duration: {config.environment.game_duration_seconds}s")
    print(f"   FPS: {config.environment.fps}")
    print(f"   Goal reward: {config.reward.goal_reward}")
    print(f"   Normalization factor: {config.reward.normalization_factor:.1f}")
    print(f"   Normalized goal reward: {config.get_normalized_reward(config.reward.goal_reward):.3f}")
    
    # Test 2: Different game settings
    print("\n2️⃣ FAST 1v1 GAME (15 seconds):")
    config.environment.game_duration_seconds = 15.0
    config.environment.num_players_per_team = 1
    config._update_reward_normalization()
    print(f"   Team size: {config.environment.num_players_per_team}v{config.environment.num_players_per_team}")
    print(f"   Game duration: {config.environment.game_duration_seconds}s")
    print(f"   Normalization factor: {config.reward.normalization_factor:.1f}")
    print(f"   Normalized goal reward: {config.get_normalized_reward(config.reward.goal_reward):.3f}")
    
    # Test 3: Bigger team, longer game
    print("\n3️⃣ BIG 3v3 GAME (120 seconds):")
    config.environment.game_duration_seconds = 120.0
    config.environment.num_players_per_team = 3
    config._update_reward_normalization()
    print(f"   Team size: {config.environment.num_players_per_team}v{config.environment.num_players_per_team}")
    print(f"   Game duration: {config.environment.game_duration_seconds}s")
    print(f"   Normalization factor: {config.reward.normalization_factor:.1f}")
    print(f"   Normalized goal reward: {config.get_normalized_reward(config.reward.goal_reward):.3f}")
    
    # Test 4: Old system preset
    print("\n4️⃣ OLD SYSTEM PRESET:")
    config.apply_preset('old_system')
    print(f"   Goal reward: {config.reward.goal_reward}")
    print(f"   Ball to goal coeff: {config.reward.ball_to_goal_coeff}")
    print(f"   Player to ball coeff: {config.reward.player_to_ball_coeff}")
    print(f"   Normalization factor: {config.reward.normalization_factor:.1f}")
    print(f"   Normalized goal reward: {config.get_normalized_reward(config.reward.goal_reward):.3f}")
    
    # Test 5: Disable normalization
    print("\n5️⃣ NO NORMALIZATION:")
    config.reward.normalize_rewards = False
    print(f"   Normalization enabled: {config.reward.normalize_rewards}")
    print(f"   Raw goal reward: {config.reward.goal_reward}")
    print(f"   'Normalized' goal reward: {config.get_normalized_reward(config.reward.goal_reward):.3f}")
    
    print("\n✅ All tests completed!")
    print("\n📊 KEY INSIGHTS:")
    print("   - Normalization keeps rewards in manageable range")
    print("   - Factor auto-adjusts based on game length and team size")
    print("   - Old system values work with new normalization")
    print("   - Can be easily disabled if needed")


if __name__ == "__main__":
    test_normalization() 