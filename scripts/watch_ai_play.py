#!/usr/bin/env python3
"""
Watch Trained AI Play Soccer

This script loads a trained RLLib model and lets you watch the AI play.
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from scripts.train_rllib import SingleAgentSoccerWrapper, create_soccer_env_wrapper

def watch_ai_play(checkpoint_path, num_games=5, game_duration=30):
    """Watch the trained AI play soccer."""
    
    print("🎮 Loading trained AI model...")
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    # Initialize Ray with same settings as training
    ray.init(
        ignore_reinit_error=True, 
        include_dashboard=False,
        num_cpus=2,
        object_store_memory=500_000_000,
        log_to_driver=False,
        _system_config={
            "automatic_object_spilling_enabled": False,
            "max_direct_call_object_size": 1000,
            "task_retry_delay_ms": 1000,
        }
    )
    
    try:
        # Register environment
        register_env("soccer_env_rllib", create_soccer_env_wrapper)
        
        # Load the trained algorithm
        algo = PPO.from_checkpoint(checkpoint_path)
        print("✅ Model loaded successfully!")
        
        # Create environment with EXACT SAME CONFIG as training
        env_config = {
            "num_players": 1,           # Team size (1 = 1v1) - matches training 
            "game_duration": game_duration,  # Game duration in seconds - matches training
            "render_training": True,    # Enable rendering to watch
            "random_positions": True,   # Random positions - matches training
            "stage": 1                  # Stage 1 - matches training
        }
        
        env = SingleAgentSoccerWrapper(env_config)
        
        print(f"\n🏆 Watching AI play {num_games} games...")
        print("🎯 Controls: Close window to stop watching")
        print("=" * 50)
        
        total_rewards = []
        
        for game in range(num_games):
            print(f"\n🥅 Game {game + 1}/{num_games}")
            
            obs, info = env.reset()
            game_reward = 0
            step_count = 0
            
            while True:
                # Get AI action
                action = algo.compute_single_action(obs)
                
                # Debug: Print action occasionally to verify AI is working
                if step_count % 50 == 0:
                    print(f"   Step {step_count}: AI Action = {action}")
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                game_reward += reward
                step_count += 1
                
                # Small delay to make it watchable (but not too slow)
                time.sleep(0.02)  # Slightly slower for better visibility
                
                if terminated or truncated:
                    break
                    
                # Safety break for runaway episodes
                if step_count > 3000:  # About 2.5 minutes at 30fps
                    print(f"   ⚠️ Episode stopped at {step_count} steps (safety limit)")
                    break
            
            total_rewards.append(game_reward)
            print(f"🏆 Game {game + 1} finished!")
            print(f"   📊 Total Reward: {game_reward:.1f}")
            print(f"   ⏱️ Steps: {step_count}")
            
            # Wait a bit between games
            time.sleep(2)
        
        # Final statistics
        print("\n" + "=" * 50)
        print("📈 FINAL STATISTICS:")
        print(f"🎯 Games Played: {num_games}")
        print(f"📊 Average Reward: {np.mean(total_rewards):.1f}")
        print(f"🏆 Best Game: {max(total_rewards):.1f}")
        print(f"📉 Worst Game: {min(total_rewards):.1f}")
        
    except Exception as e:
        print(f"❌ Error watching AI: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

def main():
    """Main function to watch AI play."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch trained AI play soccer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games to watch (default: 5)")
    parser.add_argument("--duration", type=int, default=30,  # Changed default to match training
                        help="Game duration in seconds (default: 30)")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\n💡 Available checkpoints:")
        checkpoint_dir = Path("./rllib_checkpoints")
        if checkpoint_dir.exists():
            for cp in checkpoint_dir.iterdir():
                if cp.is_dir():
                    print(f"   📂 {cp}")
        else:
            print("   📂 No checkpoints found. Train a model first!")
        return
    
    watch_ai_play(args.checkpoint, args.games, args.duration)

if __name__ == "__main__":
    main() 