#!/usr/bin/env python3

"""
Game Modes Demo

This script demonstrates all available game mode presets in the soccer environment.
Each mode showcases different physics, player sizes, and gameplay styles.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env
from soccer_env.envs.constants import GameModePresets


def demo_game_mode(mode_name, duration=15, num_players=2):
    """
    Demo a specific game mode.
    
    Args:
        mode_name: Name of the game mode preset
        duration: Demo duration in seconds
        num_players: Number of players per team
    """
    
    print(f"\n🎮 === {mode_name.upper()} MODE ===")
    print(f"📖 {GameModePresets.describe_preset(mode_name)}")
    
    # Show preset configuration
    preset = GameModePresets.get_preset(mode_name)
    print("⚙️  Configuration:")
    for key, value in preset.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 Starting {duration}s demo with {num_players}v{num_players} players...")
    print("Press ESC to skip to next mode")
    
    # Create environment with the game mode
    env_instance = env(
        render_mode="human",
        num_players_per_team=num_players,
        game_duration_seconds=duration,
        randomize_starting_positions=True,
        game_mode=mode_name,
    )
    
    # Reset environment
    observations, infos = env_instance.reset()
    
    try:
        # Let the demo run with random actions
        while env_instance.agents:
            # Random actions for demo
            import numpy as np
            actions = {
                agent: np.random.uniform(-0.3, 0.3, 4) 
                for agent in env_instance.agents
            }
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env_instance.step(actions)
            
            # Check if game ended or user quit
            if all(terminations.values()) or all(truncations.values()):
                break
                
            if any(info.get('quit', False) for info in infos.values()):
                print("⏭️  Skipping to next mode...")
                break
            
            # Small delay for smooth demo
            time.sleep(0.016)  # ~60 FPS
                
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    finally:
        env_instance.close()
    
    print(f"✅ {mode_name} demo completed!")


def interactive_mode_selection():
    """Let user select which modes to demo."""
    
    print("🎮 === SOCCER GAME MODES DEMO ===")
    print("\nAvailable Game Modes:")
    
    presets = GameModePresets.list_presets()
    for i, preset in enumerate(presets, 1):
        description = GameModePresets.describe_preset(preset)
        print(f"  {i:2d}. {preset:15s} - {description}")
    
    print(f"\n  0. Demo ALL modes")
    print(f"  q. Quit")
    
    while True:
        choice = input("\nSelect mode number (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            return []
        elif choice == '0':
            return presets
        else:
            try:
                index = int(choice) - 1
                if 0 <= index < len(presets):
                    return [presets[index]]
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Please enter a number or 'q'.")


def main():
    """Main demo function."""
    
    print("🎯 This demo showcases different game physics and player configurations!")
    print("Each mode runs for 15 seconds with random AI actions.")
    print("Press ESC during any demo to skip to the next mode.\n")
    
    # Let user choose modes
    selected_modes = interactive_mode_selection()
    
    if not selected_modes:
        print("👋 Goodbye!")
        return
    
    print(f"\n🚀 Starting demo of {len(selected_modes)} mode(s)...")
    
    # Demo each selected mode
    for mode in selected_modes:
        demo_game_mode(mode, duration=15, num_players=2)
        
        # Brief pause between modes
        if mode != selected_modes[-1]:  # Not the last mode
            print("\n⏳ Next mode in 2 seconds...")
            time.sleep(2)
    
    print("\n🎉 Demo complete! Try the modes yourself with:")
    print("   python scripts/play.py --game-mode <mode_name>")
    print("\nExample commands:")
    for mode in ["air_hockey", "giant_players", "pinball", "speed_demon"]:
        print(f"   python scripts/play.py --game-mode {mode} --time 60")
    
    print("\n📚 Happy experimenting!")


if __name__ == "__main__":
    main() 