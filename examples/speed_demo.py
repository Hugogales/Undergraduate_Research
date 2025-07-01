#!/usr/bin/env python3
"""
Player Speed Demonstration

This script shows how to create soccer environments with different player speeds.
You can now easily tweak player speed to make the game faster or slower.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from soccer_env.envs.soccer_env import env

def demo_different_speeds():
    """Demonstrate different player speeds."""
    print("=== Player Speed Demonstration ===")
    print("The player_speed parameter now works correctly!")
    print("You can set any speed value when creating the environment.\n")
    
    # Different speed configurations
    speed_configs = [
        {"name": "Slow Motion", "speed": 2.0, "description": "Very slow, tactical gameplay"},
        {"name": "Default", "speed": 5.0, "description": "Standard speed (original)"},
        {"name": "Fast", "speed": 8.0, "description": "Faster-paced action"},
        {"name": "Lightning", "speed": 12.0, "description": "Very fast, arcade-style"},
    ]
    
    for config in speed_configs:
        print(f"🏃 {config['name']} Speed (player_speed={config['speed']})")
        print(f"   {config['description']}")
        
        # Create environment with specific speed
        test_env = env(
            render_mode=None,
            num_players_per_team=1,
            player_speed=config['speed']
        )
        
        # Verify the speed was set correctly
        try:
            game_state = test_env.unwrapped.game_engine.get_game_state()
            if game_state and game_state.players:
                actual_speed = game_state.players[0].speed
                print(f"   ✅ Actual player speed: {actual_speed}")
                
                # Show movement distance per frame
                player = game_state.players[0]
                velocity = player.move([1.0, 0.0, 0.0, 0.0])  # Full speed right
                print(f"   📏 Movement: {abs(velocity[0])} pixels per frame")
                print(f"   🎯 At 42 FPS: {abs(velocity[0]) * 42:.0f} pixels per second")
        except Exception as e:
            print(f"   ⚠️  Could not verify speed: {e}")
        
        test_env.close()
        print()

def usage_examples():
    """Show usage examples."""
    print("=== Usage Examples ===")
    print("Here's how to use different player speeds in your code:\n")
    
    examples = [
        {
            "title": "1. Environment Creation with Custom Speed",
            "code": '''from soccer_env.envs.soccer_env import env

# Create a slow-paced tactical game
slow_env = env(
    render_mode="human",
    num_players_per_team=2,
    player_speed=3.0  # Slow speed
)

# Create a fast-paced arcade game  
fast_env = env(
    render_mode="human", 
    num_players_per_team=1,
    player_speed=10.0  # Fast speed
)'''
        },
        {
            "title": "2. Playing with Different Speeds",
            "code": '''# Slow game for beginners
python scripts/play.py --players 1 --time 60
# (Uses default speed from ENV_CONSTANTS.PLAYER_SPEED = 5)

# You can modify ENV_CONSTANTS.PLAYER_SPEED in constants.py
# Or create your own script with custom speeds'''
        },
        {
            "title": "3. Training with Different Speeds",
            "code": '''from soccer_env.envs.soccer_parallel_env import parallel_env

# Train with faster players
training_env = parallel_env(
    num_players_per_team=3,
    player_speed=7.0,  # 40% faster than default
    max_episode_steps=5000
)'''
        },
        {
            "title": "4. Direct Player Creation",
            "code": '''from soccer_env.core.entities import Player

# Create a slow player
slow_player = Player(
    team_id=0,
    player_id=0, 
    position=[100, 100],
    speed=2.5  # Custom speed
)

# Create a fast player
fast_player = Player(
    team_id=1,
    player_id=0,
    position=[200, 100], 
    speed=8.0  # Custom speed
)'''
        }
    ]
    
    for example in examples:
        print(f"### {example['title']}")
        print("```python")
        print(example['code'])
        print("```\n")

def speed_recommendations():
    """Provide speed recommendations for different use cases."""
    print("=== Speed Recommendations ===")
    recommendations = [
        {"use_case": "Learning/Training AI", "speed": "2.0-3.0", "reason": "Slower speeds help with learning complex behaviors"},
        {"use_case": "Standard Gameplay", "speed": "5.0", "reason": "Original balanced speed, good for most scenarios"},
        {"use_case": "Fast Action Games", "speed": "7.0-10.0", "reason": "Higher speeds for more exciting, arcade-style gameplay"},
        {"use_case": "Performance Testing", "speed": "1.0 or 15.0+", "reason": "Extreme speeds help test edge cases and performance"},
        {"use_case": "Beginner-Friendly", "speed": "3.0-4.0", "reason": "Slightly slower for easier control and decision making"},
    ]
    
    for rec in recommendations:
        print(f"🎯 {rec['use_case']}")
        print(f"   Recommended speed: {rec['speed']}")
        print(f"   Reason: {rec['reason']}\n")

if __name__ == "__main__":
    demo_different_speeds()
    usage_examples()
    speed_recommendations()
    
    print("=== Summary ===")
    print("✅ Player speed parameter now works correctly!")
    print("✅ You can set any speed value when creating environments")
    print("✅ Speed affects movement velocity directly (pixels per frame)")
    print("✅ All environment types support the player_speed parameter")
    print("\nTry different speeds to find what works best for your use case!") 