#!/usr/bin/env python3
"""
Test script to debug sprite loading issues
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_constants_import():
    """Test if constants import correctly"""
    print("Testing constants import...")
    
    try:
        from soccer_env.envs.constants import ENV_CONSTANTS, VIS_CONSTANTS
        print(f"✓ Constants imported successfully")
        print(f"  ENV_CONSTANTS type: {type(ENV_CONSTANTS)}")
        print(f"  VIS_CONSTANTS type: {type(VIS_CONSTANTS)}")
        print(f"  TEAM1_SPRITES[0]: {VIS_CONSTANTS.TEAM1_SPRITES[0] if hasattr(VIS_CONSTANTS, 'TEAM1_SPRITES') else 'NOT FOUND'}")
        print(f"  TEAM2_SPRITES[0]: {VIS_CONSTANTS.TEAM2_SPRITES[0] if hasattr(VIS_CONSTANTS, 'TEAM2_SPRITES') else 'NOT FOUND'}")
        print(f"  BALL_SPRITE: {VIS_CONSTANTS.BALL_SPRITE if hasattr(VIS_CONSTANTS, 'BALL_SPRITE') else 'NOT FOUND'}")
        return True
    except Exception as e:
        print(f"✗ Constants import failed: {e}")
        return False

def test_sprite_files_exist():
    """Test if sprite files actually exist"""
    print("\nTesting sprite file existence...")
    
    sprite_files = [
        "files/Images/Blue/characterBlue (1).png",
        "files/Images/Green/characterGreen (1).png", 
        "files/Images/Equipment/ball_soccer2.png",
        "files/Images/Backgrounds/goal.png"
    ]
    
    for sprite_file in sprite_files:
        file_path = project_root / sprite_file
        if file_path.exists():
            print(f"✓ {sprite_file} exists")
        else:
            print(f"✗ {sprite_file} NOT FOUND")

def test_player_creation():
    """Test creating a Player without sprite loading issues"""
    print("\nTesting Player creation...")
    
    try:
        from soccer_env.core.entities import Player
        print("Creating Player...")
        
        player = Player(
            team_id=0,
            player_id=0,
            position=[200, 350]
        )
        print(f"✓ Player created successfully")
        print(f"  Team ID: {player.team_id}")
        print(f"  Has sprites: {hasattr(player, 'body_image') and player.body_image is not None}")
        
        return True
    except Exception as e:
        print(f"✗ Player creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Sprite Loading Diagnostic Test")
    print("=" * 40)
    
    constants_ok = test_constants_import()
    test_sprite_files_exist()
    
    if constants_ok:
        test_player_creation()
    
    print("\nTest completed.") 