#!/usr/bin/env python3
"""
Test script to debug leg sprite assignments
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_sprite_assignments():
    """Test what sprites are assigned to each player component"""
    print("Testing sprite assignments...")
    
    try:
        from soccer_env.envs.constants import ENV_CONSTANTS
        
        print(f"\n🔍 Blue Team (Team 1) Sprite Assignments:")
        print(f"   Body sprites: {ENV_CONSTANTS.TEAM1_SPRITES}")
        print(f"   Arm sprites:  {ENV_CONSTANTS.TEAM1_ARMS}")
        print(f"   Leg sprites:  {ENV_CONSTANTS.TEAM1_LEGS}")
        
        print(f"\n🔍 Green Team (Team 2) Sprite Assignments:")
        print(f"   Body sprites: {ENV_CONSTANTS.TEAM2_SPRITES}")
        print(f"   Arm sprites:  {ENV_CONSTANTS.TEAM2_ARMS}")
        print(f"   Leg sprites:  {ENV_CONSTANTS.TEAM2_LEGS}")
        
        return True
    except Exception as e:
        print(f"✗ Could not get sprite assignments: {e}")
        return False

def analyze_sprite_files():
    """Analyze what each sprite file number typically represents"""
    print(f"\n📁 Available sprite files analysis:")
    
    blue_dir = project_root / "files/Images/Blue"
    green_dir = project_root / "files/Images/Green"
    
    print(f"\n📂 Blue sprites (files/Images/Blue/):")
    for i in range(1, 15):
        file_path = blue_dir / f"characterBlue ({i}).png"
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"   characterBlue ({i}).png - {file_size} bytes")
    
    print(f"\n📂 Green sprites (files/Images/Green/):")
    for i in range(1, 15):
        file_path = green_dir / f"characterGreen ({i}).png"
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"   characterGreen ({i}).png - {file_size} bytes")

def guess_sprite_types():
    """Guess what each sprite type is based on file sizes and patterns"""
    print(f"\n🔍 Sprite type analysis (based on file sizes):")
    
    blue_dir = project_root / "files/Images/Blue"
    
    sprite_info = {}
    for i in range(1, 15):
        file_path = blue_dir / f"characterBlue ({i}).png"
        if file_path.exists():
            file_size = file_path.stat().st_size
            sprite_info[i] = file_size
    
    # Group by file size to identify patterns
    size_groups = {}
    for num, size in sprite_info.items():
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(num)
    
    print("   Sprite groups by file size:")
    for size, numbers in sorted(size_groups.items(), reverse=True):
        print(f"   {size} bytes: characterBlue ({', '.join(map(str, numbers))}).png")
    
    # Based on typical sprite patterns:
    # - Larger files (700+ bytes) are usually body sprites
    # - Medium files (300-400 bytes) are usually limbs  
    # - Smaller files (300- bytes) are usually smaller limbs or details
    
    large_sprites = [num for num, size in sprite_info.items() if size > 600]
    medium_sprites = [num for num, size in sprite_info.items() if 350 <= size <= 600]
    small_sprites = [num for num, size in sprite_info.items() if size < 350]
    
    print(f"\n   Likely body sprites (large): {large_sprites}")
    print(f"   Likely arm/leg sprites (medium): {medium_sprites}")
    print(f"   Likely small limb sprites: {small_sprites}")

def test_current_leg_assignment():
    """Test what the current code is assigning for legs"""
    print(f"\n🦵 Current leg sprite assignment analysis:")
    
    try:
        from soccer_env.envs.constants import ENV_CONSTANTS
        
        print("For Blue team:")
        for i, leg_sprite in enumerate(ENV_CONSTANTS.TEAM1_LEGS[:5]):  # First 5 players
            print(f"   Player {i}: {leg_sprite}")
            
        print("\nFor Green team:")
        for i, leg_sprite in enumerate(ENV_CONSTANTS.TEAM2_LEGS[:5]):  # First 5 players
            print(f"   Player {i}: {leg_sprite}")
            
        # Test the logic manually
        print(f"\n🔧 Manual sprite assignment logic test:")
        available_chars = [1, 2, 3, 4, 5]
        
        for i in range(5):
            char_num = available_chars[i % len(available_chars)]
            
            if char_num == 5:
                leg_sprite_num = 14
            else:
                leg_sprite_num = 12
                
            print(f"   Player {i} (char {char_num}): should use sprite ({leg_sprite_num}) for legs")
            
    except Exception as e:
        print(f"✗ Could not analyze current assignments: {e}")

def suggest_corrections():
    """Suggest potential corrections based on analysis"""
    print(f"\n💡 Potential Issues and Suggestions:")
    
    # Based on the sprite assignment logic, we're using:
    # - characterBlue (12).png for legs of characters 1,2,3,4  
    # - characterBlue (14).png for legs of character 5
    
    print("   Current leg assignments:")
    print("   • Characters 1,2,3,4 → sprite (12)")
    print("   • Character 5 → sprite (14)")
    print()
    print("   Potential issues:")
    print("   • Sprite (12) might not be leg sprites - could be arms or other limbs")
    print("   • Sprite (14) might not be leg sprites either")
    print("   • The assignment logic might need to be corrected")
    print()
    print("   To fix, we should:")
    print("   1. Verify what sprites (11), (12), (13), (14) actually contain")
    print("   2. Look at old working code to see correct assignments")
    print("   3. Test with different sprite numbers to find the correct legs")

if __name__ == "__main__":
    print("Leg Sprite Diagnostic Test")
    print("=" * 50)
    
    test_sprite_assignments()
    analyze_sprite_files()
    guess_sprite_types()
    test_current_leg_assignment()
    suggest_corrections()
    
    print("\nTest completed.") 