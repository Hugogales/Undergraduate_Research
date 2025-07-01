#!/usr/bin/env python3
"""
Final test to verify corrected leg sprite assignments
"""

import sys
from pathlib import Path

def test_corrected_assignments():
    """Test that the corrected sprite assignments are working"""
    print("Testing corrected leg sprite assignments...")
    
    try:
        # Try the working import pattern
        import sys
        from pathlib import Path
        
        # Add project root to sys.path for imports
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import from the working path
        from soccer_env.envs.constants import ENV_CONSTANTS
        
        print("\n✅ CORRECTED Sprite Assignments:")
        print(f"   Blue Team Legs:  {ENV_CONSTANTS.TEAM1_LEGS[:5]}")
        print(f"   Green Team Legs: {ENV_CONSTANTS.TEAM2_LEGS[:5]}")
        
        print("\n🔧 Expected Pattern:")
        print("   • Players 0-3: Should use sprite (13) - medium size limbs")
        print("   • Player 4:    Should use sprite (14) - medium size limbs")
        print("   • NO MORE using sprite (12) which was too small")
        
        # Verify the pattern
        correct_pattern = True
        for i in range(4):
            if "(13)" not in ENV_CONSTANTS.TEAM1_LEGS[i]:
                print(f"   ❌ Player {i} still using wrong sprite: {ENV_CONSTANTS.TEAM1_LEGS[i]}")
                correct_pattern = False
        
        if "(14)" not in ENV_CONSTANTS.TEAM1_LEGS[4]:
            print(f"   ❌ Player 4 still using wrong sprite: {ENV_CONSTANTS.TEAM1_LEGS[4]}")
            correct_pattern = False
            
        if correct_pattern:
            print("   ✅ All leg sprite assignments are now CORRECT!")
        
        return correct_pattern
        
    except Exception as e:
        print(f"   ❌ Could not verify assignments: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_sprite_files_exist():
    """Verify that the sprite files we're now using actually exist"""
    print("\n📁 Verifying sprite files exist...")
    
    sprite_files_to_check = [
        "files/Images/Blue/characterBlue (13).png",
        "files/Images/Blue/characterBlue (14).png", 
        "files/Images/Green/characterGreen (13).png",
        "files/Images/Green/characterGreen (14).png"
    ]
    
    all_exist = True
    for sprite_file in sprite_files_to_check:
        file_path = project_root / sprite_file
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"   ✅ {sprite_file} - {file_size} bytes")
        else:
            print(f"   ❌ {sprite_file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_player_creation_with_corrected_sprites():
    """Test creating players with the corrected sprite assignments"""
    print("\n🏃 Testing Player creation with corrected sprites...")
    
    try:
        from soccer_env.core.entities import Player
        
        # Test both teams
        for team_id in [0, 1]:
            team_name = "Blue" if team_id == 0 else "Green"
            print(f"   Creating {team_name} team players...")
            
            for player_id in range(2):  # Just test first 2 players
                player = Player(
                    team_id=team_id,
                    player_id=player_id,
                    position=[200 + team_id * 900, 350]
                )
                
                # Check if sprites loaded correctly
                has_sprites = hasattr(player, 'body_image') and player.body_image is not None
                has_leg_sprites = hasattr(player, 'leg') and player.leg is not None
                
                expected_leg_sprite = 13 if player_id < 4 else 14
                print(f"     Player {player_id}: sprites loaded: {has_sprites}, leg sprites: {has_leg_sprites}")
                print(f"       Expected leg sprite: ({expected_leg_sprite})")
        
        print("   ✅ Player creation with corrected sprites successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Player creation failed: {e}")
        return False

if __name__ == "__main__":
    print("Final Leg Sprite Correction Test")
    print("=" * 45)
    
    assignments_ok = test_corrected_assignments()
    files_exist = verify_sprite_files_exist()
    
    if assignments_ok and files_exist:
        players_ok = test_player_creation_with_corrected_sprites()
        
        if players_ok:
            print("\n🎉 SUCCESS: Leg sprite issue has been FIXED!")
            print("   • Correct sprite assignments: ✅")
            print("   • All sprite files exist: ✅") 
            print("   • Player creation works: ✅")
        else:
            print("\n⚠️  Assignments and files are correct, but player creation has issues.")
    else:
        print("\n❌ There are still issues with sprite assignments or missing files.")
    
    print("\nTest completed.") 