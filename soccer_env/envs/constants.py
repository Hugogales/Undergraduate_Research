"""
Environment Constants for Soccer Game

Modify these values to customize the game behavior. These are the main configuration
parameters for the soccer environment.
"""

import random


def _generate_team_sprites(base_color, num_players=10):
    """Generate sprite lists for a team with variety."""
    sprites = []
    arms = []
    legs = []
    
    available_chars = [1, 2, 3, 4, 5]  # Available character numbers
    
    for i in range(num_players):
        # Use player_id to deterministically select character (ensures consistency)
        char_num = available_chars[i % len(available_chars)]
        
        # Build sprite paths
        body_sprite = f"files/Images/{base_color}/character{base_color} ({char_num}).png"
        
        # Character 5 uses different arm/leg sprites in the original
        if char_num == 5:
            arm_sprite = f"files/Images/{base_color}/character{base_color} (12).png"
            leg_sprite = f"files/Images/{base_color}/character{base_color} (14).png"
        else:
            arm_sprite = f"files/Images/{base_color}/character{base_color} (11).png"
            leg_sprite = f"files/Images/{base_color}/character{base_color} (13).png"
        
        sprites.append(body_sprite)
        arms.append(arm_sprite)
        legs.append(leg_sprite)
    
    return sprites, arms, legs


class EnvironmentConstants:
    """
    All environment constants in one place for easy modification.
    """
    
    # ==== SCREEN AND FIELD DIMENSIONS ====
    WIDTH = 1300                    # Total screen width in pixels
    HEIGHT = 700                    # Total screen height in pixels
    
    # Play area boundaries (excludes goal areas)
    GOAL_WIDTH = 65                 # Width of each goal area
    GOAL_HEIGHT = 175               # Height of each goal area
    
    PLAY_AREA_LEFT = GOAL_WIDTH     # Left boundary of play area (65)
    PLAY_AREA_RIGHT = WIDTH - GOAL_WIDTH  # Right boundary of play area (1235)
    PLAY_AREA_TOP = 0               # Top boundary of play area
    PLAY_AREA_BOTTOM = HEIGHT       # Bottom boundary of play area
    PLAY_AREA_WIDTH = PLAY_AREA_RIGHT - PLAY_AREA_LEFT
    PLAY_AREA_HEIGHT = HEIGHT
    
    # ==== GAME TIMING ====
    FPS = 42                        # Frames per second
    AGENT_DECISION_RATE = 42        # Number of agent decisions per second (not frames between decisions!)
    DEFAULT_GAME_DURATION = 120     # Default game duration in seconds
    
    # ==== PLAYER PROPERTIES ====
    PLAYER_RADIUS = 16              # Player collision radius in pixels
    PLAYER_SPEED = 5                # Base player movement speed
    PLAYER_POWER = 2                # Player vs player collision power
    PLAYER_HEIGHT = PLAYER_RADIUS * 2  # Player diameter for calculations
    
    # ==== BALL PROPERTIES ====
    BALL_RADIUS = 10                # Ball collision radius in pixels
    BALL_FRICTION = 0.965           # Ball velocity decay per frame (0.0-1.0)
    BALL_MAX_SPEED = 15             # Maximum ball velocity
    BALL_POWER = 0.85               # Player vs ball collision power
    KICK_SPEED = 20                 # Base kick velocity
    
    # ==== VISUAL SETTINGS ====
    # Team colors
    TEAM_1_COLOR = "Blue"           # Left team color name
    TEAM_2_COLOR = "Green"          # Right team color name
    
    # Color values (R, G, B)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)              # Team 1 player color
    GREEN = (83, 160, 23)           # Team 2 player color and field color
    BLACK = (0, 0, 0)               # Ball and text color
    RED = (255, 0, 0)               # Goal outline color
    YELLOW = (255, 255, 0)          # Play area boundary color
    
    # UI Settings
    TITLE = "2D Soccer Game"
    UI_OFFSET_X = 100               # Score box X offset from screen edge
    UI_OFFSET_Y = 20                # Score box Y offset from screen edge
    
    # ==== SPRITE PATHS ====
    # Set to None to disable sprites and use simple shapes
    BACKGROUND_SPRITE = "files/Images/Backgrounds/pitch.png"
    GOAL_SPRITE = "files/Images/Backgrounds/goal.png"
    BALL_SPRITE = "files/Images/Equipment/ball_soccer1.png"
    
    # Ball sprites (randomly selected)
    BALL_SPRITES = [
        "files/Images/Equipment/ball_soccer1.png",
        "files/Images/Equipment/ball_soccer2.png", 
        "files/Images/Equipment/ball_soccer3.png",
        "files/Images/Equipment/ball_soccer4.png"
    ]
    
    # Team sprite directories (for future player animations)
    TEAM_1_SPRITE_DIR = "files/Images/Blue"
    TEAM_2_SPRITE_DIR = "files/Images/Green"
    
    # Player sprites (using available files, extended to support up to 10 players per team)
    # Generate sprite lists for both teams with variety
    TEAM1_SPRITES, TEAM1_ARMS, TEAM1_LEGS = _generate_team_sprites("Blue", 10)
    TEAM2_SPRITES, TEAM2_ARMS, TEAM2_LEGS = _generate_team_sprites("Green", 10)
    
    # ==== GAMEPLAY SETTINGS ====
    RANDOMIZE_PLAYERS = False       # Whether to randomize starting positions
    SIMPLE_GAME = False             # Simplified game mode
    RENDER = True                   # Whether to enable rendering
    
    # Collision physics
    BALL_COLLISION_RESTITUTION = 0.8    # Ball bounce elasticity (0.0-1.0)
    PLAYER_COLLISION_RESTITUTION = 0.5  # Player collision elasticity (0.0-1.0)
    
    # Goal detection sensitivity
    GOAL_DETECTION_BUFFER = 5       # Extra pixels for goal detection
    
    # ==== DERIVED PROPERTIES ====
    @classmethod
    def get_random_ball_sprite(cls):
        """Get a random ball sprite path."""
        return random.choice(cls.BALL_SPRITES)
    
    @classmethod
    def calculate_team_positions(cls, num_players_per_team):
        """
        Calculate player starting positions for both teams.
        
        Args:
            num_players_per_team: Number of players per team (1-10)
            
        Returns:
            Tuple of (team_1_positions, team_2_positions)
        """
        team_1_positions = []
        team_2_positions = []
        
        # Calculate the number of players for each column
        back_column_players = min(3, num_players_per_team)
        middle_column_players = min(4, max(0, num_players_per_team - back_column_players))
        front_column_players = max(0, num_players_per_team - back_column_players - middle_column_players)
        
        # Calculate the vertical spacing between players
        vertical_spacing = cls.HEIGHT / 5
        
        def calculate_column_positions(start_x, num_players, offset):
            positions = []
            for i in range(num_players):
                y_position = offset + (i + 1) * vertical_spacing
                positions.append([start_x, y_position])
            return positions
        
        # Calculate the offset to center the players vertically
        offset_1 = cls.HEIGHT / 2 - (back_column_players + 1) * vertical_spacing / 2
        offset_2 = cls.HEIGHT / 2 - (middle_column_players + 1) * vertical_spacing / 2
        offset_3 = cls.HEIGHT / 2 - (front_column_players + 1) * vertical_spacing / 2
        
        # Calculate positions for each column
        back_column_x = cls.PLAY_AREA_WIDTH / 8 + cls.PLAY_AREA_LEFT
        middle_column_x = cls.PLAY_AREA_WIDTH / 4 + cls.PLAY_AREA_LEFT
        front_column_x = cls.PLAY_AREA_WIDTH * 3 / 8 + cls.PLAY_AREA_LEFT
        
        team_1_positions.extend(calculate_column_positions(back_column_x, back_column_players, offset_1))
        team_1_positions.extend(calculate_column_positions(middle_column_x, middle_column_players, offset_2))
        team_1_positions.extend(calculate_column_positions(front_column_x, front_column_players, offset_3))
        
        # Mirror positions for Team 2 across the vertical center line of the pitch
        team_2_positions = [
            [cls.PLAY_AREA_LEFT + cls.PLAY_AREA_WIDTH - (pos[0] - cls.PLAY_AREA_LEFT), pos[1]]
            for pos in team_1_positions
        ]
        
        return team_1_positions, team_2_positions
    
    @classmethod
    def get_center_position(cls):
        """Get the center position of the field."""
        return [cls.WIDTH // 2, cls.HEIGHT // 2]
    
    @classmethod
    def get_goal_positions(cls):
        """
        Get the positions of both goals.
        
        Returns:
            Tuple of (left_goal_pos, right_goal_pos)
        """
        left_goal_pos = [0, (cls.HEIGHT / 2) - (cls.GOAL_HEIGHT / 2)]
        right_goal_pos = [cls.WIDTH - cls.GOAL_WIDTH, (cls.HEIGHT / 2) - (cls.GOAL_HEIGHT / 2)]
        return left_goal_pos, right_goal_pos


# Create a global instance for easy access
ENV_CONSTANTS = EnvironmentConstants()


# ==== CONFIGURATION PRESETS ====

class GameModePresets:
    """
    Predefined game mode configurations.
    
    Each preset is a dictionary that can override default environment constants.
    Use these by passing preset_name to the environment constructor.
    """
    
    # Base preset definitions
    PRESETS = {
        "default": {
            # Standard soccer settings
            "PLAYER_SPEED": 5,
            "BALL_MAX_SPEED": 15,
            "KICK_SPEED": 15.5,
            "BALL_FRICTION": 0.965,
            "BALL_COLLISION_RESTITUTION": 0.8,
            "PLAYER_COLLISION_RESTITUTION": 0.5,
            "PLAYER_RADIUS": 15,
            "BALL_RADIUS": 8,
        },
        
        "air_hockey": {
            # Almost no friction - puck glides like air hockey
            "PLAYER_SPEED": 6,
            "BALL_MAX_SPEED": 25,
            "KICK_SPEED": 20,
            "BALL_FRICTION": 0.999,  # Almost no friction!
            "BALL_COLLISION_RESTITUTION": 0.95,
            "PLAYER_COLLISION_RESTITUTION": 0.9,
            "PLAYER_RADIUS": 12,
            "BALL_RADIUS": 6,
        },
        
        "arcade": {
            # Fast, bouncy, fun arcade action
            "PLAYER_SPEED": 8,
            "BALL_MAX_SPEED": 22,
            "KICK_SPEED": 25,
            "BALL_FRICTION": 0.98,
            "BALL_COLLISION_RESTITUTION": 1.0,  # Super bouncy!
            "PLAYER_COLLISION_RESTITUTION": 0.8,
            "PLAYER_RADIUS": 18,
            "BALL_RADIUS": 10,
        },
        
        "realistic": {
            # Realistic soccer physics
            "PLAYER_SPEED": 4,
            "BALL_MAX_SPEED": 12,
            "KICK_SPEED": 14,
            "BALL_FRICTION": 0.92,  # More realistic ball slowdown
            "BALL_COLLISION_RESTITUTION": 0.6,
            "PLAYER_COLLISION_RESTITUTION": 0.3,
            "PLAYER_RADIUS": 15,
            "BALL_RADIUS": 8,
        },
        
        "giant_players": {
            # Huge players for chaotic fun
            "PLAYER_SPEED": 4,
            "BALL_MAX_SPEED": 15,
            "KICK_SPEED": 18,
            "BALL_FRICTION": 0.96,
            "BALL_COLLISION_RESTITUTION": 0.7,
            "PLAYER_COLLISION_RESTITUTION": 0.6,
            "PLAYER_RADIUS": 35,  # Double size!
            "BALL_RADIUS": 12,
        },
        
        "pinball": {
            # Crazy bouncy pinball-style physics
            "PLAYER_SPEED": 3,
            "BALL_MAX_SPEED": 30,
            "KICK_SPEED": 35,
            "BALL_FRICTION": 0.995,
            "BALL_COLLISION_RESTITUTION": 1.2,  # Gains energy on bounce!
            "PLAYER_COLLISION_RESTITUTION": 1.0,
            "PLAYER_RADIUS": 20,
            "BALL_RADIUS": 8,
        },
        
        "slow_motion": {
            # Everything in slow motion
            "PLAYER_SPEED": 2,
            "BALL_MAX_SPEED": 8,
            "KICK_SPEED": 10,
            "BALL_FRICTION": 0.94,
            "BALL_COLLISION_RESTITUTION": 0.5,
            "PLAYER_COLLISION_RESTITUTION": 0.3,
            "PLAYER_RADIUS": 15,
            "BALL_RADIUS": 8,
        },
        
        "speed_demon": {
            # Everything super fast
            "PLAYER_SPEED": 12,
            "BALL_MAX_SPEED": 40,
            "KICK_SPEED": 45,
            "BALL_FRICTION": 0.98,
            "BALL_COLLISION_RESTITUTION": 0.9,
            "PLAYER_COLLISION_RESTITUTION": 0.7,
            "PLAYER_RADIUS": 12,
            "BALL_RADIUS": 6,
        },
        
        "tiny_players": {
            # Tiny players, big field
            "PLAYER_SPEED": 6,
            "BALL_MAX_SPEED": 18,
            "KICK_SPEED": 20,
            "BALL_FRICTION": 0.97,
            "BALL_COLLISION_RESTITUTION": 0.8,
            "PLAYER_COLLISION_RESTITUTION": 0.5,
            "PLAYER_RADIUS": 8,   # Half size!
            "BALL_RADIUS": 5,
        },
        
        "bouncy_castle": {
            # Super bouncy everything
            "PLAYER_SPEED": 6,
            "BALL_MAX_SPEED": 20,
            "KICK_SPEED": 22,
            "BALL_FRICTION": 0.99,
            "BALL_COLLISION_RESTITUTION": 1.1,
            "PLAYER_COLLISION_RESTITUTION": 1.0,
            "PLAYER_RADIUS": 20,
            "BALL_RADIUS": 10,
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> dict:
        """
        Get a preset configuration by name.
        
        Args:
            preset_name: Name of the preset to retrieve
            
        Returns:
            Dictionary of configuration values
            
        Raises:
            ValueError: If preset_name is not found
        """
        if preset_name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
        
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def list_presets(cls) -> list:
        """Get list of all available preset names."""
        return list(cls.PRESETS.keys())
    
    @classmethod
    def describe_preset(cls, preset_name: str) -> str:
        """Get a description of what a preset does."""
        descriptions = {
            "default": "Standard soccer settings with balanced gameplay",
            "air_hockey": "Almost no friction - ball glides like air hockey puck",
            "arcade": "Fast, bouncy, fun arcade action with high energy",
            "realistic": "Realistic soccer physics with proper ball behavior",
            "giant_players": "Huge players for chaotic fun and easier control",
            "pinball": "Crazy bouncy pinball-style physics with energy gain",
            "slow_motion": "Everything in slow motion for tactical gameplay",
            "speed_demon": "Everything super fast for intense action",
            "tiny_players": "Tiny players on big field for precision gameplay",
            "bouncy_castle": "Super bouncy everything for unpredictable fun"
        }
        return descriptions.get(preset_name, "No description available")
    
    @classmethod
    def apply_preset_to_constants(cls, preset_name: str) -> dict:
        """
        Apply a preset to create a modified constants dictionary.
        This doesn't modify the global ENV_CONSTANTS.
        
        Args:
            preset_name: Name of the preset to apply
            
        Returns:
            Dictionary with preset values applied to base constants
        """
        preset = cls.get_preset(preset_name)
        
        # Start with current ENV_CONSTANTS values
        modified_constants = {}
        for attr_name in dir(ENV_CONSTANTS):
            if not attr_name.startswith('_') and not callable(getattr(ENV_CONSTANTS, attr_name)):
                modified_constants[attr_name] = getattr(ENV_CONSTANTS, attr_name)
        
        # Apply preset overrides
        modified_constants.update(preset)
        
        return modified_constants


# Keep old methods for backward compatibility (but mark as deprecated)
def _deprecated_method():
    """These methods are deprecated. Use GameModePresets.get_preset() instead."""
    import warnings
    warnings.warn("Direct preset methods are deprecated. Use GameModePresets.get_preset('preset_name') instead.", 
                  DeprecationWarning, stacklevel=2)


# ==== USAGE EXAMPLES ====
"""
To modify constants:

1. Direct modification:
   ENV_CONSTANTS.PLAYER_SPEED = 10
   ENV_CONSTANTS.BALL_FRICTION = 0.9

2. Using presets:
   GameModePresets.fast_paced()
   GameModePresets.realistic()

3. Getting calculated values:
   team1_pos, team2_pos = ENV_CONSTANTS.calculate_team_positions(3)
   center = ENV_CONSTANTS.get_center_position()
   left_goal, right_goal = ENV_CONSTANTS.get_goal_positions()
""" 