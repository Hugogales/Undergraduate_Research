"""
Pygame-based renderer for the soccer environment.

This module provides rendering capabilities and human input handling for the soccer game.
"""

import pygame
import math
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import os
import sys
import time
from pathlib import Path

# Import environment constants - use only the new constants file
from ..envs.constants import ENV_CONSTANTS
from ..core.entities import Player, Ball, Goal, GameState


class SoccerRenderer:
    """
    Pygame-based renderer for soccer environment with human input support.
    """
    
    def __init__(
        self,
        field_width: float = None,
        field_height: float = None,
        screen_width: int = None,
        screen_height: int = None,
        fps: int = None
    ):
        """
        Initialize the renderer.
        
        Args:
            field_width: Width of the soccer field in game units (defaults to ENV_CONSTANTS)
            field_height: Height of the soccer field in game units (defaults to ENV_CONSTANTS)
            screen_width: Width of the screen in pixels (defaults to ENV_CONSTANTS)
            screen_height: Height of the screen in pixels (defaults to ENV_CONSTANTS)
            fps: Target frames per second (defaults to ENV_CONSTANTS)
        """
        # Use constants with fallbacks for any missing values
        self.field_width = field_width if field_width is not None else ENV_CONSTANTS.WIDTH
        self.field_height = field_height if field_height is not None else ENV_CONSTANTS.HEIGHT
        self.screen_width = screen_width if screen_width is not None else ENV_CONSTANTS.WIDTH
        self.screen_height = screen_height if screen_height is not None else ENV_CONSTANTS.HEIGHT
        self.fps = fps if fps is not None else ENV_CONSTANTS.FPS
        
        # Initialize pygame
        pygame.init()
        
        # Calculate scaling factors
        self.scale_x = self.screen_width / self.field_width
        self.scale_y = self.screen_height / self.field_height
        
        # Create display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(ENV_CONSTANTS.TITLE)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Colors from VIS_CONSTANTS
        self.colors = {
            'white': ENV_CONSTANTS.WHITE,
            'black': ENV_CONSTANTS.BLACK,
            'green': ENV_CONSTANTS.GREEN,
            'blue': ENV_CONSTANTS.BLUE,
            'red': ENV_CONSTANTS.RED,
            'yellow': ENV_CONSTANTS.YELLOW,
        }
        
        # Font for text rendering
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        
        # Action persistence like original implementation
        self.frame_counter = 0
        self.input_update_interval = int(ENV_CONSTANTS.FPS / ENV_CONSTANTS.AGENT_DECISION_RATE) if hasattr(ENV_CONSTANTS, 'AGENT_DECISION_RATE') else 3  # 3 frames between input updates
        self.last_human_actions = {}  # Store last actions for persistence
        
        # Human input tracking
        self.human_input = {
            'move_x': 0.0,
            'move_y': 0.0,
            'kick_power': 0.0,
            'kick_direction': 0.0,
            'kick_pressed': False
        }
        
        # Key mappings for human players - map to all players on each team
        self.key_mappings = {}
        
        # Team 0 (Blue) - all players controlled by WASD + Space
        team_0_mapping = {
            'up': pygame.K_w,
            'down': pygame.K_s,
            'left': pygame.K_a,
            'right': pygame.K_d,
            'kick': pygame.K_SPACE
        }
        
        # Team 1 (Red) - all players controlled by Arrow keys + Right Ctrl
        team_1_mapping = {
            'up': pygame.K_UP,
            'down': pygame.K_DOWN,
            'left': pygame.K_LEFT,
            'right': pygame.K_RIGHT,
            'kick': pygame.K_RCTRL
        }
        
        # Apply same mapping to all players on each team
        for i in range(10):  # Support up to 10 players per team
            self.key_mappings[f'team_0_player_{i}'] = team_0_mapping
            self.key_mappings[f'team_1_player_{i}'] = team_1_mapping
        
        # Load sprites if available
        self._load_sprites()
        
        # Flag for showing instructions
        self.show_instructions = True
        
        # TIME-BASED FPS THROTTLING: Track last frame time for consistent frame rate
        self.target_frame_duration = 1.0 / self.fps  # Target time per frame
        self.last_frame_time = 0.0  # Time of last throttled frame
        
        # Precise timing control for AEC steps
        self.target_step_duration = 1.0 / ENV_CONSTANTS.AGENT_DECISION_RATE if hasattr(ENV_CONSTANTS, 'AGENT_DECISION_RATE') else (1.0 / 14)
        self.last_step_time = None
        
    def _load_sprites(self):
        """Load sprite images if available, otherwise use simple shapes."""
        self.use_sprites = False
        self.sprites = {}
        
        # Try to load sprites using paths from constants
        sprite_paths = {
            'ball': getattr(ENV_CONSTANTS, 'get_random_ball_sprite', lambda: 'files/Images/Equipment/ball_soccer1.png')() if hasattr(ENV_CONSTANTS, 'get_random_ball_sprite') else getattr(ENV_CONSTANTS, 'BALL_SPRITE', 'files/Images/Equipment/ball_soccer1.png'),
            'background': getattr(ENV_CONSTANTS, 'BACKGROUND_SPRITE', 'files/Images/Backgrounds/pitch.png'),
            'goal': getattr(ENV_CONSTANTS, 'GOAL_SPRITE', 'files/Images/Backgrounds/goal.png'),
        }
        
        # Get workspace root for proper path resolution
        workspace_root = Path(__file__).resolve().parent.parent.parent
        
        for name, path in sprite_paths.items():
            if path:
                # Try both relative to workspace and absolute paths
                full_path = workspace_root / path
                if full_path.exists():
                    try:
                        self.sprites[name] = pygame.image.load(str(full_path)).convert_alpha()
                        self.use_sprites = True
                        print(f"Loaded sprite: {name} from {full_path}")
                    except pygame.error as e:
                        print(f"Failed to load sprite {name}: {e}")
                elif os.path.exists(path):
                    try:
                        self.sprites[name] = pygame.image.load(path).convert_alpha()
                        self.use_sprites = True
                        print(f"Loaded sprite: {name} from {path}")
                    except pygame.error as e:
                        print(f"Failed to load sprite {name}: {e}")
                else:
                    print(f"Sprite file not found: {path}")
        
        print(f"Sprite loading complete. Use sprites: {self.use_sprites}")
    
    def _field_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert field coordinates to screen coordinates."""
        screen_x = int(x * self.scale_x)
        screen_y = int(y * self.scale_y)
        return screen_x, screen_y
    
    def _screen_to_field(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to field coordinates."""
        field_x = screen_x / self.scale_x
        field_y = screen_y / self.scale_y
        return field_x, field_y
    
    def handle_events(self) -> Dict[str, Any]:
        """
        Handle pygame events and return human input actions.
        
        Returns:
            Dictionary containing game events and human input
        """
        events = {
            'quit': False,
            'human_actions': {}
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events['quit'] = True
        
        # Sample human input every input_update_interval frames (like original)
        # This matches the original: action_update_interval = int(FPS / AGENT_DECISION_RATE) = 3 frames
        if self.frame_counter % self.input_update_interval == 0:
            keys = pygame.key.get_pressed()
            
            # Process human input for each player and store for persistence
            for player_name, key_map in self.key_mappings.items():
                action = self._get_human_action(keys, key_map)
                self.last_human_actions[player_name] = action
        
        # Increment frame counter (this should be called once per rendered frame)
        self.frame_counter += 1
        
        # Return the current persistent actions (may be same as last few frames)
        events['human_actions'] = self.last_human_actions.copy()
        
        return events
    
    def _get_human_action(self, keys, key_map) -> List[float]:
        """
        Get human action from keyboard input.
        
        Args:
            keys: Pygame key state
            key_map: Key mapping for this player
            
        Returns:
            Action as [move_x, move_y, kick_power, kick_direction]
        """
        move_x = 0.0
        move_y = 0.0
        kick_power = 0.0
        kick_direction = 0.0
        
        # Movement
        if keys[key_map['left']]:
            move_x -= 1.0
        if keys[key_map['right']]:
            move_x += 1.0
        if keys[key_map['up']]:
            move_y -= 1.0
        if keys[key_map['down']]:
            move_y += 1.0
        
        # Normalize movement vector
        if move_x != 0 or move_y != 0:
            magnitude = math.sqrt(move_x**2 + move_y**2)
            move_x /= magnitude
            move_y /= magnitude
        
        # Kicking
        if keys[key_map['kick']]:
            kick_power = 1.0
            # Kick in movement direction or forward if not moving
            if move_x != 0 or move_y != 0:
                kick_direction = math.atan2(move_y, move_x)
            else:
                kick_direction = 0.0  # Default forward direction
        
        return [move_x, move_y, kick_power, kick_direction]
    
    def render(self, game_state: GameState, is_aec_cycle_complete: bool = False, throttle_fps: bool = True, show_human_instructions: bool = True) -> Optional[np.ndarray]:
        """
        Render the current game state with time-based FPS throttling.
        
        Args:
            game_state: Current state of the game
            is_aec_cycle_complete: Whether this render call is at the end of a complete AEC cycle (for compatibility)
            throttle_fps: Whether to apply FPS throttling (for compatibility, but now uses time-based throttling)
            show_human_instructions: Whether to show human control instructions (False for AI-only mode)
            
        Returns:
            RGB array if render mode is 'rgb_array', None otherwise
        """
        # Clear screen with black like original
        self.screen.fill(ENV_CONSTANTS.BLACK)
        
        # Draw play area with pitch image
        self._draw_field()
        
        # Draw goal grass areas first (behind everything)
        for goal in game_state.goals:
            self._draw_goal_grass(goal)
        
        # Draw players
        for player in game_state.players:
            self._draw_player(player)
            
        # Draw ball with spinning animation
        self._draw_ball(game_state.ball)
        
        # Draw goals on top
        for goal in game_state.goals:
            self._draw_goal(goal)
        
        # Draw UI
        self._draw_ui(game_state, show_human_instructions)
        
        # Update display
        pygame.display.flip()
        
        # TIME-BASED FPS THROTTLING: Only throttle based on actual time elapsed
        # This ensures consistent frame rate regardless of player count
        if throttle_fps:
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time
            
            # Only apply throttling if enough time has passed since last throttled frame
            if time_since_last_frame >= self.target_frame_duration:
                # Use pygame clock but don't throttle too frequently
                elapsed = self.clock.tick(ENV_CONSTANTS.FPS)
                self.last_frame_time = current_time
            # If not enough time has passed, just update clock without throttling
            else:
                # Still tick the clock to track time, but don't limit frame rate
                self.clock.tick()
        
        # Return RGB array if needed
        return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
    
    def _draw_field(self):
        """Draw the soccer field exactly like original."""
        # Draw pitch image in play area only
        if self.use_sprites and 'background' in self.sprites:
            # Scale background to play area size like original
            play_area_width = ENV_CONSTANTS.PLAY_AREA_RIGHT - ENV_CONSTANTS.PLAY_AREA_LEFT
            play_area_height = ENV_CONSTANTS.PLAY_AREA_BOTTOM - ENV_CONSTANTS.PLAY_AREA_TOP
            
            background = pygame.transform.scale(
                self.sprites['background'], 
                (play_area_width, play_area_height)
            )
            self.screen.blit(background, (ENV_CONSTANTS.PLAY_AREA_LEFT, ENV_CONSTANTS.PLAY_AREA_TOP))
        else:
            # Use solid green for play area
            play_area_rect = pygame.Rect(
                ENV_CONSTANTS.PLAY_AREA_LEFT,
                ENV_CONSTANTS.PLAY_AREA_TOP,
                ENV_CONSTANTS.PLAY_AREA_RIGHT - ENV_CONSTANTS.PLAY_AREA_LEFT,
                ENV_CONSTANTS.PLAY_AREA_BOTTOM - ENV_CONSTANTS.PLAY_AREA_TOP
            )
            pygame.draw.rect(self.screen, self.colors['green'], play_area_rect)
    
    def _draw_goal_grass(self, goal: Goal):
        """Draw grass area in the goal using the original draw method."""
        # Use the goal's own draw_grass method which matches the original exactly
        goal.draw_grass(self.screen)
    
    def _draw_goal(self, goal: Goal):
        """Draw a goal using the original draw method."""
        # Use the goal's own draw method which matches the original exactly
        goal.draw(self.screen)
    
    def _draw_ball(self, ball: Ball):
        """Draw the soccer ball using the original draw method."""
        # Use the ball's own draw method which matches the original exactly
        ball.draw(self.screen)
    
    def _draw_player(self, player: Player):
        """Draw a player using the original draw method."""
        # Use the player's own draw method which matches the original exactly
        player.draw(self.screen)
    
    def _draw_ui(self, game_state: GameState, show_human_instructions: bool = True):
        """Draw user interface elements exactly like original."""
        # Timer calculation - ensure time never goes negative
        time_remaining = max(0, game_state.time_remaining)  # Clamp to 0 minimum
        minutes = int(time_remaining) // 60
        seconds = int(time_remaining) % 60
        
        # Score text like original
        score_text = f"Team {ENV_CONSTANTS.TEAM_1_COLOR}: {game_state.team_0_score} | Team {ENV_CONSTANTS.TEAM_2_COLOR} : {game_state.team_1_score} | Time : {minutes:02}:{seconds:02}"
        score_surface = self.font_large.render(score_text, True, ENV_CONSTANTS.BLACK)
        
        # Calculate the position and size of the box like original
        offset_x = 100
        offset_y = 20
        box_x = offset_x
        box_y = offset_y
        box_width = score_surface.get_width() + 20  # Add some padding
        box_height = score_surface.get_height() + 10  # Add some padding
        
        # Draw the box with transparency like original
        box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        box_surface.fill((255, 255, 255, 180))  # 180 is ~70% opacity
        # Draw border on the translucent surface
        pygame.draw.rect(box_surface, ENV_CONSTANTS.BLACK, (0, 0, box_width, box_height), 2)
        # Blit the translucent box onto the main screen
        self.screen.blit(box_surface, (box_x, box_y))
        
        # Blit the text inside the box
        self.screen.blit(score_surface, (box_x + 10, box_y + 5))
        
        # Instructions for human players - only show if requested
        if show_human_instructions:
            instructions = [
                "Team 0 (Blue) ALL players: WASD to move, SPACE to kick",
                "Team 1 (Red) ALL players: Arrow keys to move, Right Ctrl to kick",
                "Press ESC to quit"
            ]
            
            # Create semi-transparent background for instructions
            instruction_bg = pygame.Surface((400, 85), pygame.SRCALPHA)
            instruction_bg.fill((0, 0, 0, 150))
            self.screen.blit(instruction_bg, (10, self.screen_height - 95))
            
            # Draw text with white outline for better visibility
            for i, instruction in enumerate(instructions):
                text_surface = self.font_medium.render(instruction, True, self.colors['white'])
                self.screen.blit(text_surface, (15, self.screen_height - 90 + i * 25))
        else:
            # Show AI watching instructions instead
            instructions = [
                "🤖 Watching AI Play",
                "Press ESC to quit"
            ]
            
            # Create semi-transparent background for instructions
            instruction_bg = pygame.Surface((200, 60), pygame.SRCALPHA)
            instruction_bg.fill((0, 0, 0, 150))
            self.screen.blit(instruction_bg, (10, self.screen_height - 70))
            
            # Draw text with white outline for better visibility
            for i, instruction in enumerate(instructions):
                text_surface = self.font_medium.render(instruction, True, self.colors['white'])
                self.screen.blit(text_surface, (15, self.screen_height - 65 + i * 25))
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit() 