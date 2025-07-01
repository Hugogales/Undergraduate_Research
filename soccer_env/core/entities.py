"""
Game entities for the soccer environment.

This module contains the classes that represent the core entities in the soccer game:
players, ball, goals, and the overall game state.
"""

import math
import numpy as np
import pygame
import random
import os
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from ..envs.constants import ENV_CONSTANTS 



@dataclass
class GameState:
    """
    Represents the complete state of a soccer game.
    """
    players: List['Player']
    ball: 'Ball'
    goals: List['Goal']
    team_0_score: int = 0
    team_1_score: int = 0
    episode_step: int = 0
    time_remaining: float = 0.0
    goal_scored: bool = False
    ball_possession: Optional[int] = None
    last_goal_scorer: Optional[int] = None


class Player:
    """
    Represents a soccer player with exact original animation and rendering capabilities.
    """
    
    def __init__(
        self,
        team_id: int,
        player_id: int,
        position: List[float] = None,
        up_key=None,
        down_key=None,
        left_key=None,
        right_key=None,
        shoot_key=None,
        field_width: float = None,
        field_height: float = None,
        speed: float = None,
        radius: float = None
    ):
        """
        Initialize a player exactly like the original Player.py
        
        Args:
            team_id: Team identifier (1 or 2 for original, 0 or 1 for AEC)
            player_id: Player identifier within the team
            position: [x, y] position of the player (optional, uses default if not provided)
            up_key: Pygame key constant for moving up
            down_key: Pygame key constant for moving down
            left_key: Pygame key constant for moving left
            right_key: Pygame key constant for moving right
            shoot_key: Pygame key constant for shooting
            field_width: Field width for boundary calculations (optional, uses ENV_CONSTANTS if not provided)
            field_height: Field height for boundary calculations (optional, uses ENV_CONSTANTS if not provided)
            speed: Player speed (optional, uses ENV_CONSTANTS if not provided)
            radius: Player radius (optional, uses ENV_CONSTANTS if not provided)
        """
        # Convert AEC team_id (0,1) to original team_id (1,2) for consistency
        self.team_id = team_id + 1 if team_id in [0, 1] else team_id
        self.player_id = player_id
        
        # Add global_id for compatibility with reward system
        self.global_id = team_id * 10 + player_id  # Simple unique identifier
        
        # Set player radius from parameter or use default
        self.radius = radius if radius is not None else ENV_CONSTANTS.PLAYER_RADIUS
        
        # Set original and current positions based on team and player ID
        if position is not None:
            self.original_position = position.copy()
        elif self.team_id == 1:
            if hasattr(ENV_CONSTANTS, 'team_1_positions') and player_id < len(ENV_CONSTANTS.team_1_positions):
                self.original_position = ENV_CONSTANTS.team_1_positions[player_id].copy()
            else:
                self.original_position = [200.0, 350.0]  # Default position
        else:
            if hasattr(ENV_CONSTANTS, 'team_2_positions') and player_id < len(ENV_CONSTANTS.team_2_positions):
                self.original_position = ENV_CONSTANTS.team_2_positions[player_id].copy()
            else:
                self.original_position = [1100.0, 350.0]  # Default position
        
        self.position = self.original_position.copy()
        
        # Movement keys
        self.up_key = up_key
        self.down_key = down_key
        if self.team_id == 1:
            self.left_key = left_key
            self.right_key = right_key
        else:
            self.left_key = right_key
            self.right_key = left_key
        
        self.shoot_key = shoot_key
        
        # Field dimensions for boundary calculations
        self.field_width = field_width if field_width is not None else ENV_CONSTANTS.WIDTH
        self.field_height = field_height if field_height is not None else ENV_CONSTANTS.HEIGHT
        
        # Player attributes from original
        self.speed = speed if speed is not None else ENV_CONSTANTS.PLAYER_SPEED
        self.velocity = [0, 0]  # (vx, vy)
        self.is_kick = False
        
        # Animation and Direction Attributes - always initialize for rendering when needed
        self.is_moving = False
        
        # Always try to load sprites if pygame is available
        try:
            # Load images exactly like original
            if self.team_id == 1:
                if hasattr(ENV_CONSTANTS, 'TEAM1_SPRITES') and player_id < len(ENV_CONSTANTS.TEAM1_SPRITES):
                    self.body_image = pygame.image.load(ENV_CONSTANTS.TEAM1_SPRITES[player_id])  # Remove .convert_alpha()
                    self.arm = pygame.image.load(ENV_CONSTANTS.TEAM1_ARMS[player_id])  # Remove .convert_alpha()
                    self.leg = pygame.image.load(ENV_CONSTANTS.TEAM1_LEGS[player_id])  # Remove .convert_alpha()
                else:
                    # Fallback sprites
                    self._load_fallback_sprites(1)
                self.direction = 0  # Degrees (0 = right, 90 = up)
            else:
                if hasattr(ENV_CONSTANTS, 'TEAM2_SPRITES') and player_id < len(ENV_CONSTANTS.TEAM2_SPRITES):
                    self.body_image = pygame.image.load(ENV_CONSTANTS.TEAM2_SPRITES[player_id])  # Remove .convert_alpha()
                    self.arm = pygame.image.load(ENV_CONSTANTS.TEAM2_ARMS[player_id])  # Remove .convert_alpha()
                    self.leg = pygame.image.load(ENV_CONSTANTS.TEAM2_LEGS[player_id])  # Remove .convert_alpha()
                else:
                    # Fallback sprites
                    self._load_fallback_sprites(2)
                self.direction = 180  # Degrees (0 = right, 90 = up)
            
            # Scale images to desired size using dynamic radius
            body_size = (self.radius * 1.5, self.radius * 2)
            leg_size = (self.radius * 1.5, self.radius * 0.75)
            arm_size = (self.radius * 3, self.radius * 0.75)
            
            self.body_image = pygame.transform.scale(self.body_image, body_size)
            self.arm = pygame.transform.scale(self.arm, arm_size)
            self.arm = pygame.transform.flip(self.arm, True, False)
            self.leg = pygame.transform.scale(self.leg, leg_size)
            
            # Initialize limb images for left and right arms and legs
            self.left_arm_image = self.arm
            self.right_arm_image = self.arm
            
            self.left_leg_image = self.leg
            self.right_leg_image = self.leg
            
            # Animation properties using dynamic radius
            self.arm_angle = 0
            self.arm_direction = 1  # 1 for forward, -1 for backward
            self.arm_min_angle = -40  # Minimum rotation angle for arms
            self.arm_max_angle = 40  # Maximum rotation angle for arms
            self.arm_speed = 9    # Degrees per frame for arms
            
            self.left_arm_angle = 0
            self.right_arm_angle = 0
            
            self.leg_direction = -1  # Opposite to arms for natural walking
            self.leg_size = 0
            self.left_leg_size = 0
            self.right_leg_size = 0
            self.leg_min_size = -20  # Minimum stretch value for legs
            self.leg_max_size = 20  # Maximum stretch value for legs
            self.leg_speed = 4.5     # Units per frame for legs
            self.leg_angle = 0
            
            # Constants to tweak limb positions relative to the body using dynamic radius
            self.ARM_OFFSET_X = -self.radius * 0.1  # Adjust as needed
            self.ARM_OFFSET_Y = self.radius * 0.6   # Adjust as needed
            
            self.LEG_OFFSET_X = self.radius * 3/4
            self.LEG_OFFSET_Y = self.radius / 2
            
            # Flip right leg image for mirroring
            self.right_leg_image = pygame.transform.flip(self.leg, True, False)
        except Exception as e:
            print(f"Warning: Failed to load Player sprites: {e}")
            # Initialize fallback attributes for compatibility
            self.is_moving = False
            self.direction = 0 if self.team_id == 1 else 180
            self.arm_angle = 0
            self.left_arm_angle = 0
            self.right_arm_angle = 0
            self.leg_size = 0
            self.left_leg_size = 0
            self.right_leg_size = 0
            self.leg_angle = 0
    
    def _load_fallback_sprites(self, team_id):
        """Load fallback sprites if original sprites are not found."""
        try:
            # Create simple colored rectangles as fallback using dynamic radius
            body_size = (int(self.radius * 1.5), int(self.radius * 2))
            arm_size = (int(self.radius * 3), int(self.radius * 0.75))
            leg_size = (int(self.radius * 1.5), int(self.radius * 0.75))
            
            if team_id == 1:
                color = (0, 0, 255)  # Blue
            else:
                color = (83, 160, 23)  # Green (matching ENV_CONSTANTS.GREEN)
            
            self.body_image = pygame.Surface(body_size, pygame.SRCALPHA)
            self.body_image.fill(color)
            
            self.arm = pygame.Surface(arm_size, pygame.SRCALPHA)
            self.arm.fill(color)
            
            self.leg = pygame.Surface(leg_size, pygame.SRCALPHA)
            self.leg.fill(color)
            
        except Exception as e:
            print(f"Warning: Failed to create fallback sprites: {e}")
            # Create minimal surfaces
            self.body_image = pygame.Surface((32, 32), pygame.SRCALPHA)
            self.arm = pygame.Surface((48, 12), pygame.SRCALPHA)
            self.leg = pygame.Surface((24, 12), pygame.SRCALPHA)
    
    def handle_movement(self, keys):
        """
        Handle movement from keyboard input exactly like original.
        
        Args:
            keys: Pygame key state
            
        Returns:
            The velocity of the player
        """
        arr = [
            1 if keys[self.up_key] else 0,
            1 if keys[self.down_key] else 0,
            1 if keys[self.left_key] else 0,
            1 if keys[self.right_key] else 0,
            1 if keys[self.shoot_key] else 0
        ]
        
        return self.move(arr)
    
    def move(self, arr):
        """
        Move the player based on the input array exactly like original.
        
        Args:
            arr: array of 4 or 5 elements representing either:
                 - [move_x, move_y, kick_power, kick_direction] (4 elements) 
                 - [up, down, left, right, shoot] (5 elements, original format)
            
        Returns:
            The velocity of the player
        """
        vx, vy = 0, 0
        
        if len(arr) == 4:
            # New format: [move_x, move_y, kick_power, kick_direction]
            move_x, move_y, kick_power, kick_direction = arr
            
            # Apply movement directly
            vx = move_x * self.speed
            vy = move_y * self.speed
            
            # Handle shooting
            if kick_power > 0.1:  # Minimum kick threshold
                self.is_kick = True
            else:
                self.is_kick = False
                
        elif len(arr) == 5:
            # Original format: [up, down, left, right, shoot]
            if self.team_id == 1:
                if arr[0] == 1:  # up
                    vy -= self.speed
                if arr[1] == 1:  # down
                    vy += self.speed
                if arr[2] == 1:  # left
                    vx -= self.speed
                if arr[3] == 1:  # right
                    vx += self.speed
            elif self.team_id == 2:
                if arr[1] == 1:  # up
                    vy -= self.speed
                if arr[0] == 1:  # down
                    vy += self.speed
                if arr[3] == 1:  # right
                    vx -= self.speed
                if arr[2] == 1:  # left
                    vx += self.speed
            
            if arr[4] == 1:  # shoot
                self.is_kick = True
            else:
                self.is_kick = False
        else:
            # Invalid array length, use no action
            vx, vy = 0, 0
            self.is_kick = False
        
        # Normalize velocity to maintain constant speed exactly like original
        if vx != 0 or vy != 0:
            magnitude = math.sqrt(vx ** 2 + vy ** 2)
            vx = (vx / magnitude) * self.speed
            vy = (vy / magnitude) * self.speed
        
        self.velocity = [vx, vy]
        
        # Always update movement state and direction for animations
        self.is_moving = (vx != 0 or vy != 0)
        if self.is_moving:
            self.direction = math.degrees(math.atan2(vy, vx))
            self.leg_angle = math.degrees(math.atan2(vy, vx))
        
        # Update position with boundary checks exactly like original
        new_x = self.position[0] + vx
        new_y = self.position[1] + vy
        
        in_left_goal = (new_x - self.radius <= ENV_CONSTANTS.GOAL_WIDTH) and \
                       (new_y >= ENV_CONSTANTS.HEIGHT / 2 - ENV_CONSTANTS.GOAL_HEIGHT / 2) and \
                       (new_y <= ENV_CONSTANTS.HEIGHT / 2 + ENV_CONSTANTS.GOAL_HEIGHT / 2)
        
        in_right_goal = (new_x + self.radius >= ENV_CONSTANTS.WIDTH - ENV_CONSTANTS.GOAL_WIDTH) and \
                        (new_y >= ENV_CONSTANTS.HEIGHT / 2 - ENV_CONSTANTS.GOAL_HEIGHT / 2) and \
                        (new_y <= ENV_CONSTANTS.HEIGHT / 2 + ENV_CONSTANTS.GOAL_HEIGHT / 2)
        
        # Calculate allowed vertical boundaries based on position exactly like original
        if in_left_goal or in_right_goal:
            # Calculate the vertical boundaries of the goal
            goal_top = (ENV_CONSTANTS.HEIGHT / 2) - (ENV_CONSTANTS.GOAL_HEIGHT / 2) + self.radius
            goal_bottom = (ENV_CONSTANTS.HEIGHT / 2) + (ENV_CONSTANTS.GOAL_HEIGHT / 2) - self.radius
            # Restrict vertical movement within the goal
            new_y = max(goal_top, min(new_y, goal_bottom))
            new_x = max(self.radius, min(new_x, ENV_CONSTANTS.WIDTH - self.radius))
        else:
            # Restrict vertical movement within the entire screen
            new_y = max(self.radius, min(new_y, ENV_CONSTANTS.HEIGHT - self.radius))
            new_x = max(ENV_CONSTANTS.PLAY_AREA_LEFT + self.radius, min(new_x, ENV_CONSTANTS.PLAY_AREA_RIGHT - self.radius))
        
        self.position = [new_x, new_y]
        
        return self.velocity
    
    def reset(self):
        """
        Reset the player's position to the original position and velocity exactly like original.
        """
        self.position = self.original_position.copy()
        self.velocity = [0, 0]
        
        if ENV_CONSTANTS.RENDER:
            self.is_moving = False
    
    # VISUALIZATION METHODS - exactly like original
    
    def update_animation(self):
        """Update player animation exactly like original."""
        # Skip the RENDER check - we should always render if this method is called
        
        if self.is_moving:
            # Update arm angle
            self.arm_angle += self.arm_direction * self.arm_speed
            if self.arm_angle > self.arm_max_angle or self.arm_angle < self.arm_min_angle:
                self.arm_direction *= -1  # Reverse direction
            
            # Set left and right arm angles to be opposite
            self.left_arm_angle = self.arm_angle
            self.right_arm_angle = -self.arm_angle
            
            # Update leg size for stretching
            self.leg_size += self.leg_direction * self.leg_speed
            if self.leg_size > self.leg_max_size or self.leg_size < self.leg_min_size:
                self.leg_direction *= -1  # Reverse direction
            
            # Set left and right leg sizes to be opposite
            self.left_leg_size = self.leg_size
            self.right_leg_size = -self.leg_size
        
        else:
            # Reset angles when not moving
            self.arm_angle = 0
            self.left_arm_angle = 0
            self.right_arm_angle = 0
            self.leg_size = 0
            self.left_leg_size = 0
            self.right_leg_size = 0
    
    def calculate_arm_position(self, offset_x, offset_y):
        """Calculate arm position exactly like original."""
        rad = math.radians(self.direction)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] + offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y
    
    def calculate_leg_position(self, offset_x, offset_y):
        """Calculate leg position exactly like original."""
        rad = math.radians(self.leg_angle)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] + offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y
    
    def draw(self, surface):
        """
        Draw the player with full animation exactly like original.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Skip the RENDER check - we should always render if this method is called
        import pygame
        
        # Check if sprites are properly loaded
        if not hasattr(self, 'body_image') or self.body_image is None:
            # Try to reload sprites one more time before falling back
            try:
                self._ensure_sprites_loaded()
            except:
                pass
        
        if not hasattr(self, 'body_image') or self.body_image is None:
            # Fallback to drawing a simple circle with correct team colors
            if self.team_id == 1:
                color = (0, 0, 255)  # Blue for team 1
            else:
                color = (83, 160, 23)  # Green for team 2 (matching ENV_CONSTANTS.GREEN)
            pygame.draw.circle(surface, color, (int(self.position[0]), int(self.position[1])), self.radius)
            return
        
        # Leg scaling factors based on leg sizes
        left_leg_scale_factor = max(0.1, abs(self.left_leg_size / 10))  # Ensure minimum scale
        right_leg_scale_factor = max(0.1, abs(self.right_leg_size / 10))  # Ensure minimum scale
        
        # Scale leg images (stretch along the y-axis)
        left_leg_scaled = pygame.transform.scale(
            self.left_leg_image,
            (int(self.left_leg_image.get_width() * left_leg_scale_factor), self.left_leg_image.get_height())
        )
        right_leg_scaled = pygame.transform.scale(
            self.right_leg_image,
            (int(self.right_leg_image.get_width() * right_leg_scale_factor), self.right_leg_image.get_height())
        )
        
        # Calculate leg positions
        if self.leg_size > 0:
            left_leg_pos = self.calculate_leg_position(self.LEG_OFFSET_X, self.LEG_OFFSET_Y)
            rotated_left_leg = pygame.transform.rotate(left_leg_scaled, -self.leg_angle)
            right_leg_pos = self.calculate_leg_position(-self.LEG_OFFSET_X, -self.LEG_OFFSET_Y)
            rotated_right_leg = pygame.transform.rotate(right_leg_scaled, -self.leg_angle)
        else:
            left_leg_pos = self.calculate_leg_position(-self.LEG_OFFSET_X, self.LEG_OFFSET_Y)
            rotated_left_leg = pygame.transform.rotate(left_leg_scaled, -self.leg_angle + 180)
            right_leg_pos = self.calculate_leg_position(self.LEG_OFFSET_X, -self.LEG_OFFSET_Y)
            rotated_right_leg = pygame.transform.rotate(right_leg_scaled, -self.leg_angle + 180)
        
        # Draw left leg
        left_leg_rect = rotated_left_leg.get_rect(center=left_leg_pos)
        surface.blit(rotated_left_leg, left_leg_rect)
        
        # Draw right leg
        right_leg_rect = rotated_right_leg.get_rect(center=right_leg_pos)
        surface.blit(rotated_right_leg, right_leg_rect)
        
        # Draw left arm
        left_arm_angle = -self.left_arm_angle + self.direction + 90
        rotated_left_arm = pygame.transform.rotate(self.left_arm_image, -left_arm_angle)
        left_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, -self.ARM_OFFSET_Y)
        left_arm_rect = rotated_left_arm.get_rect(center=left_arm_pos)
        surface.blit(rotated_left_arm, left_arm_rect)
        
        # Draw right arm
        right_arm_angle = self.right_arm_angle + self.direction - 90
        rotated_right_arm = pygame.transform.rotate(self.right_arm_image, -right_arm_angle)
        right_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, self.ARM_OFFSET_Y)
        right_arm_rect = rotated_right_arm.get_rect(center=right_arm_pos)
        surface.blit(rotated_right_arm, right_arm_rect)
        
        # Rotate body based on direction
        rotated_body = pygame.transform.rotate(self.body_image, -self.direction)
        body_rect = rotated_body.get_rect(center=self.position)
        surface.blit(rotated_body, body_rect)
        
        # Update animations
        self.update_animation()
    
    def _ensure_sprites_loaded(self):
        """Ensure sprites are properly loaded for this player."""
        import pygame
        
        # Determine which sprite files to use
        if self.team_id == 1:
            sprite_files = {
                'body': ENV_CONSTANTS.TEAM1_SPRITES[min(self.player_id, len(ENV_CONSTANTS.TEAM1_SPRITES) - 1)],
                'arm': ENV_CONSTANTS.TEAM1_ARMS[min(self.player_id, len(ENV_CONSTANTS.TEAM1_ARMS) - 1)],
                'leg': ENV_CONSTANTS.TEAM1_LEGS[min(self.player_id, len(ENV_CONSTANTS.TEAM1_LEGS) - 1)]
            }
        else:
            sprite_files = {
                'body': ENV_CONSTANTS.TEAM2_SPRITES[min(self.player_id, len(ENV_CONSTANTS.TEAM2_SPRITES) - 1)],
                'arm': ENV_CONSTANTS.TEAM2_ARMS[min(self.player_id, len(ENV_CONSTANTS.TEAM2_ARMS) - 1)],
                'leg': ENV_CONSTANTS.TEAM2_LEGS[min(self.player_id, len(ENV_CONSTANTS.TEAM2_LEGS) - 1)]
            }
        
        # Load sprite images
        self.body_image = pygame.image.load(sprite_files['body'])
        self.arm = pygame.image.load(sprite_files['arm'])
        self.leg = pygame.image.load(sprite_files['leg'])
        
        # Scale images to desired size using dynamic radius
        body_size = (self.radius * 1.5, self.radius * 2)
        leg_size = (self.radius * 1.5, self.radius * 0.75)
        arm_size = (self.radius * 3, self.radius * 0.75)
        
        self.body_image = pygame.transform.scale(self.body_image, body_size)
        self.arm = pygame.transform.scale(self.arm, arm_size)
        self.arm = pygame.transform.flip(self.arm, True, False)
        self.leg = pygame.transform.scale(self.leg, leg_size)
        
        # Initialize limb images for left and right arms and legs
        self.left_arm_image = self.arm
        self.right_arm_image = self.arm
        
        self.left_leg_image = self.leg
        self.right_leg_image = self.leg
        
        # Flip right leg image for mirroring
        self.right_leg_image = pygame.transform.flip(self.leg, True, False)
    
    # Properties for compatibility with current codebase
    @property
    def x(self) -> float:
        """Get x position."""
        return self.position[0]
    
    @x.setter
    def x(self, value: float) -> None:
        """Set x position."""
        self.position[0] = value
    
    @property
    def y(self) -> float:
        """Get y position."""
        return self.position[1]
    
    @y.setter
    def y(self, value: float) -> None:
        """Set y position."""
        self.position[1] = value
    
    @property
    def vx(self) -> float:
        """Get x velocity."""
        return self.velocity[0]
    
    @vx.setter
    def vx(self, value: float) -> None:
        """Set x velocity."""
        self.velocity[0] = value
    
    @property
    def vy(self) -> float:
        """Get y velocity."""
        return self.velocity[1]
    
    @vy.setter
    def vy(self, value: float) -> None:
        """Set y velocity."""
        self.velocity[1] = value
    
    def distance_to(self, other) -> float:
        """
        Calculate distance to another object (ball or player).
        
        Args:
            other: Object with x and y properties or position attribute
            
        Returns:
            Distance in pixels
        """
        if hasattr(other, 'position'):
            other_x, other_y = other.position[0], other.position[1]
        else:
            other_x, other_y = other.x, other.y
        
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)
    
    def distance_to_ball(self, ball: 'Ball') -> float:
        """
        Calculate distance to the ball.
        
        Args:
            ball: Ball object
            
        Returns:
            Distance in pixels
        """
        return self.distance_to(ball)
    
    def distance_to_goal(self, goal: 'Goal') -> float:
        """
        Calculate distance to a goal.
        
        Args:
            goal: Goal object
            
        Returns:
            Distance in pixels
        """
        return math.sqrt((self.x - goal.x)**2 + (self.y - goal.y)**2)
    
    def collides_with(self, other) -> bool:
        """
        Check if this player collides with another object.
        
        Args:
            other: Object with position and radius
            
        Returns:
            True if objects are colliding
        """
        distance = self.distance_to(other)
        return distance < (self.radius + other.radius)
    
    def reset_to_start(self) -> None:
        """Reset player to starting position (compatibility method)."""
        self.reset()


class Ball:
    """
    Represents the soccer ball with exact original physics and animation.
    """
    
    def __init__(
        self,
        position: List[float],
        radius: float = None,
        friction: float = None,
        max_speed: float = None,
        field_width: float = None,
        field_height: float = None
    ):
        """
        Initialize the ball exactly like the original Ball.py
        
        Args:
            position: [x, y] initial position of the ball
            radius: Ball radius (optional, uses ENV_CONSTANTS if not provided)
            friction: Ball friction coefficient (optional)
            max_speed: Maximum ball speed (optional)
            field_width: Field width for boundary calculations (optional, uses ENV_CONSTANTS if not provided)
            field_height: Field height for boundary calculations (optional, uses ENV_CONSTANTS if not provided)
        """
        self.position = position.copy()
        self.orginal_position = position.copy()  # Keep original typo for compatibility
        self.radius = radius if radius is not None else ENV_CONSTANTS.BALL_RADIUS
        self.velocity = [0, 0]  # (vx, vy)
        self.last_hit_player_id = None
        
        # Field dimensions for boundary calculations
        self.field_width = field_width if field_width is not None else ENV_CONSTANTS.WIDTH
        self.field_height = field_height if field_height is not None else ENV_CONSTANTS.HEIGHT
        
        # Physics properties
        self.friction = friction if friction is not None else ENV_CONSTANTS.BALL_FRICTION
        self.max_speed = max_speed if max_speed is not None else ENV_CONSTANTS.BALL_MAX_SPEED
        self.kick_speed = ENV_CONSTANTS.KICK_SPEED if hasattr(ENV_CONSTANTS, 'KICK_SPEED') else 15.5
        
        # Compatibility properties
        self.last_touched_by = None
        self.possession_player = None
        
        # Add angle attribute for rotating animation (required by original Ball.py)
        self.angle = 0
        
        # Always load sprites if pygame is available (for rendering when needed)
        try:
            # Load ball image exactly like original - check if sprite path exists
            if hasattr(ENV_CONSTANTS, 'BALL_SPRITE') and ENV_CONSTANTS.BALL_SPRITE:
                self.original_image = pygame.image.load(ENV_CONSTANTS.BALL_SPRITE)  # Remove .convert_alpha()
                self.original_image = pygame.transform.scale(self.original_image, (self.radius * 2, self.radius * 2))
            else:
                self.original_image = None
            self.angle = 0
        except (ImportError, pygame.error, FileNotFoundError, AttributeError) as e:
            self.original_image = None
            self.angle = 0
    
    def update_position(self):
        """
        Update the ball's position based on its velocity and handle boundary collisions exactly like original.
        """
        # Update position based on velocity
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]
        
        # Determine if the ball is in a goal exactly like original
        in_left_goal = (new_x - self.radius <= ENV_CONSTANTS.GOAL_WIDTH) and \
                      (new_y >= ENV_CONSTANTS.HEIGHT / 2 - ENV_CONSTANTS.GOAL_HEIGHT / 2) and \
                      (new_y <= ENV_CONSTANTS.HEIGHT / 2 + ENV_CONSTANTS.GOAL_HEIGHT / 2)
        
        in_right_goal = (new_x + self.radius >= ENV_CONSTANTS.WIDTH - ENV_CONSTANTS.GOAL_WIDTH) and \
                       (new_y >= ENV_CONSTANTS.HEIGHT / 2 - ENV_CONSTANTS.GOAL_HEIGHT / 2) and \
                       (new_y <= ENV_CONSTANTS.HEIGHT / 2 + ENV_CONSTANTS.GOAL_HEIGHT / 2)
        
        # Calculate allowed vertical boundaries based on position exactly like original
        if in_left_goal or in_right_goal:
            # Calculate the vertical boundaries of the goal
            goal_top = (ENV_CONSTANTS.HEIGHT / 2) - (ENV_CONSTANTS.GOAL_HEIGHT / 2) + self.radius
            goal_bottom = (ENV_CONSTANTS.HEIGHT / 2) + (ENV_CONSTANTS.GOAL_HEIGHT / 2) - self.radius
            # Restrict vertical movement within the goal
            new_y = max(goal_top, min(new_y, goal_bottom))
            new_x = max(self.radius, min(new_x, ENV_CONSTANTS.WIDTH - self.radius))
        else:
            # Bouncing off the play area walls exactly like original
            if new_x <= ENV_CONSTANTS.PLAY_AREA_LEFT + self.radius:
                new_x = ENV_CONSTANTS.PLAY_AREA_LEFT + 2 * self.radius
                self.velocity[0] = -self.velocity[0]
            elif new_x >= ENV_CONSTANTS.PLAY_AREA_RIGHT - self.radius:
                new_x = ENV_CONSTANTS.PLAY_AREA_RIGHT - 2 * self.radius
                self.velocity[0] = -self.velocity[0]
            
            if new_y <= self.radius:
                new_y = self.radius
                self.velocity[1] = -self.velocity[1]
            elif new_y >= ENV_CONSTANTS.HEIGHT - self.radius:
                new_y = ENV_CONSTANTS.HEIGHT - 2 * self.radius
                self.velocity[1] = -self.velocity[1]
            
            # Restrict vertical movement within the entire screen
            new_y = max(self.radius, min(new_y, ENV_CONSTANTS.HEIGHT - self.radius))
            new_x = max(ENV_CONSTANTS.PLAY_AREA_LEFT + self.radius, min(new_x, ENV_CONSTANTS.PLAY_AREA_RIGHT - self.radius))
        
        # Update position
        self.position = [new_x, new_y]
        
        # Apply friction to slow down the ball exactly like original
        self.velocity[0] *= self.friction
        self.velocity[1] *= self.friction
        
        # Cap the ball's speed exactly like original
        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed > self.max_speed:
            self.velocity[0] = (self.velocity[0] / speed) * self.max_speed
            self.velocity[1] = (self.velocity[1] / speed) * self.max_speed
    
    def reset(self):
        """
        Reset the ball's position to the original position and velocity exactly like original.
        """
        self.position = self.orginal_position.copy()
        self.velocity = [0, 0]
        self.last_hit_player_id = None
        self.last_touched_by = None
        self.possession_player = None
    
    def draw(self, surface):
        """
        Draw the ball on the given surface exactly like original.
        
        Args:
            surface: The Pygame surface to draw on
        """
        # Skip the RENDER check - we should always render if this method is called
        import pygame
        
        if not hasattr(self, 'original_image') or self.original_image is None:
            # Fallback to drawing a simple circle
            pygame.draw.circle(surface, (255, 255, 255), (int(self.position[0]), int(self.position[1])), self.radius)
            return
        
        # Calculate the rotation angle based on the velocity exactly like original
        self.angle += (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5 * 4
        # Rotate the original image
        image = pygame.transform.rotate(self.original_image, self.angle)
        # Get the new rect and center it
        new_rect = image.get_rect(center=(self.position[0], self.position[1]))
        # Draw the rotated image on the surface
        surface.blit(image, new_rect.topleft)
    
    # Properties for compatibility with current codebase
    @property
    def x(self) -> float:
        """Get x position."""
        return self.position[0]
    
    @x.setter
    def x(self, value: float) -> None:
        """Set x position."""
        self.position[0] = value
    
    @property
    def y(self) -> float:
        """Get y position."""
        return self.position[1]
    
    @y.setter
    def y(self, value: float) -> None:
        """Set y position."""
        self.position[1] = value
    
    @property
    def vx(self) -> float:
        """Get x velocity."""
        return self.velocity[0]
    
    @vx.setter
    def vx(self, value: float) -> None:
        """Set x velocity."""
        self.velocity[0] = value
    
    @property
    def vy(self) -> float:
        """Get y velocity."""
        return self.velocity[1]
    
    @vy.setter
    def vy(self, value: float) -> None:
        """Set y velocity."""
        self.velocity[1] = value
    
    def kick(self, direction: float, power: float, kicker: 'Player') -> None:
        """
        Apply a kick to the ball (for compatibility).
        
        Args:
            direction: Kick direction in radians
            power: Kick power (0.0 to 1.0)
            kicker: Player who kicked the ball
        """
        kick_speed = power * self.kick_speed
        
        self.velocity[0] = math.cos(direction) * kick_speed
        self.velocity[1] = math.sin(direction) * kick_speed
        
        # Cap velocity to max speed
        velocity_magnitude = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if velocity_magnitude > self.max_speed:
            scale = self.max_speed / velocity_magnitude
            self.velocity[0] *= scale
            self.velocity[1] *= scale
        
        self.last_touched_by = getattr(kicker, 'global_id', None)
        self.last_hit_player_id = (kicker.team_id, kicker.player_id)
        self.possession_player = None
    
    def distance_to(self, other) -> float:
        """
        Calculate distance to another entity.
        
        Args:
            other: Another entity with x, y attributes
            
        Returns:
            Distance in pixels
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def collides_with(self, other) -> bool:
        """
        Check collision with another circular entity.
        
        Args:
            other: Another entity with x, y, radius attributes
            
        Returns:
            True if colliding
        """
        distance = self.distance_to(other)
        return distance < (self.radius + other.radius)
    
    def reset_to_start(self) -> None:
        """Reset ball to starting position (compatibility method)."""
        self.reset()


class Goal:
    """
    Represents a goal area for scoring exactly like the original.
    """
    
    def __init__(
        self,
        position: List[float],
        width: float,
        height: float,
        invert_image: bool = False
    ):
        """
        Initialize a goal exactly like the original Goal.py
        
        Args:
            position: (x, y) position of the top-left corner of the goal
            width: Width of the goal area
            height: Height of the goal area
            invert_image: Whether to flip the goal image horizontally
        """
        self.position = position
        self.width = width
        self.height = height
        
        # Create boundary square within the goal exactly like original
        self.inner_height = self.height - 2 * ENV_CONSTANTS.PLAYER_HEIGHT
        self.inner_position = [
            self.position[0],
            self.position[1] + ENV_CONSTANTS.PLAYER_HEIGHT
        ]
        
        # Create pygame rect (pygame should already be imported at module level)
        self.inner_rect = pygame.Rect(
            self.inner_position[0],
            self.inner_position[1],
            self.width,
            self.inner_height
        )
        
        # Compatibility properties
        self.x, self.y = position[0] + width/2, position[1] + height/2  # Center for compatibility
        self.team_id = 0 if position[0] == 0 else 1  # Determine team based on position
        
        # Calculate goal boundaries for collision detection
        self.left = self.position[0]
        self.right = self.position[0] + width
        self.top = self.position[1]
        self.bottom = self.position[1] + height
        
        # Always try to load sprites if pygame is available
        try:
            self.color = ENV_CONSTANTS.GREEN if hasattr(ENV_CONSTANTS, 'GREEN') else (83, 160, 23)
            
            # Load goal image exactly like original - check if sprite path exists
            if hasattr(ENV_CONSTANTS, 'GOAL_SPRITE') and ENV_CONSTANTS.GOAL_SPRITE:
                original_image = pygame.image.load(ENV_CONSTANTS.GOAL_SPRITE)  # Remove .convert_alpha()
                if invert_image:
                    original_image = pygame.transform.flip(original_image, True, False)
                # Scale exactly like original: width * 1.1, height * 1.1
                self.original_image = pygame.transform.scale(original_image, (int(self.width * 1.1), int(self.height * 1.1)))
            else:
                self.original_image = None
        except Exception as e:
            self.original_image = None
            self.color = ENV_CONSTANTS.GREEN if hasattr(ENV_CONSTANTS, 'GREEN') else (83, 160, 23)
    
    def check_goal(self, ball):
        """
        Check if the entire ball has entered this goal area exactly like original.
        
        Args:
            ball: Instance of Ball
            
        Returns:
            True if the entire ball is in the goal, else False
        """
        # Define goal boundaries exactly like original
        left, top = self.inner_position
        right = left + self.width
        bottom = top + self.height
        
        # Ensure the entire ball is within the inner goal area exactly like original
        if self.position[0] == 0:
            # Left goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] + ball.radius) <= right and \
                       (ball.position[1] + 5 * ball.radius) >= top and \
                       (ball.position[1] - 5 * ball.radius) <= bottom
        else:
            # Right goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] - ball.radius) >= left and \
                       (ball.position[1] + 5 * ball.radius) >= top and \
                       (ball.position[1] - 5 * ball.radius) <= bottom
        
        return in_goal_x
    
    def draw(self, surface):
        """
        Draw the goal and its inner boundary on the given surface exactly like original.
        
        Args:
            surface: The Pygame surface to draw on
        """
        # Skip the RENDER check - we should always render if this method is called
        import pygame
        
        if not hasattr(self, 'original_image') or self.original_image is None:
            # Fallback to drawing a simple rectangle
            pygame.draw.rect(surface, (139, 69, 19), (self.position[0], self.position[1], self.width, self.height))  # Brown goal
            return
        
        # Draw outer goal exactly like original
        new_rect = self.original_image.get_rect(center=(self.position[0] + self.width/2, self.position[1] + self.height/2))
        # Draw the image on the surface
        surface.blit(self.original_image, new_rect.topleft)
    
    def draw_grass(self, surface):
        """
        Draw the grass area in the goal exactly like original.
        
        Args:
            surface: The Pygame surface to draw on
        """
        # Skip the RENDER check - we should always render if this method is called
        import pygame
        
        # Draw green goal area exactly like original
        grass_color = ENV_CONSTANTS.GREEN if hasattr(ENV_CONSTANTS, 'GREEN') else (83, 160, 23)
        pygame.draw.rect(surface, grass_color, (self.position[0], self.position[1], self.width, self.height))
    
    def contains_ball(self, ball: Ball) -> bool:
        """
        Check if the ball is inside the goal using original logic (compatibility method).
        
        Args:
            ball: Ball to check
            
        Returns:
            True if ball scored a goal
        """
        return self.check_goal(ball)
    
    def distance_to_ball(self, ball: Ball) -> float:
        """
        Calculate distance from goal center to ball.
        
        Args:
            ball: Ball object
            
        Returns:
            Distance in pixels
        """
        return math.sqrt((self.x - ball.x)**2 + (self.y - ball.y)**2) 