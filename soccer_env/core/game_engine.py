"""
Game Engine for the soccer environment.

This module contains the main game engine that manages the game state,
handles physics simulation, and implements game rules.
"""

import math
import random
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
import sys
from pathlib import Path
from ..envs.constants import ENV_CONSTANTS
from .entities import Player, Ball, Goal, GameState
from .physics import PhysicsEngine


class GameEngine:
    """
    Main game engine that orchestrates the soccer simulation.
    """
    
    def __init__(
        self,
        field_width: float = None,
        field_height: float = None,
        num_players_per_team: int = 2,
        ball_friction: float = None,
        player_speed: float = None,
        game_duration: float = None,
        simulation_mode: bool = False,
        preset_config: dict = None,
    ):
        """
        Initialize the game engine using environment constants.
        
        Args:
            simulation_mode: If True, use fixed time steps for training (instead of real-time)
            preset_config: Dictionary of preset configuration values to override defaults
        """
        # Use environment constants with fallbacks
        self.field_width = field_width if field_width is not None else ENV_CONSTANTS.WIDTH
        self.field_height = field_height if field_height is not None else ENV_CONSTANTS.HEIGHT
        self.num_players_per_team = num_players_per_team
        self.total_players = num_players_per_team * 2
        self.game_duration = game_duration if game_duration is not None else getattr(ENV_CONSTANTS, 'DEFAULT_GAME_DURATION', 120)
        self.player_speed = player_speed if player_speed is not None else ENV_CONSTANTS.PLAYER_SPEED
        self.simulation_mode = simulation_mode
        
        # Store preset config for use in entity creation
        self.preset_config = preset_config or {}
        
        # Get values from preset or use defaults
        self.player_radius = self.preset_config.get('PLAYER_RADIUS', ENV_CONSTANTS.PLAYER_RADIUS)
        self.ball_radius = self.preset_config.get('BALL_RADIUS', ENV_CONSTANTS.BALL_RADIUS)
        self.ball_max_speed = self.preset_config.get('BALL_MAX_SPEED', ENV_CONSTANTS.BALL_MAX_SPEED)
        self.kick_speed = self.preset_config.get('KICK_SPEED', ENV_CONSTANTS.KICK_SPEED)
        self.ball_collision_restitution = self.preset_config.get('BALL_COLLISION_RESTITUTION', ENV_CONSTANTS.BALL_COLLISION_RESTITUTION)
        self.player_collision_restitution = self.preset_config.get('PLAYER_COLLISION_RESTITUTION', ENV_CONSTANTS.PLAYER_COLLISION_RESTITUTION)
        
        # Physics properties from preset or parameters
        self.ball_friction = ball_friction if ball_friction is not None else self.preset_config.get('BALL_FRICTION', ENV_CONSTANTS.BALL_FRICTION)
        
        # Initialize physics engine
        self.physics = PhysicsEngine(
            field_width=self.field_width,
            field_height=self.field_height,
            ball_friction=self.ball_friction
        )
        
        # Game state
        self.players: List[Player] = []
        self.ball: Optional[Ball] = None
        self.goals: List[Goal] = []
        self.current_state: Optional[GameState] = None
        self.previous_state: Optional[GameState] = None
        
        # Agent actions
        self.agent_actions: Dict[int, List[float]] = {}
        
        # Game statistics
        self.team_0_score = 0
        self.team_1_score = 0
        self.episode_step = 0
        self.time_remaining = self.game_duration
        self.goal_scored_this_step = False
        self.last_goal_scorer = None
        
        # REAL-TIME TRACKING: Track actual elapsed time for consistent game timing
        self.last_update_time = None  # Time of last game update
        self.target_fps = ENV_CONSTANTS.FPS  # Target frame rate for fallback
        
        # Initialize game entities
        self._initialize_entities()
        
    def _initialize_entities(self) -> None:
        """Initialize players, ball, and goals using environment constants."""
        # Clear existing entities
        self.players.clear()
        self.goals.clear()
        
        # Get team positions from constants
        if hasattr(ENV_CONSTANTS, 'calculate_team_positions'):
            team_1_positions, team_2_positions = ENV_CONSTANTS.calculate_team_positions(self.num_players_per_team)
        elif hasattr(ENV_CONSTANTS, 'team_1_positions') and hasattr(ENV_CONSTANTS, 'team_2_positions'):
            # Use static position lists if available
            team_1_positions = ENV_CONSTANTS.team_1_positions[:self.num_players_per_team]
            team_2_positions = ENV_CONSTANTS.team_2_positions[:self.num_players_per_team]
        else:
            # Use the position calculation algorithm from the constants
            team_1_positions, team_2_positions = self._calculate_positions_like_original()
        
        # Create players using calculated positions
        for i in range(self.num_players_per_team):
            # Team 1 (left side, team_id=0)
            team1_pos = team_1_positions[i] if i < len(team_1_positions) else [100, 200 + i * 100]
            player1 = Player(
                team_id=0,
                player_id=i,
                position=team1_pos,
                field_width=ENV_CONSTANTS.WIDTH,
                field_height=ENV_CONSTANTS.HEIGHT,
                speed=self.player_speed,
                radius=self.player_radius  # Use preset radius
            )
            self.players.append(player1)
            
        for i in range(self.num_players_per_team):
            # Team 2 (right side, team_id=1)
            team2_pos = team_2_positions[i] if i < len(team_2_positions) else [1200, 200 + i * 100]
            player2 = Player(
                team_id=1,
                player_id=i,
                position=team2_pos,
                field_width=ENV_CONSTANTS.WIDTH,
                field_height=ENV_CONSTANTS.HEIGHT,
                speed=self.player_speed,
                radius=self.player_radius  # Use preset radius
            )
            self.players.append(player2)
        
        # Create ball at center with small random offset
        random_x = random.random() * 5 - 2.5
        random_y = random.random() * 5 - 2.5
        ball_position = [ENV_CONSTANTS.WIDTH // 2 + random_y, ENV_CONSTANTS.HEIGHT // 2 + random_x]
        self.ball = Ball(
            position=ball_position,
            field_width=ENV_CONSTANTS.WIDTH,
            field_height=ENV_CONSTANTS.HEIGHT,
            radius=self.ball_radius,  # Use preset radius
            max_speed=self.ball_max_speed,  # Use preset max speed
            friction=self.ball_friction  # This already comes from preset or parameter
        )
        
        # Create goals using environment constants exactly like original
        left_goal_pos = [0, (ENV_CONSTANTS.HEIGHT / 2) - (ENV_CONSTANTS.GOAL_HEIGHT / 2)]
        right_goal_pos = [ENV_CONSTANTS.WIDTH - ENV_CONSTANTS.GOAL_WIDTH, (ENV_CONSTANTS.HEIGHT / 2) - (ENV_CONSTANTS.GOAL_HEIGHT / 2)]
        
        # Left goal exactly like original
        self.goals.append(Goal(
            left_goal_pos, 
            ENV_CONSTANTS.GOAL_WIDTH, 
            ENV_CONSTANTS.GOAL_HEIGHT, 
            invert_image=True
        ))
        
        # Right goal exactly like original
        self.goals.append(Goal(
            right_goal_pos, 
            ENV_CONSTANTS.GOAL_WIDTH, 
            ENV_CONSTANTS.GOAL_HEIGHT,
            invert_image=False
        ))
        
        # Initialize game state
        self._update_game_state()
    
    def _calculate_positions_like_original(self):
        """
        Calculate player positions using the same algorithm as the original implementation.
        Supports team sizes from 0 to 10 players.
        """
        # Initialize position lists for both teams
        team_1_positions = []
        team_2_positions = []
        
        # Use NUMBER_OF_PLAYERS (total players per team) like the original
        NUMBER_OF_PLAYERS = self.num_players_per_team
        
        # Calculate the number of players for each column (exactly like original)
        back_column_players = min(3, NUMBER_OF_PLAYERS)
        middle_column_players = min(4, max(0, NUMBER_OF_PLAYERS - back_column_players))
        front_column_players = max(0, NUMBER_OF_PLAYERS - back_column_players - middle_column_players)

        # Calculate the vertical spacing between players (exactly like original)
        vertical_spacing = ENV_CONSTANTS.HEIGHT / 5

        # Function to calculate positions for a column (exactly like original)
        def calculate_column_positions(start_x, num_players, offset):
            positions = []
            for i in range(num_players):
                y_position = offset + (i + 1) * vertical_spacing
                positions.append([start_x, y_position])
            return positions

        # Calculate the offset to center the players vertically (exactly like original)
        offset_1 = ENV_CONSTANTS.HEIGHT / 2 - (back_column_players + 1) * vertical_spacing / 2
        offset_2 = ENV_CONSTANTS.HEIGHT / 2 - (middle_column_players + 1) * vertical_spacing / 2
        offset_3 = ENV_CONSTANTS.HEIGHT / 2 - (front_column_players + 1) * vertical_spacing / 2

        # Calculate positions for each column (exactly like original)
        # Note: the original has a space in "self. PLAY_AREA_WIDTH" which I'll replicate
        PLAY_AREA_WIDTH = ENV_CONSTANTS.PLAY_AREA_RIGHT - ENV_CONSTANTS.PLAY_AREA_LEFT
        back_column_x = PLAY_AREA_WIDTH / 8 + ENV_CONSTANTS.PLAY_AREA_LEFT
        middle_column_x = PLAY_AREA_WIDTH / 4 + ENV_CONSTANTS.PLAY_AREA_LEFT
        front_column_x = PLAY_AREA_WIDTH * 3 / 8 + ENV_CONSTANTS.PLAY_AREA_LEFT

        team_1_positions.extend(calculate_column_positions(back_column_x, back_column_players, offset_1))
        team_1_positions.extend(calculate_column_positions(middle_column_x, middle_column_players, offset_2))
        team_1_positions.extend(calculate_column_positions(front_column_x, front_column_players, offset_3))
        
        # Mirror positions for Team 2 across the vertical center line of the pitch (exactly like original)
        team_2_positions = [
            [ENV_CONSTANTS.PLAY_AREA_LEFT + PLAY_AREA_WIDTH - (pos[0] - ENV_CONSTANTS.PLAY_AREA_LEFT), pos[1]]
            for pos in team_1_positions
        ]
        
        return team_1_positions, team_2_positions
    
    def reset(
        self, 
        randomize_positions: bool = False,
        random_state: Optional[np.random.Generator] = None
    ) -> GameState:
        """
        Reset the game to initial state.
        
        Args:
            randomize_positions: Whether to randomize starting positions
            random_state: Random number generator for reproducibility
            
        Returns:
            Initial game state
        """
        # Reset scores and time
        self.team_0_score = 0
        self.team_1_score = 0
        self.episode_step = 0
        self.time_remaining = self.game_duration
        self.goal_scored_this_step = False
        self.last_goal_scorer = None
        
        # RESET REAL-TIME TRACKING: Initialize timing for consistent game progression
        self.last_update_time = None  # Will be set on first step call
        
        # Reset entities to starting positions
        for player in self.players:
            if randomize_positions and random_state is not None:
                # Randomize positions within team areas
                if player.team_id == 0:
                    x = random_state.uniform(5, self.field_width * 0.4)
                else:
                    x = random_state.uniform(self.field_width * 0.6, self.field_width - 5)
                y = random_state.uniform(5, self.field_height - 5)
                player.position = [x, y]
                player.start_x, player.start_y = x, y
            else:
                player.reset_to_start()
        
        # Reset ball
        if randomize_positions and random_state is not None:
            # Randomize ball position in center area
            x = random_state.uniform(self.field_width * 0.3, self.field_width * 0.7)
            y = random_state.uniform(self.field_height * 0.3, self.field_height * 0.7)
            self.ball.position = [x, y]
            self.ball.start_x, self.ball.start_y = x, y
        else:
            self.ball.reset_to_start()
        
        # Clear agent actions
        self.agent_actions.clear()
        
        # Update game state
        self._update_game_state()
        
        return self.current_state
    
    def set_agent_action(self, agent_id: int, action: List[float]) -> None:
        """
        Set action for a specific agent.
        
        Args:
            agent_id: Agent identifier
            action: Action values [move_x, move_y, kick_power, kick_direction]
        """
        self.agent_actions[agent_id] = action
    
    def step(self, update_time: bool = True) -> None:
        """
        Execute one simulation step with real-time based timing.
        
        Args:
            update_time: Whether to update the game timer (should be True only once per complete AEC cycle)
        """
        # Store current state as previous
        self.previous_state = self._copy_current_state()
        
        # Process agent actions for all players
        self._process_agent_actions()
        
        # Update ball's movement exactly like original
        self.ball.update_position()
        
        # Handle collisions exactly like original
        self._handle_collisions()
        
        # Check for goals exactly like original
        goal1, goal2 = self._check_goals()
        
        # Update game state - exactly like original Game.py
        self.goal_scored_this_step = goal1 or goal2
        if goal1:
            # Ball went into left goal (goal1) -> team2 scores -> team_1_score increases
            self.team_1_score += 1
            self.last_goal_scorer = 1
            self._reset_positions()  # Immediate reset like original
        elif goal2:
            # Ball went into right goal (goal2) -> team1 scores -> team_0_score increases
            self.team_0_score += 1
            self.last_goal_scorer = 0
            self._reset_positions()  # Immediate reset like original
        
        # TIMING UPDATE: Use simulation mode or real-time based timing
        if update_time:
            if self.simulation_mode:
                # Fixed time step mode for training/simulation
                delta_time = 1.0 / self.target_fps
                self.time_remaining -= delta_time
                self.episode_step += 1
            else:
                # Real-time mode for human gameplay
                current_time = time.time()
                
                if self.last_update_time is None:
                    # First update - use target FPS as fallback
                    delta_time = 1.0 / self.target_fps
                    self.last_update_time = current_time
                else:
                    # Use actual elapsed time since last update
                    delta_time = current_time - self.last_update_time
                    self.last_update_time = current_time
                    
                    # Clamp delta_time to reasonable bounds to prevent issues
                    # (e.g., if game is paused or system lags heavily)
                    min_delta = 1.0 / (self.target_fps * 2)  # Half target FPS = max 2x speed
                    max_delta = 1.0 / (self.target_fps / 4)  # Quarter target FPS = max 4x slowdown
                    delta_time = max(min_delta, min(max_delta, delta_time))
                
                self.time_remaining -= delta_time
                self.episode_step += 1
        
        # Clear agent actions for next step
        self.agent_actions.clear()
        
        # Update current game state
        self._update_game_state()
    
    def _process_agent_actions(self) -> None:
        """Apply actions for all agents."""
        for i, player in enumerate(self.players):
            if i in self.agent_actions:
                action = self.agent_actions[i]
                
                # Apply movement
                player.move(action)
                
                # Handle ball interactions
                if len(action) >= 4 and self.ball.collides_with(player):
                    kick_power = action[2]
                    kick_direction = action[3]
                    
                    if kick_power > 0.1:  # Minimum kick threshold
                        self.ball.kick(kick_direction, kick_power, player)
    
    def _handle_collisions(self):
        """
        Handle collision between players and the ball, and between players themselves exactly like original.
        """
        for player in self.players:
            # Check collision between player and ball exactly like original
            self._check_ball_collision(player, self.ball)
            
            # Check collision between player and player exactly like original
            for other_player in self.players:
                if player == other_player:
                    continue
                self._check_player_collision(player, other_player)
    
    def _check_ball_collision(self, player: Player, ball: Ball):
        """
        Check and handle collision between a player and the ball exactly like original.
        
        Args:
            player: Instance of Player
            ball: Instance of Ball
        """
        distance = math.hypot(ball.position[0] - player.position[0],
                              ball.position[1] - player.position[1])
        min_distance = player.radius + ball.radius
        
        if distance < min_distance:
            # Calculate the overlap
            overlap = min_distance - distance
            
            # Calculate the direction from player to ball
            if distance == 0:
                # Prevent division by zero; assign arbitrary direction
                direction_x, direction_y = 1, 0
            else:
                direction_x = (ball.position[0] - player.position[0]) / distance
                direction_y = (ball.position[1] - player.position[1]) / distance
            
            # Adjust ball position to prevent sticking exactly like original
            ball.position[0] += direction_x * 2 * overlap
            ball.position[1] += direction_y * 2 * overlap
            
            # Calculate relative velocity
            relative_velocity_x = ball.velocity[0] - player.velocity[0]
            relative_velocity_y = ball.velocity[1] - player.velocity[1]
            
            # Calculate the velocity along the direction of collision
            velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y
            
            if velocity_along_normal > 0:
                return  # They are moving away from each other
            
            # Define restitution (elasticity) from preset
            restitution = self.ball_collision_restitution
            
            # Ball power from constants
            ball_power = ENV_CONSTANTS.BALL_POWER if hasattr(ENV_CONSTANTS, 'BALL_POWER') else 0.95
            kick_speed = self.kick_speed  # Use preset kick speed
            
            # Update ball's velocity based on collision exactly like original
            base_velocity = (1 + restitution) * velocity_along_normal * ball_power
            base_velocity = -kick_speed if player.is_kick else base_velocity
            
            ball.velocity[0] -= base_velocity * direction_x
            ball.velocity[1] -= base_velocity * direction_y
            
            ball.last_hit_player_id = (player.team_id + 1, player.player_id)  # Convert to original format (1,2)
    
    def _check_player_collision(self, player1: Player, player2: Player):
        """
        Check and handle collision between two players exactly like original.
        
        Args:
            player1: Instance of Player
            player2: Instance of Player
        """
        distance = math.hypot(player2.position[0] - player1.position[0],
                              player2.position[1] - player1.position[1])
        min_distance = player1.radius + player2.radius
        
        if distance < min_distance:
            # Calculate the overlap
            overlap = min_distance - distance
            
            # Calculate the direction from player1 to player2
            if distance == 0:
                # Prevent division by zero; assign arbitrary direction
                direction_x, direction_y = 1, 0
            else:
                direction_x = (player2.position[0] - player1.position[0]) / distance
                direction_y = (player2.position[1] - player1.position[1]) / distance
            
            # Adjust positions to prevent sticking (push players apart equally) exactly like original
            player1.position[0] -= direction_x * (overlap / 2)
            player1.position[1] -= direction_y * (overlap / 2)
            player2.position[0] += direction_x * (overlap / 2)
            player2.position[1] += direction_y * (overlap / 2)
            
            # Note: Original doesn't implement velocity changes for player-player collisions
            # The original Game.py has the velocity change code commented out
    
    def _check_goals(self) -> Tuple[bool, bool]:
        """
        Check if ball has entered any goal exactly like original.
        
        Returns:
            Tuple of (goal1_scored, goal2_scored) where goal1 is left goal, goal2 is right goal
        """
        goal1_scored = False  # Left goal (Team 1 scores against Team 2)
        goal2_scored = False  # Right goal (Team 2 scores against Team 1)
        
        # Check left goal (goal1) - Team 1 scores here
        if self.goals[0].check_goal(self.ball):
            goal1_scored = True
        
        # Check right goal (goal2) - Team 2 scores here
        if self.goals[1].check_goal(self.ball):
            goal2_scored = True
        
        return goal1_scored, goal2_scored
    
    def _reset_positions(self) -> None:
        """Reset all entities to starting positions after a goal exactly like original."""
        # Reset ball to center with small random offset exactly like original
        random_x = random.random() * 5 - 2.5
        random_y = random.random() * 5 - 2.5
        self.ball.position = [ENV_CONSTANTS.WIDTH // 2 + random_y, ENV_CONSTANTS.HEIGHT // 2 + random_x]
        self.ball.velocity = [0, 0]
        self.ball.angle = 0
        
        # Reset players to original positions exactly like original
        for player in self.players:
            player.reset_to_start()
    
    def _update_game_state(self) -> None:
        """Update the current game state."""
        # Determine ball possession
        ball_possession = None
        min_distance = float('inf')
        
        for i, player in enumerate(self.players):
            distance = player.distance_to(self.ball)
            if distance < min_distance:
                min_distance = distance
                if distance < player.radius + self.ball.radius + 5:  # Possession threshold
                    ball_possession = i
        
        # Create game state
        self.current_state = GameState(
            players=self.players.copy(),
            ball=self.ball,
            goals=self.goals.copy(),
            team_0_score=self.team_0_score,
            team_1_score=self.team_1_score,
            episode_step=self.episode_step,
            time_remaining=self.time_remaining,
            goal_scored=self.goal_scored_this_step,
            ball_possession=ball_possession,
            last_goal_scorer=self.last_goal_scorer,
        )
    
    def _copy_current_state(self) -> GameState:
        """Create a copy of the current game state."""
        if self.current_state is None:
            return None
            
        # Create shallow copy of players (no need to recreate Player objects with sprite loading)
        players_copy = []
        for player in self.players:
            # Create a simple copy without triggering sprite loading
            import copy
            player_copy = copy.copy(player)
            # Update position and velocity to current values
            player_copy.position = [player.x, player.y]
            player_copy.velocity = [player.vx, player.vy]
            players_copy.append(player_copy)
        
        # Create copy of ball (lightweight, no sprite loading issues)
        ball_copy = Ball(position=[self.ball.x, self.ball.y], field_width=ENV_CONSTANTS.WIDTH, field_height=ENV_CONSTANTS.HEIGHT)
        ball_copy.vx, ball_copy.vy = self.ball.vx, self.ball.vy
        ball_copy.angle = self.ball.angle
        
        # Goals don't change so we can reuse them
        return GameState(
            players=players_copy,
            ball=ball_copy,
            goals=self.goals,
            team_0_score=self.team_0_score,
            team_1_score=self.team_1_score,
            episode_step=self.episode_step,
            time_remaining=self.time_remaining,
            goal_scored=self.goal_scored_this_step,
            last_goal_scorer=self.last_goal_scorer
        )
    
    def get_game_state(self) -> Optional[GameState]:
        """Get the current game state."""
        return self.current_state
    
    def get_previous_state(self) -> Optional[GameState]:
        """Get the previous game state."""
        return self.previous_state 