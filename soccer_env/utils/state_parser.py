"""
State Parser for converting game state to agent observations.

This module handles the conversion of the complete game state into
individual agent observations following the EXACT original format from the old codebase.
"""

import math
import numpy as np
from typing import List, Optional
from ..core.entities import GameState, Player, Ball, Goal


class StateParser:
    """
    Parses game state into observations for individual agents.
    Based EXACTLY on the original state parser design - no changes!
    """
    
    def __init__(self):
        """Initialize the state parser with the same field dimensions as the old system."""
        # Field dimensions for normalization - match old system exactly
        # From old system: PLAY_AREA_WIDTH and PLAY_AREA_HEIGHT
        self.field_width = 1170.0  # ENV_PARAMS.PLAY_AREA_RIGHT - ENV_PARAMS.PLAY_AREA_LEFT = 1300 - 130 = 1170
        self.field_height = 700.0  # ENV_PARAMS.PLAY_AREA_HEIGHT = 700
    
    def get_agent_observation(
        self, 
        game_state: GameState, 
        agent_id: int
    ) -> np.ndarray:
        """
        Get observation for a specific agent using the EXACT original state format.
        
        The state vector includes (EXACTLY like old system):
        - Vectors to players in own team (relative positions, normalized)
        - Vectors to players in opposing team (relative positions, normalized)  
        - Vector to ball (relative position, normalized)
        - Ball velocity (normalized)
        - Vectors to the two goals (opponent's goal first, then own goal, normalized)
        - Raycasts (north, south, east, west distances, normalized)
        
        NO EXTRA FEATURES - matches old system exactly!
        
        Args:
            game_state: Current game state
            agent_id: ID of the agent (0 to num_players-1)
            
        Returns:
            Observation vector as numpy array
        """
        if game_state is None or agent_id >= len(game_state.players):
            # Return zero observation if invalid
            return np.zeros(self.get_observation_size(len(game_state.players) if game_state else 4))
        
        player = game_state.players[agent_id]
        ball = game_state.ball
        goals = game_state.goals
        team_id = player.team_id
        
        # Precompute goal center points EXACTLY like old system (NORMALIZED!)
        goal_centers = {
            1: [
                (goals[0].x + goals[0].width / 2) / self.field_width,    # ✅ NORMALIZED!
                (goals[0].y + goals[0].height / 2) / self.field_height   # ✅ NORMALIZED!
            ],
            2: [
                (goals[1].x + goals[1].width / 2) / self.field_width if len(goals) > 1 else (goals[0].x + goals[0].width / 2) / self.field_width,    # ✅ NORMALIZED!
                (goals[1].y + goals[1].height / 2) / self.field_height if len(goals) > 1 else (goals[0].y + goals[0].height / 2) / self.field_height   # ✅ NORMALIZED!
            ]
        }
        
        # Get own goal area and normalize (EXACTLY like old system)
        own_goal = goals[0] if team_id == 1 else goals[1] if len(goals) > 1 else goals[0]
        own_goal_x = own_goal.x / self.field_width
        own_goal_y = own_goal.y / self.field_height
        own_goal_width = own_goal.width / self.field_width
        own_goal_height = own_goal.height / self.field_height
        
        # Get player's own position and normalize (EXACTLY like old system)
        px, py = player.x, player.y
        px /= self.field_width
        py /= self.field_height
        
        # Determine if we need to flip positions for consistency (team 2 gets flipped view)
        flip = (team_id == 2)
        
        # Function to flip positions if necessary (EXACTLY like old system)
        def flip_position(x, y):
            return 1.0 - x, y  # Flip horizontally in normalized coordinates
        
        # Flip player's position if necessary
        if flip:
            px, py = flip_position(px, py)
        
        # Vectors to teammates (relative positions, normalized) - EXACTLY like old system
        own_team_vectors = []
        for other_player in game_state.players:
            if other_player == player:
                continue  # Skip the player itself
            if other_player.team_id == team_id:
                ox, oy = other_player.x, other_player.y
                ox /= self.field_width
                oy /= self.field_height
                
                if flip:
                    ox, oy = flip_position(ox, oy)
                    rel_y = -oy + py
                else:
                    rel_y = oy - py
                
                rel_x = ox - px
                own_team_vectors.extend([rel_x, rel_y])
        
        # Vectors to opponents (relative positions, normalized) - EXACTLY like old system
        opponent_team_vectors = []
        for other_player in game_state.players:
            if other_player.team_id != team_id:
                ox, oy = other_player.x, other_player.y
                ox /= self.field_width
                oy /= self.field_height
                
                if flip:
                    ox, oy = flip_position(ox, oy)
                    rel_y = oy - py
                else:
                    rel_y = -oy + py
                
                rel_x = ox - px
                opponent_team_vectors.extend([rel_x, rel_y])
        
        # Vector to ball (relative position, normalized) - EXACTLY like old system
        ball_x, ball_y = ball.x, ball.y
        ball_x /= self.field_width
        ball_y /= self.field_height
        
        if flip:
            ball_x, ball_y = flip_position(ball_x, ball_y)
            rel_ball_y = -ball_y + py
        else:
            rel_ball_y = ball_y - py
        
        rel_ball_x = ball_x - px
        ball_vector = [rel_ball_x, rel_ball_y]
        
        # Ball velocity (normalized) - EXACTLY like old system
        max_ball_speed = 15.0  # ENV_PARAMS.BALL_MAX_SPEED from old system
        ball_vx, ball_vy = ball.vx, ball.vy
        ball_vx /= max_ball_speed
        ball_vy /= max_ball_speed
        
        if flip:
            ball_vx = -ball_vx  # Flip horizontal velocity
            ball_vy = -ball_vy  # Flip vertical velocity
        
        ball_velocity = [ball_vx, ball_vy]
        
        # Vectors to goals (opponent's goal first, then own goal, normalized) - EXACTLY like old system
        opponent_team_id = 2 if team_id == 1 else 1
        opponent_goal_center = goal_centers[opponent_team_id]
        own_goal_center = goal_centers[team_id]
        
        if flip:
            opponent_goal_center = flip_position(*opponent_goal_center)
            own_goal_center = flip_position(*own_goal_center)
        
        # Compute relative vectors to goals (EXACTLY like old system)
        rel_opponent_goal_x = opponent_goal_center[0] - px
        rel_own_goal_x = own_goal_center[0] - px
        
        if flip:
            rel_opponent_goal_y = -opponent_goal_center[1] + py
            rel_own_goal_y = -own_goal_center[1] + py
        else:
            rel_opponent_goal_y = opponent_goal_center[1] - py
            rel_own_goal_y = own_goal_center[1] - py
        
        goal_vectors = [rel_opponent_goal_x, rel_opponent_goal_y, rel_own_goal_x, rel_own_goal_y]
        
        # Raycasts (north, south, east, west distances) - EXACTLY like old system
        raycasts = self._calculate_raycasts_exact_old_system(px, py, flip, goals, team_id, own_goal_x, own_goal_y, own_goal_width, own_goal_height)
        
        # Assemble the state vector (EXACTLY like old system)
        state_vector = (
            own_team_vectors +
            opponent_team_vectors +
            ball_vector +
            ball_velocity +
            goal_vectors +
            raycasts
        )
        
        return np.array(state_vector, dtype=np.float32)
    
    def _calculate_raycasts_exact_old_system(
        self, 
        px: float, 
        py: float, 
        flip: bool, 
        goals: List[Goal], 
        team_id: int,
        own_goal_x: float,
        own_goal_y: float, 
        own_goal_width: float,
        own_goal_height: float
    ) -> List[float]:
        """
        Calculate raycast distances to walls/boundaries EXACTLY like the old system.
        This is a direct port of the old raycast logic.
        """
        # Flip player's x position if necessary (EXACTLY like old system)
        if flip:
            px = 1.0 - px
        
        # Get opponent goal area and normalize (EXACTLY like old system)
        opponent_goal = goals[1] if team_id == 1 else goals[0] if len(goals) > 1 else goals[0]
        opponent_goal_x = opponent_goal.x / self.field_width
        opponent_goal_y = opponent_goal.y / self.field_height
        opponent_goal_width = opponent_goal.width / self.field_width
        opponent_goal_height = opponent_goal.height / self.field_height
        
        # Determine player's area (EXACTLY like old system)
        in_own_goal_area = (own_goal_x <= px <= own_goal_x + own_goal_width and
                           own_goal_y <= py <= own_goal_y + own_goal_height)
        
        in_opponent_goal_area = (opponent_goal_x <= px <= opponent_goal_x + opponent_goal_width and
                                opponent_goal_y <= py <= opponent_goal_y + opponent_goal_height)
        
        in_middle_strip = (not in_own_goal_area and not in_opponent_goal_area and
                          own_goal_y <= py <= own_goal_y + own_goal_height)
        
        # Calculate raycast distances (EXACTLY like old system)
        if in_own_goal_area or in_opponent_goal_area:
            # Player is in goal area
            goal_x = own_goal_x if in_own_goal_area else opponent_goal_x
            goal_y = own_goal_y if in_own_goal_area else opponent_goal_y
            goal_height = own_goal_height if in_own_goal_area else opponent_goal_height
            
            # North (up): distance to top of goal
            north_distance = (goal_y + goal_height) - py
            # South (down): distance to bottom of goal
            south_distance = py - goal_y
            # East (right): distance calculation (EXACTLY like old system)
            east_distance = 1 - px + 2 * own_goal_width
            # West (left): distance calculation (EXACTLY like old system)
            west_distance = px
        else:
            # Player is in field
            north_distance = 1.0 - py
            south_distance = py
            
            if in_middle_strip:
                east_distance = 1.0 - px + 2 * own_goal_width
                west_distance = px
            else:
                east_distance = 1.0 - px + own_goal_width
                west_distance = px - own_goal_width
        
        # Apply flipping (EXACTLY like old system)
        if flip:
            north_distance, south_distance = south_distance, north_distance
            east_distance, west_distance = west_distance, east_distance
        
        return [north_distance, south_distance, east_distance, west_distance]
    
    def get_observation_size(self, total_players: int) -> int:
        """
        Get the size of the observation vector based on EXACT original format.
        
        Args:
            total_players: Total number of players in the game
            
        Returns:
            Size of observation vector
        """
        players_per_team = total_players // 2
        
        # EXACTLY like old system: STATE_SIZE = 12 + 2 * (2 * NUMBER_OF_PLAYERS - 1)
        # But let's calculate it step by step to match the old format:
        size = (
            2 * (players_per_team - 1) +  # Teammates (excluding self)
            2 * players_per_team +         # Opponents  
            2 +                            # Ball position
            2 +                            # Ball velocity
            4 +                            # Goal vectors (opponent + own)
            4                              # Raycasts (north, south, east, west)
        )
        
        return size 