"""
Reward Calculator for the soccer environment.

This module calculates rewards for agents based on game state changes,
directly ported from the original working RewardFunction.py
"""

import math
import numpy as np
from typing import List, Optional
from ..core.entities import GameState, Player, Ball


class RewardCalculator:
    """
    Calculates rewards for agents based on game state changes.
    EXACT PORT from the original working RewardFunction.py
    """
    
    def __init__(
        self,
        # Goal rewards (EXACT values from old/src/params.py)
        goal_reward: float = 400.0,                   # Old: GOAL_REWARD = 400
        goal_conceded_penalty: float = -400.0,       # Old: -GOAL_REWARD = -400
        
        # Dense reward coefficients (ORIGINAL training values - mostly disabled)
        player_to_ball_reward_coeff: float = 0.0,     # Old TRAINING: 0.0 (disabled)
        ball_to_goal_reward_coeff: float = 0.1,       # Old TRAINING: 0.1 (enabled)
        distance_to_teammates_coeff: float = 0.0,     # Old TRAINING: 0.0 (disabled)
        
        # NOT in original system - disabled
        ball_possession_reward: float = 0.0,          # NOT in original system - disabled
        out_of_bounds_penalty: float = 0.0,          # NOT in original system - disabled
        
        # Settings
        dense_rewards: bool = True,
        max_ball_distance: float = 150.0,
        max_goal_distance: float = 150.0,
    ):
        """
        Initialize the reward calculator exactly like the original.
        """
        self.goal_reward = goal_reward
        self.goal_conceded_penalty = goal_conceded_penalty
        self.player_to_ball_reward_coeff = player_to_ball_reward_coeff
        self.ball_to_goal_reward_coeff = ball_to_goal_reward_coeff
        self.distance_to_teammates_coeff = distance_to_teammates_coeff
        self.ball_possession_reward = ball_possession_reward
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.dense_rewards = dense_rewards
        self.max_ball_distance = max_ball_distance
        self.max_goal_distance = max_goal_distance
        
        # Distance reward cap from old system
        self.distance_reward_cap = 80.0
    
    def calculate_rewards(
        self, 
        current_state: GameState, 
        previous_state: Optional[GameState]
    ) -> List[float]:
        """
        Calculate rewards using EXACT logic from original calculate_rewards method.
        
        Args:
            current_state: Current game state
            previous_state: Previous game state (can be None for first step)
            
        Returns:
            List of rewards for each agent
        """
        if current_state is None:
            return [0.0] * 4  # Default for 4 players
        
        # Determine goal flags exactly like original Game.py
        goal1 = False  # Ball went into goal2 (right goal) -> team1 scored
        goal2 = False  # Ball went into goal1 (left goal) -> team2 scored
        
        if previous_state is not None and current_state.goal_scored:
            # EXACT mapping from original Game.py:
            # When ball goes into goal2 (right): score_team1 += 1; return (True, False) -> goal1=True
            # When ball goes into goal1 (left): score_team2 += 1; return (False, True) -> goal2=True
            #
            # In new system: team_0 = team1, team_1 = team2
            
            if current_state.team_0_score > previous_state.team_0_score:
                # team_0_score increased = score_team1 increased = ball went into goal2 (right)
                goal1 = True  # Original returns (True, False) -> goal1=True
            elif current_state.team_1_score > previous_state.team_1_score:
                # team_1_score increased = score_team2 increased = ball went into goal1 (left)  
                goal2 = True  # Original returns (False, True) -> goal2=True
        
        # Use EXACT reward calculation from original calculate_rewards
        rewards = []
        
        for player in current_state.players:
            total_reward = 0
            
            # Dense rewards (if enabled) - EXACT from original
            if self.dense_rewards:
                # Reward for player moving towards the ball
                total_reward += self.calculate_player_to_ball_velocity_reward(player, current_state)
                
                # Reward for ball moving towards the opponent's goal
                total_reward += self.calculate_ball_to_goal_velocity_reward(player, current_state)
                
                # Reward for player being far away from teammates
                total_reward += self.calculate_distance_to_players_reward(player, current_state)
            
            # Goal rewards - EXACT logic from original
            if goal1 and player.team_id == 1:
                # Reward team 1 for scoring
                total_reward += self.goal_reward
            if goal2 and player.team_id == 2:
                # Reward team 2 for scoring
                total_reward += self.goal_reward
            if goal1 and player.team_id == 2:
                # Penalize team 2 for opponent scoring
                total_reward -= self.goal_reward
            if goal2 and player.team_id == 1:
                # Penalize team 1 for opponent scoring
                total_reward -= self.goal_reward
            
            rewards.append(total_reward)
        
        return rewards
    
    def calculate_player_to_ball_velocity_reward(self, player: Player, game_state: GameState) -> float:
        """
        EXACT copy from original calculate_player_to_ball_velocity_reward
        """
        ball = game_state.ball
        
        # Vector from player to ball
        vector_to_ball = [
            ball.x - player.x,
            ball.y - player.y
        ]
        
        # Normalize vector_to_ball
        distance_to_ball = math.hypot(vector_to_ball[0], vector_to_ball[1])
        if distance_to_ball == 0:
            return 0  # Avoid division by zero; no reward
        
        unit_vector_to_ball = [vector_to_ball[0] / distance_to_ball, vector_to_ball[1] / distance_to_ball]
        
        # Player's velocity vector
        player_velocity = [player.vx, player.vy]
        
        # Compute dot product
        dot_product = player_velocity[0] * unit_vector_to_ball[0] + player_velocity[1] * unit_vector_to_ball[1]
        
        # Set the reward proportional to the dot product
        reward = dot_product * self.player_to_ball_reward_coeff
        
        return reward
    
    def calculate_ball_to_goal_velocity_reward(self, player: Player, game_state: GameState) -> float:
        """
        EXACT copy from original calculate_ball_to_goal_velocity_reward
        """
        ball = game_state.ball
        team_id = player.team_id
        
        # Precompute goal center points (EXACT like original)
        goal_centers = {
            1: [
                game_state.goals[0].x + game_state.goals[0].width / 2,
                game_state.goals[0].y + game_state.goals[0].height / 2
            ],
            2: [
                game_state.goals[1].x + game_state.goals[1].width / 2 if len(game_state.goals) > 1 else game_state.goals[0].x + game_state.goals[0].width / 2,
                game_state.goals[1].y + game_state.goals[1].height / 2 if len(game_state.goals) > 1 else game_state.goals[0].y + game_state.goals[0].height / 2
            ]
        }
        
        # Determine opponent's goal center - EXACT from original
        opponent_team_id = 2 if team_id == 1 else 1
        opponent_goal_center = goal_centers[opponent_team_id]
        
        # Vector from ball to opponent's goal - EXACT from original
        vector_to_goal = [
            opponent_goal_center[0] - ball.x,
            opponent_goal_center[1] - ball.y
        ]
        
        # Normalize vector_to_goal - EXACT from original
        distance_to_goal = math.hypot(vector_to_goal[0], vector_to_goal[1])
        if distance_to_goal == 0:
            return 0  # Ball is at the goal; no reward needed
        
        unit_vector_to_goal = [vector_to_goal[0] / distance_to_goal, vector_to_goal[1] / distance_to_goal]
        
        # Ball's velocity vector - EXACT from original
        ball_velocity = [ball.vx, ball.vy]
        
        # Compute dot product - EXACT from original
        dot_product = ball_velocity[0] * unit_vector_to_goal[0] + ball_velocity[1] * unit_vector_to_goal[1]
        
        # Set the reward proportional to the dot product - EXACT from original
        reward = dot_product * self.ball_to_goal_reward_coeff
        
        return reward
    
    def calculate_distance_to_players_reward(self, player: Player, game_state: GameState) -> float:
        """
        EXACT copy from original calculate_distance_to_players_reward
        """
        distance_reward = 0
        
        for other_player in game_state.players:  # within the same team
            if other_player != player and other_player.team_id == player.team_id:
                distance = math.hypot(player.x - other_player.x, player.y - other_player.y)
                distance_reward += distance
        
        distance_reward /= len(game_state.players) - 1
        
        return self.distance_to_teammates_coeff * min(distance_reward, self.distance_reward_cap)
    
    def calculate_team_rewards(
        self, 
        current_state: GameState, 
        previous_state: Optional[GameState]
    ) -> tuple[float, float]:
        """
        Calculate team-level rewards.
        
        Args:
            current_state: Current game state
            previous_state: Previous game state
            
        Returns:
            Tuple of (team_0_reward, team_1_reward)
        """
        individual_rewards = self.calculate_rewards(current_state, previous_state)
        
        team_0_reward = 0.0
        team_1_reward = 0.0
        team_0_count = 0
        team_1_count = 0
        
        for i, player in enumerate(current_state.players):
            player_team = player.team_id - 1 if player.team_id > 0 else player.team_id
            
            if player_team == 0:
                team_0_reward += individual_rewards[i]
                team_0_count += 1
            else:
                team_1_reward += individual_rewards[i]
                team_1_count += 1
        
        # Average rewards
        if team_0_count > 0:
            team_0_reward /= team_0_count
        if team_1_count > 0:
            team_1_reward /= team_1_count
        
        return team_0_reward, team_1_reward
    
    def get_reward_info(
        self, 
        current_state: GameState, 
        previous_state: Optional[GameState]
    ) -> dict:
        """
        Get detailed reward information for analysis.
        
        Args:
            current_state: Current game state
            previous_state: Previous game state
            
        Returns:
            Dictionary with reward breakdown
        """
        if current_state is None:
            return {}
        
        rewards = self.calculate_rewards(current_state, previous_state)
        team_0_reward, team_1_reward = self.calculate_team_rewards(current_state, previous_state)
        
        info = {
            'individual_rewards': rewards,
            'team_0_reward': team_0_reward,
            'team_1_reward': team_1_reward,
            'goal_scored': current_state.goal_scored,
            'ball_possession': current_state.ball_possession,
            'team_0_score': current_state.team_0_score,
            'team_1_score': current_state.team_1_score,
            }
        
        return info 