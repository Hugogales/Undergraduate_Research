"""
Action Parser for converting agent actions to game actions.

This module handles the conversion of raw agent actions (continuous or discrete)
into the appropriate format for the game engine.
"""

import math
import numpy as np
from typing import List, Union
from gymnasium import spaces


class ActionParser:
    """
    Parses agent actions into game-compatible format.
    """
    
    def __init__(self, continuous: bool = True):
        """
        Initialize the action parser.
        
        Args:
            continuous: Whether to use continuous or discrete action space
        """
        self.continuous = continuous
        
        # Discrete action mappings
        if not continuous:
            self._setup_discrete_actions()
    
    def _setup_discrete_actions(self) -> None:
        """Setup discrete action mappings."""
        # Movement directions (8 directions + no movement)
        self.move_actions = [
            [0, 0],      # 0: No movement
            [1, 0],      # 1: Right
            [1, 1],      # 2: Right-Down
            [0, 1],      # 3: Down
            [-1, 1],     # 4: Left-Down
            [-1, 0],     # 5: Left
            [-1, -1],    # 6: Left-Up
            [0, -1],     # 7: Up
            [1, -1],     # 8: Right-Up
        ]
        
        # Kick actions (8 directions + no kick)
        self.kick_actions = [
            [0, 0],      # 0: No kick
            [1, 0],      # 1: Kick Right
            [1, 1],      # 2: Kick Right-Down
            [0, 1],      # 3: Kick Down
            [-1, 1],     # 4: Kick Left-Down
            [-1, 0],     # 5: Kick Left
            [-1, -1],    # 6: Kick Left-Up
            [0, -1],     # 7: Kick Up
            [1, -1],     # 8: Kick Right-Up
        ]
        
        # Combined action space: 9 move actions × 9 kick actions = 81 total
        # But we'll use a simplified version: 18 actions (matching old system)
        # Actions 0-8: Movement only (no kick)
        # Actions 9-17: Movement + kick in same direction
        # Action 17: No movement + kick (missing from current implementation)
        self.discrete_action_map = {}
        
        # Movement only actions (0-8)
        for i in range(9):
            self.discrete_action_map[i] = {
                'move': self.move_actions[i],
                'kick_power': 0.0,
                'kick_direction': 0.0
            }
        
        # Movement + kick actions (9-16)
        for i in range(8):
            action_id = 9 + i
            direction = self.move_actions[i + 1]  # Skip no-movement
            self.discrete_action_map[action_id] = {
                'move': direction,
                'kick_power': 0.8,  # Default kick power
                'kick_direction': math.atan2(direction[1], direction[0])
            }
        
        # Action 17: No movement + kick (matching old system exactly)
        self.discrete_action_map[17] = {
            'move': [0, 0],  # No movement
            'kick_power': 0.8,  # Default kick power
            'kick_direction': 0.0  # Kick forward
        }
    
    def parse_action(self, action: Union[int, List[float], np.ndarray], agent_id: int) -> List[float]:
        """
        Parse agent action into game-compatible format.
        
        Args:
            action: Raw action from agent (discrete int or continuous array)
            agent_id: ID of the agent performing the action
            
        Returns:
            Parsed action as [move_x, move_y, kick_power, kick_direction]
        """
        if self.continuous:
            return self._parse_continuous_action(action)
        else:
            return self._parse_discrete_action(action)
    
    def _parse_continuous_action(self, action: Union[List[float], np.ndarray]) -> List[float]:
        """
        Parse continuous action.
        
        Args:
            action: Continuous action array [move_x, move_y, kick_power, kick_direction]
            
        Returns:
            Parsed action as [move_x, move_y, kick_power, kick_direction]
        """
        if isinstance(action, (list, np.ndarray)) and len(action) >= 4:
            move_x = np.clip(action[0], -1.0, 1.0)
            move_y = np.clip(action[1], -1.0, 1.0)
            kick_power = np.clip(action[2], 0.0, 1.0)
            kick_direction = np.clip(action[3], -math.pi, math.pi)
            
            return [move_x, move_y, kick_power, kick_direction]
        else:
            # Default action if invalid
            return [0.0, 0.0, 0.0, 0.0]
    
    def _parse_discrete_action(self, action: Union[int, List[int], np.ndarray]) -> List[float]:
        """
        Parse discrete action.
        
        Args:
            action: Discrete action ID (can be int, list, array, or numpy scalar from RLLib)
            
        Returns:
            Parsed action as [move_x, move_y, kick_power, kick_direction]
        """
        # Handle case where RLLib returns action as array/list instead of int
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # Scalar numpy array (like np.array(5))
                action = action.item()  # Extract the scalar value
            elif len(action) > 0:
                action = int(action[0])  # Extract first element and convert to int
            else:
                action = 0  # Default action
        elif isinstance(action, list):
            if len(action) > 0:
                action = int(action[0])  # Extract first element and convert to int
            else:
                action = 0  # Default action
        elif not isinstance(action, (int, np.integer)):
            action = int(action)  # Convert to int if possible
        
        if action in self.discrete_action_map:
            action_data = self.discrete_action_map[action]
            
            move_x, move_y = action_data['move']
            kick_power = action_data['kick_power']
            kick_direction = action_data['kick_direction']
            
            return [float(move_x), float(move_y), kick_power, kick_direction]
        else:
            # Default action if invalid
            return [0.0, 0.0, 0.0, 0.0]
    
    def get_action_space(self) -> spaces.Space:
        """
        Get the action space for agents.
        
        Returns:
            Gymnasium action space
        """
        if self.continuous:
            # Continuous: [move_x, move_y, kick_power, kick_direction]
            return spaces.Box(
                low=np.array([-1.0, -1.0, 0.0, -math.pi]),
                high=np.array([1.0, 1.0, 1.0, math.pi]),
                dtype=np.float32
            )
        else:
            # Discrete: 18 possible actions
            return spaces.Discrete(18)
    
    def action_to_string(self, action: Union[int, List[float], np.ndarray]) -> str:
        """
        Convert action to human-readable string.
        
        Args:
            action: Action to convert
            
        Returns:
            Human-readable action description
        """
        parsed = self.parse_action(action, 0)  # Agent ID doesn't matter for parsing
        move_x, move_y, kick_power, kick_direction = parsed
        
        # Movement description
        if abs(move_x) < 0.1 and abs(move_y) < 0.1:
            move_desc = "Stand"
        else:
            move_desc = f"Move({move_x:.1f}, {move_y:.1f})"
        
        # Kick description
        if kick_power < 0.1:
            kick_desc = "No Kick"
        else:
            kick_angle_deg = math.degrees(kick_direction)
            kick_desc = f"Kick(power={kick_power:.1f}, angle={kick_angle_deg:.0f}°)"
        
        return f"{move_desc} + {kick_desc}"
    
    def sample_random_action(self, random_state: np.random.Generator = None) -> Union[int, np.ndarray]:
        """
        Sample a random action.
        
        Args:
            random_state: Random number generator
            
        Returns:
            Random action in the appropriate format
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        if self.continuous:
            return random_state.uniform(
                [-1.0, -1.0, 0.0, -math.pi],
                [1.0, 1.0, 1.0, math.pi]
            ).astype(np.float32)
        else:
            return random_state.integers(0, 18)  # 18 actions (0-17)
    
    def normalize_action(self, action: Union[int, List[float], np.ndarray]) -> Union[int, np.ndarray]:
        """
        Normalize action to ensure it's within valid bounds.
        
        Args:
            action: Action to normalize
            
        Returns:
            Normalized action
        """
        if self.continuous:
            if isinstance(action, (list, np.ndarray)) and len(action) >= 4:
                normalized = np.array([
                    np.clip(action[0], -1.0, 1.0),
                    np.clip(action[1], -1.0, 1.0),
                    np.clip(action[2], 0.0, 1.0),
                    np.clip(action[3], -math.pi, math.pi)
                ], dtype=np.float32)
                return normalized
            else:
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            if isinstance(action, (int, np.integer)):
                return int(np.clip(action, 0, 17))
            else:
                return 0 