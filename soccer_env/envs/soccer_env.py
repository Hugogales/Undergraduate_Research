"""
Soccer Environment

This module implements a soccer environment following PettingZoo's Parallel API standards.
The environment supports both cooperative and competitive multi-agent reinforcement learning.

NOTE: This environment uses the Parallel API for natural soccer gameplay where all agents act simultaneously.
"""

import functools
import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Optional, Dict, List, Tuple, Any

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

# Fix absolute imports to be relative to src
import sys
import os

# Get path to src directory
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import with relative paths
from ..core.game_engine import GameEngine
from ..core.physics import PhysicsEngine
from ..core.entities import Player, Ball, Goal
from ..utils.state_parser import StateParser
from ..utils.reward_calculator import RewardCalculator
from ..utils.action_parser import ActionParser
from ..utils.renderer import SoccerRenderer
from ..envs.constants import ENV_CONSTANTS


def env(render_mode: Optional[str] = None, **kwargs):
    """
    Factory function for creating the soccer environment with wrappers.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', 'ansi', or None)
        **kwargs: Additional arguments passed to the environment
        
    Returns:
        Wrapped soccer environment
    """
    environment = raw_env(render_mode=render_mode, **kwargs)
    
    # Add standard PettingZoo wrappers for parallel environments
    if render_mode == "ansi":
        environment = wrappers.CaptureStdoutWrapper(environment)
    
    return environment


def parallel_env(render_mode: Optional[str] = None, **kwargs):
    """
    Alias for env() - maintained for backward compatibility.
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', 'ansi', or None)
        **kwargs: Additional arguments passed to the environment
        
    Returns:
        Soccer environment
    """
    return env(render_mode=render_mode, **kwargs)


class raw_env(ParallelEnv):
    """
    Soccer Environment using Parallel API
    
    A multi-agent soccer environment where teams compete to score goals.
    All agents act simultaneously, making it ideal for real-time soccer simulation.
    
    Features:
    - Support for variable team sizes (1v1, 2v2, 3v3, 5v5, etc.)
    - Continuous and discrete action spaces
    - Human vs AI gameplay with WASD and arrow key controls
    - Training and evaluation modes
    - Proper timing that's independent of team size
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "name": "soccer_v0",
        "is_parallelizable": True,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        num_players_per_team: int = 2,
        field_width: float = None,
        field_height: float = None,
        max_episode_steps: int = None,
        game_duration_seconds: float = 120.0,
        continuous_actions: bool = True,
        randomize_starting_positions: bool = True,
        ball_friction: float = None,
        player_speed: float = None,
        game_mode: str = "default",
        **kwargs
    ):
        """
        Initialize the soccer environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', 'ansi', or None)
            num_players_per_team: Number of players per team
            field_width: Width of the soccer field (defaults to ENV_CONSTANTS.WIDTH)
            field_height: Height of the soccer field (defaults to ENV_CONSTANTS.HEIGHT)
            max_episode_steps: Maximum steps per episode (calculated from game_duration if None)
            game_duration_seconds: Duration of game in seconds (default: 120)
            continuous_actions: Whether to use continuous action space
            randomize_starting_positions: Whether to randomize starting positions
            ball_friction: Ball friction coefficient (defaults to ENV_CONSTANTS.BALL_FRICTION or game_mode preset)
            player_speed: Base player movement speed (defaults to ENV_CONSTANTS.PLAYER_SPEED or game_mode preset)
            game_mode: Game mode preset ('default', 'air_hockey', 'arcade', 'realistic', 'giant_players', 
                      'pinball', 'slow_motion', 'speed_demon', 'tiny_players', 'bouncy_castle')
        """
        super().__init__()
        
        # Apply game mode preset first
        from ..envs.constants import GameModePresets
        try:
            preset_config = GameModePresets.apply_preset_to_constants(game_mode)
            self.game_mode = game_mode
            print(f"🎮 Game Mode: {game_mode} - {GameModePresets.describe_preset(game_mode)}")
        except ValueError as e:
            print(f"⚠️  {e}")
            print(f"Using default game mode instead.")
            preset_config = GameModePresets.apply_preset_to_constants("default")
            self.game_mode = "default"
        
        # Use preset values as defaults, but allow parameter overrides
        if field_width is None:
            field_width = preset_config.get('WIDTH', ENV_CONSTANTS.WIDTH)
        if field_height is None:
            field_height = preset_config.get('HEIGHT', ENV_CONSTANTS.HEIGHT)
        if ball_friction is None:
            ball_friction = preset_config.get('BALL_FRICTION', ENV_CONSTANTS.BALL_FRICTION)
        if player_speed is None:
            player_speed = preset_config.get('PLAYER_SPEED', ENV_CONSTANTS.PLAYER_SPEED)
        
        # Store preset config for use in game components
        self.preset_config = preset_config
        
        # Environment parameters
        self.num_players_per_team = num_players_per_team
        self.total_agents = num_players_per_team * 2
        self.field_width = field_width
        self.field_height = field_height
        self.continuous_actions = continuous_actions
        self.randomize_starting_positions = randomize_starting_positions
        self.game_duration_seconds = game_duration_seconds
        
        # Calculate max_episode_steps from game duration if not provided
        if max_episode_steps is None:
            fps = ENV_CONSTANTS.FPS if hasattr(ENV_CONSTANTS, 'FPS') else 42
            self.max_episode_steps = int(game_duration_seconds * fps)
            print(f"Calculated max_episode_steps: {self.max_episode_steps} from {game_duration_seconds}s game at {fps} FPS")
        else:
            self.max_episode_steps = max_episode_steps
        
        # Create agent names - this defines the agents in the environment
        self.possible_agents = []
        for team in ["team_0", "team_1"]:
            for i in range(num_players_per_team):
                agent_name = f"{team}_player_{i}"
                self.possible_agents.append(agent_name)
        
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        
        # Initialize game components
        self.game_engine = GameEngine(
            field_width=field_width,
            field_height=field_height,
            num_players_per_team=num_players_per_team,
            ball_friction=ball_friction,
            player_speed=player_speed,
            game_duration=game_duration_seconds,
            simulation_mode=(render_mode is None),  # Use simulation mode for headless training
            preset_config=self.preset_config,  # Pass preset config for additional properties
        )
        
        self.physics_engine = PhysicsEngine()
        self.state_parser = StateParser()
        self.reward_calculator = RewardCalculator()
        self.action_parser = ActionParser(continuous=continuous_actions)
        
        # Rendering
        self.render_mode = render_mode
        self.renderer: Optional[SoccerRenderer] = None
        if self.render_mode == "human":
            self.renderer = SoccerRenderer(
                field_width=self.field_width,
                field_height=self.field_height
            )
            self.renderer.show_instructions = True
        
        # Episode tracking
        self.episode_step = 0
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """
        Get observation space for an agent.
        
        Uses the StateParser to determine the correct observation size
        based on the original state format from the old codebase.
        """
        obs_dim = self.state_parser.get_observation_size(self.total_agents)
        
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """
        Get action space for an agent.
        
        Actions:
        - Movement: [x_velocity, y_velocity] (continuous) or discrete directions
        - Ball interaction: [kick_power, kick_direction] (continuous) or discrete actions
        """
        # Use the action parser's action space (handles both continuous and discrete correctly)
        return self.action_parser.get_action_space()
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Get observation for a specific agent.
        """
        if agent not in self.agents:
            return np.zeros(self.observation_space(agent).shape[0])
        
        game_state = self.game_engine.get_game_state()
        if game_state is None:
            # Return zero observation if game state is invalid
            return np.zeros(self.observation_space(agent).shape[0])
            
        agent_idx = self.agent_name_mapping[agent]
        observation = self.state_parser.get_agent_observation(
            game_state,
            agent_idx
        )
        return observation.astype(np.float32)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observations, infos)
        """
        # Reset agents
        self.agents = self.possible_agents[:]
        
        # Initialize random state if needed
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        else:
            # Create np_random if it doesn't exist
            if not hasattr(self, 'np_random'):
                self.np_random, _ = seeding.np_random(None)
        
        # Reset game state
        self.game_engine.reset(
            randomize_positions=self.randomize_starting_positions,
            random_state=self.np_random
        )
        
        # Reset internal tracking
        self.episode_step = 0
        
        # Prepare observations and infos
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        if self.render_mode == "human":
            self.render()
            
        return observations, infos
    
    def step(
        self, 
        actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Dict]
    ]:
        """
        Execute one step in the environment with actions from all agents.
        
        Args:
            actions: Dict of actions for each agent
            
        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent
            terminations: Dict of termination flags for each agent
            truncations: Dict of truncation flags for each agent
            infos: Dict of info dicts for each agent
        """
        # Handle human input before processing actions
        if self.render_mode == 'human' and self.renderer:
            events = self.renderer.handle_events()
            if events['quit']:
                # Set all agents to terminated to end the episode gracefully
                terminations = {agent: True for agent in self.agents}
                truncations = {agent: False for agent in self.agents}
                observations = {agent: self.observe(agent) for agent in self.agents}
                rewards = {agent: 0.0 for agent in self.agents}
                infos = {agent: {"quit": True} for agent in self.agents}
                self.agents = []  # Clear agents to end episode
                return observations, rewards, terminations, truncations, infos
            
            # Only override with human input if we're actually in human play mode
            # For AI-only mode (watching AI), don't override AI actions with human input
            # We can detect AI-only mode by checking if all actions are numpy arrays (AI) vs keyboard input
            is_ai_only_mode = all(isinstance(action, np.ndarray) for action in actions.values())
            
            # Store the AI-only mode flag for the renderer
            self._is_ai_only_mode = is_ai_only_mode
            
            if not is_ai_only_mode:
                # Override actions with human input if available (normal human play mode)
                for agent_name, human_action in events['human_actions'].items():
                    if agent_name in actions:
                        actions[agent_name] = human_action
        
        # Parse and set actions for all agents
        for agent, action in actions.items():
            if agent in self.agents:
                agent_idx = self.agent_name_mapping[agent]
                parsed_action = self.action_parser.parse_action(action, agent_idx)
                self.game_engine.set_agent_action(agent_idx, parsed_action)
        
        # Step the game forward
        self.game_engine.step()
        game_state = self.game_engine.get_game_state()
        # NOTE: Don't increment episode_step here - GameEngine already handles it
        
        # Handle case where game_state might be None
        if game_state is None:
            # Return empty results if game state is invalid
            observations = {agent: self.observe(agent) for agent in self.agents}
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}  # End episode
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {"error": "Invalid game state"} for agent in self.agents}
            self.agents = []  # Clear agents to end episode
            return observations, rewards, terminations, truncations, infos
        
        # Calculate rewards for all agents
        all_rewards = self.reward_calculator.calculate_rewards(
            game_state, 
            self.game_engine.get_previous_state()
        )
        
        # Check for termination conditions
        time_up = game_state.episode_step >= self.max_episode_steps
        goal_scored = getattr(game_state, 'goal_scored', False)
        
        # Episodes should terminate when goals are scored for proper RL training
        terminated = goal_scored
        truncated = time_up
        
        # If goal was scored, we want immediate termination for learning
        if goal_scored:
            print(f"🥅 Goal scored! Episode terminated. Score: {getattr(game_state, 'team_0_score', 0)}-{getattr(game_state, 'team_1_score', 0)}")
        
        # Prepare return dictionaries
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for i, agent in enumerate(self.agents):
            observations[agent] = self.observe(agent)
            rewards[agent] = all_rewards[i] if i < len(all_rewards) else 0.0
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "episode_step": game_state.episode_step,
                "team_0_score": getattr(game_state, 'team_0_score', 0),
                "team_1_score": getattr(game_state, 'team_1_score', 0),
                "time_remaining": max(0, self.max_episode_steps - game_state.episode_step),
                "ball_possession": getattr(game_state, 'ball_possession', None),
            }
        
        # Remove agents if episode is done
        if terminated or truncated:
            self.agents = []
        
        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminations, truncations, infos
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
            
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode in ["human", "rgb_array"]:
            return self._render_graphical()
    
    def _render_ansi(self) -> str:
        """
        Render the environment as ASCII text.
        """
        game_state = self.game_engine.get_game_state()
        
        if game_state is None:
            output = f"Step: {self.episode_step}\n"
            output += "Game state not available\n"
            print(output)
            return output
        
        output = f"Step: {self.episode_step}\n"
        output += f"Score: Team 0: {getattr(game_state, 'team_0_score', 0)} - Team 1: {getattr(game_state, 'team_1_score', 0)}\n"
        output += f"Ball position: ({getattr(game_state.ball, 'x', 0):.2f}, {getattr(game_state.ball, 'y', 0):.2f})\n"
        
        if hasattr(game_state, 'players') and game_state.players:
            for i, player in enumerate(game_state.players):
                team = "Team 0" if i < self.num_players_per_team else "Team 1"
                output += f"{team} Player {i}: ({getattr(player, 'x', 0):.2f}, {getattr(player, 'y', 0):.2f})\n"
        
        print(output)
        return output
    
    def _render_graphical(self) -> Optional[np.ndarray]:
        """
        Render the environment using the SoccerRenderer.
        """
        if self.renderer is None:
            self.renderer = SoccerRenderer(
                field_width=self.field_width, 
                field_height=self.field_height
            )
            self.renderer.show_instructions = True

        game_state = self.game_engine.get_game_state()
        if game_state is None:
            return None

        # Detect if we're in AI-only mode by checking if there was human input detected
        # If the renderer has no human actions, we're likely in AI-only mode
        show_human_instructions = True
        if hasattr(self, '_is_ai_only_mode'):
            show_human_instructions = not self._is_ai_only_mode

        # Use consistent time-based FPS throttling
        rgb_array = self.renderer.render(game_state, throttle_fps=True, show_human_instructions=show_human_instructions)
        return rgb_array
    
    def close(self) -> None:
        """
        Close the environment and clean up resources.
        """
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# Maintain aliases for backward compatibility
SoccerEnv = raw_env
SoccerParallelEnv = raw_env 