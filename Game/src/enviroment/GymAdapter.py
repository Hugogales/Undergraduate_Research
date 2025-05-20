import gym
import numpy as np
from gym import spaces
from enviroment.Game import Game
from AI.StateParser import StateParser
from AI.RewardFunction import RewardFunction
from params import EnvironmentHyperparameters, AIHyperparameters
from AI.randmodel import RandomModel
from functions.league import League

class SoccerEnv(gym.Env):
    """
    SoccerEnv: An OpenAI Gym wrapper for the soccer game environment.
    
    This class implements the standard Gym interface (reset, step, render) while
    wrapping the existing Game class. It preserves all functionality of the original
    environment while allowing it to be used with standard RL libraries.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, log_name=None, competing_model=None, current_stage=None, league=None):
        super(SoccerEnv, self).__init__()
        
        # Initialize environment parameters
        self.env_params = EnvironmentHyperparameters()
        self.ai_params = AIHyperparameters()
        
        # Set current stage if provided, otherwise use the one from AI parameters
        self.current_stage = current_stage if current_stage is not None else self.ai_params.current_stage
        
        # Create Game instance
        self.game = Game(log_name=log_name)
        
        # Set up action and observation spaces
        # For team 1 players
        num_players_team1 = len(self.game.team_1)
        
        # Each player has 5 possible actions: [up, down, left, right, shoot]
        # Each action is binary (0 or 1)
        self.action_space = spaces.MultiDiscrete([2] * 5 * num_players_team1)
        
        # Observation space is based on the state parser's output
        # This is a simplified representation - adjust based on actual state size
        state_dim = self._get_state_dimension()
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'), 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Track internal state
        self.current_state = None
        self.current_reward = 0
        self.current_done = False
        self.info = {}
        
        # Frame counter for action rate
        self.frame_counter = 0
        self.action_update_interval = int(self.env_params.FPS / self.env_params.AGENT_DECISION_RATE)
        
        # Current actions for both teams
        self.current_actions_team1 = [[0,0,0,0,0]] * num_players_team1
        self.current_actions_team2 = [[0,0,0,0,0]] * len(self.game.team_2)
        
        # Accumulated values between action states
        self.accumulated_rewards_team1 = [0] * num_players_team1
        self.accumulated_rewards_team2 = [0] * len(self.game.team_2)
        self.accumulated_dones = []
        self.total_ball_distance = 0
        
        # For tracking ball position between steps
        self.previous_ball_position = self.game.ball.position.copy()
        
        # Store competing model if provided
        self.competing_model = competing_model
        
        # Store league if provided or create a new one if in stages 3 or 4
        self.league = league
        
        # If competing model is not provided and we're in stages 1 or 2, use RandomModel
        if self.competing_model is None and (self.current_stage == 1 or self.current_stage == 2):
            self.competing_model = RandomModel()
            
        # If no competing model and we're in stages 3 or 4 and no league, we'll need a league later
        self.competing_model_rating = None
    
    def _get_state_dimension(self):
        """
        Determine the dimension of the state space based on the state parser's output.
        """
        # Get a sample state to determine its dimension
        states = self.game.state_parser.parse_state()
        states_team1 = states[:len(self.game.team_1)]
        
        # Flatten the state to get its dimension
        flat_state = np.array(states_team1).flatten()
        return len(flat_state)
    
    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        
        Returns:
            observation (object): The initial observation of the space.
        """
        # Reset the game
        self.game.reset_game()
        
        # Reset internal trackers
        self.frame_counter = 0
        self.total_ball_distance = 0
        self.accumulated_rewards_team1 = [0] * len(self.game.team_1)
        self.accumulated_rewards_team2 = [0] * len(self.game.team_2)
        self.accumulated_dones = []
        self.previous_ball_position = self.game.ball.position.copy()
        
        # If in stages 3 or 4 and we have a league, sample a new competing model
        if (self.current_stage >= 3) and self.league is not None:
            self.competing_model, self.competing_model_rating = self.league.sample_player()
        
        # Get initial state
        states = self.game.state_parser.parse_state()
        states_team1 = states[:len(self.game.team_1)]
        self.current_state = np.array(states_team1).flatten()
        
        return self.current_state
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action (array_like): An action provided by the agent
                                 (array of actions for all team 1 players)
        
        Returns:
            observation (object): Agent's observation of the current environment
            reward (float): Amount of reward returned after previous action
            done (bool): Whether the episode has ended
            info (dict): Contains auxiliary diagnostic information
        """
        # Convert flat action array to per-player actions
        self._process_actions(action)
        
        # Run a single frame
        goal1, goal2 = self._run_single_frame()
        
        # Get the new state
        states = self.game.state_parser.parse_state()
        states_team1 = states[:len(self.game.team_1)]
        self.current_state = np.array(states_team1).flatten()
        
        # Check if episode is done
        done = not (self.game.timer > 0) or goal1 or goal2
        self.accumulated_dones.append(done)
        
        # If it's time to update actions or game is done, process accumulated rewards
        if self.frame_counter % self.action_update_interval == 0 or done:
            reward = sum(self.accumulated_rewards_team1)
            episode_done = any(self.accumulated_dones)
            
            # Update stats
            self.game.stats.calculate_stats(
                reward, 
                [0] * len(self.game.team_1),  # Dummy entropy values
                self.total_ball_distance
            )
            
            # Reset accumulators
            self.total_ball_distance = 0
            self.accumulated_rewards_team1 = [0] * len(self.game.team_1)
            self.accumulated_dones = []
        else:
            # Return accumulated reward so far if it's not an action frame
            reward = sum(self.accumulated_rewards_team1)
            episode_done = any(self.accumulated_dones)
        
        # Prepare info dict
        info = {
            'score_team1': self.game.score_team1,
            'score_team2': self.game.score_team2,
            'time_remaining': self.game.timer,
            'ball_position': self.game.ball.position.copy(),
            'ball_velocity': self.game.ball.velocity.copy(),
            'goal1': goal1,
            'goal2': goal2,
            'competing_model_rating': self.competing_model_rating
        }
        
        return self.current_state, reward, episode_done, info
    
    def set_competing_model(self, model, rating=None):
        """
        Change the competing model without restarting the environment.
        
        Args:
            model: The new competing model
            rating: Optional ELO rating of the competing model
        """
        self.competing_model = model
        self.competing_model_rating = rating
    
    def set_league(self, league):
        """
        Set a league to sample competing models from.
        
        Args:
            league: An instance of League
        """
        self.league = league
    
    def update_league(self, model, rating=None):
        """
        Update the league with a new model.
        
        Args:
            model: The model to add to the league
            rating: Optional ELO rating of the model
        """
        if self.league is not None:
            self.league.update(model, rating)
    
    def set_stage(self, stage):
        """
        Change the current training stage.
        
        Args:
            stage: The new training stage (1-4)
        """
        self.current_stage = stage
        
        # If moving to stages 1 or 2, ensure we have a RandomModel
        if stage <= 2 and not isinstance(self.competing_model, RandomModel):
            self.competing_model = RandomModel()
            self.competing_model_rating = None
    
    def _process_actions(self, action):
        """
        Process the action array from gym interface to the game's format
        
        Args:
            action: Flattened array of actions for team 1 players
        """
        num_players = len(self.game.team_1)
        actions_per_player = 5  # [up, down, left, right, shoot]
        
        # Reshape actions into per-player format
        for i in range(num_players):
            player_actions = action[i*actions_per_player:(i+1)*actions_per_player]
            self.current_actions_team1[i] = player_actions.tolist() if hasattr(player_actions, 'tolist') else list(player_actions)
    
    def _run_single_frame(self):
        """
        Run a single frame of the game.
        
        Returns:
            tuple: (goal1, goal2) indicating if goals were scored
        """
        # Update simulation time
        self.game.simulation_time += self.game.delta_time
        self.game.timer = self.game.env_params.GAME_DURATION - self.game.simulation_time
        
        # Handle players' movement for team 1
        for i, player in enumerate(self.game.team_1):
            move_input = self.current_actions_team1[i]
            player.move(move_input)
        
        # Handle actions for team 2 based on current stage and competing model
        if self.competing_model is not None:
            # Use the competing model to get actions (either RandomModel or a learned model)
            states = self.game.state_parser.parse_state()
            states_team2 = states[len(self.game.team_1):]
            actions2, _ = self.competing_model.get_actions(states_team2)
            self.current_actions_team2 = actions2
        
        # Apply the actions to team 2 players
        for i, player in enumerate(self.game.team_2):
            move_input = self.current_actions_team2[i]
            player.move(move_input)
        
        # Store previous ball position
        self.previous_ball_position = self.game.ball.position.copy()
        
        # Update ball's movement
        self.game.ball.update_position()
        
        # Handle collisions
        self.game.handle_collisions()
        
        # Calculate ball distance moved
        self.total_ball_distance += np.linalg.norm(
            np.array(self.game.ball.position) - np.array(self.previous_ball_position)
        )
        
        # Check for goals
        goal1, goal2 = self.game.check_goals()
        
        # Calculate rewards
        rewards = self.game.reward_function.calculate_rewards(goal1, goal2)
        self.accumulated_rewards_team1 = [a + b for a, b in zip(self.accumulated_rewards_team1, rewards[:len(self.game.team_1)])]
        self.accumulated_rewards_team2 = [a + b for a, b in zip(self.accumulated_rewards_team2, rewards[len(self.game.team_1):])]
        
        # Increment frame counter
        self.frame_counter += 1
        
        return goal1, goal2
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): The mode to render with.
        """
        # Check if rendering is enabled in parameters
        if self.env_params.RENDER:
            self.game.render()
    
    def close(self):
        """
        Clean up any resources used by the environment.
        """
        if self.game.log_name is not None:
            self.game.logger.close()
    
    # Additional methods to preserve original functionality
    
    def run_episode(self, model, competing_model=None, current_stage=None):
        """
        Run a full episode using the existing Game.run() method.
        This preserves compatibility with existing code.
        
        Args:
            model: The model for team 1
            competing_model: The model for team 2
            current_stage: The current training stage
            
        Returns:
            Same output as Game.run()
        """
        # Update current stage if provided
        if current_stage is not None:
            self.current_stage = current_stage
            
        # Update competing model if provided
        if competing_model is not None:
            self.competing_model = competing_model
        
        # Use the original Game.run method
        return self.game.run(model, self.competing_model, self.current_stage)
    
    def run_play_mode(self):
        """
        Run the game in play mode, allowing human players.
        
        Returns:
            Same output as Game.run_play()
        """
        return self.game.run_play()
    
    def replay_from_log(self, states):
        """
        Replay the game from logged states.
        
        Args:
            states: The states to replay
        """
        self.game.replay(states) 