# soccer_env.py

import gym
from gym import spaces
import numpy as np
import math
import json
import os
from multiprocessing import Lock

from params import EnvironmentHyperparameters, VisualHyperparametters
from enviroment.Ball import Ball
from enviroment.Player import Player
from enviroment.Goal import Goal
from functions.Logger import Logger
from AI.StateParser import StateParser
from AI.RewardFunction import RewardFunction 

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()
if ENV_PARAMS.RENDER:
    VIS_PARAMS = VisualHyperparametters()

class SoccerEnv(gym.Env):
    """
    Custom Gym environment for the soccer game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(SoccerEnv, self).__init__()
        
        # Initialize game components
        self._initialize_game()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(9)  # 9 actions: 8 directions + no movement

        # Observation space size depends on the number of players and features
        # Each player has position (x, y) and velocity (vx, vy)
        # Ball has position and velocity
        num_players = ENV_PARAMS.NUMBER_OF_PLAYERS * 2  # 2 teams
        self.state_size = (num_players * 4) + 4  # positions and velocities + ball
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.state_size,), dtype=np.float32)
        
        # Initialize state parser and reward function
        self.state_parser = StateParser()
        self.reward_function = RewardFunction(self)
        
        # Initialize logging
        self.log_dir = ENV_PARAMS.log_name
        self.log_interval = ENV_PARAMS.log_interval
        self.game_counter = 0  # Counts completed games
        self.lock = Lock()  # To handle concurrent access if needed
        
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # Accumulate game states
        self.logger = Logger()

    def _initialize_game(self):
        """
        Initializes or resets the game components.
        """
        # Initialize Players
        self.team_1 = []
        self.team_2 = []
        for i in range(ENV_PARAMS.NUMBER_OF_PLAYERS):
            # AI-controlled teams
            self.team_1.append(
                Player(
                    team_id=1,
                    player_id=i
                ))
            self.team_2.append(
                Player(
                    team_id=2,
                    player_id=i
                ))
        self.players = self.team_1 + self.team_2

        # Initialize Ball
        self.ball = Ball(
            position=[ENV_PARAMS.WIDTH // 2, ENV_PARAMS.HEIGHT // 2],
        )

        # Initialize Goals
        goal_height = ENV_PARAMS.GOAL_HEIGHT
        self.goal1 = Goal(
            position=[0, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Left goal
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
            invert_image=True,
        )
        self.goal2 = Goal(
            position=[ENV_PARAMS.WIDTH - ENV_PARAMS.GOAL_WIDTH, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Right goal
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
        )

        # Initialize Scores
        self.score_team1 = 0
        self.score_team2 = 0

        # Initialize Timer
        self.timer = ENV_PARAMS.GAME_DURATION
        self.simulation_time = 0
        self.delta_time = 1 / ENV_PARAMS.FPS

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        self._initialize_game()
        self.done = False
        self.current_game_states = []  # Reset game states

        # Return initial observations
        return self._get_observation()

    def step(self, action_dict):
        """
        Executes actions for all agents and returns the next state, rewards, done, and info.
        
        :param action_dict: A dictionary mapping agent IDs to actions.
        :return: observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), {}, self.done, {}
        
        # Apply actions to players
        for agent_id, action in action_dict.items():
            player = self._get_player_from_agent_id(agent_id)
            if player:
                # Swap action directions for Team 2 to maintain consistency
                if player.team_id == 2:
                    action = self._swap_action(action)
                move_input = self._action_to_input(action)
                player.move(move_input)
        
        # Update ball's movement
        self.ball.update_position()

        # Handle collisions
        self.handle_collisions()

        # Check for goals
        goal1_scored, goal2_scored = self.check_goals()

        # Calculate rewards
        rewards = self.reward_function.calculate_rewards(goal1_scored, goal2_scored)

        # Capture current state for logging
        current_state = self._get_current_state()
        self.current_game_states.append(current_state)

        # Check if the game is done
        self.simulation_time += self.delta_time
        self.timer = ENV_PARAMS.GAME_DURATION - self.simulation_time
        if self.timer <= 0:
            self.done = True

            # Increment game counter
            self.game_counter += 1

            # Log the game if it's time
            if self.log_dir is not None and self.game_counter % self.log_interval == 0:
                self._log_game()

        return self._get_observation(), rewards, self.done, {}

    def render(self, mode='human'):
        """
        Rendering is not required for RLlib training.
        """
        pass

    def close(self):
        """
        Cleans up resources.
        """
        pass

    def _get_observation(self):
        """
        Generates the observation for all agents.
        
        :return: A dictionary mapping agent IDs to observations.
        """
        observations = {}
        states = self.state_parser.parse_state(self.players, self.ball)
        for player_id, state in states:
            player = self._get_player_by_id(player_id)
            if player:
                agent_id = self._get_agent_id(player)
                observations[agent_id] = np.array(state, dtype=np.float32)
        return observations

    def _get_current_state(self):
        """
        Captures the current game state for logging.
        
        :return: A dictionary representing the current state.
        """
        state = {
            'time': self.timer,
            'players': [],
            'ball': {
                'position': self.ball.position.copy(),
                'velocity': self.ball.velocity.copy()
            }
        }
        for player in self.players:
            player_state = {
                'team_id': player.team_id,
                'player_id': player.player_id,
                'position': player.position.copy(),
                'velocity': player.velocity.copy()
            }
            state['players'].append(player_state)
        return state

    def _log_game(self):
        """
        Logs the entire game state to a JSON file named gameX.json.
        """
        filename = os.path.join(self.log_dir, f"game{self.game_counter}.json")
        with open(filename, 'w') as f:
            json.dump(self.current_game_states, f, indent=4)
        print(f"Game {self.game_counter} logged to {filename}")

    def _swap_action(self, action):
        """
        Swaps the action directions for Team 2 to maintain consistency.
        
        :param action: Integer representing the action.
        :return: Swapped action integer.
        """
        # Define action mapping to swap directions
        swap_mapping = {
            0: 1,  # Up <-> Down
            1: 0,
            2: 3,  # Left <-> Right
            3: 2,
            4: 6,  # Up-Left <-> Down-Right
            5: 7,  # Up-Right <-> Down-Left
            6: 4,
            7: 5,
            8: 8   # No movement
        }
        return swap_mapping.get(action, 8)

    def _action_to_input(self, action):
        """
        Converts a discrete action into movement input.
        
        :param action: Integer representing the action.
        :return: A list [up, down, left, right]
        """
        action_mapping = {
            0: [1, 0, 0, 0],  # Up
            1: [0, 1, 0, 0],  # Down
            2: [0, 0, 1, 0],  # Left
            3: [0, 0, 0, 1],  # Right
            4: [1, 0, 1, 0],  # Up-Left
            5: [1, 0, 0, 1],  # Up-Right
            6: [0, 1, 1, 0],  # Down-Left
            7: [0, 1, 0, 1],  # Down-Right
            8: [0, 0, 0, 0]   # No movement
        }
        return action_mapping.get(action, [0, 0, 0, 0])

    def _get_player_from_agent_id(self, agent_id):
        """
        Retrieves the Player object corresponding to the given agent ID.
        
        :param agent_id: String representing the agent ID.
        :return: Player object or None.
        """
        # Agent ID format: "team_{team_id}_player_{player_id}"
        try:
            parts = agent_id.split('_')
            team_id = int(parts[1])
            player_id = int(parts[3])
            for player in self.players:
                if player.team_id == team_id and player.player_id == player_id:
                    return player
        except (IndexError, ValueError):
            return None
        return None

    def _get_player_by_id(self, player_id):
        """
        Retrieves the Player object by player_id.
        
        :param player_id: Integer representing the player ID.
        :return: Player object or None.
        """
        for player in self.players:
            if player.player_id == player_id:
                return player
        return None

    def _get_agent_id(self, player):
        """
        Generates a unique agent ID for a player.
        
        :param player: Player object.
        :return: String representing the agent ID.
        """
        return f"team_{player.team_id}_player_{player.player_id}"

    def handle_collisions(self):
        """
        Handles collision between players and the ball, and between players themselves.
        """
        for player in self.players:
            # Check collision between player and ball
            self.check_collision(player, self.ball)

            # Check collision between player and player
            for other_player in self.players:
                if player == other_player:
                    continue
                self.check_player_collision(player, other_player)

    def check_collision(self, player, ball):
        """
        Checks and handles collision between a player and the ball.

        :param player: Instance of Player.
        :param ball: Instance of Ball.
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

            # Adjust ball position to prevent sticking
            ball.position[0] += direction_x * overlap
            ball.position[1] += direction_y * overlap

            # Calculate relative velocity
            relative_velocity_x = ball.velocity[0] - player.velocity[0]
            relative_velocity_y = ball.velocity[1] - player.velocity[1]

            # Calculate the velocity along the direction of collision
            velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y

            if velocity_along_normal > 0:
                return  # They are moving away from each other

            # Define restitution (elasticity)
            restitution = 0.7  # 0 = inelastic, 1 = perfectly elastic

            # Update ball's velocity based on collision
            ball.velocity[0] -= (1 + restitution) * velocity_along_normal * direction_x
            ball.velocity[1] -= (1 + restitution) * velocity_along_normal * direction_y

    def check_player_collision(self, player1, player2):
        """
        Checks and handles collision between two players.

        :param player1: Instance of Player.
        :param player2: Instance of Player.
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

            # Adjust positions to prevent sticking (push players apart equally)
            player1.position[0] -= direction_x * (overlap / 2)
            player1.position[1] -= direction_y * (overlap / 2)
            player2.position[0] += direction_x * (overlap / 2)
            player2.position[1] += direction_y * (overlap / 2)

            # Calculate relative velocity
            relative_velocity_x = player2.velocity[0] - player1.velocity[0]
            relative_velocity_y = player2.velocity[1] - player1.velocity[1]

            # Calculate the velocity along the direction of collision
            velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y

            if velocity_along_normal > 0:
                return  # They are moving away from each other

            # Define restitution (elasticity) less than ball's
            restitution = 0.7  # Players push each other less strongly

            # Update players' velocities based on collision
            player1.velocity[0] -= (1 + restitution) * velocity_along_normal * direction_x
            player1.velocity[1] -= (1 + restitution) * velocity_along_normal * direction_y
            player2.velocity[0] += (1 + restitution) * velocity_along_normal * direction_x
            player2.velocity[1] += (1 + restitution) * velocity_along_normal * direction_y
