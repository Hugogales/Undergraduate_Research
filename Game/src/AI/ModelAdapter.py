import numpy as np
from enviroment.GymAdapter import SoccerEnv
from params import AIHyperparameters
from params import EnvironmentHyperparameters

class ModelAdapter:
    """
    ModelAdapter: A class to adapt existing AI models to work with the OpenAI Gym interface.
    
    This adapter translates between the Gym environment's action/observation format
    and the format expected by the existing AI models.
    """
    
    def __init__(self, model):
        """
        Initialize the adapter with an existing AI model.
        
        Args:
            model: An instance of an AI model (PPO, MAAC, HUGO, etc.)
        """
        self.model = model
        self.memories = []
        self.env_params = EnvironmentHyperparameters()
    
    def act(self, observation):
        """
        Get actions from the model given an observation from the Gym environment.
        
        Args:
            observation: The observation from the Gym environment (flattened state)
        
        Returns:
            actions: Actions in the format expected by the Gym environment
        """
        # Reshape observation to the format expected by the model
        num_players = self._get_num_players_from_observation(observation)
        
        # Reshape the flat observation into a list of player states
        states = self._reshape_observation_to_states(observation, num_players)
        
        # Get actions from the model
        actions, entropys = self.model.get_actions(states)
        
        # Convert actions to format expected by Gym environment
        flattened_actions = self._flatten_actions(actions)
        
        return flattened_actions, entropys
    
    def store_reward(self, reward, done):
        """
        Store reward in the model's memory.
        
        Args:
            reward: The reward value
            done: Whether the episode is done
        """
        # Distribute the reward among players if needed
        num_players = len(self.model.memories[0][2]) if self.model.memories else 1
        rewards = [reward / num_players] * num_players
        
        self.model.store_rewards(rewards, done)
    
    def update(self):
        """
        Update the model (train).
        """
        return self.model.update()
    
    def _get_num_players_from_observation(self, observation):
        """
        Determine the number of players from the observation size.
        This is a simple heuristic - adjust based on your state structure.
        
        Args:
            observation: Flattened state from the Gym environment
        
        Returns:
            int: Number of players
        """
        # Use the number of players from environment parameters
        return self.env_params.NUMBER_OF_PLAYERS
    
    def _reshape_observation_to_states(self, observation, num_players):
        """
        Reshape the flattened observation into player states.
        
        Args:
            observation: Flattened state from the Gym environment
            num_players: Number of players
            
        Returns:
            list: List of player states
        """
        # This is a placeholder. Implement based on your specific state structure.
        # For example:
        state_size = len(observation) // num_players
        states = []
        
        for i in range(num_players):
            start = i * state_size
            end = (i + 1) * state_size
            state = observation[start:end].tolist() if hasattr(observation, 'tolist') else list(observation[start:end])
            states.append(state)
        
        return states
    
    def _flatten_actions(self, actions):
        """
        Flatten player actions into a single array as expected by Gym.
        
        Args:
            actions: List of actions per player
            
        Returns:
            array: Flattened actions
        """
        # Convert list of player actions to flat array
        flat_actions = []
        for player_action in actions:
            flat_actions.extend(player_action)
        
        return np.array(flat_actions) 