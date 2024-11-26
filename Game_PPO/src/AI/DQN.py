import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from params import AIHyperparameters, EnvironmentHyperparameters
import os
from torch.cuda.amp import GradScaler, autocast

class DQNAgent:
    def __init__(self):
        AI_PARAMS = AIHyperparameters()

        self.state_size = AI_PARAMS.STATE_SIZE  # Number of state features
        self.action_size = AI_PARAMS.ACTION_SIZE  # Number of possible actions
        # Hyperparameters
        self.gamma = AI_PARAMS.gamma  # Discount rate
        self.epsilon = AI_PARAMS.epsilon  # Exploration rate
        self.epsilon_min = AI_PARAMS.epsilon_min  # Minimum exploration rate
        self.epsilon_decay = AI_PARAMS.epsilon_decay  # Decay rate for exploration
        self.learning_rate = AI_PARAMS.learning_rate
        self.batch_size = AI_PARAMS.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ENV_PARAMS = EnvironmentHyperparameters()

        # Replay memory
        max_memory = 10000  # Increased replay memory size
        self.memory = deque(maxlen=max_memory)  # Replay memory

        # Neural network for approximating Q-values
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model
        self.target_model.to(self.device)
        self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Added weight decay for L2 regularization

        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler()

        # Target network update parameters
        self.target_update_freq = AI_PARAMS.target_update_freq  # e.g., every 1000 steps
        self.step_count = 0  # To track when to update target network

    def _build_model(self):
        # Define a more balanced feedforward neural network
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        model = model.to(self.device)
        return model

    def update_target_model(self):
        """
        Updates the target network by copying weights from the primary model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state):
        """
        Selects an action using an ε-greedy policy.
        
        :param state: The current state.
        :return: The selected action index.
        """
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action
            action = random.randrange(self.action_size)
        else:
            # Exploit: choose the best action based on current Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def _action_to_input(self, action):
        """
        Maps action index to movement input.
        
        :param action: The action index.
        :return: The movement input as a list.
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

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory.
        
        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state.
        :param done: Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Trains the network using mini-batches from replay memory.
        Also updates the target network periodically.
        """
        # Train only if enough samples are available in memory
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to train

        # Sample a mini-batch from the replay memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)

            # Compute the target Q-value
            with torch.no_grad():
                # Use the target network for computing the target
                target_q_values = self.target_model(next_state_tensor)
                max_target_q = torch.max(target_q_values).item()
                target = reward + (self.gamma * max_target_q * (1 - int(done)))

            # Get current Q-values from the primary network
            current_q_values = self.model(state_tensor)
            target_f = current_q_values.clone().detach()
            target_f[action] = target

            # Store states and targets for batch training
            states.append(state_tensor)
            targets.append(target_f)

        # Convert lists to tensors
        states_tensor = torch.stack(states).to(self.device)
        targets_tensor = torch.stack(targets).to(self.device)

        # Forward pass with autocast for mixed precision
        with autocast():
            outputs = self.model(states_tensor)
            loss = self.criterion(outputs, targets_tensor)

        # Backward pass and optimization with scaler
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Increment step count and update target network if needed
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_model()
            print("Target network updated.")


    def save_model(self, model_name="DQN_model"):
        """
        Saves the model's state dictionary to the specified path.
        
        :param model_name: The name of the model file.
        """
        path = f"files/Models/{model_name}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_name="DQN_model", test=False):
        """
        Loads the model's state dictionary from the specified path.
        
        :param model_name: The name of the model file.
        :param test: If True, sets the model to evaluation mode; else, training mode.
        """
        path = f"files/Models/{model_name}.pth"
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            if test:
                self.model.eval()  # Set the model to evaluation mode
            else:
                self.model.train()  # Set the model to training mode
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} does not exist.")
