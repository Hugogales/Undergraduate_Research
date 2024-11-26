import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from params import AIHyperparameters
import os

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_size, 526),
            nn.ReLU(),
            nn.Linear(526, 526),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(526, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(526, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
       
        AI_PARAMS = AIHyperparameters()
        self.temperature = AI_PARAMS.temperature

    def forward(self, x):
        x = self.common(x)
        action_probs = torch.softmax(self.actor(x)/ self.temperature, dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self):
        AI_PARAMS = AIHyperparameters()

        self.state_size = AI_PARAMS.STATE_SIZE
        self.action_size = AI_PARAMS.ACTION_SIZE
        self.gamma = AI_PARAMS.gamma
        self.epsilon_clip = AI_PARAMS.epsilon_clip
        self.K_epochs = AI_PARAMS.K_epochs
        self.learning_rate = AI_PARAMS.learning_rate
        self.c_entropy = AI_PARAMS.c_entropy
        self.max_grad_norm = AI_PARAMS.max_grad_norm
        self.c_value = AI_PARAMS.c_value
        self.lam = AI_PARAMS.lam

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy network and old policy network
        self.policy = self._build_model().to(self.device)
        self.policy_old = self._build_model().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Loss function
        self.MseLoss = nn.MSELoss()

    def _build_model(self):
        return ActorCriticNetwork(self.state_size, self.action_size)

    def select_action(self, state):
        try :
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs, state_value = self.policy_old(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        except ValueError as e:
            print(e)
            print(f"state tensor: {state_tensor}")
            print(f"action_probs: {action_probs}")
            raise ValueError("Break")

        return action.item(), action_log_prob.item(), state_value.item()
    
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

    def update(self, memories):
        """
        Performs PPO update using experiences stored in the provided memories.

        :param memories: List of Memory instances (one for each agent)
        """
        # Combine experiences from all memories
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        state_values = []
        returns = []
        advantages = []

        for memory in memories:
            # Convert lists to tensors
            old_states = torch.FloatTensor(memory.states).to(self.device)
            old_actions = torch.LongTensor(memory.actions).to(self.device)
            old_log_probs = torch.FloatTensor(memory.log_probs).to(self.device)
            state_values_tensor = torch.FloatTensor(memory.state_values).to(self.device).squeeze()

            # Compute returns and advantages for this memory
            rewards = memory.rewards
            dones = memory.dones
            returns_tensor, advantages_tensor = self.compute_gae(rewards, dones, state_values_tensor)

            # Append to the combined lists
            states.append(old_states)
            actions.append(old_actions)
            log_probs.append(old_log_probs)
            state_values.append(state_values_tensor)
            returns.append(returns_tensor)
            advantages.append(advantages_tensor)

            # Clear memory after processing
            memory.clear()

        # Concatenate experiences from both agents
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        returns = torch.cat(returns, dim=0)
        advantages = torch.cat(advantages, dim=0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO policy update
        for _ in range(self.K_epochs):
            # Get action probabilities and1 state values from the policy network
            action_probs, state_values_new = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()

            # Calculate the ratios
            ratios = torch.exp(action_log_probs - log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

            # Calculate loss
            loss = -torch.min(surr1, surr2).mean() + \
                   self.c_value * self.MseLoss(state_values_new.squeeze(), returns) - \
                   self.c_entropy * dist_entropy.mean()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())

    def compute_gae(self, rewards, dones, values):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        :param rewards: List of rewards for an episode.
        :param dones: List of done flags for an episode.
        :param values: Tensor of state values.
        :return: Tensors of returns and advantages.
        """
        gamma = self.gamma
        advantages = []
        gae = 0
        values = values.detach().cpu().numpy()
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i + 1]
            else:
                next_value = values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + values
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
        

    def save_model(self, model_name="PPO_model_giant"):
        path = f"files/Models/{model_name}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, model_name="PPO_model_big", test=False):
        path = f"files/Models/{model_name}.pth"
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy_old.load_state_dict(self.policy.state_dict())
            if test:
                self.policy.eval()
                self.policy_old.eval()
            else:
                self.policy.train()
                self.policy_old.train()
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} does not exist.")
