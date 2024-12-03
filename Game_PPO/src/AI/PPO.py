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
        self.mini_batch_size = AI_PARAMS.batch_size
        self.min_learning_rate = AI_PARAMS.min_learning_rate
        self.episodes = AI_PARAMS.episodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize policy network and old policy network
        self.policy = self._build_model().to(self.device)
        self.policy_old = self._build_model().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # Loss function
        self.MseLoss = nn.MSELoss()

    def lr_lambda(self, epoch):
        initial_lr = self.learning_rate
        final_lr = self.min_learning_rate
        total_epochs = self.episodes
        lr = final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
        return max(lr / initial_lr, final_lr / initial_lr)


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
            0: [1, 0, 0, 0, 0],  # Up - drible
            1: [0, 1, 0, 0, 0],  # Down - drible
            2: [0, 0, 1, 0, 0],  # Left - drible
            3: [0, 0, 0, 1, 0],  # Right - drible
            4: [1, 0, 1, 0, 0],  # Up-Left - drible
            5: [1, 0, 0, 1, 0],  # Up-Right - drible
            6: [0, 1, 1, 0, 0],  # Down-Left - drible
            7: [0, 1, 0, 1, 0],  # Down-Right - drible
            8: [0, 0, 0, 0, 0],  # No movement - drible
            9: [1, 0, 0, 0, 1],  # Up - shoot
            10: [0, 1, 0, 0, 1],  # Down - shoot
            11: [0, 0, 1, 0, 1],  # Left - shoot
            12: [0, 0, 0, 1, 1],  # Right - shoot
            13: [1, 0, 1, 0, 1],  # Up-Left - shoot
            14: [1, 0, 0, 1, 1],  # Up-Right - shoot
            15: [0, 1, 1, 0, 1],  # Down-Left - shoot
            16: [0, 1, 0, 1, 1],  # Down-Right - shoot
            17: [0, 0, 0, 0, 1]  # No movement - shoot
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

        # Concatenate experiences from all agents
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        returns = torch.cat(returns, dim=0)
        advantages = torch.cat(advantages, dim=0)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Shuffle the data
        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)
        states = states[indices]
        actions = actions[indices]
        log_probs = log_probs[indices]
        returns = returns[indices]
        advantages = advantages[indices]

        # Define mini-batch size
        mini_batch_size = self.mini_batch_size  # e.g., 64
        num_mini_batches = dataset_size // mini_batch_size

        # PPO policy update with mini-batching
        for _ in range(self.K_epochs):
            for i in range(num_mini_batches):
                # Define the start and end of the mini-batch
                start = i * mini_batch_size
                end = start + mini_batch_size

                # Slice the mini-batch
                mini_states = states[start:end]
                mini_actions = actions[start:end]
                mini_log_probs = log_probs[start:end]
                mini_returns = returns[start:end]
                mini_advantages = advantages[start:end]

                # Forward pass
                action_probs, state_values_new = self.policy(mini_states)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages

                # Calculate loss
                loss = -torch.min(surr1, surr2).mean() + \
                    self.c_value * self.MseLoss(state_values_new.squeeze(), mini_returns) - \
                    self.c_entropy * dist_entropy.mean()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # Handle any remaining data not fitting into mini-batches
            remainder = dataset_size % mini_batch_size
            if remainder != 0:
                start = num_mini_batches * mini_batch_size
                mini_states = states[start:]
                mini_actions = actions[start:]
                mini_log_probs = log_probs[start:]
                mini_returns = returns[start:]
                mini_advantages = advantages[start:]

                # Forward pass
                action_probs, state_values_new = self.policy(mini_states)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages

                # Calculate loss
                loss = -torch.min(surr1, surr2).mean() + \
                    self.c_value * self.MseLoss(state_values_new.squeeze(), mini_returns) - \
                    self.c_entropy * dist_entropy.mean()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
        
        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()


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
