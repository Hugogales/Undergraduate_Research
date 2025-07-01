"""
PPO (Proximal Policy Optimization) Agent for Soccer Environment

This module implements a PPO agent adapted from the old codebase
but modernized for the new environment structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from typing import List, Dict, Optional, Tuple


class Memory:
    """Memory buffer for storing experience."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def clear(self):
        """Clear all stored experiences."""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
    
    def size(self):
        """Get the number of stored experiences."""
        return len(self.states)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO - supports both discrete and continuous actions."""
    
    def __init__(self, state_size: int, action_size: int = 18, hidden_size: int = 256, discrete_actions: bool = True):
        super(ActorCriticNetwork, self).__init__()
        
        self.discrete_actions = discrete_actions
        self.action_size = action_size
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        if discrete_actions:
            # Actor head for discrete actions (outputs logits)
            self.actor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
            )
        else:
            # Actor head for continuous actions (outputs mean and std)
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
                nn.Tanh()  # Actions are in [-1, 1] range
            )
            
            self.actor_std = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
                nn.Softplus()  # Ensure positive std
            )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network."""
        shared_features = self.shared(state)
        
        if self.discrete_actions:
            action_logits = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_logits, value
        else:
            action_mean = self.actor_mean(shared_features)
            action_std = self.actor_std(shared_features) + 1e-6
            value = self.critic(shared_features)
            return action_mean, action_std, value
    
    def get_action_and_value(self, state):
        """Get action and value for a given state."""
        if self.discrete_actions:
            action_logits, value = self.forward(state)
            
            # Create categorical distribution
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return action, log_prob, entropy, value
        else:
            action_mean, action_std, value = self.forward(state)
            
            # Create normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return action, log_prob, entropy, value
    
    def evaluate_actions(self, state, action):
        """Evaluate actions for PPO update."""
        if self.discrete_actions:
            action_logits, value = self.forward(state)
            
            # Create categorical distribution
            dist = torch.distributions.Categorical(logits=action_logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return log_prob, entropy, value
        else:
            action_mean, action_std, value = self.forward(state)
            
            # Create normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return log_prob, entropy, value


class PPOAgent:
    """PPO Agent for soccer environment."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 4,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        epsilon_clip: float = 0.2,
        k_epochs: int = 4,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space  
            hidden_size: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            lambda_: GAE lambda parameter
            epsilon_clip: PPO clipping parameter
            k_epochs: Number of training epochs per update
            batch_size: Mini-batch size
            vf_coef: Value function loss coefficient
            ent_coef: Entropy loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"PPO Agent using device: {self.device}")
        
        # Hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy = ActorCriticNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = Memory()
        
        print(f"PPO Network parameters: {sum(p.numel() for p in self.policy.parameters())}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Select action for given state.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action_and_value(state_tensor)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()
        
        # Clamp actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action, log_prob, value
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        log_prob: float, 
        reward: float, 
        done: bool,
        value: float
    ):
        """Store transition in memory."""
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.log_probs.append(log_prob)
        self.memory.rewards.append(reward)
        self.memory.dones.append(done)
        self.memory.values.append(value)
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO."""
        if self.memory.size() == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        dones = torch.BoolTensor(self.memory.dones).to(self.device)
        values = torch.FloatTensor(self.memory.values).to(self.device)
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
        # Mini-batch training
        batch_size = min(self.batch_size, len(states))
        num_batches = len(states) // batch_size
        
        for epoch in range(self.k_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                log_probs, entropy, state_values = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Actor loss (PPO objective)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch_advantages
                actor_loss_batch = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss_batch = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss_batch = -entropy.mean()
                
                # Total loss
                loss = (actor_loss_batch + 
                       self.vf_coef * critic_loss_batch + 
                       self.ent_coef * entropy_loss_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                actor_loss += actor_loss_batch.item()
                critic_loss += critic_loss_batch.item()
                entropy_loss += entropy_loss_batch.item()
        
        # Clear memory
        self.memory.clear()
        
        # Return metrics
        total_updates = self.k_epochs * num_batches
        return {
            'total_loss': total_loss / total_updates,
            'actor_loss': actor_loss / total_updates,
            'critic_loss': critic_loss / total_updates,
            'entropy_loss': entropy_loss / total_updates,
        }
    
    def _compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_value = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = ~dones[step]
                next_value = 0  # Terminal state
            else:
                next_non_terminal = ~dones[step]
                next_value = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * self.lambda_ * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        return advantages, returns
    
    def save(self, filepath: str):
        """Save agent model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Agent loaded from {filepath}")
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        if training:
            self.policy.train()
        else:
            self.policy.eval() 