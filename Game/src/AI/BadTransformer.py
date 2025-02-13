import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from params import AIHyperparameters
import copy
import os

# ---------------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------------
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]


# ---------------------------------------------------------------------------------
# NEW: Attention-based Actor-Critic Network
# ---------------------------------------------------------------------------------
class AttentionActorCriticNetwork(nn.Module):
    """
    1) Each agent's state -> MLP -> embedding
    2) Self-Attention among embeddings
    3) Actor head: distribution per agent
    4) Critic head: value per agent
    """
    def __init__(self, state_size, action_size, num_heads=4, embedding_dim=384):
        super(AttentionActorCriticNetwork, self).__init__()

        AI_PARAMS = AIHyperparameters()
        self.temperature = AI_PARAMS.temperature
        self.action_size = action_size
        self.emb_dim = embedding_dim

        # 1) MLP to transform each agent's state into an embedding
        self.embedding_mlp = nn.Sequential(
            nn.Linear(state_size, 768),
            nn.ReLU(),
            nn.Linear(768, embedding_dim),
            nn.ReLU()
        )

        # 2) Multi-head self-attention block
        #    We treat each agent as a 'token' in the sequence dimension
        self.attention_block = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True  # so input can be [batch_size, seq_len, embed_dim]
        )

        # 3) Actor head: transforms each post-attention embedding -> action logits
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, 768),
            nn.ReLU(),
            nn.Linear(768, action_size)
        )

        # 4) Critic head: transforms each post-attention embedding -> scalar value
        #    We'll produce one value per agent
        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")


    def forward(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs: [batch_size, num_agents, action_size]
          state_values: [batch_size, num_agents]
        """
        B, N, D = x.shape

        # 1) Agent-level embeddings: shape [B, N, emb_dim]
        embeddings = self.embedding_mlp(x)  # [B, N, emb_dim]

        # 2) Self-attention
        #    If using nn.MultiheadAttention with batch_first=True,
        #    we can directly pass [B, N, emb_dim] as (query, key, value)
        attn_output, _ = self.attention_block(embeddings, embeddings, embeddings)
        # attn_output is [B, N, emb_dim]

        # 3) Flatten to apply actor/critic heads
        flat = attn_output.reshape(B*N, self.emb_dim)

        # Actor head
        logits = self.actor(flat)  # [B*N, action_size]
        action_probs = torch.softmax(logits / self.temperature, dim=-1)
        action_probs = action_probs.reshape(B, N, self.action_size)  # unflatten

        # Critic head
        values = self.critic(flat).squeeze(-1)  # [B*N]
        values = values.reshape(B, N)  # unflatten to [B, N]

        return action_probs, values


# ---------------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------------
class BadTransformerPPOAgent:
    def __init__(self, mode="train"):
        AI_PARAMS = AIHyperparameters()

        self.mode = "train"
        self.memories = []

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

        # # not changed: building policy networks
        # -------------------------------------------------------------
        self.policy = self._build_model().to(self.device)
        self.policy_old = self._build_model().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # # not changed: Optimizer
        # -------------------------------------------------------------
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # # not changed: Loss function
        # -------------------------------------------------------------
        self.MseLoss = nn.MSELoss()

    # # not changed
    def lr_lambda(self, epoch):
        initial_lr = self.learning_rate
        final_lr = self.min_learning_rate
        total_epochs = self.episodes
        lr = final_lr + (initial_lr - final_lr) * (1 - epoch / total_epochs)
        return max(lr / initial_lr, final_lr / initial_lr)

    # ---------------------------------------------------------------------------------
    # # changed: _build_model now returns our new attention-based network
    # ---------------------------------------------------------------------------------
    def _build_model(self):
        return AttentionActorCriticNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            num_heads=4,        # Feel free to adjust
            embedding_dim=128   # Feel free to adjust
        )

    # ---------------------------------------------------------------------------------
    # # changed: we now select actions for all agents in one forward pass
    # ---------------------------------------------------------------------------------
    def select_action(self, states):
        """
        states shape: [num_agents, state_dim]
        This function will add a batch dimension of size 1, pass it through policy_old,
        and return a list of (action, logprob, value, entropy) for each agent.
        """
        states_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)  # [1, num_agents, state_dim]
        with torch.no_grad():
            action_probs, values = self.policy_old(states_tensor)  # shapes: [1, N, action_size], [1, N]
        action_probs = action_probs[0]  # [N, action_size]
        values = values[0]             # [N]

        # For each agent, sample an action
        actions = []
        log_probs = []
        entropies = []
        for i in range(action_probs.size(0)):
            dist = torch.distributions.Categorical(action_probs[i])
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            entropies.append(dist.entropy().item())

        # Values is a vector of shape [N], one value per agent
        return actions, log_probs, values.cpu().numpy(), entropies

    # # not changed: Just mapping integer actions -> environment inputs
    def _action_to_input(self, action):
        action_mapping = {
            0: [1, 0, 0, 0, 0],   # Up - drible
            1: [0, 1, 0, 0, 0],   # Down - drible
            2: [0, 0, 1, 0, 0],   # Left - drible
            3: [0, 0, 0, 1, 0],   # Right - drible
            4: [1, 0, 1, 0, 0],   # Up-Left - drible
            5: [1, 0, 0, 1, 0],   # Up-Right - drible
            6: [0, 1, 1, 0, 0],   # Down-Left - drible
            7: [0, 1, 0, 1, 0],   # Down-Right - drible
            8: [0, 0, 0, 0, 0],   # No movement - drible
            9: [1, 0, 0, 0, 1],   # Up - shoot
            10: [0, 1, 0, 0, 1],  # Down - shoot
            11: [0, 0, 1, 0, 1],  # Left - shoot
            12: [0, 0, 0, 1, 1],  # Right - shoot
            13: [1, 0, 1, 0, 1],  # Up-Left - shoot
            14: [1, 0, 0, 1, 1],  # Up-Right - shoot
            15: [0, 1, 1, 0, 1],  # Down-Left - shoot
            16: [0, 1, 0, 1, 1],  # Down-Right - shoot
            17: [0, 0, 0, 0, 1]   # No movement - shoot
        }
        return action_mapping.get(action, [0, 0, 0, 0])

    # ---------------------------------------------------------------------------------
    # # changed: get_actions merges states into one batch, calls select_action once
    # ---------------------------------------------------------------------------------
    def get_actions(self, states):
        """
        states: list of shape [num_agents, state_dim] if we stacked them,
                or a list of length N each of shape [state_dim].
        Return: (actions, entropies)
          - actions is a list (len num_agents) of the discrete moves
          - entropies is a list (len num_agents) of the entropies
        """
        # Convert states into a single array [num_agents, state_dim]
        state_tensor = np.array(states)  # shape [num_agents, state_dim]

        # Single call to select_action
        actions_indices, log_probs, values, entropies = self.select_action(state_tensor)

        # If in train mode, store them into memory
        if self.mode == "train":
            for i in range(len(actions_indices)):
                self.memories[i].states.append(states[i])
                self.memories[i].actions.append(actions_indices[i])
                self.memories[i].log_probs.append(log_probs[i])
                self.memories[i].state_values.append(values[i])

        # Convert discrete actions to environment input
        actions = []
        for a_idx in actions_indices:
            actions.append(self._action_to_input(a_idx))

        return actions, entropies

    # # not changed: store_rewards
    def store_rewards(self, rewards, done):
        if self.mode != "train":
            return
        for i in range(len(rewards)):
            self.memories[i].rewards.append(rewards[i])
            self.memories[i].dones.append(done)

    # ---------------------------------------------------------------------------------
    # # changed: update to reshape states -> [T, num_agents, state_dim] in each mini-batch
    # ---------------------------------------------------------------------------------
    def update(self):
        if self.mode != "train":
            return

        # Combine experiences from all memories
        states = []
        actions = []
        log_probs = []
        returns_list = []
        advantages_list = []

        for memory in self.memories:
            old_states = torch.FloatTensor(memory.states).to(self.device)
            old_actions = torch.LongTensor(memory.actions).to(self.device)
            old_log_probs = torch.FloatTensor(memory.log_probs).to(self.device)
            state_values_tensor = torch.FloatTensor(memory.state_values).to(self.device).squeeze()

            # Compute returns and advantages
            rewards = memory.rewards
            dones = memory.dones
            returns_tensor, advantages_tensor = self.compute_gae(rewards, dones, state_values_tensor)

            states.append(old_states)
            actions.append(old_actions)
            log_probs.append(old_log_probs)
            returns_list.append(returns_tensor)
            advantages_list.append(advantages_tensor)

            memory.clear()

        # We now have lists from each agent. We do a vertical stack because each memory is T timesteps for that agent
        # If we had N agents, each memory is length T, total is T*N
        states = torch.stack(states, dim=1)      # shape [T, N, state_dim]
        actions = torch.stack(actions, dim=1)    # shape [T, N]
        log_probs = torch.stack(log_probs, dim=1)   # shape [T, N]
        returns_list = torch.stack(returns_list, dim=1)   # shape [T, N]
        advantages_list = torch.stack(advantages_list, dim=1)   # shape [T, N]

        # Flatten T and N -> single dimension for shuffle
        T, N, D = states.shape[0], states.shape[1], states.shape[2]
        # flatten states to [T*N, D]
        states_flat = states.view(T*N, D)
        actions_flat = actions.view(T*N)
        log_probs_flat = log_probs.view(T*N)
        returns_flat = returns_list.view(T*N)
        advantages_flat = advantages_list.view(T*N)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-5)

        # Shuffle
        dataset_size = T*N
        indices = torch.randperm(dataset_size)
        states_flat = states_flat[indices]
        actions_flat = actions_flat[indices]
        log_probs_flat = log_probs_flat[indices]
        returns_flat = returns_flat[indices]
        advantages_flat = advantages_flat[indices]

        # Mini-batches
        mini_batch_size = self.mini_batch_size
        num_mini_batches = dataset_size // mini_batch_size

        for _ in range(self.K_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size

                mb_states = states_flat[start:end]       # [MB, D]
                mb_actions = actions_flat[start:end]     # [MB]
                mb_log_probs = log_probs_flat[start:end] # [MB]
                mb_returns = returns_flat[start:end]     # [MB]
                mb_advantages = advantages_flat[start:end]

                # Reshape mini-batch to [MB//N, N, D] for forward pass (if MB is multiple of N)
                # Because we want [batch_size, num_agents, state_dim] for attention
                # If MB not divisible by N, you need more advanced handling or bucket your mini-batches carefully.
                # For simplicity, assume MB is divisible by N here.
                if mini_batch_size % N != 0:
                    # fallback: skip or handle remainder carefully
                    continue

                mb_size = mb_states.shape[0]  # should be mini_batch_size
                time_steps = mb_size // N
                mb_states_reshaped = mb_states.view(time_steps, N, D)
                
                # Forward pass
                action_probs, state_values_new = self.policy(mb_states_reshaped)
                # action_probs: [time_steps, N, action_size]
                # state_values_new: [time_steps, N]

                # Flatten back [time_steps*N, action_size]
                action_probs_flat = action_probs.view(mb_size, self.action_size)
                state_values_flat = state_values_new.view(mb_size)

                # Build new distributions
                dist = torch.distributions.Categorical(action_probs_flat)
                new_log_probs = dist.log_prob(mb_actions)
                dist_entropy = dist.entropy()

                # Compute ratio
                ratios = torch.exp(new_log_probs - mb_log_probs)

                # Surrogate objectives
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mb_advantages

                # PPO loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_flat, mb_returns)
                entropy_loss = dist_entropy.mean()

                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * entropy_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # Handle leftover if any (similar approach)...

        # After K epochs, update policy_old
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()

    # # not changed
    def compute_gae(self, rewards, dones, values):
        if self.mode != "train":
            return
        gamma = self.gamma
        advantages = []
        gae = 0
        values = values.detach().cpu().numpy()
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i + 1] if i+1 < len(values) else 0
            else:
                next_value = values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + values
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)

    # # not changed
    def save_model(self, model_name="PPO_model_giant"):
        path = f"files/Models/{model_name}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    # # not changed
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

    # # not changed
    def memory_prep(self, number_of_agents):
        if self.mode != "train":
            return
        for memory in self.memories:
            memory.clear()
        self.memories = []
        for _ in range(number_of_agents):
            self.memories.append(Memory())

    # # not changed
    def clone(self):
        return copy.deepcopy(self)

    def assign_device(self, device):
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)