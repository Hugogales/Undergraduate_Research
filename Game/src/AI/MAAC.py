import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from params import AIHyperparameters, EnvironmentHyperparameters
import copy
import os

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.action_state_values = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.action_state_values[:]


class AttentionActorCriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, num_heads=4, embedding_dim=384):
        super(AttentionActorCriticNetwork, self).__init__()

        AI_PARAMS = AIHyperparameters()
        self.temperature = AI_PARAMS.temperature
        self.action_size = action_size
        self.emb_dim = embedding_dim

        self.actor = nn.Sequential(
            nn.Linear(state_size, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 384),
            nn.LeakyReLU(),
            nn.Linear(384, action_size),
        )

        self.critic_embedding = nn.Sequential(
            nn.Linear(state_size + action_size, 768),
            nn.LeakyReLU(),
            nn.Linear(768, embedding_dim),
        )

        # 2) Multi-head self-attention block
        #    We treat each agent as a 'token' in the sequence dimension
        self.critic_attention_block = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True  # so input can be [batch_size, seq_len, embed_dim]
        )

        # 3) Actor head: transforms each post-attention embedding -> action logits
        self.critic_out = nn.Sequential(
            nn.Linear(embedding_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")


    def actor_forward(self, x):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs: [batch_size, num_agents, action_size]
        """
        action_probs = torch.softmax(self.actor(x)/ self.temperature, dim=-1) # [B, N, action_size]
        return action_probs
    
    def critic_forward(self, x, action_idx):
        """
        x.shape = [batch_size, num_agents, state_dim]
          - batch_size is # of timesteps or mini-batch size
          - num_agents is the number of agents
          - state_dim is the dimension of each agent's state
        Returns:
          action_probs: [batch_size, num_agents, action_size]
        """
        B, N, D = x.shape
        action_one_hot = torch.zeros(B, N, self.action_size).to(x.device) # [B, N, action_size]
        action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1) # [B, N, action_size]
        critic_input = torch.cat([x, action_one_hot], dim=-1) # [B, N, state_size + action_size]
        critic_input = critic_input.reshape(B*N, -1) # [B*N, state_size + action_size]
        critic_input = self.critic_embedding(critic_input) # [B*N, embedding_dim]
        critic_input = critic_input.reshape(B, N, -1) # [B, N, embedding_dim]
        attn_output, _ = self.critic_attention_block(critic_input, critic_input, critic_input) # [B, N, embedding_dim]
        values = self.critic_out(attn_output).squeeze(-1) # [B, N]
        values = values.reshape(B, N) # [B, N]
        return values


    def multi_agent_baseline(self, x, action_idx): # this was hard
        """
        x.shape = [B, N, state_dim]
        action_idx.shape = [B, N]
        Returns:
        baseline_values: [B, N]
        """
        B, N, state_dim = x.shape

        with torch.no_grad():
            #  Get actor probabilities (for weighting)
            full_action_probs = self.actor_forward(x)  # [B, N, A]

            #  Embedding for chosen action (one per agent/batch):
            chosen_action_one_hot = torch.zeros(B, N, self.action_size, device=x.device)
            chosen_action_one_hot.scatter_(-1, action_idx.unsqueeze(-1), 1)
            chosen_critic_input = torch.cat([x, chosen_action_one_hot], dim=-1)  # [B, N, state_dim + A]
            chosen_critic_input = chosen_critic_input.view(B*N, -1)              # [B*N, state_dim + A]
            chosen_emb = self.critic_embedding(chosen_critic_input)              # [B*N, emb_dim]
            chosen_emb = chosen_emb.view(B, N, -1)                               # [B, N, emb_dim]

            # Embedding for every possible action for every agent:
            x_expanded = x.unsqueeze(2).repeat(1, 1, self.action_size, 1)
            range_actions = torch.arange(self.action_size, device=x.device).view(1, 1, -1)
            action_range = range_actions.repeat(B, N, 1)  # [B, N, A]
            all_action_one_hot = torch.zeros(B, N, self.action_size, self.action_size, device=x.device)
            all_action_one_hot.scatter_( -1, action_range.unsqueeze(-1), 1)
            critic_input_all = torch.cat([x_expanded, all_action_one_hot], dim=-1)

            # Flatten to feed into critic_embedding
            critic_input_all = critic_input_all.view(B*N*self.action_size, -1)  # [B*N*A, state_dim + A]
            all_actions_emb = self.critic_embedding(critic_input_all)           # [B*N*A, emb_dim]
            all_actions_emb = all_actions_emb.view(B, N, self.action_size, -1)  # [B, N, A, emb_dim]

            # For each agent i, build the final “attention input”
            baseline_values = torch.zeros(B, N, device=x.device)

            for i in range(N):
                # Build a list of embeddings for all agents:
                agent_emb_list = chosen_emb.clone()  # shape => [B, N, emb_dim]

                # Replace i-th agent’s embedding with the “all possible actions” embeddings
                agent_emb_list = agent_emb_list.unsqueeze(2).repeat(1, 1, self.action_size, 1)
                agent_emb_list[:, i, :, :] = all_actions_emb[:, i, :, :]

                # Flatten to pass through attention -> [B*A, N, emb_dim]
                agent_emb_list = agent_emb_list.permute(0, 2, 1, 3)  # => [B, A, N, emb_dim]
                agent_emb_list = agent_emb_list.reshape(B*self.action_size, N, -1)  # => [B*A, N, emb_dim]

                # Pass through attention
                attn_output, _ = self.critic_attention_block(agent_emb_list, agent_emb_list, agent_emb_list) # [B*A, N, emb_dim]

                # Extract the i-th agent’s embedding 
                agent_output = attn_output[:, i, :]  # [B*A, emb_dim]

                agent_values = self.critic_out(agent_output).squeeze(-1)  # [B*A]
                agent_values = agent_values.view(B, self.action_size) # [B, A]
                agent_probs = full_action_probs[:, i, :]  # [B, A]
                agent_baseline = (agent_values * agent_probs).sum(dim=1)  # [B]
                baseline_values[:, i] = agent_baseline

            return baseline_values

class MAAC:
    def __init__(self, mode="train"):
        AI_PARAMS = AIHyperparameters()
        ENV_PARAMS = EnvironmentHyperparameters()

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
        self.td_dif_N = AI_PARAMS.TD_difference_N
        
        self.number_of_agents = ENV_PARAMS.NUMBER_OF_PLAYERS
        self.mini_batch_size = int(self.mini_batch_size // self.number_of_agents) # batches will be grouped by number of agents
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
        return AttentionActorCriticNetwork(self.state_size, self.action_size)

    def select_action(self, state):
        try :
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_probs = self.policy_old.actor_forward(state_tensor) # [1, num_agents, action_size]

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample() # [1, num_agents]
            action_log_prob = dist.log_prob(action) # [1, num_agents]
            entropy = dist.entropy() # [1, num_agents]
        except ValueError as e:
            print(e)
            print(f"state tensor: {state_tensor}")
            print(f"action_probs: {action_probs}")
            raise ValueError("Break")

        return action.item(), action_log_prob.item(), entropy.item()
    
    def _action_to_input(self, action):
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

    def update(self):
        """
        B = batch size or number of timesteps
        N = number of agents
        G = number of games
        """

        if self.mode != "train":
            return

        # Combine experiences from all memories
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        gae_returns = []
        advantages = []

        for i in range(0, len(self.memories), self.number_of_agents): # for each game
            old_states = [] # [N, B, state_size]
            old_actions = [] # [N, B]
            old_log_probs = [] # [N, B]
            rewards = [] # [N, B]
            dones = [] # [N, B]

            for j in range(self.number_of_agents): # for each agent
                old_states.append(self.memories[i+j].states)
                old_actions.append(self.memories[i+j].actions)
                old_log_probs.append(self.memories[i+j].log_probs)
                rewards.append(self.memories[i+j].rewards)
                dones.append(self.memories[i+j].dones)
            
            # Convert lists to tensors and reshape
            old_states = torch.FloatTensor(old_states).permute(1, 0, 2).to(self.device) # [B, N, state_size]
            old_actions = torch.LongTensor(old_actions).permute(1, 0).to(self.device) # [B, N]
            old_log_probs = torch.FloatTensor(old_log_probs).permute(1, 0).to(self.device) # [B, N]
            rewards = torch.FloatTensor(rewards).permute(1, 0).to(self.device) # [B, N]
            dones = torch.FloatTensor(dones).permute(1, 0).to(self.device) # [B, N]

            multi_agent_baseline = self.policy_old.multi_agent_baseline(old_states, old_actions) # [B, N]
            gae_returns_tensor, advantages_tensor = self.compute_gae(rewards, dones, multi_agent_baseline) # [B, N], [B, N]

            # Append to the combined lists
            states.append(old_states) 
            actions.append(old_actions)
            log_probs.append(old_log_probs)
            advantages.append(advantages_tensor)
            gae_returns.append(gae_returns_tensor)

            # Clear memory after processing
            for j in range(self.number_of_agents):
                self.memories[i+j]

        # Concatenate experiences from all agents
        states = torch.cat(states, dim=0) # [B*G , N, state_size]
        actions = torch.cat(actions, dim=0) # [B*G , N]
        log_probs = torch.cat(log_probs, dim=0) # [B*G , N]
        advantages = torch.cat(advantages, dim=0) # [B*G , N] 
        gae_returns = torch.cat(gae_returns, dim=0) # [B*G , N]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Shuffle the data
        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)
        states = states[indices]
        actions = actions[indices]
        log_probs = log_probs[indices]
        advantages = advantages[indices]
        gae_returns = gae_returns[indices]

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
                mini_states = states[start:end] # [mini_batch_size, N, state_size]
                mini_actions = actions[start:end] # [mini_batch_size, N]
                mini_log_probs = log_probs[start:end] # [mini_batch_size, N]
                mini_advantages = advantages[start:end] # [mini_batch_size, N]
                mini_gae_returns = gae_returns[start:end] # [mini_batch_size, N]
                
                # Forward pass
                action_probs = self.policy.actor_forward(mini_states) # [mini_batch_size, N, action_size]
                state_values_new = self.policy.critic_forward(mini_states, mini_actions) # [mini_batch_size, N]
                dist = torch.distributions.Categorical(action_probs) # [mini_batch_size, N]
                action_log_probs = dist.log_prob(mini_actions) # [mini_batch_size, N]
                dist_entropy = dist.entropy() # [mini_batch_size, N]
                
                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs) # [mini_batch_size, N]

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages

                # Calculate loss
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns) 
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean()

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
                mini_advantages = advantages[start:]
                mini_gae_returns = gae_returns[start:]

                # Forward pass
                action_probs = self.policy.actor_forward(mini_states)
                state_values_new = self.policy.critic_forward(mini_states, mini_actions)
                dist = torch.distributions.Categorical(action_probs)
                action_log_probs = dist.log_prob(mini_actions)
                dist_entropy = dist.entropy()

                # Calculate the ratios
                ratios = torch.exp(action_log_probs - mini_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * mini_advantages
                surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * mini_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values_new.squeeze(), mini_gae_returns)
                loss = actor_loss + self.c_value * critic_loss - self.c_entropy * dist_entropy.mean()
                # Calculate loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
        
        # Update old policy parameters with new policy parameters
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()

    
    def compute_gae(self, rewards, dones, baseline_values):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
    
        :param rewards: List of rewards for an episode. # [B, N]
        :param dones: List of done flags for an episode. # [B, N]
        :param baseline_values: Tensor of state baseline_values. # [B, N]
        :return: Tensors of returns.
        """
        if self.mode != "train":
            return

        # reshaping
        baseline_values = baseline_values.reshape(-1) # [B*N]
        dones = dones.reshape(-1) # [B*N]
        rewards = rewards.reshape(-1) # [B*N]

        baseline_values = baseline_values.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()

        gamma = self.gamma
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else baseline_values[i + 1]
            else:
                next_value = baseline_values[i + 1]
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - baseline_values[i]
            gae = delta + gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages)
        returns = advantages + baseline_values

        #reshaping and converting to tensor [B, N]
        returns = torch.FloatTensor(returns).reshape(-1, self.number_of_agents).to(self.device)
        advantages = torch.FloatTensor(advantages).reshape(-1, self.number_of_agents).to(self.device)
        return returns, advantages

    
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

    
    def memory_prep(self, number_of_agents):
        if self.mode != "train":
            return
        
        for memory in self.memories:
            memory.clear()

        self.memories = []
        for _ in range(number_of_agents):
            self.memories.append(Memory())

    
    def get_actions(self, states):
        actions = []
        entropies = []
        for i, state in enumerate(states):
            action, log_prob, entropy = self.select_action(state)
            if self.mode == "train":
                self.memories[i].states.append(state)
                self.memories[i].actions.append(action)
                self.memories[i].log_probs.append(log_prob)

            actions.append(self._action_to_input(action))
            entropies.append(entropy)
        
        return actions, entropies
    

    def store_rewards(self, rewards, done):
        if self.mode != "train":
            return

        for i in range(len(rewards)):
            self.memories[i].rewards.append(rewards[i])
            self.memories[i].dones.append(done)
        

    def clone(self):
        return copy.deepcopy(self)
    

    def assign_device(self, device):
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)