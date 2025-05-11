#!/usr/bin/env python
"""
Rainbow DQN Agent implementation.
"""

import random
import collections
import torch
import torch.nn as nn
from config import DEVICE, LEARNING_RATE, N_STEPS, GAMMA, CAPACITY, BATCH_SIZE
from models import DuelingDQN
from replay import PrioritizedReplay

class Agent:
    """
    Rainbow DQN agent with the following components:
    - Dueling architecture
    - Noisy Networks for exploration
    - N-step returns
    - Prioritized Experience Replay
    - Double Q-learning
    """
    def __init__(self, env):
        obs_dim = env._get_obs().shape
        self.policy_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
        self.target_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplay(CAPACITY)
        self.n_steps_buffer = collections.deque(maxlen=N_STEPS)
        self.step_counter = 0
        # 增加epsilon探索参数
        self.eps_start = 1.0  # 保持不变
        self.eps_end = 0.05   # 降低结束值，增加探索
        self.eps_decay_steps = 500_000  # 延长衰减时间，给予更多探索
        self.epsilon = self.eps_start
        self.env = env  # Store environment reference

    def get_valid_actions(self, state):
        """Get list of valid actions based on current board state and curriculum stage."""
        # Get the current board state from the observation
        board = state[2].numpy()  # Get the empty cells plane
        valid_actions = []
        
        # Get the current valid cell IDs based on curriculum stage
        current_ids = self.env.get_current_ids()
        
        # For each valid cell ID in the current curriculum stage
        for action, cell_id in enumerate(current_ids):
            if board[cell_id] == 1:  # If the cell is empty
                valid_actions.append(action)
        
        return valid_actions

    @torch.no_grad()
    def select_action(self, state):
        valid_actions = self.get_valid_actions(state)
        
        if not valid_actions:  # If no valid actions, return a random action (shouldn't happen in normal play)
            return random.randrange(self.policy_net.adv.out_features)
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        state = state.unsqueeze(0).to(DEVICE)
        q_values = self.policy_net(state) # Get Q-values for all actions in current curriculum space
        
        # Create a mask where True indicates an invalid action.
        # Initialize assuming all actions are invalid.
        # The q_values tensor has shape (1, num_actions_for_curriculum)
        invalid_action_mask = torch.ones_like(q_values, dtype=torch.bool)
        
        if valid_actions: # Ensure valid_actions is not empty before trying to index
            # Mark the valid actions as False (i.e., not invalid) in the mask.
            # valid_actions contains indices relative to the current action space.
            invalid_action_mask[0, valid_actions] = False 
        
        # Set Q-values of invalid actions to negative infinity.
        # This ensures they won't be chosen by argmax.
        masked_q_values = q_values.masked_fill(invalid_action_mask, float('-inf'))
        
        # Select the action with the highest Q-value among the (effectively) valid ones.
        action = masked_q_values.argmax(1).item()
        
        return action

    def update_epsilon(self):
        self.epsilon = max(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * 
            min(1.0, self.step_counter / self.eps_decay_steps)
        )

    def _calc_multistep_return(self):
        R = 0.0
        for i, trans in enumerate(self.n_steps_buffer):
            R += (GAMMA ** i) * trans[2]   # index 2 is reward
        return R

    def push(self, transition):
        self.n_steps_buffer.append(transition)
        if len(self.n_steps_buffer) < N_STEPS:
            return
        s0, a0, _, _, _ = self.n_steps_buffer[0]   # action is index 1
        _,  _, _, sn, dn = self.n_steps_buffer[-1]
        R = torch.tensor(self._calc_multistep_return(), dtype=torch.float32)
        self.memory.push(s0, a0, R, sn, dn)

    def optimize(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(BATCH_SIZE)

        states      = states.to(DEVICE)
        next_states = next_states.to(DEVICE)
        # Ensure actions is int64 type
        actions     = actions.to(torch.int64).unsqueeze(1).to(DEVICE)
        rewards     = rewards.to(DEVICE)
        dones       = dones.to(DEVICE)

        q_values      = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            targets = rewards + (GAMMA ** N_STEPS) * next_q * (1 - dones)

        loss = (q_values - targets).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(idxs, prios.detach())
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict()) 