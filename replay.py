#!/usr/bin/env python
"""
Prioritized Replay Buffer implementation for Rainbow DQN.
"""

import numpy as np
import torch
from config import DEVICE

class PrioritizedReplay:
    """
    Prioritized Experience Replay buffer.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = 0.4   # Importance sampling correction (starts low, anneals to 1)
        self.beta_step = 1e-6  # Increment for beta at each sampling

    def push(self, *args):
        """
        Add a new experience to the buffer.
        Uses the maximum priority seen so far for new experiences.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((*args,))
        else:
            self.buffer[self.pos] = (*args,)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences with probability proportional to their priority.
        Returns a tuple of tensors for (state, action, reward, next_state, done),
        along with the indices sampled and importance sampling weights.
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** -self.beta
        self.beta = min(1.0, self.beta + self.beta_step)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        batch = list(zip(*samples))
        return (*map(torch.stack, batch), indices, weights)

    def update_priorities(self, indices, priorities):
        """
        Update priorities for the given indices.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio.detach().cpu().item() 