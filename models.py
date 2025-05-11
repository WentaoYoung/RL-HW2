#!/usr/bin/env python
"""
Neural network models for Rainbow DQN.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE

class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_eps', torch.empty(out_features))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self):
        self.weight_eps.normal_()
        self.bias_eps.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias   = self.bias_mu + self.bias_sigma   * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with NoisyNet
    """
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h = obs_shape[0], obs_shape[1]  # 3×80 -> flatten
        self.fc1 = nn.Linear(c*h, 256)
        self.noisy1 = NoisyLinear(256, 256)
        # value and advantage streams
        self.val = NoisyLinear(256, 1)
        self.adv = NoisyLinear(256, n_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        value = self.val(x)
        adv   = self.adv(x)
        q = value + adv - adv.mean(1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise() 