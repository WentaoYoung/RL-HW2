#!/usr/bin/env python
"""
Rainbow DQN实现的配置。
包含整个项目中使用的所有超参数和常量。
"""

import torch

# ----------  超参数 ----------
BOARD_SIZE_CENTRAL = 4               # 每个正方形 = 4×4
CURRICULUM_SIMPLE_EPISODES = 10_000  # 首先在4×4方块上玩
CURRICULUM_WIN_THRESHOLD = 50        # 进阶课程所需的连续胜利
CURRICULUM_STAGES = 4                # 课程阶段数量（包括roleplay）
N_STEPS = 3                          # n步回报的步数
GAMMA = 0.99                         # 折扣因子
CAPACITY = 50_000                    # 回放缓冲区容量
BATCH_SIZE = 128                     # 训练的批量大小
LEARNING_RATE = 1e-4                 # 学习率
TARGET_SYNC = 5_000                  # 目标网络同步频率（步数）
EPS_TERMINAL = 0.5                   # 胜利的奖励
EPS_INVALID = -0.05                  # 无效移动/失误的惩罚
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 