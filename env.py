#!/usr/bin/env python
"""
Environment for Cross Tic-Tac-Toe game.
"""

import random
import numpy as np
import torch
from config import BOARD_SIZE_CENTRAL, EPS_INVALID, EPS_TERMINAL

class CrossTicTacToeEnv:
    """
    Board layout:
            0 1 2 3
            4 5 6 7
            8 9 10 11
            12 13 14 15
16 17 18 19 20 21 22 23 24 25 26 27
28 29 30 31 32 33 34 35 36 37 38 39
40 41 42 43 44 45 46 47 48 49 50 51
52 53 54 55 56 57 58 59 60 61 62 63
            64 65 66 67
            68 69 70 71
            72 73 74 75
            76 77 78 79

    = 80 legal cells.  index&rarr;(r,c) mapping stored in self.id2rc
    """
    SHIFTS = [-1, 1, 4, -4, 3, -3, 5, -5]           # 8 von-Neumann diagonal neighbours in 1-D layout per square
    WIN_LINES = None                                 # filled at init

    def __init__(self, curriculum=False):
        self.curriculum = curriculum
        self.simple_mode = curriculum               # start simple
        self.curriculum_stage = 1 if curriculum else 3  # Start at stage 1 if curriculum enabled
        self.reset()

    # -----  board helpers  -----
    @staticmethod
    def _make_cross_ids():
        ids, id2rc, rc2id = [], {}, {}
        # central square coordinates offset so everything positive
        coords = []
        for dr in range(-4, 8):     # generous range
            for dc in range(-4, 8):
                # define five 4×4 squares in plus shape
                central = (0 <= dr < 4 and 0 <= dc < 4)
                north   = (-4 <= dr < 0 and 0 <= dc < 4)
                south   = (4 <= dr < 8 and 0 <= dc < 4)
                west    = (0 <= dr < 4 and -4 <= dc < 0)
                east    = (0 <= dr < 4 and 4 <= dc < 8)
                if central or north or south or west or east:
                    coords.append((dr+4, dc+4))     # shift origin to positive
        coords = sorted(list(set(coords)))
        for idx, (r, c) in enumerate(coords):
            ids.append(idx)
            id2rc[idx] = (r, c)
            rc2id[(r, c)] = idx
        return ids, id2rc, rc2id

    ids, id2rc, rc2id = _make_cross_ids.__func__()   # static build
    
    # Define curriculum board layouts
    @classmethod
    def _init_curriculum_boards(cls):
        # Simple curriculum layout – just the central 4×4 (16 cells)
        cls.ids_simple = [i for i, (r, c) in cls.id2rc.items() if 4 <= r < 8 and 4 <= c < 8]
        
        # Medium curriculum layout – central plus one arm (32 cells)
        cls.ids_medium = [i for i, (r, c) in cls.id2rc.items() if (4 <= r < 8 and 4 <= c < 8) or 
                                                                (4 <= r < 8 and 0 <= c < 4)]
    
    @property
    def action_space(self):
        if self.curriculum_stage == 1:
            return len(self.ids_simple)
        elif self.curriculum_stage == 2:
            return len(self.ids_medium)
        else:
            return len(self.ids)
    
    def get_current_ids(self):
        if self.curriculum_stage == 1:
            return self.ids_simple
        elif self.curriculum_stage == 2:
            return self.ids_medium
        else:
            return self.ids

    def reset(self):
        self.turn = 1                                # 1 = agent (X), -1 = opponent (O)
        self.done = False
        self.winner = 0
        self.board = np.zeros(len(self.ids), dtype=np.int8)
        self.invalid_moves_this_ep = 0  # Add tracking for invalid moves
        return self._get_obs()

    def _get_obs(self):
        # map board to three-plane 3×80 tensor
        obs = np.stack([(self.board == 1),
                        (self.board == -1),
                        (self.board == 0)], axis=0).astype(np.float32)
        return torch.from_numpy(obs)

    # ----------  Step ----------
    def step(self, action: int):
        """
        action &isin; [0, action_space). 
        Perform the quirky ½-probability placement rule, then switch turn.
        The agent is always player 1 (X) and moves first each episode.
        """
        assert not self.done, "Episode finished"
        reward = 0.0
        random_choice = False

        # map to global action id
        ids_list = self.get_current_ids()
        cell_id = ids_list[action]

        chosen = cell_id
        placed = None
        if random.random() < 0.5:
            placed = chosen
        else:
            random_choice = True
            # pick among 8 neighbours
            neighbours = []
            for shift in self.SHIFTS:
                n = cell_id+shift
                if 0 <= n < len(self.ids):
                    neighbours.append(n)
                else:
                    neighbours.append(None)         # outside board
            placed = random.choice(neighbours)

        # attempt placement
        valid_placement = (placed is not None and self.board[placed] == 0)
        
        if not valid_placement:
            # Handle invalid placement
            self.invalid_moves_this_ep += 1
            if not random_choice:  # Only penalize direct invalid choices
                reward = EPS_INVALID
        else:
            # Handle valid placement
            self.board[placed] = self.turn
            
            # 添加中间奖励 - 计算潜在获胜线
            if self.turn == 1:  # 只为智能体（玩家1）添加形状奖励
                reward += self._calculate_shape_reward(placed)
                
            if self._check_win(placed):
                self.done = True
                self.winner = self.turn
                reward = EPS_TERMINAL

        # switch turn (opponent is random policy)
        self.turn *= -1
        if not self.done:
            self._opponent_move()

        if self.done and self.winner == 1:
            reward  = 1.0
        elif self.done and self.winner == -1:
            reward -= 1.0

        return self._get_obs(), reward, self.done, {}

    def _calculate_shape_reward(self, last_id):
        """计算基于棋型的中间奖励"""
        r, c = self.id2rc[last_id]
        player = 1  # 智能体总是玩家1
        shape_reward = 0.0
        
        # 定义回报权重
        weights = {
            2: 0.05,  # 2连，小奖励
            3: 0.1,   # 3连，更大奖励（如果课程阶段>1）
        }
        
        def count_line(dr, dc):
            cnt = 1
            for d in (1, -1):
                rr, cc = r, c
                while True:
                    rr += dr * d
                    cc += dc * d
                    nid = self.rc2id.get((rr, cc), None)
                    if nid is None or self.board[nid] != player:
                        break
                    cnt += 1
            return cnt
            
        # 检查所有方向的潜在连线
        for dr, dc in [(1,0), (0,1), (1,1), (1,-1)]:
            line_length = count_line(dr, dc)
            if line_length >= 2:
                shape_reward += weights.get(line_length, 0)
                
        return shape_reward

    # ----------  Opponent ----------
    def _opponent_move(self):
        legal = np.where(self.board == 0)[0]
        if len(legal) == 0:
            self.done = True
            return
        opp_action = int(random.choice(legal))
        self.board[opp_action] = self.turn
        if self._check_win(opp_action):
            self.done = True
            self.winner = self.turn
        self.turn *= -1

    # ----------  Win check ----------
    def _check_win(self, last_id: int) -> bool:
        r, c = self.id2rc[last_id]
        player = self.board[last_id]

        # 根据课程阶段设置合理的胜利条件
        # 在4×4的棋盘上，不可能有5连，所以第一阶段最多只能要求4连
        row_col_need = 4  # 行/列都需要4连
        if self.curriculum_stage == 1:
            diag_need = 4  # 第一阶段对角线也只能要求4连
        else:
            diag_need = 5  # 第二三阶段才可能有5连对角线

        def count_line(dr, dc):
            cnt = 1  # 当前位置算一个
            for d in (1, -1):  # 两个方向
                rr, cc = r, c
                while True:
                    rr += dr * d
                    cc += dc * d
                    nid = self.rc2id.get((rr, cc), None)
                    if nid is None or self.board[nid] != player:
                        break
                    cnt += 1
            return cnt

        # 检查行/列
        for dr, dc in [(1,0), (0,1)]:
            if count_line(dr, dc) >= row_col_need:
                return True
        # 检查对角线
        for dr, dc in [(1,1), (1,-1)]:
            if count_line(dr, dc) >= diag_need:
                return True
        return False 

# 在文件末尾添加RoleplayEnv类
class RoleplayEnv(CrossTicTacToeEnv):
    """继承自CrossTicTacToeEnv的环境，使用历史模型作为对手"""
    def __init__(self, curriculum=False):
        super().__init__(curriculum=curriculum)
        self.historical_opponent = None
        
    def set_historical_opponent(self, opponent):
        """设置历史模型对手"""
        self.historical_opponent = opponent
        
    def _opponent_move(self):
        """覆盖对手移动方法，使用历史模型（如果可用）"""
        legal = np.where(self.board == 0)[0]
        if len(legal) == 0:
            self.done = True
            return
            
        if self.historical_opponent is not None:
            # 使用历史模型进行移动
            opp_action = self.historical_opponent.select_action(self.board)
        else:
            # 使用随机策略
            opp_action = int(random.choice(legal))
            
        self.board[opp_action] = self.turn
        if self._check_win(opp_action):
            self.done = True
            self.winner = self.turn
        self.turn *= -1 