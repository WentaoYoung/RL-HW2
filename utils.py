#!/usr/bin/env python
"""
Utility functions and classes for the Rainbow DQN implementation.
"""

import time
import csv
import collections
from pathlib import Path
from typing import List

class StatsTracker:
    """
    Tracks training statistics and logs them to a CSV file.
    Logs: ep, time(s), reward, win, loss, draw, invalids, moves
    Saves to CSV every `flush_every` episodes.
    """
    headers = ["episode", "elapsed_sec", "reward", "win", "loss", "draw",
               "invalid_moves", "moves", "epsilon"]

    def __init__(self, out_path="stats.csv", flush_every=100, append=False):
        self.path = Path(out_path)
        self.flush_every = flush_every
        self.start = time.time()
        self.rows: List[List] = []
        
        # 添加记录最近500条记录的列表，用于计算统计数据
        self.recent_stats = collections.deque(maxlen=500)
        
        # Write header or append based on mode
        if not append:
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.headers)
        elif not self.path.exists():
            # If append mode but file doesn't exist, create it with header
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(self.headers)

    def log(self, episode, reward, winner, invalids, moves, epsilon):
        """
        Log stats for an episode.
        """
        row = [
            episode,
            round(time.time()-self.start, 2),
            round(float(reward), 3),
            int(winner == 1),
            int(winner == -1),
            int(winner == 0),
            invalids,
            moves,
            round(float(epsilon), 4),
        ]
        self.rows.append(row)
        self.recent_stats.append(row)  # 同时添加到最近统计数据
        if episode % self.flush_every == 0:
            with self.path.open("a", newline="") as f:
                csv.writer(f).writerows(self.rows)
            self.rows.clear()

    def close(self):
        """
        Close the stats tracker and flush any remaining data.
        """
        if self.rows:
            with self.path.open("a", newline="") as f:
                csv.writer(f).writerows(self.rows)
            self.rows.clear() 

class StatsTrackerExtended(StatsTracker):
    """StatsTracker的扩展版本，支持额外的列"""
    
    def __init__(self, out_path="stage4_roleplay_stats.csv", flush_every=100, append=False, 
                 extra_headers=None):
        self.extra_headers = extra_headers or ["curriculum_stage", "opponent_level", "opponent_id"]
        extended_headers = self.headers + self.extra_headers
        
        # 存储原始标题
        original_headers = self.headers
        
        # 临时替换标题以进行父类初始化
        self.headers = extended_headers
        
        # 调用父类初始化方法
        super().__init__(out_path, flush_every, append)
        
        # 恢复原始标题（父类期望这些）
        self.headers = original_headers
        
        # 使用列表而不是deque来支持切片
        self.recent_stats = []
        self.max_recent = 500
    
    def log(self, episode, reward, winner, invalids, moves, epsilon, extra_values=None):
        """
        记录一个集的统计数据，带有额外的值。
        """
        extra_values = extra_values or {}
        
        # 构建与父类相同的基本行
        row = [
            episode,
            round(time.time()-self.start, 2),
            round(float(reward), 3),
            int(winner == 1),
            int(winner == -1),
            int(winner == 0),
            invalids,
            moves,
            round(float(epsilon), 4),
        ]
        
        # 按照extra_headers中的顺序添加额外的值
        for header in self.extra_headers:
            row.append(extra_values.get(header, ""))
        
        self.rows.append(row)
        
        # 添加到最近的统计数据并维持最大长度
        self.recent_stats.append(row)
        if len(self.recent_stats) > self.max_recent:
            self.recent_stats = self.recent_stats[-self.max_recent:]
            
        if episode % self.flush_every == 0:
            with self.path.open("a", newline="") as f:
                csv.writer(f).writerows(self.rows)
            self.rows.clear() 