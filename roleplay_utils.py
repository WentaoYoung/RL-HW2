#!/usr/bin/env python
"""
Utility functions and classes for roleplay training.
Contains functionality, classes and helper functions for playing against historical models.
"""

import torch
import numpy as np
from pathlib import Path
import random
import os
import glob

from env import CrossTicTacToeEnv
from models import DuelingDQN
from config import DEVICE

class HistoricalModelOpponent:
    """Wrapper for using a historical model as opponent"""
    def __init__(self, model, curriculum_stage, env):
        self.model = model
        self.curriculum_stage = curriculum_stage
        self.env = env
        self.device = next(model.parameters()).device
        
    def select_action(self, board):
        """
        Select action using the historical model
        
        Args:
            board: Current board state
            
        Returns:
            action_id: Global board index for the selected action
        """
        # Create observation for opponent (from opponent's perspective)
        obs = np.stack([
            (board == -1),  # Opponent's pieces become 1 (first player)
            (board == 1),   # Agent's pieces become -1 (second player)
            (board == 0)    # Empty spaces remain the same
        ], axis=0).astype(np.float32)
        
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        
        # Get Q-values from historical model
        with torch.no_grad():
            q_values = self.model(obs)
        
        # Get the current valid cell IDs based on curriculum stage
        if self.curriculum_stage == 1:
            valid_ids = self.env.ids_simple
        elif self.curriculum_stage == 2:
            valid_ids = self.env.ids_medium
        else:
            valid_ids = self.env.ids
        
        # Map from action index to board index and vice versa
        action_to_board_idx = {i: board_idx for i, board_idx in enumerate(valid_ids)}
        board_idx_to_action = {board_idx: i for i, board_idx in enumerate(valid_ids)}
        
        # Get legal moves (empty cells)
        legal_indices = np.where(board == 0)[0]
        
        # Filter to only legal moves in the current action space
        legal_actions = [board_idx_to_action[idx] for idx in legal_indices if idx in board_idx_to_action]
        
        if not legal_actions:
            # If no legal actions in the current action space, use random
            return int(random.choice(legal_indices))
        
        # Get best legal action
        legal_q_values = q_values[0, legal_actions]
        best_action_idx = legal_actions[legal_q_values.argmax().item()]
        
        return action_to_board_idx[best_action_idx]

def load_historical_model(checkpoint_path):
    """Load a historical model from a checkpoint file."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract curriculum stage from the filename if available
    curriculum_stage = 3  # Default to full board
    if "stage" in Path(checkpoint_path).stem:
        try:
            stage_str = Path(checkpoint_path).stem.split("stage")[1][0]
            curriculum_stage = int(stage_str)
        except:
            print("Could not parse curriculum stage from filename, defaulting to stage 3")
    
    # Create a dummy environment to get the right observation shape and action space
    env = CrossTicTacToeEnv(curriculum=True)
    env.curriculum_stage = curriculum_stage
    CrossTicTacToeEnv._init_curriculum_boards()
    
    # Create the model with the correct dimensions
    obs_dim = env._get_obs().shape
    model = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()  # Set to evaluation mode
    
    return model, curriculum_stage, env

def get_sorted_opponent_checkpoints(checkpoints_dir="checkpoints"):
    """Get a sorted list of opponent checkpoints from oldest to newest"""
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    
    if not checkpoint_files:
        return []
        
    # Include both stage3 and stage4 models for full board play
    suitable_checkpoints = []
    for file in checkpoint_files:
        filename = Path(file).stem
        # Include both stage3 and stage4 models (they both play on full board)
        if ("stage3" in filename or "stage4" in filename) and "ep" in filename:
            try:
                ep_num = int(filename.split("ep")[1].split("_")[0])
                suitable_checkpoints.append((ep_num, file))
            except:
                continue
    
    # Sort by episode number (ascending)
    suitable_checkpoints.sort()
    return [file for _, file in suitable_checkpoints]

def get_checkpoint_id(checkpoint_path):
    """Extract a short identifier from the checkpoint filename"""
    filename = Path(checkpoint_path).stem
    if "ep" in filename:
        try:
            # Extract episode number
            ep_num = filename.split("ep")[1].split("_")[0]
            # If it contains stage info, add that too
            if "stage" in filename:
                stage = filename.split("stage")[1][0]
                return f"s{stage}ep{ep_num}"
            return f"ep{ep_num}"
        except:
            pass
    # Fallback to just using the filename
    return filename[:10]  # Limit length

def print_stats_summary(stats, episode_rewards, opponent_idx, opponent_ids, opponent_checkpoints, curriculum_stage, window=500):
    """Print detailed training statistics summary"""
    # Ensure there's enough data to calculate statistics
    if len(stats.recent_stats) == 0:
        return
        
    # Get the most recent statistics
    recent_window = stats.recent_stats[-min(window, len(stats.recent_stats)):]
    
    # Win rate statistics
    win_count = sum(1 for row in recent_window if row[3] == 1)  # win column
    loss_count = sum(1 for row in recent_window if row[4] == 1)  # loss column
    draw_count = sum(1 for row in recent_window if row[5] == 1)  # draw column
    
    win_percentage = (win_count / len(recent_window)) * 100
    loss_percentage = (loss_count / len(recent_window)) * 100
    draw_percentage = (draw_count / len(recent_window)) * 100
    
    # Invalid moves statistics
    invalid_moves = [row[6] for row in recent_window]  # invalid_moves column
    invalid_avg = np.mean(invalid_moves) if invalid_moves else 0
    
    # Reward statistics
    last_rewards = episode_rewards[-window:] if len(episode_rewards) >= window else episode_rewards
    avg_reward = np.mean(last_rewards)
    min_reward = np.min(last_rewards)
    max_reward = np.max(last_rewards)
    
    # Get current training step
    current_step = recent_window[-1][0]  # episode column
    
    # Current opponent information
    current_opponent_id = opponent_ids[opponent_idx]
    current_opponent_path = Path(opponent_checkpoints[opponent_idx]).name
    
    # Print statistics summary
    print("\n" + "="*80)
    print(f"TRAINING STATISTICS SUMMARY (episode {current_step}, last {len(recent_window)} games)")
    print("-"*80)
    print(f"Win rate: {win_percentage:.1f}% ({win_count}/{len(recent_window)})")
    print(f"Loss rate: {loss_percentage:.1f}% ({loss_count}/{len(recent_window)})")
    print(f"Draw rate: {draw_percentage:.1f}% ({draw_count}/{len(recent_window)})")
    print(f"Average invalid moves per game: {invalid_avg:.2f}")
    print("-"*80)
    print(f"Reward stats: avg={avg_reward:.3f}, min={min_reward:.3f}, max={max_reward:.3f}")
    print("-"*80)
    print(f"Curriculum stage: {curriculum_stage}")
    print(f"Current opponent: {opponent_idx+1}/{len(opponent_checkpoints)} (ID: {current_opponent_id})")
    print(f"Opponent model: {current_opponent_path}")
    print("="*80 + "\n") 