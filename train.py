#!/usr/bin/env python
"""
Unified training framework, supporting standard training (first 3 stages) and adversarial model training (4th stage)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import os
import glob
import time
import csv

from config import TARGET_SYNC, CURRICULUM_WIN_THRESHOLD, CURRICULUM_STAGES, DEVICE, LEARNING_RATE
from env import CrossTicTacToeEnv, RoleplayEnv
from agent import Agent
from models import DuelingDQN
from utils import StatsTracker, StatsTrackerExtended

def extract_stage_from_checkpoint(checkpoint_path):
    """Extract curriculum stage from checkpoint filename"""
    curriculum_stage = 1
    if "stage" in Path(checkpoint_path).stem:
        try:
            stage_str = Path(checkpoint_path).stem.split("stage")[1][0]
            curriculum_stage = int(stage_str)
        except:
            print("Failed to parse curriculum stage from filename, defaulting to stage 1")
    return curriculum_stage

def load_checkpoint(agent, checkpoint_path, env=None):
    """Load checkpoint and return starting episode and curriculum stage"""
    start_episode = 1
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Checkpoint file {checkpoint_path} does not exist")
        return start_episode, None
    
    # Load checkpoint file
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return start_episode, None
            
    # Extract curriculum stage from filename (if available)
    curriculum_stage = None
    if "stage" in checkpoint_path.stem:
        try:
            stage_str = checkpoint_path.stem.split("stage")[1][0]
            curriculum_stage = int(stage_str)
        except:
            print("Failed to parse curriculum stage from filename")
    
    # Load curriculum stage from checkpoint (if available)
    if curriculum_stage is None and 'curriculum_stage' in checkpoint:
        curriculum_stage = checkpoint['curriculum_stage']
    
    # If environment is provided and curriculum stage is found, set the environment's curriculum stage
    if env and curriculum_stage is not None:
        env.curriculum_stage = curriculum_stage
            
    # Create networks with the correct dimensions for the current curriculum stage
    if env:
        obs_dim = env._get_obs().shape
        agent.policy_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
        agent.target_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
            
        # Load weights
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=LEARNING_RATE)
            
        # If optimizer state is saved, load it
        if 'optimizer' in checkpoint:
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            
        # Load training state
        if 'episode' in checkpoint:
            start_episode = checkpoint['episode'] + 1
        if 'step_counter' in checkpoint:
            agent.step_counter = checkpoint['step_counter']
        if 'epsilon' in checkpoint:
            agent.epsilon = checkpoint['epsilon']
            
    return start_episode, checkpoint

def save_checkpoint(agent, save_dir, ep, curriculum_stage, extra_data=None, is_final=False):
    """Save checkpoint, supports extra data"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    prefix = "agent_final" if is_final else f"agent_ep{ep}"
    filename = f"{prefix}_stage{curriculum_stage}"
    
    # Add extra data to filename
    if extra_data:
        for key, value in extra_data.items():
            if key and value:
                filename += f"_{key}{value}"
    
    ckpt_path = save_dir / f"{filename}.pt"
    
    # Prepare checkpoint data
    checkpoint = {
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode': ep,
        'step_counter': agent.step_counter,
        'epsilon': agent.epsilon,
        'curriculum_stage': curriculum_stage
    }
    
    # Add extra data to checkpoint
    if extra_data:
        checkpoint.update(extra_data)
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path

def select_latest_checkpoint(checkpoints_dir="checkpoints", stage_filter=None):
    """Find the latest checkpoint, optionally filtered by stage"""
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    
    if not checkpoint_files:
        return None
    
    # First check for final models - these have highest priority
    final_models = [f for f in checkpoint_files if "final" in Path(f).stem.lower()]
    
    # If stage filter is specified, apply it
    if stage_filter:
        stage_filter_str = f"stage{stage_filter}"
        final_models = [f for f in final_models if stage_filter_str in Path(f).stem.lower()]
    
    # If we have a final model, use it
    if final_models:
        return final_models[0]
    
    # Otherwise, find checkpoint with highest episode number
    latest_ep = 0
    latest_file = None
    
    for file in checkpoint_files:
        filename = Path(file).stem
        if "ep" in filename:
            # Apply stage filter
            if stage_filter and f"stage{stage_filter}" not in filename:
                continue
                
            try:
                ep_num = int(filename.split("ep")[1].split("_")[0])
                if ep_num > latest_ep:
                    latest_ep = ep_num
                    latest_file = file
            except:
                continue
    
    return latest_file

def train(episodes=200_000, 
          curriculum=True, 
          save_dir="checkpoints", 
          log_csv="stats.csv", 
          resume_from=None,
          roleplay_mode=False,
          stats_interval=2000,
          opponent_pool_update_interval=5000,
          consecutive_wins_threshold=CURRICULUM_WIN_THRESHOLD):
    """
    Unified training framework, supporting standard training and roleplay training
    
    Args:
        episodes: Number of episodes to train (total across all stages including roleplay)
        curriculum: Whether to use curriculum learning
        save_dir: Directory to save checkpoints
        log_csv: Path to save statistics
        resume_from: Path to checkpoint to resume from
        roleplay_mode: Whether to use historical models as opponents after stage 3
        stats_interval: How often to print detailed statistics (in episodes)
        opponent_pool_update_interval: How often to update opponent pool (only in roleplay mode)
        consecutive_wins_threshold: Number of consecutive wins needed to advance
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize stage tracking
    in_roleplay_phase = False
    
    # Initially start with regular environment regardless of roleplay_mode
    env = CrossTicTacToeEnv(curriculum=curriculum)
    curriculum_stage = 1  # Always start at stage 1 unless resuming
    
    CrossTicTacToeEnv._init_curriculum_boards()
    
    # Initialize agent
    agent = Agent(env)
    
    # Set initial state
    start_episode = 1
    opponent_idx = 0
    opponent_checkpoints = []
    opponent_ids = []
    
    # If no resume path specified, try to automatically find latest checkpoint
    if resume_from is None:
        # For both modes, look for any latest checkpoint
        resume_from = select_latest_checkpoint(save_dir)
        
        if resume_from:
            print(f"Automatically selected latest checkpoint: {Path(resume_from).name}")
    
    # Resume from checkpoint
    extra_checkpoint_data = None
    if resume_from:
        start_episode, extra_checkpoint_data = load_checkpoint(agent, resume_from, env)
        
        # Load curriculum stage
        if extra_checkpoint_data and 'curriculum_stage' in extra_checkpoint_data:
            curriculum_stage = extra_checkpoint_data['curriculum_stage']
        else:
            curriculum_stage = extract_stage_from_checkpoint(resume_from)
        
        # Check if we should be in roleplay phase
        if curriculum_stage == 4:
            in_roleplay_phase = True
            
            # Switch to roleplay environment
            if roleplay_mode:
                # Save current agent state to load into new environment
                policy_state = agent.policy_net.state_dict()
                target_state = agent.target_net.state_dict()
                optimizer_state = agent.optimizer.state_dict()
                epsilon = agent.epsilon
                step_counter = agent.step_counter
                
                # Create roleplay environment
                actual_board_stage = 3  # Roleplay uses the full board
                if extra_checkpoint_data and 'board_size_stage' in extra_checkpoint_data:
                    actual_board_stage = extra_checkpoint_data['board_size_stage']
                
                # Switch to roleplay env
                env = RoleplayEnv(curriculum=curriculum)
                env.curriculum_stage = actual_board_stage
                
                # Recreate agent with new environment
                agent = Agent(env)
                agent.policy_net.load_state_dict(policy_state)
                agent.target_net.load_state_dict(target_state)
                agent.optimizer.load_state_dict(optimizer_state)
                agent.epsilon = epsilon
                agent.step_counter = step_counter
                
                # Handle opponent-specific information
                if 'opponent_idx' in extra_checkpoint_data:
                    opponent_idx = extra_checkpoint_data['opponent_idx']
            
        # Print resume information        
        stage_names = {1: "simple (central 4×4)", 2: "medium (central + one arm)", 3: "full cross", 4: "against historical models"}
        stage_name = stage_names.get(curriculum_stage, f"stage {curriculum_stage}")
        
        print(f"[Resume] Loaded checkpoint from {Path(resume_from)}")
        print(f"[Resume] Starting from episode {start_episode}, curriculum stage {curriculum_stage}: {stage_name}")
        print(f"[Resume] Epsilon: {agent.epsilon:.4f}, Step counter: {agent.step_counter}")
    
    # Ensure training will continue even if resuming from a higher episode count
    if start_episode > episodes:
        print(f"WARNING: start_episode ({start_episode}) > episodes ({episodes})")
        print("Adjusting episodes to continue training...")
        episodes = start_episode + 100000  # Add 100k more episodes
        print(f"New training target: {episodes} episodes")
    
    # Initialize stats tracker
    if in_roleplay_phase:
        stats = StatsTrackerExtended(out_path=log_csv, append=(start_episode > 1))
    else:
        stats = StatsTracker(out_path=log_csv, append=(start_episode > 1))

    # Handle roleplay preparation for both initial and resumed roleplay phase
    if in_roleplay_phase and roleplay_mode:
        from roleplay_utils import get_sorted_opponent_checkpoints, get_checkpoint_id, load_historical_model, HistoricalModelOpponent
        
        # Get sorted opponent checkpoints (from oldest to newest)
        opponent_checkpoints = get_sorted_opponent_checkpoints(save_dir)
        
        if not opponent_checkpoints:
            raise ValueError(f"No suitable opponent checkpoints found in {save_dir}")
        
        # Create checkpoint ID list for reference
        opponent_ids = [get_checkpoint_id(cp) for cp in opponent_checkpoints]
        
        print(f"[Roleplay] Found {len(opponent_checkpoints)} opponent checkpoints:")
        for i, (chk, chk_id) in enumerate(zip(opponent_checkpoints, opponent_ids)):
            print(f"  {i+1}: {Path(chk).name} (ID: {chk_id})")
            
        # Ensure opponent index is valid
        opponent_idx = min(opponent_idx, len(opponent_checkpoints) - 1)
            
        # Initialize current opponent
        current_opponent_checkpoint = opponent_checkpoints[opponent_idx]
        current_opponent_id = opponent_ids[opponent_idx]
        historical_model, historical_model_stage, _ = load_historical_model(current_opponent_checkpoint)
        
        # Create opponent wrapper and set it in the environment
        opponent = HistoricalModelOpponent(historical_model, historical_model_stage, env)
        env.set_historical_opponent(opponent)
        
        print(f"[Roleplay] Starting with opponent model: {Path(current_opponent_checkpoint).name}")
        print(f"[Roleplay] Opponent curriculum stage: {historical_model_stage}")
        print(f"[Roleplay] Opponent level: {opponent_idx+1}/{len(opponent_checkpoints)} (ID: {current_opponent_id})")
    
    # Print training info
    stage_names = {1: "simple (central 4×4)", 2: "medium (central + one arm)", 3: "full cross", 4: "against historical models"}
    mode_description = "Roleplay mode enabled" if roleplay_mode else "Standard training mode"
    stage_name = stage_names.get(curriculum_stage, f"stage {curriculum_stage}")
    
    print(f"[Training] {mode_description}, from episode {start_episode} to {episodes}")
    print(f"[Training] Current curriculum stage: {curriculum_stage} ({stage_name})")
    if roleplay_mode and not in_roleplay_phase:
        print(f"[Training] Will proceed to roleplay (stage 4) after completing stage 3")
    
    # Initialize training statistics
    win_streak = 0
    current_opponent_episodes = 0  # Only for roleplay phase
    episode_rewards = []
    last_pool_update = start_episode  # Only for roleplay phase
    
    # Training loop
    for ep in range(start_episode, episodes + 1):
        # For roleplay phase, periodically update opponent pool
        if in_roleplay_phase and roleplay_mode and opponent_pool_update_interval > 0 and ep - last_pool_update >= opponent_pool_update_interval:
            from roleplay_utils import get_sorted_opponent_checkpoints, get_checkpoint_id
            
            old_pool_size = len(opponent_checkpoints)
            new_opponent_checkpoints = get_sorted_opponent_checkpoints(save_dir)
            new_opponent_ids = [get_checkpoint_id(cp) for cp in new_opponent_checkpoints]
            
            if len(new_opponent_checkpoints) > old_pool_size:
                print(f"[Roleplay] Updating opponent pool...")
                print(f"[Roleplay] Found {len(new_opponent_checkpoints) - old_pool_size} new opponent model(s)")
                
                for i in range(old_pool_size, len(new_opponent_checkpoints)):
                    print(f"  New opponent {i+1}: {Path(new_opponent_checkpoints[i]).name} (ID: {new_opponent_ids[i]})")
                
                opponent_checkpoints = new_opponent_checkpoints
                opponent_ids = new_opponent_ids
                opponent_idx = min(opponent_idx, len(opponent_checkpoints) - 1)
                
                current_opponent_checkpoint = opponent_checkpoints[opponent_idx]
                current_opponent_id = opponent_ids[opponent_idx]
                
                print(f"[Roleplay] Updated opponent pool now has {len(opponent_checkpoints)} models")
            
            last_pool_update = ep
        
        # Reset environment and start new episode
        state = env.reset()
        ep_reward, moves = 0.0, 0
        
        # Single episode training loop
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push((
                state.cpu(),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                next_state.cpu(),
                torch.tensor(done, dtype=torch.float32)
            ))
            agent.optimize()
            agent.step_counter += 1
            agent.update_epsilon()
            state = next_state
            ep_reward += reward
            moves += 1
            if done:
                break

        # Update statistics
        if in_roleplay_phase:
            stats.log(ep, ep_reward, env.winner, env.invalid_moves_this_ep, moves, agent.epsilon,
                     extra_values={'curriculum_stage': curriculum_stage,
                                  'opponent_level': opponent_idx + 1 if opponent_checkpoints else 0,
                                  'opponent_id': current_opponent_id if opponent_checkpoints else ""})
        else:
            stats.log(ep, ep_reward, env.winner, env.invalid_moves_this_ep, moves, agent.epsilon)
        
        episode_rewards.append(ep_reward)

        # Update win streak counter
        if env.winner == 1:  # Agent won
            win_streak += 1
        else:
            win_streak = 0

        # Handle advancement logic
        if in_roleplay_phase and roleplay_mode:
            # Roleplay phase: progress opponent difficulty based on consecutive wins
            current_opponent_episodes += 1
            if opponent_checkpoints and win_streak >= consecutive_wins_threshold and opponent_idx < len(opponent_checkpoints) - 1:
                opponent_idx += 1
                current_opponent_checkpoint = opponent_checkpoints[opponent_idx]
                current_opponent_id = opponent_ids[opponent_idx]
                historical_model, historical_model_stage, _ = load_historical_model(current_opponent_checkpoint)
                
                # Update opponent
                opponent = HistoricalModelOpponent(historical_model, historical_model_stage, env)
                env.set_historical_opponent(opponent)
                
                print(f"[Roleplay] Achieved {win_streak} consecutive wins!")
                print(f"[Roleplay] Advanced to opponent level {opponent_idx+1}/{len(opponent_checkpoints)}")
                print(f"[Roleplay] New opponent model: {Path(current_opponent_checkpoint).name} (ID: {current_opponent_id})")
                
                # Reset counters
                win_streak = 0
                current_opponent_episodes = 0
        elif curriculum and env.curriculum_stage < 3 and win_streak >= consecutive_wins_threshold:
            # Standard training mode: progress curriculum stage based on consecutive wins
            env.curriculum_stage += 1
            curriculum_stage = env.curriculum_stage
            win_streak = 0
            
            # Create new networks for the new action space
            old_policy_state = agent.policy_net.state_dict()
            old_target_state = agent.target_net.state_dict()
            
            # Initialize new networks with the expanded action space
            obs_dim = env._get_obs().shape
            agent.policy_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
            agent.target_net = DuelingDQN(obs_dim, env.action_space).to(DEVICE)
            
            # Transfer weights for shared layers (not the final action layer)
            for name, param in old_policy_state.items():
                if 'adv' not in name:  # Don't copy advantage layer weights
                    agent.policy_net.state_dict()[name].copy_(param)
            
            for name, param in old_target_state.items():
                if 'adv' not in name:  # Don't copy advantage layer weights
                    agent.target_net.state_dict()[name].copy_(param)
            
            # Reset optimizer with new network parameters
            agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=LEARNING_RATE)
            
            stage_names = {1: "simple (central 4×4)", 2: "medium (central + one arm)", 3: "full cross"}
            print(f"[Curriculum] Advanced to stage {env.curriculum_stage}: {stage_names[env.curriculum_stage]} at episode {ep}")
        
        # Handle transition to roleplay mode after completing stage 3
        elif roleplay_mode and not in_roleplay_phase and curriculum_stage == 3 and win_streak >= consecutive_wins_threshold:
            print(f"[Curriculum] Completed stage 3 with {win_streak} consecutive wins at episode {ep}")
            print(f"[Curriculum] Transitioning to roleplay mode (stage 4)")
            
            # Save current agent state for stage 3
            save_checkpoint(agent, save_dir, ep, curriculum_stage)
            
            # Transition to roleplay mode
            in_roleplay_phase = True
            curriculum_stage = 4
            win_streak = 0
            
            # Save current agent state to load into new environment
            policy_state = agent.policy_net.state_dict()
            target_state = agent.target_net.state_dict()
            optimizer_state = agent.optimizer.state_dict()
            epsilon = agent.epsilon
            step_counter = agent.step_counter
            
            # Switch to roleplay env
            env = RoleplayEnv(curriculum=curriculum)
            env.curriculum_stage = 3  # Roleplay uses the full board
            
            # Recreate agent with new environment
            agent = Agent(env)
            agent.policy_net.load_state_dict(policy_state)
            agent.target_net.load_state_dict(target_state)
            agent.optimizer.load_state_dict(optimizer_state)
            agent.epsilon = epsilon
            agent.step_counter = step_counter
            
            # Switch to extended stats tracker
            old_stats = stats
            stats = StatsTrackerExtended(out_path=log_csv, append=True)
            stats.recent_stats = old_stats.recent_stats
            
            # Load opponents
            from roleplay_utils import get_sorted_opponent_checkpoints, get_checkpoint_id, load_historical_model, HistoricalModelOpponent
            
            # Get sorted opponent checkpoints (from oldest to newest)
            opponent_checkpoints = get_sorted_opponent_checkpoints(save_dir)
            
            if not opponent_checkpoints:
                raise ValueError(f"No suitable opponent checkpoints found in {save_dir}")
            
            # Create checkpoint ID list for reference
            opponent_ids = [get_checkpoint_id(cp) for cp in opponent_checkpoints]
            
            print(f"[Roleplay] Found {len(opponent_checkpoints)} opponent checkpoints:")
            for i, (chk, chk_id) in enumerate(zip(opponent_checkpoints, opponent_ids)):
                print(f"  {i+1}: {Path(chk).name} (ID: {chk_id})")
                
            # Initialize current opponent (start with the earliest/weakest)
            current_opponent_checkpoint = opponent_checkpoints[0]
            current_opponent_id = opponent_ids[0]
            historical_model, historical_model_stage, _ = load_historical_model(current_opponent_checkpoint)
            
            # Create opponent wrapper and set it in the environment
            opponent = HistoricalModelOpponent(historical_model, historical_model_stage, env)
            env.set_historical_opponent(opponent)
            
            print(f"[Roleplay] Starting with opponent model: {Path(current_opponent_checkpoint).name}")
            print(f"[Roleplay] Opponent curriculum stage: {historical_model_stage}")
            print(f"[Roleplay] Opponent level: 1/{len(opponent_checkpoints)} (ID: {current_opponent_id})")

        # Target network sync
        if agent.step_counter % TARGET_SYNC == 0:
            agent.sync_target()

        # Print detailed statistics
        if in_roleplay_phase and ep % stats_interval == 0:
            from roleplay_utils import print_stats_summary
            print_stats_summary(stats, episode_rewards, opponent_idx, opponent_ids, opponent_checkpoints, curriculum_stage)
        elif ep % 500 == 0:  # Periodic console print
            last = episode_rewards[-500:] if len(episode_rewards) >= 500 else episode_rewards
            
            # Calculate recent win rate
            if isinstance(stats, StatsTrackerExtended):
                recent_window = stats.recent_stats[-min(500, len(stats.recent_stats)):]
                win_count = sum(1 for row in recent_window if row[3] == 1)  # win column
                win_percentage = (win_count / len(recent_window)) * 100
                invalid_moves = [row[6] for row in recent_window]  # invalid_moves column
                invalid_avg = np.mean(invalid_moves) if invalid_moves else 0
            elif stats.recent_stats:
                win_count = sum(1 for row in stats.recent_stats if row[3] == 1)  # win column
                win_percentage = (win_count / len(stats.recent_stats)) * 100
                invalid_moves = [row[6] for row in stats.recent_stats]  # invalid_moves column
                invalid_avg = np.mean(invalid_moves) if invalid_moves else 0
            else:
                win_percentage = 0
                invalid_avg = 0
            
            # Build output
            status = f"Ep {ep:>6} | avgR/500={np.mean(last):.3f} | " \
                     f"win%={win_percentage:.1f}% | invalid/ep={invalid_avg:.2f} | " \
                     f"eps={agent.epsilon:.3f} | Stage={curriculum_stage}"
                     
            if in_roleplay_phase and opponent_checkpoints:
                status += f" | Opp={opponent_idx+1}/{len(opponent_checkpoints)} ({current_opponent_id})"
                
            print(status)
        
        # Periodic save
        if ep % 5_000 == 0:
            # Add extra data for roleplay phase
            extra_data = None
            if in_roleplay_phase and opponent_checkpoints:
                extra_data = {
                    'opponent_idx': opponent_idx,
                    'opponent_id': current_opponent_id,
                    'board_size_stage': env.curriculum_stage
                }
                
            save_checkpoint(agent, save_dir, ep, curriculum_stage, extra_data)

    # Final save
    stats.close()
    
    # Add extra data for roleplay phase
    extra_data = None
    if in_roleplay_phase and opponent_checkpoints:
        extra_data = {
            'opponent_idx': opponent_idx,
            'opponent_id': current_opponent_id,
            'board_size_stage': env.curriculum_stage
        }
        
    save_checkpoint(agent, save_dir, episodes, curriculum_stage, extra_data, is_final=True)
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rainbow DQN agent")
    parser.add_argument("--episodes", type=int, default=200_000, help="Number of episodes to train (total across all stages)")
    parser.add_argument("--stats", type=str, default="stats.csv", help="Where to store training statistics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--roleplay", action="store_true", help="Progress to roleplay mode after completing stage 3")
    parser.add_argument("--consecutive-wins", type=int, default=CURRICULUM_WIN_THRESHOLD, 
                       help=f"Consecutive wins needed to advance (default: {CURRICULUM_WIN_THRESHOLD})")
    parser.add_argument("--stats-interval", type=int, default=2000, 
                       help="How often to print detailed statistics (in episodes)")
    parser.add_argument("--pool-update-interval", type=int, default=5000, 
                       help="How often to update opponent pool (only in roleplay mode)")
    
    args = parser.parse_args()
    
    # Choose correct CSV file based on roleplay flag
    log_csv = args.stats
    if args.roleplay and log_csv == "stats.csv":
        log_csv = "stage4_roleplay_stats.csv"
    
    train(
        episodes=args.episodes, 
        log_csv=log_csv, 
        resume_from=args.resume, 
        curriculum=not args.no_curriculum,
        roleplay_mode=args.roleplay,
        stats_interval=args.stats_interval,
        opponent_pool_update_interval=args.pool_update_interval,
        consecutive_wins_threshold=args.consecutive_wins
    ) 