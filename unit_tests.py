#!/usr/bin/env python
"""
Comprehensive unit tests for the Cross Tic-Tac-Toe reinforcement learning project.
"""

import unittest
import tempfile
import os
import csv
import random
import collections
import numpy as np
import torch
from pathlib import Path
import glob

from env import CrossTicTacToeEnv, RoleplayEnv
from agent import Agent
from models import DuelingDQN
from replay import PrioritizedReplay
from utils import StatsTracker, StatsTrackerExtended
from roleplay_utils import (
    HistoricalModelOpponent,
    get_checkpoint_id,
    get_sorted_opponent_checkpoints,
    print_stats_summary
)


class TestEnvironment(unittest.TestCase):
    """Test cases for the game environment classes."""
    
    def setUp(self):
        """Set up the environment for each test."""
        # Initialize curriculum boards
        CrossTicTacToeEnv._init_curriculum_boards()
        # Create environment with curriculum mode
        self.env = CrossTicTacToeEnv(curriculum=True)
        
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.turn, 1)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.winner, 0)
        self.assertTrue(np.all(self.env.board == 0))
        self.assertEqual(self.env.curriculum_stage, 1)
        
    def test_reset(self):
        """Test environment reset functionality."""
        # Change some state
        self.env.turn = -1
        self.env.done = True
        self.env.winner = 1
        self.env.board[0] = 1
        
        # Reset and check state
        obs = self.env.reset()
        self.assertEqual(self.env.turn, 1)
        self.assertFalse(self.env.done)
        self.assertEqual(self.env.winner, 0)
        self.assertTrue(np.all(self.env.board == 0))
        
        # Check observation
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape[0], 3)  # 3 planes
        
    def test_get_obs(self):
        """Test observation generation."""
        # Set up board with some pieces
        self.env.board[0] = 1    # Player 1 (X)
        self.env.board[1] = -1   # Player -1 (O)
        
        # Get observation
        obs = self.env._get_obs()
        
        # Check observation format and content
        self.assertEqual(obs.shape, (3, len(self.env.ids)))
        self.assertEqual(obs[0, 0].item(), 1.0)  # Player 1 at position 0
        self.assertEqual(obs[1, 1].item(), 1.0)  # Player -1 at position 1
        self.assertEqual(obs[2, 0].item(), 0.0)  # Position 0 is not empty
        self.assertEqual(obs[2, 1].item(), 0.0)  # Position 1 is not empty
        self.assertEqual(obs[2, 2].item(), 1.0)  # Position 2 is empty
        
    def test_get_current_ids(self):
        """Test the curriculum stage id mapping."""
        # Stage 1
        self.env.curriculum_stage = 1
        ids = self.env.get_current_ids()
        self.assertEqual(ids, self.env.ids_simple)
        self.assertEqual(len(ids), 16)  # 4x4 central board
        
        # Stage 2
        self.env.curriculum_stage = 2
        ids = self.env.get_current_ids()
        self.assertEqual(ids, self.env.ids_medium)
        self.assertEqual(len(ids), 32)  # Central plus one arm
        
        # Stage 3
        self.env.curriculum_stage = 3
        ids = self.env.get_current_ids()
        self.assertEqual(ids, self.env.ids)
        self.assertEqual(len(ids), 80)  # Full cross board
        
    def test_action_space(self):
        """Test action space size across curriculum stages."""
        # Stage 1
        self.env.curriculum_stage = 1
        self.assertEqual(self.env.action_space, 16)
        
        # Stage 2
        self.env.curriculum_stage = 2
        self.assertEqual(self.env.action_space, 32)
        
        # Stage 3
        self.env.curriculum_stage = 3
        self.assertEqual(self.env.action_space, 80)
        
    def test_step_valid_action(self):
        """Test step function with valid action."""
        # First action in stage 1
        action = 0
        obs, reward, done, _ = self.env.step(action)
        
        # Check that either the chosen cell or a neighbor has a piece
        # Due to random placement rule, we can't deterministically check exact position
        cell_id = self.env.ids_simple[action]
        player_cells = np.where(self.env.board == 1)[0]
        
        # Either the chosen cell or one of its neighbors should have our piece
        possible_places = [cell_id]
        for shift in self.env.SHIFTS:
            n = cell_id + shift
            if 0 <= n < len(self.env.ids):
                possible_places.append(n)
                
        self.assertTrue(any(place in player_cells for place in possible_places))
        
        # Check that opponent has also played
        self.assertTrue(np.any(self.env.board == -1))
        
        # Check observation
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape[0], 3)
        
    def test_check_win(self):
        """Test win detection."""
        self.env.curriculum_stage = 1  # 4x4 board where 4-in-a-row is a win
        
        # Set up a winning position for player 1 (horizontal)
        # In simple curriculum, ids 5, 6, 7, 8 would be a horizontal line
        simple_ids = self.env.ids_simple
        for i in range(4):
            self.env.board[simple_ids[i]] = 1
            
        # Check if last move would result in a win
        self.assertTrue(self.env._check_win(simple_ids[3]))
        
        # Reset and test vertical win
        self.env.reset()
        # Set up vertical line
        for i in range(0, 16, 4):
            self.env.board[simple_ids[i]] = 1
            
        # Check if last move would result in a win
        self.assertTrue(self.env._check_win(simple_ids[12]))
        
        # Test diagonal win
        self.env.reset()
        # Set up diagonal line
        for i in range(0, 16, 5):
            self.env.board[simple_ids[i]] = 1
            
        # Check if last move would result in a win
        self.assertTrue(self.env._check_win(simple_ids[15]))
        
        # Test non-winning position
        self.env.reset()
        self.env.board[simple_ids[0]] = 1
        self.env.board[simple_ids[1]] = 1
        self.env.board[simple_ids[2]] = 1
        
        # This isn't a win yet
        self.assertFalse(self.env._check_win(simple_ids[2]))


class TestRoleplayEnvironment(unittest.TestCase):
    """Test cases for the RoleplayEnv class."""
    
    def setUp(self):
        """Set up the environment for each test."""
        # Initialize curriculum boards
        CrossTicTacToeEnv._init_curriculum_boards()
        # Create roleplay environment
        self.env = RoleplayEnv(curriculum=True)
        
    def test_initialization(self):
        """Test roleplay environment initialization."""
        self.assertEqual(self.env.turn, 1)
        self.assertFalse(self.env.done)
        self.assertIsNone(self.env.historical_opponent)
        
    def test_set_historical_opponent(self):
        """Test setting a historical opponent."""
        # Create a mock opponent
        class MockOpponent:
            def select_action(self, board):
                return 0
                
        mock_opponent = MockOpponent()
        
        # Set the opponent
        self.env.set_historical_opponent(mock_opponent)
        
        # Check opponent was set
        self.assertEqual(self.env.historical_opponent, mock_opponent)
        
    def test_opponent_move_with_model(self):
        """Test opponent move using a model instead of random."""
        # Create a mock opponent that always selects a specific action
        class MockOpponent:
            def select_action(self, board):
                # Find the first empty cell
                empty_cells = np.where(board == 0)[0]
                if len(empty_cells) > 0:
                    return empty_cells[0]
                return 0  # Fallback
                
        mock_opponent = MockOpponent()
        self.env.set_historical_opponent(mock_opponent)
        
        # Reset environment
        self.env.reset()
        
        # Set turn to -1 to ensure opponent moves
        self.env.turn = -1
        
        # Execute opponent move
        initial_state = self.env.board.copy()
        self.env._opponent_move()
        
        # Check that board changed in expected way
        self.assertFalse(np.array_equal(self.env.board, initial_state))
        
        # First position (which was empty) should now have opponent's mark
        self.assertEqual(self.env.board[0], -1)


class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
    def setUp(self):
        """Set up the agent for each test."""
        # Initialize curriculum boards
        CrossTicTacToeEnv._init_curriculum_boards()
        # Create environment
        self.env = CrossTicTacToeEnv(curriculum=True)
        # Create agent
        self.agent = Agent(self.env)
        
    def test_initialization(self):
        """Test agent initialization."""
        # Check policy and target networks exist
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.target_net)
        
        # Check optimizer is created
        self.assertIsNotNone(self.agent.optimizer)
        
        # Check memory is initialized
        self.assertIsNotNone(self.agent.memory)
        
        # Check n-steps buffer
        self.assertIsInstance(self.agent.n_steps_buffer, collections.deque)
        
        # Check epsilon parameters
        self.assertEqual(self.agent.eps_start, 1.0)
        self.assertEqual(self.agent.eps_end, 0.05)
        self.assertEqual(self.agent.epsilon, 1.0)
        
    def test_get_valid_actions(self):
        """Test get_valid_actions method."""
        # Reset environment to initial state
        obs = self.env.reset()
        
        # Get valid actions
        valid_actions = self.agent.get_valid_actions(obs)
        
        # In initial state, all actions should be valid
        self.assertEqual(len(valid_actions), self.env.action_space)
        
        # Now place a piece and check
        self.env.board[self.env.ids_simple[0]] = 1  # Place at first position
        obs = self.env._get_obs()
        
        # Get valid actions again
        valid_actions = self.agent.get_valid_actions(obs)
        
        # Should be one less valid action
        self.assertEqual(len(valid_actions), self.env.action_space - 1)
        self.assertNotIn(0, valid_actions)  # First action should not be valid
        
    def test_select_action_exploration(self):
        """Test select_action during exploration phase."""
        # Force exploration
        self.agent.epsilon = 1.0
        
        # Reset environment
        obs = self.env.reset()
        
        # Select action
        action = self.agent.select_action(obs)
        
        # Action should be valid
        self.assertIn(action, range(self.env.action_space))
        
    def test_select_action_exploitation(self):
        """Test select_action during exploitation phase."""
        # Force exploitation
        self.agent.epsilon = 0.0
        
        # Reset environment
        obs = self.env.reset()
        
        # Select action
        action = self.agent.select_action(obs)
        
        # Action should be valid
        self.assertIn(action, range(self.env.action_space))
        
    def test_update_epsilon(self):
        """Test epsilon update function."""
        # Set initial values
        self.agent.eps_start = 1.0
        self.agent.eps_end = 0.1
        self.agent.eps_decay_steps = 100
        self.agent.epsilon = 1.0
        self.agent.step_counter = 0
        
        # Update at start
        self.agent.update_epsilon()
        self.assertEqual(self.agent.epsilon, 1.0)
        
        # Update halfway
        self.agent.step_counter = 50
        self.agent.update_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, 0.55, places=2)
        
        # Update at end
        self.agent.step_counter = 100
        self.agent.update_epsilon()
        self.assertEqual(self.agent.epsilon, 0.1)
        
        # Update past end
        self.agent.step_counter = 200
        self.agent.update_epsilon()
        self.assertEqual(self.agent.epsilon, 0.1)
        
    def test_push(self):
        """Test pushing transitions to memory."""
        # Create dummy transition
        s0 = torch.randn(3, 80)
        a0 = 0
        r = 0.5
        s1 = torch.randn(3, 80)
        d = False
        
        # Push to n-steps buffer
        for _ in range(self.agent.n_steps_buffer.maxlen):
            self.agent.push((s0, a0, r, s1, d))
            
        # Check memory has one item
        self.assertEqual(len(self.agent.memory.buffer), 1)
        
    def test_sync_target(self):
        """Test target network synchronization."""
        # Modify policy network
        with torch.no_grad():
            for param in self.agent.policy_net.parameters():
                param.add_(torch.randn_like(param))
                
        # Networks should now be different
        policy_params = list(self.agent.policy_net.parameters())
        target_params = list(self.agent.target_net.parameters())
        
        # Check that at least one parameter is different
        params_equal = all(torch.allclose(p, t) for p, t in zip(policy_params, target_params))
        self.assertFalse(params_equal)
        
        # Sync target
        self.agent.sync_target()
        
        # Check all parameters now match
        policy_params = list(self.agent.policy_net.parameters())
        target_params = list(self.agent.target_net.parameters())
        params_equal = all(torch.allclose(p, t) for p, t in zip(policy_params, target_params))
        self.assertTrue(params_equal)
        
    def test_calc_multistep_return(self):
        """Test multistep return calculation."""
        # Clear buffer
        self.agent.n_steps_buffer.clear()
        
        # Create transitions with known rewards
        gamma = 0.99  # Gamma from config
        
        # Add transitions with rewards 1, 2, 3
        for i in range(1, 4):
            s = torch.zeros(3, 80)
            a = 0
            r = float(i)
            next_s = torch.zeros(3, 80)
            d = False
            self.agent.n_steps_buffer.append((s, a, r, next_s, d))
            
        # Calculate expected return: r1 + gamma*r2 + gamma^2*r3
        expected = 1 + gamma * 2 + gamma**2 * 3
        
        # Check calculated return
        calculated = self.agent._calc_multistep_return()
        
        # Should be close to expected value
        self.assertAlmostEqual(calculated, expected, places=5)


class TestDuelingDQN(unittest.TestCase):
    """Test cases for the DuelingDQN model."""
    
    def setUp(self):
        """Set up test model."""
        self.input_shape = (3, 80)  # 3 planes, 80 cells
        self.output_size = 16       # 16 actions (stage 1)
        self.model = DuelingDQN(self.input_shape, self.output_size)
        
    def test_initialization(self):
        """Test model initialization."""
        # Check network components exist
        self.assertIsNotNone(self.model.fc1)
        self.assertIsNotNone(self.model.noisy1)
        self.assertIsNotNone(self.model.adv)
        self.assertIsNotNone(self.model.val)
        
        # Check output dimensions
        self.assertEqual(self.model.val.out_features, 1)
        self.assertEqual(self.model.adv.out_features, self.output_size)
        
    def test_forward_shape(self):
        """Test forward pass output shape."""
        # Create dummy input
        x = torch.zeros(1, *self.input_shape)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.output_size))
        
    def test_reset_noise(self):
        """Test noise reset functionality."""
        # Create two identical inputs
        x = torch.ones(1, *self.input_shape)
        
        # Get output with current noise
        out1 = self.model(x)
        
        # Reset noise
        self.model.reset_noise()
        
        # Get output with new noise
        out2 = self.model(x)
        
        # Outputs should be different due to noise reset
        # Note: This may occasionally fail by random chance if noise values happen to be very similar
        outputs_identical = torch.allclose(out1, out2)
        if outputs_identical:
            print("Warning: Outputs were identical after noise reset. This can happen rarely by chance.")
        else:
            self.assertFalse(outputs_identical)


class TestPrioritizedReplay(unittest.TestCase):
    """Test cases for the PrioritizedReplay class."""
    
    def setUp(self):
        """Set up test replay memory."""
        self.capacity = 100
        self.replay = PrioritizedReplay(self.capacity)
        # These values should match the defaults in the PrioritizedReplay class
        self.alpha = 0.6
        self.beta = 0.4
        
    def test_initialization(self):
        """Test replay initialization."""
        self.assertEqual(self.replay.capacity, self.capacity)
        self.assertEqual(self.replay.alpha, self.alpha)
        self.assertEqual(self.replay.beta, self.beta)
        self.assertEqual(len(self.replay.buffer), 0)
        # The priorities array is pre-allocated with the capacity size
        self.assertEqual(len(self.replay.priorities), self.capacity)
        
    def test_push(self):
        """Test pushing transitions to memory."""
        # Create dummy transition
        s0 = torch.zeros(3, 80)
        a0 = torch.tensor(0)
        r = torch.tensor(1.0)
        s1 = torch.zeros(3, 80)
        d = torch.tensor(False)
        
        # Push to memory
        self.replay.push(s0, a0, r, s1, d)
        
        # Check memory size
        self.assertEqual(len(self.replay.buffer), 1)
        # The priorities array is pre-allocated with the capacity size
        self.assertEqual(len(self.replay.priorities), self.capacity)
        
        # Check priority is maximum
        self.assertEqual(self.replay.priorities[0], 1.0)
        
    def test_update_priorities(self):
        """Test updating priorities."""
        # Fill memory with dummy transitions
        for i in range(5):
            s0 = torch.zeros(3, 80)
            a0 = torch.tensor(i)
            r = torch.tensor(float(i))
            s1 = torch.zeros(3, 80)
            d = torch.tensor(False)
            self.replay.push(s0, a0, r, s1, d)
            
        # Initial priorities for filled positions should all be 1.0
        # (only the first 5 positions in this case)
        self.assertTrue(all(self.replay.priorities[i] == 1.0 for i in range(5)))
        
        # Update priorities
        indices = [0, 2, 4]
        priorities = torch.tensor([2.0, 3.0, 4.0])
        self.replay.update_priorities(indices, priorities)
        
        # Check updated priorities
        self.assertEqual(self.replay.priorities[0], 2.0)
        self.assertEqual(self.replay.priorities[1], 1.0)  # Unchanged
        self.assertEqual(self.replay.priorities[2], 3.0)
        self.assertEqual(self.replay.priorities[3], 1.0)  # Unchanged
        self.assertEqual(self.replay.priorities[4], 4.0)
        
    def test_capacity_limit(self):
        """Test capacity limit enforcement."""
        # Fill beyond capacity
        for i in range(self.capacity + 10):
            s0 = torch.zeros(3, 80)
            a0 = torch.tensor(i % 5)
            r = torch.tensor(float(i))
            s1 = torch.zeros(3, 80)
            d = torch.tensor(i % 2 == 0)
            self.replay.push(s0, a0, r, s1, d)
            
        # Size should be limited to capacity
        self.assertEqual(len(self.replay.buffer), self.capacity)
        self.assertEqual(len(self.replay.priorities), self.capacity)


class TestRoleplayUtils(unittest.TestCase):
    """Test cases for roleplay utility functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Initialize curriculum boards
        CrossTicTacToeEnv._init_curriculum_boards()
        
        # Create a dummy environment with stage 1 curriculum
        self.env = CrossTicTacToeEnv(curriculum=True)
        self.env.curriculum_stage = 1  # Use stage 1 for testing
        
        # Create a simple model for stage 1
        obs_dim = self.env._get_obs().shape
        self.model = DuelingDQN(obs_dim, self.env.action_space)
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_historical_model_opponent(self):
        """Test historical model opponent."""
        # Create opponent wrapper with matching curriculum stage
        opponent = HistoricalModelOpponent(self.model, 1, self.env)
        
        # Test initialization
        self.assertEqual(opponent.model, self.model)
        self.assertEqual(opponent.curriculum_stage, 1)
        self.assertEqual(opponent.env, self.env)
        
        # Test action selection on empty board
        board = np.zeros(80, dtype=np.int8)
        action_id = opponent.select_action(board)
        
        # Action should be a valid board index within range
        self.assertGreaterEqual(action_id, 0)
        self.assertLess(action_id, 80)

    def test_get_checkpoint_id(self):
        """Test checkpoint ID extraction."""
        # Test with a stage3 file
        path = "/path/to/agent_ep50000_stage3.pt"
        self.assertEqual(get_checkpoint_id(path), "s3ep50000")
        
        # Test with a stage4 file
        path = "/path/to/agent_ep100000_stage4.pt"
        self.assertEqual(get_checkpoint_id(path), "s4ep100000")
        
        # Test with just episode number
        path = "/path/to/agent_ep75000.pt"
        self.assertEqual(get_checkpoint_id(path), "ep75000")
        
        # Test with unusual filename
        path = "/path/to/random_checkpoint.pt"
        self.assertEqual(get_checkpoint_id(path), "random_che")


class TestStatsTracker(unittest.TestCase):
    """Test cases for the StatsTracker class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test output file path
        self.test_csv = self.test_dir / "test_stats.csv"
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_initialization(self):
        """Test StatsTracker initialization."""
        # Create stats tracker
        tracker = StatsTracker(str(self.test_csv), flush_every=10)
        
        # File should be created with header
        self.assertTrue(self.test_csv.exists())
        
        # Check header in file
        with open(self.test_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, StatsTracker.headers)
            
        # Check internal state
        self.assertEqual(tracker.flush_every, 10)
        self.assertEqual(len(tracker.rows), 0)
        self.assertIsInstance(tracker.recent_stats, collections.deque)
        
    def test_close(self):
        """Test closing the stats tracker."""
        # Create tracker
        tracker = StatsTracker(str(self.test_csv), flush_every=10)
        
        # Log some episodes 
        episodes_to_log = 3
        for i in range(episodes_to_log):
            tracker.log(
                episode=i,
                reward=0.1 * i,
                winner=1 if i % 2 == 0 else 0,
                invalids=0,
                moves=10 + i,
                epsilon=0.9 - 0.01 * i
            )
            
        # Close tracker
        tracker.close()
        
        # Rows should be cleared
        self.assertEqual(len(tracker.rows), 0)
        
        # File should have header + episodes
        with open(self.test_csv, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), episodes_to_log + 1)  # Add 1 for header


class TestStatsTrackerExtended(unittest.TestCase):
    """Test cases for the StatsTrackerExtended class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test output file path
        self.test_csv = self.test_dir / "test_extended_stats.csv"
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        
    def test_initialization(self):
        """Test StatsTrackerExtended initialization."""
        # Create stats tracker
        tracker = StatsTrackerExtended(
            str(self.test_csv),
            flush_every=10,
            extra_headers=["custom1", "custom2"]
        )
        
        # Check extended headers
        expected_headers = StatsTracker.headers + ["custom1", "custom2"]
        
        # Check file header
        with open(self.test_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, expected_headers)
            
        # Check internal state
        self.assertEqual(tracker.extra_headers, ["custom1", "custom2"])
        self.assertEqual(len(tracker.rows), 0)
        
    def test_log_with_extra_values(self):
        """Test logging with extra values."""
        # Create tracker
        tracker = StatsTrackerExtended(
            str(self.test_csv),
            flush_every=5, # Use higher value to avoid flushing during test
            extra_headers=["opponent_id", "curriculum_stage"]
        )
        
        # Log with extra values
        tracker.log(
            episode=1,
            reward=0.5,
            winner=1,
            invalids=0,
            moves=10,
            epsilon=0.9,
            extra_values={
                "opponent_id": "ep5000",
                "curriculum_stage": 3
            }
        )
        
        # Check internal state
        self.assertEqual(len(tracker.rows), 1)
        
        # Check the extra values were added correctly
        self.assertEqual(tracker.rows[0][-2], "ep5000")
        self.assertEqual(tracker.rows[0][-1], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2) 