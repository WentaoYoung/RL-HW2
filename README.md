# Cross Tic-Tac-Toe Rainbow DQN

This project implements a Rainbow DQN agent for playing a modified version of Tic-Tac-Toe called "Cross" Tic-Tac-Toe. The game is played on a cross-shaped board, and has special rules for placement and winning conditions.

## Game Rules

- The board is a cross shape made of 5 squares, with each square being 4×4 in size.
- Players take turns placing pieces on the board.
- When a player selects a square:
  - With 50% chance, the piece is placed at the chosen square.
  - With 50% chance, the piece is randomly placed in one of the 8 adjacent squares.
  - If the randomly selected square is occupied or outside the board, the player's move is forfeited.
- To win, a player needs:
  - 4 in a row/column, or
  - 5 in a diagonal.

## Rainbow DQN Implementation

The Rainbow DQN implementation includes:
- Dueling architecture
- Noisy Networks for exploration
- N-step returns
- Prioritized Experience Replay
- Double Q-learning

## Curriculum Learning

The agent is trained using curriculum learning with four stages:
1. Simple: Central 4×4 square only (16 cells)
2. Medium: Central square plus one arm (32 cells)
3. Full cross: All 80 cells
4. Roleplay: Playing against historical models from previous stages

The agent advances to the next curriculum stage after achieving 50 consecutive wins.

## Project Structure

- `config.py`: Contains all hyperparameters and constants
- `env.py`: Cross Tic-Tac-Toe environment with RoleplayEnv extension
- `models.py`: Neural network models with Dueling DQN and NoisyNet
- `replay.py`: Prioritized experience replay buffer
- `agent.py`: Rainbow DQN agent implementation
- `utils.py`: Utilities including stats tracking
- `train.py`: Unified training framework for all stages
- `roleplay_utils.py`: Utilities for roleplay training (stage 4)
- `unit_tests.py`: Comprehensive unit tests for all components
- `report.ipynb`: Jupyter notebook with training results and analysis

## How to Use

### Training

```bash
# Standard training (stages 1-3)
python train.py --episodes 200000 --stats stats.csv

# Roleplay training (stage 4)
python train.py --roleplay --episodes 300000 --stats stage4_roleplay_stats.csv
```

Options:
- `--episodes`: Number of episodes to train (default: 200000)
- `--stats`: Where to save training statistics (default: stats.csv)
- `--resume`: Path to checkpoint file to resume training from
- `--no-curriculum`: Disable curriculum learning
- `--roleplay`: Use historical models as opponents (stage 4)
- `--consecutive-wins`: Consecutive wins needed to advance (default: 50)
- `--stats-interval`: How often to print detailed statistics (default: 2000)
- `--pool-update-interval`: How often to update opponent pool (default: 5000)

### Resuming Training

```bash
# Resume standard training
python train.py --resume checkpoints/agent_ep50000_stage2.pt

# Resume roleplay training
python train.py --roleplay --resume checkpoints/agent_stage4_opps3ep45000.pt
```

### Running Tests

The project includes comprehensive unit tests for all components. To run the tests:

```bash
python unit_tests.py
```

The test suite includes tests for:
- `TestEnvironment`: Board initialization, observation generation, curriculum stages, action spaces, win detection
- `TestRoleplayEnvironment`: Testing against historical models
- `TestAgent`: Action selection, epsilon decay, replay buffer, target network synchronization
- `TestDuelingDQN`: Neural network architecture and noise reset
- `TestPrioritizedReplay`: Experience replay memory
- `TestRoleplayUtils`: Historical model opponents and checkpoint management
- `TestStatsTracker/TestStatsTrackerExtended`: Statistics tracking

## Training Workflow

The recommended training workflow is:
1. Train through stages 1-3 with `python train.py --episodes 200000`
2. Train stage 4 (roleplay) using `python train.py --roleplay --episodes 300000`

The unified framework automatically handles:
- Checkpoint loading and saving
- Curriculum progression
- Training statistics tracking
- Opponent selection for roleplay mode

## Implementation Details

### Environment
- The cross-shaped board is implemented as a flattened array of 80 cells
- Different curriculum stages use subsets of this board
- The environment provides observations as 3-channel tensors (player pieces, opponent pieces, empty spaces)

### Agent
- The agent uses epsilon-greedy exploration with Noisy Networks
- N-step returns for more efficient learning
- Target network synchronized periodically for stable training
- Prioritized experience replay for efficient learning from important transitions

### Roleplay Mode
- In stage 4, the agent plays against checkpoints of previous versions of itself
- Opponents are selected from a pool that grows as training progresses
- As the agent improves, it faces increasingly difficult opponents

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Environment Setup

You can recreate the environment using the provided `env.yml` file:

```bash
conda env create -f env.yml
conda activate rl-hw2
``` 