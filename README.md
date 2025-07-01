Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
venv\Scripts\activate    
# Multi-Agent Soccer Environment

A comprehensive multi-agent soccer environment built with PettingZoo standards and optimized for reinforcement learning research. This environment supports both cooperative and competitive multi-agent training scenarios.

## 🏈 Features

- **PettingZoo Compliance**: Full compatibility with PettingZoo's AEC and Parallel APIs
- **Multi-Agent Support**: Configurable team sizes (1v1, 2v2, 3v3, etc.)
- **Action Spaces**: Both continuous and discrete action spaces
- **Physics Simulation**: Realistic ball physics and player collisions
- **Rendering**: Beautiful pygame-based visualization
- **RLlib Integration**: Built-in support for Ray RLlib algorithms
- **Reward Engineering**: Configurable dense and sparse reward structures
- **Observation Spaces**: Rich observations including relative positions, velocities, and game state

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd soccer-env

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from soccer_env.envs.soccer_parallel_env import parallel_env

# Create a 2v2 soccer environment
env = parallel_env(
    num_players_per_team=2,
    continuous_actions=True,
    render_mode="human"
)

# Run a random episode
observations, infos = env.reset()

while env.agents:
    actions = {}
    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample()
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()
```

### Test the Environment

```bash
# Run basic functionality tests
python src/examples/test_basic_env.py

# Or use the console command
soccer-env-test
```

## 🧠 Training with RLlib

The environment comes with built-in RLlib integration supporting multiple algorithms:

### Basic Training

```bash
# Train with PPO (default)
python src/examples/rllib_training.py \
    --algorithm PPO \
    --num-players-per-team 2 \
    --continuous-actions \
    --max-iterations 1000

# Train with MAPPO (centralized training)
python src/examples/rllib_training.py \
    --algorithm MAPPO \
    --num-players-per-team 3 \
    --max-timesteps 2000000

# Train with IPPO (independent policies)
python src/examples/rllib_training.py \
    --algorithm IPPO \
    --num-players-per-team 2 \
    --train-both-teams
```

### Evaluation

```bash
# Evaluate a trained model
python src/examples/rllib_training.py \
    --evaluate \
    --checkpoint-path ./results/checkpoint_path \
    --algorithm PPO \
    --eval-episodes 10 \
    --render
```

## 🎮 Environment Details

### Observation Space

Each agent receives observations containing:
- **Own state**: Position, velocity (4 values)
- **Ball state**: Position, velocity (4 values)  
- **Other players**: Positions of all other players (2 × (n-1) values)
- **Goals**: Goal positions (4 values)
- **Game info**: Time remaining, scores, ball possession (4 values)
- **Relative info**: Distances and angles to key objects (6 values)

### Action Space

#### Continuous Actions (default)
- `move_x`: Horizontal movement [-1, 1]
- `move_y`: Vertical movement [-1, 1]
- `kick_power`: Ball kick power [0, 1]
- `kick_direction`: Ball kick direction [-π, π]

#### Discrete Actions
- 17 discrete actions combining movement and kicking
- Actions 0-8: Movement only (8 directions + no movement)
- Actions 9-16: Movement + kick in same direction

### Reward Structure

#### Sparse Rewards
- **Goal scored**: +100 for scoring team, -100 for conceding team
- **Goal scorer bonus**: Additional +100 for the player who scored

#### Dense Rewards (optional)
- **Ball possession**: +0.1 per step with ball control
- **Ball proximity**: Reward for getting closer to ball
- **Goal approach**: Reward for advancing toward opponent's goal with ball
- **Teammate spacing**: Penalty for crowding teammates

## 🔧 Configuration Options

### Environment Parameters

```python
env = parallel_env(
    num_players_per_team=2,           # Players per team
    field_width=100.0,                # Field width
    field_height=60.0,                # Field height
    max_episode_steps=3000,           # Max steps per episode
    continuous_actions=True,          # Action space type
    randomize_starting_positions=True, # Randomize initial positions
    ball_friction=0.965,              # Ball physics
    player_speed=2.0,                 # Base player speed
    render_mode="human",              # Rendering mode
)
```

### Reward Configuration

```python
from soccer_env.utils.reward_calculator import RewardCalculator

reward_calc = RewardCalculator(
    goal_reward=100.0,                # Goal scoring reward
    goal_conceded_penalty=-100.0,     # Goal conceded penalty
    ball_possession_reward=0.1,       # Ball possession reward
    dense_rewards=True,               # Enable dense rewards
)
```

## 📊 Algorithms and Performance

The environment has been tested with various multi-agent RL algorithms:

### Supported Algorithms
- **PPO**: Proximal Policy Optimization with team-based policies
- **MAPPO**: Multi-Agent PPO with centralized training
- **IPPO**: Independent PPO with separate policies per agent
- **Compatible with**: SAC, TD3, DQN, and other RLlib algorithms

### Training Tips
1. **Start simple**: Begin with 1v1 or 2v2 before scaling up
2. **Reward shaping**: Use dense rewards initially, then transition to sparse
3. **Curriculum learning**: Gradually increase episode length and complexity
4. **Opponent diversity**: Train against multiple opponent policies

## 🎯 Use Cases

### Research Applications
- **Multi-agent coordination**: Team strategy emergence
- **Competitive training**: Self-play and population-based training
- **Transfer learning**: Adapting policies across team sizes
- **Curriculum learning**: Progressive task difficulty

### Educational Use
- **RL fundamentals**: Learn multi-agent RL concepts
- **Game AI**: Develop intelligent game agents
- **Algorithm comparison**: Benchmark different MARL algorithms

## 🧪 Testing and Validation

### Automated Testing
```bash
# Run all tests
pytest tests/

# Test PettingZoo compliance
python src/examples/test_basic_env.py
```

### Manual Testing
```bash
# Quick environment test
python -c "from soccer_env.envs.soccer_parallel_env import parallel_env; \
           env = parallel_env(); \
           obs, _ = env.reset(); \
           print('Environment working!'); \
           env.close()"
```

## 📈 Performance Metrics

Track training progress with these key metrics:
- **Episode reward**: Total reward per episode
- **Goal rate**: Goals scored per episode
- **Ball possession time**: Percentage of time with ball control
- **Episode length**: Steps until episode termination
- **Win rate**: Percentage of games won (if applicable)

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/
flake8 src/

# Run type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) multi-agent RL framework
- Integrated with [Ray RLlib](https://docs.ray.io/en/latest/rllib/) for scalable training
- Inspired by [Google Research Football](https://github.com/google-research/football) environment

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the [documentation](docs/)
- Join our [Discord community](link-to-discord)

---

**Happy training! ⚽🤖**

# Soccer Environment

A multi-agent soccer environment for reinforcement learning research.

## Getting Started

### Prerequisites

- Python 3.7+
- pygame
- numpy
- gymnasium
- pettingzoo

### Installation

Install the required packages:

```bash
pip install pygame numpy gymnasium pettingzoo
```

## Running the Environment

### Playing the Game

To play the soccer game with human controls:

```bash
python play.py --players 1
```

Options:
- `--players`: Number of players per team (default: 1)
- `--steps`: Maximum steps to run (default: 3000)

### Controls

- **Blue Team (Team 0)**:
  - Movement: WASD keys
  - Kick: SPACE

- **Red Team (Team 1)**:
  - Movement: Arrow keys
  - Kick: Right CTRL

## Training Agents

### Training with AEC API

```bash
python src/train.py --players 1 --episodes 100 --render
```

### Training with Parallel API

```bash
python src/train_parallel.py --players 1 --episodes 100 --render
```

Options:
- `--players`: Number of players per team
- `--episodes`: Number of episodes to train for
- `--render`: Enable rendering during training

## Replaying Games

To replay a recorded game:

```bash
python src/replay.py path/to/game_file.pkl
```

## Directory Structure

- `src/`: Main package directory
  - `core/`: Core game components (physics, entities, game engine)
  - `envs/`: Environment implementations (AEC and Parallel)
  - `utils/`: Utility modules (renderer, reward calculator, etc.)
  - `play.py`: Script to play the game
  - `train.py`: Script to train agents using AEC API
  - `train_parallel.py`: Script to train agents using Parallel API
  - `replay.py`: Script to replay recorded games

## License

This project is licensed under the MIT License - see the LICENSE file for details.
