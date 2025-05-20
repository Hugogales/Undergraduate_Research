# Soccer Environment - OpenAI Gym Interface

This document explains how to use the Soccer Environment with the OpenAI Gym interface, particularly focusing on the 4-stage training process and League integration.

## Overview

The Soccer Environment has been refactored to support the OpenAI Gym interface, which allows easy integration with popular reinforcement learning libraries like Stable Baselines3, RLlib, and others. The original functionality is fully preserved, so you can still use the environment in the traditional way.

## The 4-Stage Training Process

The environment supports the established 4-stage training process:

1. **Stage 1**: Training against a random opponent with fixed player positions
2. **Stage 2**: Training against a random opponent with randomized player positions
3. **Stage 3**: Training against a copy of the model (self-play) with randomized player positions
4. **Stage 4**: Training against a copy of the model with fixed player positions

The Gym adapter fully supports this training progression, either through the original training functions or via the Gym interface.

## League System Integration

For stages 3 and 4, the environment can use the League system to manage competing models:

- The League keeps track of a set of models that can be used as opponents
- Models in the League are sampled according to their ELO ratings
- New models can be added to the League over time
- This creates a curriculum of opponents of varying skill levels

The SoccerEnv class now integrates with the League system, allowing you to:
- Create an environment with a League
- Sample competing models from the League
- Update the League with new models

## Getting Started

### Using the Gym Interface

To use the Soccer Environment as a Gym environment, you need to:

1. Import the `SoccerEnv` class from `enviroment.GymAdapter`
2. Create an instance of `SoccerEnv` with optional parameters for current_stage and competing_model
3. Use the standard Gym methods: `reset()`, `step()`, and `render()`

Example:

```python
from enviroment.GymAdapter import SoccerEnv
from AI.randmodel import RandomModel

# Create the environment (defaults to stage 1 with a RandomModel)
env = SoccerEnv()

# Or specify the stage and competing model explicitly
competing_model = RandomModel()
env = SoccerEnv(competing_model=competing_model, current_stage=1)

# Reset the environment
observation = env.reset()

# Run a single step
action = [0, 0, 0, 0, 0]  # Example action for one player
next_observation, reward, done, info = env.step(action)

# Render the environment
env.render()

# Close the environment when done
env.close()
```

### Using with the League System

To use the Soccer Environment with the League system:

```python
from enviroment.GymAdapter import SoccerEnv
from functions.league import League
from functions.ELO import ELO
from AI.HUGO import HUGO

# Create ELO environment
elo_env = ELO()

# Create a base model for opponents
base_opponent = HUGO(mode="test")

# Create a league with the base opponent
league = League(elo_env, base_opponent)

# Add models to the league
model1 = HUGO(mode="test")
model1.load_model("HUGO_v100_sub0")
league.add_player(model1, elo_env.init_rating())

# Create environment with the league
env = SoccerEnv(current_stage=3, league=league)

# Now when you reset the environment, it will sample a competing model from the league
observation = env.reset()

# ... run the environment as usual ...

# You can update the league with new models
env.update_league(new_model, new_rating)
```

### Changing the Competing Model Dynamically

You can change the competing model without creating a new environment:

```python
from enviroment.GymAdapter import SoccerEnv
from AI.PPO import PPOAgent

# Create environment
env = SoccerEnv(current_stage=4)

# Create a new competing model
new_model = PPOAgent(mode="test")
new_model.load_model("PPO_v30_sub4")

# Set the competing model
env.set_competing_model(new_model)

# Continue using the environment with the new competing model
observation = env.reset()
```

### Multi-Stage Training with Gym

To properly implement the 4-stage training with the Gym interface:

```python
from enviroment.GymAdapter import SoccerEnv
from AI.randmodel import RandomModel
from AI.HUGO import HUGO
from params import EnvironmentHyperparameters, AIHyperparameters
from functions.league import League
from functions.ELO import ELO

# Initialize parameters
ENV_PARAMS = EnvironmentHyperparameters()
AI_PARAMS = AIHyperparameters()

# Stage 1: Random opponent, fixed positions
AI_PARAMS.current_stage = 1
ENV_PARAMS.RANDOMIZE_PLAYERS = False
env1 = SoccerEnv(current_stage=1)  # Will use RandomModel by default

# Train here with your RL algorithm
# ...

# Stage 2: Random opponent, random positions
AI_PARAMS.current_stage = 2
ENV_PARAMS.RANDOMIZE_PLAYERS = True
env2 = SoccerEnv(current_stage=2)  # Will use RandomModel by default

# Train here with your RL algorithm
# ...

# Create a league for stages 3 and 4
elo_env = ELO()
base_opponent = HUGO(mode="test")
league = League(elo_env, base_opponent)

# Add your trained model to the league
model_copy = HUGO(mode="test")
model_copy.load_model("your_saved_model")  # Load your trained model
league.add_player(model_copy, elo_env.init_rating())

# Stage 3: League-based opponents, random positions
AI_PARAMS.current_stage = 3
ENV_PARAMS.RANDOMIZE_PLAYERS = True
env3 = SoccerEnv(current_stage=3, league=league)

# Train here with your RL algorithm
# ...

# Stage 4: League-based opponents, fixed positions
AI_PARAMS.current_stage = 4
ENV_PARAMS.RANDOMIZE_PLAYERS = False
env4 = SoccerEnv(current_stage=4, league=league)

# Train here with your RL algorithm
# ...
```

See the full example in `examples/train_with_stable_baselines.py` for a complete implementation.

## Testing with Rendering

To test the environment with rendering enabled, you can:

1. Set the `RENDER` parameter to `True` in the environment parameters:

```python
ENV_PARAMS = EnvironmentHyperparameters()
ENV_PARAMS.RENDER = True
```

2. Use the provided test script:

```bash
python examples/test_gym_with_league.py
```

This script demonstrates:
- Using the environment with rendering enabled
- Testing all 4 stages
- Using the League system
- Changing competing models dynamically

## MLP Models and Per-Player Processing

When using third-party RL libraries like Stable Baselines3, a challenge arises because these libraries typically use a single MLP to process the entire state, while our architecture processes each player's state separately.

Here are approaches to handle this:

1. **Custom Policy Networks**: Implement a custom policy network for libraries like Stable Baselines3 that separates the input into player states and processes each independently.

2. **ModelAdapter Approach**: Use our ModelAdapter to wrap existing models that handle per-player processing.

3. **Input Restructuring**: Restructure the input state in a way that MLPs can effectively learn player relationships, despite processing the combined state.

For best compatibility with our existing models, we recommend using the ModelAdapter approach which preserves the per-player processing architecture.

## Using with Existing AI Models

The existing AI models can be used with the Gym interface through the `ModelAdapter` class:

```python
from AI.ModelAdapter import ModelAdapter
from AI.PPO import PPOAgent
from enviroment.GymAdapter import SoccerEnv

# Create your model
model = PPOAgent(mode="train")

# Wrap it with the adapter
adapter = ModelAdapter(model)

# Create the environment
env = SoccerEnv()

# Use the adapter with the environment
observation = env.reset()
action, _ = adapter.act(observation)
next_observation, reward, done, info = env.step(action)
adapter.store_reward(reward, done)
```

## Preserving Original Functionality

The original functionality is fully preserved. You can still use the environment in the traditional way:

```python
from enviroment.Game import Game
from AI.PPO import PPOAgent

# Create a game instance
game = Game()

# Run the game with your models
model1 = PPOAgent(mode="train")
model2 = PPOAgent(mode="test")
game.run(model1, model2, current_stage=1)
```

In addition, the `SoccerEnv` class provides methods that mirror the original Game class methods:

- `run_episode(model, competing_model, current_stage)`: Run a full episode
- `run_play_mode()`: Run in human-player mode
- `replay_from_log(states)`: Replay from logged states

## Using the Original Training Functions with Gym

You can use the original training functions but have them internally use the Gym interface by setting:

```python
ENV_PARAMS = EnvironmentHyperparameters()
ENV_PARAMS.USE_GYM = True
```

Then run your training as usual:

```python
from main import train_PPO
train_PPO()  # Will use the Gym interface if USE_GYM is True
```

## More Information

For more details on how to use the environment, see the example scripts in the `examples/` directory. 