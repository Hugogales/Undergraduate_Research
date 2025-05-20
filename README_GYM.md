# Soccer Environment - OpenAI Gym Interface

This document explains how to use the Soccer Environment with the OpenAI Gym interface.

## Overview

The Soccer Environment has been refactored to support the OpenAI Gym interface, which allows easy integration with popular reinforcement learning libraries like Stable Baselines3, RLlib, and others. The original functionality is fully preserved, so you can still use the environment in the traditional way.

## Getting Started

### Using the Gym Interface

To use the Soccer Environment as a Gym environment, you need to:

1. Import the `SoccerEnv` class from `enviroment.GymAdapter`
2. Create an instance of `SoccerEnv`
3. Use the standard Gym methods: `reset()`, `step()`, and `render()`

Example:

```python
from enviroment.GymAdapter import SoccerEnv

# Create the environment
env = SoccerEnv()

# Reset the environment
observation = env.reset()

# Run a single step
action = [0, 0, 0, 0, 0]  # Example action
next_observation, reward, done, info = env.step(action)

# Render the environment
env.render()

# Close the environment when done
env.close()
```

### Using with Stable Baselines3

The environment can be easily used with Stable Baselines3. See the example in `examples/train_with_stable_baselines.py`.

Basic usage:

```python
from stable_baselines3 import PPO
from enviroment.GymAdapter import SoccerEnv

# Create the environment
env = SoccerEnv()

# Create and train a PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model
model.save("soccer_ppo")
```

### Configuration

You can configure the environment by modifying the parameters in `params.py`. In addition, a new parameter `USE_GYM` has been added to `EnvironmentHyperparameters` that allows you to choose between the original interface and the Gym interface when training:

```python
# In params.py
self.USE_GYM = True  # Set to True to use the Gym interface
```

## Existing Models

The existing AI models can be used with the Gym interface through the `ModelAdapter` class. This adapter translates between the Gym interface and your custom model interface:

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

## Action and Observation Spaces

The Gym interface defines standard action and observation spaces:

- **Action Space**: Multi-discrete space for all players' actions
- **Observation Space**: Box space for the flattened state of all players

The `ModelAdapter` class takes care of translating between the Gym format and the format expected by your models.

## Training with Gym Interface

You can train your models using the Gym interface by either:

1. Setting `ENV_PARAMS.USE_GYM = True` and running the training as usual
2. Using a custom training loop with the Gym interface
3. Using a third-party RL library like Stable Baselines3

## More Information

For more details on how to use the environment, see the example scripts in the `examples/` directory. 