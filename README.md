# Undergraduate Research 

This research is for testing a new TAAC algorithm which uses attention in the actor to promote collaboration between agents in soccer environment

# HyperParams:

## environment
- Environment Hyperparameters:
- AGENT_DECISION_RATE: 14
- BALL_FRICTION: 0.965
- BALL_HEIGHT: 20
- BALL_MAX_SPEED: 15
- BALL_POWER: 0.95
- BALL_RADIUS: 10
- CAP_FPS: False
- FPS: 42
- GAME_DURATION: 75
- GOAL_HEIGHT: 182.0
- GOAL_WIDTH: 65.0
- HEIGHT: 700
- KICK_SPEED: 15.5
- NUMBER_OF_GAMES: 4
- NUMBER_OF_PLAYERS: 3
- PLAYER_HEIGHT: 32
- PLAYER_POWER: 2
- PLAYER_RADIUS: 16
- PLAYER_SPEED: 5
- PLAY_AREA_WIDTH: 1170.0
- STATS_UPDATE_INTERVAL: 1000
- WIDTH: 1300

## AI Models

## TAAC
- BALL_TO_GOAL_REWARD_COEFF: 0.1
- DISTANCE_REWARD_CAP: 130
- DISTANCE_REWARD_COEFF: 0.002
- GOAL_REWARD: 400
- K_epochs: 30
- PLAYER_TO_BALL_REWARD_COEFF: 0.0
- batch_size: 16384
- c_entropy: 0.0005
- c_value: 1
- episodes: 50000
- epsilon_clip: 0.1
- gamma: 0.985
- lam: 0.985
- learning_rate: 1e-05
- max_grad_norm: 2000
- max_oppenents: 15
- min_learning_rate: 1e-06
- opposing_model_freeze_time: 750
- positive_reward_coef: 1
- similarity_loss_cap: -0.5
- similarity_loss_coef: 0.01


MAAC:
- ACTION_SIZE: 18
- DISTANCE_REWARD_CAP: 130
- DISTANCE_REWARD_COEFF: 0.000
- BALL_TO_GOAL_REWARD_COEFF: 0.2
- GOAL_REWARD: 400
- K_epochs: 25 
- PLAYER_TO_BALL_REWARD_COEFF: 0.0004
- batch_size: 16384
- c_entropy: 0.0045
- c_value: 1
- episodes: 50000
- epsilon_clip: 0.1
- gamma: 0.985
- lam: 0.985
- learning_rate: 3e-06
- max_grad_norm: 2000
- min_learning_rate: 3e-07
- opposing_model_freeze_time: 2000
- positive_reward_coef: 1

PPO:
- BALL_TO_GOAL_REWARD_COEFF: 0.1
- DISTANCE_REWARD_CAP: 130
- DISTANCE_REWARD_COEFF: 0.002
- GOAL_REWARD: 400
- K_epochs: 30
- PLAYER_TO_BALL_REWARD_COEFF: 0.0
- STATE_SIZE: 22
- batch_size: 16384
- c_entropy: 0.0005
- c_value: 1
- episodes: 50000
- epsilon_clip: 0.1
- gamma: 0.985
- lam: 0.985
- learning_rate: 1e-05
- max_grad_norm: 2000
- max_oppenents: 15
- min_learning_rate: 1e-06
- opposing_model_freeze_time: 750
- positive_reward_coef: 

### NOTE 
*this repo is unfinished and the code is not very clean*
