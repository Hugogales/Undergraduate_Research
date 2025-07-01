# Scripts Directory

This directory contains the main training and utility scripts for the soccer environment.

## Main Training Scripts

### `train_rllib.py` - Primary Training Script 🏆
- **Purpose**: Train AI agents using RLLib with the original reward system
- **Rewards**: Uses ONLY the 4 original reward components from the old system:
  - Goal rewards: +400/-400
  - Player→Ball velocity: 0.0004 coefficient  
  - Ball→Goal velocity: 0.1 coefficient
  - Distance to teammates: 0.002 coefficient
- **Features**: Optimized for Windows, minimal resources, single agent (1v1)
- **Usage**: `python scripts/train_rllib.py`

### `train_single_agent.py` - Alternative Training Method
- **Purpose**: Simple single-agent training without RLLib
- **Usage**: `python scripts/train_single_agent.py`

## Utility Scripts

### `watch_ai_play.py` - Watch Trained AI 🎮
- **Purpose**: Load a trained model and watch it play visually
- **Usage**: `python scripts/watch_ai_play.py --checkpoint ./rllib_checkpoints/episode_700 --games 3`

### `play.py` - Human vs Human/AI Play 🕹️
- **Purpose**: Play soccer manually with keyboard controls
- **Usage**: `python scripts/play.py`

### `replay.py` - Replay Recorded Games 📹
- **Purpose**: Replay previously recorded game sessions
- **Usage**: `python scripts/replay.py`

## Quick Start

1. **Train an AI**: `python scripts/train_rllib.py`
2. **Watch it play**: `python scripts/watch_ai_play.py --checkpoint ./rllib_checkpoints/episode_700`
3. **Play manually**: `python scripts/play.py`

## Configuration

The main training script uses the original reward system by default. To modify training parameters, edit the `Config` class in `train_rllib.py` around line 520.

## Removed Scripts

The following scripts were removed to simplify the codebase:
- `train_rllib_original_rewards.py` (functionality moved to main script)
- `train_rllib_balanced_rewards.py` (removed - will work on new rewards later)
- `test_original_rewards.py` (removed)
- `test_balanced_rewards.py` (removed)
- `analyze_rewards.py` (removed)

All functionality now uses the original reward system by default. 