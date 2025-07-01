"""
Soccer Environment

A multi-agent soccer environment for reinforcement learning research.
Follows PettingZoo Parallel API standards for natural simultaneous gameplay.
"""

# Register the environment with gymnasium
from gymnasium.envs.registration import register

register(
    id="Soccer-v0",
    entry_point="soccer_env.envs.soccer_env:env",
)

# Lazy loading functions to avoid circular imports
def env(*args, **kwargs):
    """Create a soccer environment."""
    from .envs.soccer_env import env as _env
    return _env(*args, **kwargs)

def raw_env(*args, **kwargs):
    """Create a raw soccer environment."""
    from .envs.soccer_env import raw_env as _raw_env
    return _raw_env(*args, **kwargs)

def parallel_env(*args, **kwargs):
    """Create a parallel soccer environment (alias for env)."""
    return env(*args, **kwargs)

# Export for compatibility
SoccerEnv = env
SoccerParallelEnv = env

__all__ = [
    'env', 'raw_env', 'parallel_env',
    'SoccerEnv', 'SoccerParallelEnv'
] 