"""
Soccer Environment Modules

This package contains the soccer environment implementation using PettingZoo Parallel API.
"""

# Lazy loading functions
def env(*args, **kwargs):
    """Create a soccer environment."""
    from .soccer_env import env as _env
    return _env(*args, **kwargs)

def raw_env(*args, **kwargs):
    """Create a raw soccer environment."""
    from .soccer_env import raw_env as _raw_env
    return _raw_env(*args, **kwargs)

def parallel_env(*args, **kwargs):
    """Create a parallel soccer environment (alias for env)."""
    return env(*args, **kwargs)

# Export for compatibility
SoccerEnv = env
SoccerParallelEnv = env

__all__ = ['env', 'raw_env', 'parallel_env', 'SoccerEnv', 'SoccerParallelEnv'] 