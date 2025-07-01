"""
Core soccer environment components.

This package contains the core game engine, physics simulation, and entity classes
that power the soccer environment.
"""

# Don't eagerly import to avoid circular dependencies
# These can be imported directly when needed:
# from .game_engine import GameEngine
# from .physics import PhysicsEngine  
# from .entities import Player, Ball, Goal, GameState

__all__ = ["GameEngine", "PhysicsEngine", "Player", "Ball", "Goal", "GameState"] 