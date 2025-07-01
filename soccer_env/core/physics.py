"""
Physics Engine for the soccer environment.

This module handles all physics simulation including collisions,
boundary checking, and entity movement updates.
"""

import math
from typing import List
from .entities import Player, Ball, Goal


class PhysicsEngine:
    """
    Handles physics simulation for the soccer environment.
    """
    
    def __init__(
        self,
        field_width: float = 100.0,
        field_height: float = 60.0,
        ball_friction: float = 0.965,
        player_friction: float = 0.9,
        restitution: float = 0.8,
    ):
        """
        Initialize the physics engine.
        
        Args:
            field_width: Width of the soccer field
            field_height: Height of the soccer field
            ball_friction: Ball friction coefficient
            player_friction: Player friction coefficient
            restitution: Collision elasticity (0 = inelastic, 1 = elastic)
        """
        self.field_width = field_width
        self.field_height = field_height
        self.ball_friction = ball_friction
        self.player_friction = player_friction
        self.restitution = restitution
        
        # Field boundaries (accounting for goal areas)
        self.goal_width = field_width * 0.05
        self.left_boundary = self.goal_width
        self.right_boundary = field_width - self.goal_width
        self.top_boundary = 0
        self.bottom_boundary = field_height
    
    def step(self, players: List[Player], ball: Ball, goals: List[Goal]) -> None:
        """
        Execute one physics simulation step.
        
        Args:
            players: List of all players
            ball: The ball object
            goals: List of goal objects
        """
        # Update positions based on velocities
        self._update_positions(players, ball)
        
        # Handle collisions
        self._handle_player_player_collisions(players)
        self._handle_player_ball_collisions(players, ball)
        
        # Apply boundary constraints
        self._apply_boundary_constraints(players, ball, goals)
        
        # Apply friction
        self._apply_friction(players, ball)
    
    def _update_positions(self, players: List[Player], ball: Ball) -> None:
        """Update positions based on current velocities."""
        # Update players
        for player in players:
            player.update_position()
        
        # Update ball
        ball.update_position()
    
    def _handle_player_player_collisions(self, players: List[Player]) -> None:
        """Handle collisions between players."""
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                player1, player2 = players[i], players[j]
                
                if player1.collides_with(player2):
                    self._resolve_player_collision(player1, player2)
    
    def _handle_player_ball_collisions(self, players: List[Player], ball: Ball) -> None:
        """Handle collisions between players and the ball."""
        for player in players:
            if player.collides_with(ball):
                self._resolve_player_ball_collision(player, ball)
    
    def _resolve_player_collision(self, player1: Player, player2: Player) -> None:
        """
        Resolve collision between two players.
        
        Args:
            player1: First player
            player2: Second player
        """
        # Calculate collision vector
        dx = player2.x - player1.x
        dy = player2.y - player1.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            # Prevent division by zero
            dx, dy = 1, 0
            distance = 1
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Separate overlapping players
        overlap = (player1.radius + player2.radius) - distance
        if overlap > 0:
            separation = overlap / 2
            player1.x -= nx * separation
            player1.y -= ny * separation
            player2.x += nx * separation
            player2.y += ny * separation
        
        # Calculate relative velocity
        rel_vx = player2.vx - player1.vx
        rel_vy = player2.vy - player1.vy
        
        # Calculate relative velocity along collision normal
        speed = rel_vx * nx + rel_vy * ny
        
        # Only resolve if objects are approaching
        if speed < 0:
            return
        
        # Calculate collision impulse
        impulse = 2 * speed / 2  # Assuming equal mass
        impulse *= self.restitution
        
        # Apply impulse to velocities
        player1.vx += impulse * nx
        player1.vy += impulse * ny
        player2.vx -= impulse * nx
        player2.vy -= impulse * ny
    
    def _resolve_player_ball_collision(self, player: Player, ball: Ball) -> None:
        """
        Resolve collision between player and ball.
        
        Args:
            player: Player object
            ball: Ball object
        """
        # Calculate collision vector
        dx = ball.x - player.x
        dy = ball.y - player.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            # Prevent division by zero
            dx, dy = 1, 0
            distance = 1
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Separate overlapping objects
        overlap = (player.radius + ball.radius) - distance
        if overlap > 0:
            # Move ball away from player
            ball.x += nx * overlap
            ball.y += ny * overlap
        
        # Calculate relative velocity
        rel_vx = ball.vx - player.vx
        rel_vy = ball.vy - player.vy
        
        # Calculate relative velocity along collision normal
        speed = rel_vx * nx + rel_vy * ny
        
        # Only resolve if objects are approaching
        if speed < 0:
            return
        
        # Ball is lighter than player, so player dominates the collision
        # Transfer some of player's velocity to ball
        ball_mass = 0.1
        player_mass = 1.0
        total_mass = ball_mass + player_mass
        
        # Calculate collision impulse
        impulse = 2 * speed / total_mass
        impulse *= self.restitution
        
        # Apply impulse (ball gets most of the impulse)
        ball.vx -= impulse * player_mass * nx
        ball.vy -= impulse * player_mass * ny
        player.vx += impulse * ball_mass * nx
        player.vy += impulse * ball_mass * ny
        
        # Update ball possession tracking
        ball.last_touched_by = player.global_id
    
    def _apply_boundary_constraints(
        self, 
        players: List[Player], 
        ball: Ball, 
        goals: List[Goal]
    ) -> None:
        """
        Apply field boundary constraints to players and ball.
        
        Args:
            players: List of all players
            ball: The ball object
            goals: List of goal objects
        """
        # Constrain players to field
        for player in players:
            # Horizontal boundaries
            if player.x - player.radius < self.left_boundary:
                player.x = self.left_boundary + player.radius
                player.vx = max(0, player.vx)  # Stop leftward movement
            elif player.x + player.radius > self.right_boundary:
                player.x = self.right_boundary - player.radius
                player.vx = min(0, player.vx)  # Stop rightward movement
            
            # Vertical boundaries
            if player.y - player.radius < self.top_boundary:
                player.y = self.top_boundary + player.radius
                player.vy = max(0, player.vy)  # Stop upward movement
            elif player.y + player.radius > self.bottom_boundary:
                player.y = self.bottom_boundary - player.radius
                player.vy = min(0, player.vy)  # Stop downward movement
        
        # Constrain ball to field (with special handling for goals)
        self._constrain_ball_to_field(ball, goals)
    
    def _constrain_ball_to_field(self, ball: Ball, goals: List[Goal]) -> None:
        """
        Constrain ball to field boundaries with goal area handling.
        
        Args:
            ball: The ball object
            goals: List of goal objects
        """
        # Check if ball is in a goal area
        in_goal = False
        for goal in goals:
            if goal.contains_ball(ball):
                in_goal = True
                break
        
        # Horizontal boundaries
        if not in_goal:
            if ball.x - ball.radius < self.left_boundary:
                ball.x = self.left_boundary + ball.radius
                ball.vx = -ball.vx * self.restitution  # Bounce
            elif ball.x + ball.radius > self.right_boundary:
                ball.x = self.right_boundary - ball.radius
                ball.vx = -ball.vx * self.restitution  # Bounce
        else:
            # Allow ball to enter goal areas
            if ball.x - ball.radius < 0:
                ball.x = ball.radius
                ball.vx = -ball.vx * self.restitution
            elif ball.x + ball.radius > self.field_width:
                ball.x = self.field_width - ball.radius
                ball.vx = -ball.vx * self.restitution
        
        # Vertical boundaries
        if ball.y - ball.radius < self.top_boundary:
            ball.y = self.top_boundary + ball.radius
            ball.vy = -ball.vy * self.restitution  # Bounce
        elif ball.y + ball.radius > self.bottom_boundary:
            ball.y = self.bottom_boundary - ball.radius
            ball.vy = -ball.vy * self.restitution  # Bounce
    
    def _apply_friction(self, players: List[Player], ball: Ball) -> None:
        """
        Apply friction to players and ball.
        
        Args:
            players: List of all players
            ball: The ball object
        """
        # Apply friction to players
        for player in players:
            player.vx *= self.player_friction
            player.vy *= self.player_friction
            
            # Stop very small movements
            if abs(player.vx) < 0.01:
                player.vx = 0
            if abs(player.vy) < 0.01:
                player.vy = 0
        
        # Apply friction to ball (already handled in ball.update_position())
        # The ball applies its own friction coefficient 