from src.params import EnvironmentHyperparameters
import pygame
import math
import os

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()

if not ENV_PARAMS.RENDER:
    # Disable Pygame support if rendering is disabled
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class Ball:
    def __init__(self, position, color):
        """
        Initializes the Ball object.

        :param position: [x, y] initial position of the ball.
        :param color: Color of the ball.
        """
        self.position = position
        self.color = color
        self.radius = ENV_PARAMS.BALL_RADIUS
        self.velocity = [0, 0]  # (vx, vy)

    def update_position(self):
        """
        Updates the ball's position based on its velocity and handles boundary collisions.
        """
        # Update position based on velocity
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]

        # Determine if the ball is in a goal
        in_left_goal = (new_x - self.radius <= ENV_PARAMS.GOAL_WIDTH) and (new_y >= ENV_PARAMS.HEIGHT / 2 - ENV_PARAMS.GOAL_HEIGHT / 2) and (new_y <= ENV_PARAMS.HEIGHT / 2 + ENV_PARAMS.GOAL_HEIGHT / 2)
        in_right_goal = (new_x + self.radius >= ENV_PARAMS.WIDTH - ENV_PARAMS.GOAL_WIDTH) and (new_y >= ENV_PARAMS.HEIGHT / 2 - ENV_PARAMS.GOAL_HEIGHT / 2) and (new_y <= ENV_PARAMS.HEIGHT / 2 + ENV_PARAMS.GOAL_HEIGHT / 2)

        # Calculate allowed vertical boundaries based on position
        if in_left_goal or in_right_goal:
            # Calculate the vertical boundaries of the goal
            goal_top = (ENV_PARAMS.HEIGHT / 2) - (ENV_PARAMS.GOAL_HEIGHT / 2) + self.radius
            goal_bottom = (ENV_PARAMS.HEIGHT / 2) + (ENV_PARAMS.GOAL_HEIGHT / 2) - self.radius
            # Restrict vertical movement within the goal
            new_y = max(goal_top, min(new_y, goal_bottom))
            new_x = max(self.radius, min(new_x, ENV_PARAMS.WIDTH - self.radius))
        else:
            # Bouncing off the play area walls
            if new_x <= ENV_PARAMS.PLAY_AREA_LEFT + self.radius:
                new_x = ENV_PARAMS.PLAY_AREA_LEFT + self.radius
                self.velocity[0] = -self.velocity[0]
            elif new_x >= ENV_PARAMS.PLAY_AREA_RIGHT - self.radius:
                new_x = ENV_PARAMS.PLAY_AREA_RIGHT - self.radius
                self.velocity[0] = -self.velocity[0]

            if new_y <= self.radius:
                new_y = self.radius
                self.velocity[1] = -self.velocity[1]
            elif new_y >= ENV_PARAMS.HEIGHT - self.radius:
                new_y = ENV_PARAMS.HEIGHT - self.radius
                self.velocity[1] = -self.velocity[1]

            # Restrict vertical movement within the entire screen
            new_y = max(self.radius, min(new_y, ENV_PARAMS.HEIGHT - self.radius))
            new_x = max(ENV_PARAMS.PLAY_AREA_LEFT + self.radius, min(new_x, ENV_PARAMS.PLAY_AREA_RIGHT - self.radius))

        # Update position
        self.position = [new_x, new_y]

        # Apply friction to slow down the ball
        self.velocity[0] *= ENV_PARAMS.BALL_FRICTION
        self.velocity[1] *= ENV_PARAMS.BALL_FRICTION

        # Cap the ball's speed
        speed = math.hypot(self.velocity[0], self.velocity[1])
        if speed > ENV_PARAMS.BALL_MAX_SPEED:
            self.velocity[0] = (self.velocity[0] / speed) * ENV_PARAMS.BALL_MAX_SPEED
            self.velocity[1] = (self.velocity[1] / speed) * ENV_PARAMS.BALL_MAX_SPEED

    def draw(self, surface):
        """
        Draws the ball on the given surface.

        :param surface: The Pygame surface to draw on.
        """
        pygame.draw.circle(surface, self.color, (int(self.position[0]), int(self.position[1])), self.radius)