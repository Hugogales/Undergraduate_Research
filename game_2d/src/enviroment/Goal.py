from src.params import EnvironmentHyperparameters
import pygame
import math
import os

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()

if not ENV_PARAMS.RENDER:
    # Disable Pygame support if rendering is disabled
    os.environ["SDL_VIDEODRIVER"] = "dummy"

class Goal:
    def __init__(self, position, width, height, color=ENV_PARAMS.GOAL_COLOR):
        """
        Initializes the Goal object.

        :param position: (x, y) position of the top-left corner of the goal.
        :param width: Width of the goal area.
        :param height: Height of the goal area.
        :param color: Color of the goal outline.
        """
        self.position = position
        self.width = width
        self.height = height
        self.color = color

        # Create boundary square within the goal
        self.inner_height = self.height - 2 * ENV_PARAMS.PLAYER_HEIGHT
        self.inner_position = [
            self.position[0],
            self.position[1] + ENV_PARAMS.PLAYER_HEIGHT
        ]
        self.inner_rect = pygame.Rect(
            self.inner_position[0],
            self.inner_position[1],
            self.width,
            self.inner_height
        )

    def draw(self, surface):
        """
        Draws the goal and its inner boundary on the given surface.

        :param surface: The Pygame surface to draw on.
        """
        # Draw outer goal
        pygame.draw.rect(surface, self.color, 
                         (self.position[0], self.position[1], self.width, self.height), 2)  # 2-pixel thick outline
        # Draw inner boundary
        pygame.draw.rect(surface, self.color, self.inner_rect, 2)

    def check_goal(self, ball):
        """
        Checks if the entire ball has entered this goal area.

        :param ball: Instance of Ball.
        :return: True if the entire ball is in the goal, else False.
        """
        # Define goal boundaries
        left, top = self.inner_position
        right = left + self.width
        bottom = top + self.inner_height

        # Ensure the entire ball is within the inner goal area
        if self.position[0] == 0:
            # Left goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] + ball.radius) <= right
        else:
            # Right goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] - ball.radius) >= left

        return in_goal_x 