from src.params import EnvironmentHyperparameters
import pygame
import math
import os

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()

if not ENV_PARAMS.RENDER:
    # Disable Pygame support if rendering is disabled
    os.environ["SDL_VIDEODRIVER"] = "dummy"

class Player:
    def __init__(self, position, color, up_key, down_key, left_key, right_key):
        """
        Initializes the Player object.

        :param position: [x, y] initial position of the player.
        :param color: Color of the player.
        :param up_key: Pygame key constant for moving up.
        :param down_key: Pygame key constant for moving down.
        :param left_key: Pygame key constant for moving left.
        :param right_key: Pygame key constant for moving right.
        """
        self.position = position
        self.color = color
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key
        self.radius = ENV_PARAMS.PLAYER_RADIUS
        self.speed = ENV_PARAMS.PLAYER_SPEED
        self.velocity = [0, 0]  # (vx, vy)

    def handle_movement(self, keys, up = False, right = False):
        """
        Updates the player's position based on pressed keys.

        :param keys: The current state of all keyboard buttons.
        :return: (vx, vy) movement vector.
        """
        vx, vy = 0, 0

        if up:
            vy -= self.speed
        if right:
            vx += self.speed

        if keys[self.up_key]:
            vy -= self.speed 
        if keys[self.down_key]:
            vy += self.speed
        if keys[self.left_key]:
            vx -= self.speed
        if keys[self.right_key]:
            vx += self.speed

        # Update position
        new_x = self.position[0] + vx
        new_y = self.position[1] + vy

        # Determine if the player is in a goal
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
            # Restrict vertical movement within the entire screen
            new_y = max(self.radius, min(new_y, ENV_PARAMS.HEIGHT - self.radius))
            new_x = max(ENV_PARAMS.PLAY_AREA_LEFT + self.radius, min(new_x, ENV_PARAMS.PLAY_AREA_RIGHT - self.radius))

        # Update position
        self.position = [new_x, new_y]

        # Update velocity
        self.velocity = [vx, vy]

        return self.velocity

    def draw(self, surface):
        """
        Draws the player on the given surface.

        :param surface: The Pygame surface to draw on.
        """
        pygame.draw.circle(surface, self.color, (int(self.position[0]), int(self.position[1])), self.radius)
