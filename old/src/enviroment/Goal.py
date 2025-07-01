from params import EnvironmentHyperparameters, VisualHyperparametters
import pygame
import math
import os

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()


if not ENV_PARAMS.RENDER:
    # Disable Pygame support if rendering is disabled
    os.environ["SDL_VIDEODRIVER"] = "dummy"
else:
    VIS_PARAMS = VisualHyperparametters()

class Goal:
    def __init__(self, position, width, height, invert_image = False):
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

        if ENV_PARAMS.RENDER:
            self.color = VIS_PARAMS.GREEN
            # Load ball image
            original_image = pygame.image.load(VIS_PARAMS.GOAL_SPRITE).convert_alpha()
            if invert_image:
                original_image = pygame.transform.flip(original_image, True, False)
            self.original_image = pygame.transform.scale(original_image, (self.width * 1.1 , self.height  * 1.1))

    def check_goal(self, ball):
        """
        Checks if the entire ball has entered this goal area.

        :param ball: Instance of Ball.
        :return: True if the entire ball is in the goal, else False.
        """
        # Define goal boundaries
        left, top = self.inner_position
        right = left + self.width
        bottom = top + self.height

        # Ensure the entire ball is within the inner goal area
        if self.position[0] == 0:
            # Left goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] + ball.radius) <= right and (ball.position[1] + 5*  ball.radius) >= top and (ball.position[1] - 5 *ball.radius) <= bottom
        else:
            # Right goal: entire ball must be within inner goal boundaries
            in_goal_x = (ball.position[0] - ball.radius) >= left and (ball.position[1] + 5*  ball.radius) >= top and (ball.position[1] - 5 * ball.radius) <= bottom

        return in_goal_x 
       
    def draw(self, surface):
        """
        Draws the goal and its inner boundary on the given surface.

        :param surface: The Pygame surface to draw on.
        """
        # Draw outer goal
        new_rect = self.original_image.get_rect(center=(self.position[0]+ self.width/2, self.position[1] + self.height/2))
        # Draw the rotated image on the surface
        surface.blit(self.original_image, new_rect.topleft)
    
    def draw_grass(self, surface):
        """
        Draws the grass area in the goal.

        :param surface: The Pygame surface to draw on.
        """
        # Draw green goal area
        pygame.draw.rect(surface, VIS_PARAMS.GREEN, (self.position[0], self.position[1], self.width, self.height))
        
        