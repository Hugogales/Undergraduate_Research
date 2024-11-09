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


class Player:
    def __init__(
        self,
        team_id,
        player_id,
        up_key=None,
        down_key=None,
        left_key=None,
        right_key=None
    ):
        """
        Initializes the Player object with images, positions, and animation parameters.

        :param team_id: Integer representing the team (1 or 2).
        :param player_id: Integer representing the player within the team.
        :param up_key: Pygame key constant for moving up.
        :param down_key: Pygame key constant for moving down.
        :param left_key: Pygame key constant for moving left.
        :param right_key: Pygame key constant for moving right.
        """
        # Set original and current positions based on team and player ID
        if team_id == 1:
            self.original_position = ENV_PARAMS.team_1_positions[player_id]
        else:
            self.original_position = ENV_PARAMS.team_2_positions[player_id]
        self.position = self.original_position.copy()



        if ENV_PARAMS.RENDER:
            self.color = VIS_PARAMS.BLUE if team_id == 1 else VIS_PARAMS.GREEN

            # Load and flip images based on team
            if team_id == 1:
                self.body_image = pygame.image.load(VIS_PARAMS.TEAM1_SPRITES[player_id]).convert_alpha()
                self.arm_image = pygame.image.load(VIS_PARAMS.TEAM1_ARMS[player_id]).convert_alpha()
                self.leg_image = pygame.image.load(VIS_PARAMS.TEAM1_LEGS[player_id]).convert_alpha()
            else:
                # Reverse the images horizontally for team 2
                self.body_image = pygame.image.load(VIS_PARAMS.TEAM2_SPRITES[player_id]).convert_alpha()
                self.body_image = pygame.transform.flip(self.body_image, True, False)
                self.arm_image = pygame.image.load(VIS_PARAMS.TEAM2_ARMS[player_id]).convert_alpha()
                self.arm_image = pygame.transform.flip(self.arm_image, True, False)
                self.leg_image = pygame.image.load(VIS_PARAMS.TEAM2_LEGS[player_id]).convert_alpha()
                self.leg_image = pygame.transform.flip(self.leg_image, True, False)

            # Scale images to desired size
            self.body_image = pygame.transform.scale(
                self.body_image,
                (ENV_PARAMS.PLAYER_RADIUS * 2, ENV_PARAMS.PLAYER_RADIUS * 2)
            )
            self.arm_image = pygame.transform.scale(
                self.arm_image,
                (ENV_PARAMS.PLAYER_RADIUS, ENV_PARAMS.PLAYER_RADIUS)
            )
            self.leg_image = pygame.transform.scale(
                self.leg_image,
                (ENV_PARAMS.PLAYER_RADIUS, ENV_PARAMS.PLAYER_RADIUS)
            )

            # Initialize rotated images
            self.rotated_body_image = self.body_image
            self.rotated_arm_image = self.arm_image
            self.rotated_leg_image = self.leg_image

            # Get rects for positioning
            self.body_rect = self.rotated_body_image.get_rect(center=self.position)
            self.arm_rect = self.rotated_arm_image.get_rect(center=self.position)
            self.leg_rect = self.rotated_leg_image.get_rect(center=self.position)

            # Animation and Direction Attributes
            self.direction = 0  # Degrees (0 = right, 90 = up)
            self.is_moving = False

            # Animation parameters for arms and legs
            self.arm_angle = 0
            self.arm_direction = 1  # 1 for forward, -1 for backward
            self.arm_min_angle = -30  # Minimum rotation angle for arms
            self.arm_max_angle = 30   # Maximum rotation angle for arms
            self.arm_speed = 2        # Degrees per frame for arms

            self.leg_angle = 0
            self.leg_direction = 1  # 1 for forward, -1 for backward
            self.leg_min_angle = -30  # Minimum rotation angle for legs
            self.leg_max_angle = 30   # Maximum rotation angle for legs
            self.leg_speed = 2        # Degrees per frame for legs

            # Constants to tweak limb positions relative to the body
            self.ARM_OFFSET_X = 15  # Horizontal offset for arms
            self.ARM_OFFSET_Y = -20  # Vertical offset for arms
            self.LEG_OFFSET_X = 10   # Horizontal offset for legs
            self.LEG_OFFSET_Y = 25   # Vertical offset for legs

            # Movement keys
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key

        # Player attributes
        self.radius = ENV_PARAMS.PLAYER_RADIUS
        self.speed = ENV_PARAMS.PLAYER_SPEED
        self.velocity = [0, 0]  # (vx, vy)

    def handle_movement(self, keys):
        """
        Updates the player's velocity based on pressed keys and determines if the player is moving.

        :param keys: The current state of all keyboard buttons.
        :return: (vx, vy) movement vector.
        """
        up, down, left, right = False, False, False, False

        if keys[self.up_key]:
            up = True
        if keys[self.down_key]:
            down = True
        if keys[self.left_key]:
            left = True
        if keys[self.right_key]:
            right = True

        return self.move(up, down, left, right)

    def move(self, up=False, down=False, left=False, right=False):
        """
        Updates the player's position based on the given movement vector and handles boundary conditions.

        :param up: Move up.
        :param down: Move down.
        :param left: Move left.
        :param right: Move right.
        :return: (vx, vy) movement vector.
        """
        vx, vy = 0, 0

        if up:
            vy -= self.speed
        if down:
            vy += self.speed
        if left:
            vx -= self.speed
        if right:
            vx += self.speed

        # Determine if the player is moving
        if ENV_PARAMS.RENDER:
            self.is_moving = not (vx == 0 and vy == 0)
            # Calculate direction if moving
            if self.is_moving:
                self.direction = math.degrees(math.atan2(-vy, vx))  # Negative vy because Pygame's y-axis is inverted

            # Update position with boundary checks
            new_x = self.position[0] + vx
            new_y = self.position[1] + vy

        # Determine if the player is in a goal
        in_left_goal = (
            (new_x - self.radius <= ENV_PARAMS.GOAL_WIDTH) and
            (ENV_PARAMS.HEIGHT / 2 - ENV_PARAMS.GOAL_HEIGHT / 2 <= new_y <= ENV_PARAMS.HEIGHT / 2 + ENV_PARAMS.GOAL_HEIGHT / 2)
        )
        in_right_goal = (
            (new_x + self.radius >= ENV_PARAMS.WIDTH - ENV_PARAMS.GOAL_WIDTH) and
            (ENV_PARAMS.HEIGHT / 2 - ENV_PARAMS.GOAL_HEIGHT / 2 <= new_y <= ENV_PARAMS.HEIGHT / 2 + ENV_PARAMS.GOAL_HEIGHT / 2)
        )

        if in_left_goal or in_right_goal:
            # Calculate the vertical boundaries of the goal
            goal_top = (ENV_PARAMS.HEIGHT / 2) - (ENV_PARAMS.GOAL_HEIGHT / 2) + self.radius
            goal_bottom = (ENV_PARAMS.HEIGHT / 2) + (ENV_PARAMS.GOAL_HEIGHT / 2) - self.radius
            # Restrict vertical movement within the goal
            new_y = max(goal_top, min(new_y, goal_bottom))
            new_x = max(self.radius, min(new_x, ENV_PARAMS.WIDTH - self.radius))
        else:
            # Restrict vertical and horizontal movement within the play area
            new_y = max(self.radius, min(new_y, ENV_PARAMS.HEIGHT - self.radius))
            new_x = max(ENV_PARAMS.PLAY_AREA_LEFT + self.radius, min(new_x, ENV_PARAMS.PLAY_AREA_RIGHT - self.radius))

        # Update position
        self.position = [new_x, new_y]

        # Update velocity
        self.velocity = [vx, vy]

        return self.velocity

    def reset(self):
        """
        Resets the player's position to the original position and stops movement.
        """
        self.position = self.original_position.copy()
        self.velocity = [0, 0]
        self.is_moving = False
        self.direction = 0

    def update_animation(self):
        """
        Updates the angles of the arms and legs to simulate walking motion.
        """
        if not ENV_PARAMS.RENDER:
            return

        if self.is_moving:
            # Update arm angle
            self.arm_angle += self.arm_direction * self.arm_speed
            if self.arm_angle > self.arm_max_angle:
                self.arm_angle = self.arm_max_angle
                self.arm_direction = -1
            elif self.arm_angle < self.arm_min_angle:
                self.arm_angle = self.arm_min_angle
                self.arm_direction = 1

            # Update leg angle
            self.leg_angle += self.leg_direction * self.leg_speed
            if self.leg_angle > self.leg_max_angle:
                self.leg_angle = self.leg_max_angle
                self.leg_direction = -1
            elif self.leg_angle < self.leg_min_angle:
                self.leg_angle = self.leg_min_angle
                self.leg_direction = 1
        else:
            # Reset angles when not moving
            self.arm_angle = 0
            self.leg_angle = 0

    def update(self, keys):
        """
        Updates the player's state, including movement and animations.

        :param keys: The current state of all keyboard buttons.
        """
        self.handle_movement(keys)
        self.update_animation()

    def draw(self, surface):
        """
        Draws the player on the given surface with rotation and animations.

        :param surface: The Pygame surface to draw on.
        """
        if not ENV_PARAMS.RENDER:
            return

        # Rotate body based on direction
        self.rotated_body_image = pygame.transform.rotate(self.body_image, self.direction)
        self.body_rect = self.rotated_body_image.get_rect(center=self.position)
        surface.blit(self.rotated_body_image, self.body_rect)

        # Rotate arms based on arm_angle and direction
        total_arm_angle = self.arm_angle + self.direction
        self.rotated_arm_image = pygame.transform.rotate(self.arm_image, total_arm_angle)
        self.arm_rect = self.rotated_arm_image.get_rect(
            center=(
                self.position[0] + self.ARM_OFFSET_X * math.cos(math.radians(self.direction)),
                self.position[1] + self.ARM_OFFSET_Y * math.sin(math.radians(self.direction))
            )
        )
        surface.blit(self.rotated_arm_image, self.arm_rect)

        # Rotate legs based on leg_angle and direction
        total_leg_angle = self.leg_angle + self.direction
        self.rotated_leg_image = pygame.transform.rotate(self.leg_image, total_leg_angle)
        self.leg_rect = self.rotated_leg_image.get_rect(
            center=(
                self.position[0] + self.LEG_OFFSET_X * math.cos(math.radians(self.direction)),
                self.position[1] + self.LEG_OFFSET_Y * math.sin(math.radians(self.direction))
            )
        )
        surface.blit(self.rotated_leg_image, self.leg_rect)
