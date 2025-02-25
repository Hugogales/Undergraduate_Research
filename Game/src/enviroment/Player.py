from params import EnvironmentHyperparameters, VisualHyperparametters
import random
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
        right_key=None,
        shoot_key=None
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

        self.team_id = team_id
        self.player_id = player_id

        if team_id == 1:
            self.original_position = ENV_PARAMS.team_1_positions[player_id]
        else:
            self.original_position = ENV_PARAMS.team_2_positions[player_id]
        self.position = self.original_position.copy()

        # Movement keys
        self.up_key = up_key
        self.down_key = down_key
        if team_id == 1:
            self.left_key = left_key
            self.right_key = right_key
        else:
            self.left_key = right_key
            self.right_key = left_key
        
        self.shoot_key = shoot_key

        # Player attributes
        self.radius = ENV_PARAMS.PLAYER_RADIUS
        self.speed = ENV_PARAMS.PLAYER_SPEED
        self.velocity = [0, 0]  # (vx, vy)
        self.is_kick = False

        # Animation and Direction Attributes

        if ENV_PARAMS.RENDER:
            self.is_moving = False

            # Load images
            if team_id == 1:
                self.body_image = pygame.image.load(VIS_PARAMS.TEAM1_SPRITES[player_id]).convert_alpha()
                self.arm = pygame.image.load(VIS_PARAMS.TEAM1_ARMS[player_id]).convert_alpha()
                self.leg = pygame.image.load(VIS_PARAMS.TEAM1_LEGS[player_id]).convert_alpha()
                self.direction = 0  # Degrees (0 = right, 90 = up)
            else:
                self.body_image = pygame.image.load(VIS_PARAMS.TEAM2_SPRITES[player_id]).convert_alpha()
                self.arm = pygame.image.load(VIS_PARAMS.TEAM2_ARMS[player_id]).convert_alpha()
                self.leg = pygame.image.load(VIS_PARAMS.TEAM2_LEGS[player_id]).convert_alpha()
                self.direction = 180  # Degrees (0 = right, 90 = up)

            # Scale images to desired size
            body_size = (ENV_PARAMS.PLAYER_RADIUS *1.5 , ENV_PARAMS.PLAYER_RADIUS * 2)
            leg_size = (ENV_PARAMS.PLAYER_RADIUS *1.5, ENV_PARAMS.PLAYER_RADIUS * 0.75)
            arm_size = (ENV_PARAMS.PLAYER_RADIUS *3, ENV_PARAMS.PLAYER_RADIUS * 0.75)

            self.body_image = pygame.transform.scale(self.body_image, body_size)
            self.arm = pygame.transform.scale(self.arm, arm_size)
            self.arm = pygame.transform.flip(self.arm, True, False)
            self.leg = pygame.transform.scale(self.leg, leg_size)

            # Initialize limb images for left and right arms and legs
            self.left_arm_image = self.arm
            self.right_arm_image = self.arm

            self.left_leg_image = self.leg
            self.right_leg_image = self.leg

            self.arm_angle = 0
            self.arm_direction = 1  # 1 for forward, -1 for backward
            self.arm_min_angle = -40  # Minimum rotation angle for arms
            self.arm_max_angle = 40  # Maximum rotation angle for arms
            self.arm_speed = 9    # Degrees per frame for arms

            self.left_arm_angle = 0
            self.right_arm_angle = 0

            self.leg_direction = -1  # Opposite to arms for natural walking
            self.leg_size = 0
            self.left_leg_size = 0
            self.right_leg_size = 0
            self.leg_min_size = -20  # Minimum stretch value for legs
            self.leg_max_size = 20  # Maximum stretch value for legs
            self.leg_speed =  4.5     # Units per frame for legs
            self.leg_angle = 0

            # Constants to tweak limb positions relative to the body
            self.ARM_OFFSET_X = -ENV_PARAMS.PLAYER_RADIUS * 0.1  # Adjust as needed
            self.ARM_OFFSET_Y = ENV_PARAMS.PLAYER_RADIUS * 0.6   # Adjust as needed

            self.LEG_OFFSET_X = ENV_PARAMS.PLAYER_RADIUS * 3/4
            self.LEG_OFFSET_Y = ENV_PARAMS.PLAYER_RADIUS / 2

            # Flip right leg image for mirroring
            self.right_leg_image = pygame.transform.flip(self.leg, True, False)

    def _use_keys():
        """
        Returns the keys used by the player for movement.
        """


    def handle_movement(self, keys):
        arr = [
            1 if keys[self.up_key] else 0,
            1 if keys[self.down_key] else 0,
            1 if keys[self.left_key] else 0,
            1 if keys[self.right_key] else 0,
            1 if keys[self.shoot_key] else 0]
            
        return self.move(arr)
       
    def move(self, arr):
        """
        moves the player based on the input array
        :param arr: array of 4 elements representing the movement keys (up, down, left, right)

        :return: the velocity of the player
        """
        vx, vy = 0, 0

        if self.team_id == 1:
            if arr[0] == 1: # up
                vy -= self.speed
            if arr[1] == 1: # down
                vy += self.speed
            if arr[2] == 1: # left
                vx -= self.speed
            if arr[3] == 1: # right
                vx += self.speed
        elif self.team_id == 2:
            if arr[1] == 1: # up
                vy -= self.speed
            if arr[0] == 1: # down
                vy += self.speed
            if arr[3] == 1: # right
                vx -= self.speed
            if arr[2] == 1: # left
                vx += self.speed
        if arr[4] == 1: # shoot
            self.is_kick = True
        else:
            self.is_kick = False

                # Normalize velocity to maintain constant speed
        if vx != 0 or vy != 0:
            magnitude = math.sqrt(vx ** 2 + vy ** 2)
            vx = (vx / magnitude) * self.speed
            vy = (vy / magnitude) * self.speed
        
        self.velocity = [vx, vy]

        if ENV_PARAMS.RENDER:
            self.is_moving = (vx != 0 or vy != 0)
            if self.is_moving:
                self.direction = math.degrees(math.atan2(vy, vx))
                self.leg_angle = math.degrees(math.atan2(vy, vx))

        # Update position with boundary checks
        new_x = self.position[0] + vx
        new_y = self.position[1] + vy

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


        self.position = [new_x, new_y]

        return self.velocity

    def reset(self):
        """
        Resets the player's position to the original position and velocity.
        """
        self.position = self.original_position.copy()
    
        self.velocity = [0, 0]

        if ENV_PARAMS.RENDER:
            self.is_moving = False

    # VISUALIZATION METHODS

    def update_animation(self):
        if not ENV_PARAMS.RENDER:
            return

        if self.is_moving:
            # Update arm angle
            self.arm_angle += self.arm_direction * self.arm_speed
            if self.arm_angle > self.arm_max_angle or self.arm_angle < self.arm_min_angle:
                self.arm_direction *= -1  # Reverse direction

            # Set left and right arm angles to be opposite
            self.left_arm_angle = self.arm_angle
            self.right_arm_angle = -self.arm_angle

            # Update leg size for stretching
            self.leg_size += self.leg_direction * self.leg_speed
            if self.leg_size > self.leg_max_size or self.leg_size < self.leg_min_size:
                self.leg_direction *= -1  # Reverse direction

            # Set left and right leg sizes to be opposite
            self.left_leg_size = self.leg_size
            self.right_leg_size = -self.leg_size

        else:
            # Reset angles when not moving
            self.arm_angle = 0
            self.left_arm_angle = 0
            self.right_arm_angle = 0
            self.leg_size = 0
            self.left_leg_size = 0
            self.right_leg_size = 0

    def calculate_arm_position(self, offset_x, offset_y):
        rad = math.radians(self.direction)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] + offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y

    def calculate_leg_position(self, offset_x, offset_y):
        rad = math.radians(self.leg_angle)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] + offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y

    def draw(self, surface):
        if not ENV_PARAMS.RENDER:
            return

        # Leg scaling factors based on leg sizes
        left_leg_scale_factor = abs( self.left_leg_size / 10 )# Adjust divisor as needed
        right_leg_scale_factor = abs( self.right_leg_size / 10)  # Adjust divisor as needed

        # Scale leg images (stretch along the y-axis)
        left_leg_scaled = pygame.transform.scale(
            self.left_leg_image,
            (int(self.left_leg_image.get_width() * left_leg_scale_factor), self.left_leg_image.get_height())
        )
        right_leg_scaled = pygame.transform.scale(
            self.right_leg_image,
            (int(self.right_leg_image.get_width() * right_leg_scale_factor), self.right_leg_image.get_height() )
        )

        # Calculate leg positions
        if self.leg_size > 0:
            left_leg_pos = self.calculate_leg_position(self.LEG_OFFSET_X, self.LEG_OFFSET_Y)
            rotated_left_leg = pygame.transform.rotate(left_leg_scaled, -self.leg_angle)
            right_leg_pos = self.calculate_leg_position(-self.LEG_OFFSET_X, -self.LEG_OFFSET_Y)
            rotated_right_leg = pygame.transform.rotate(right_leg_scaled, -self.leg_angle)
        else:
            left_leg_pos = self.calculate_leg_position(-self.LEG_OFFSET_X, self.LEG_OFFSET_Y)
            rotated_left_leg = pygame.transform.rotate(left_leg_scaled, -self.leg_angle + 180)
            right_leg_pos = self.calculate_leg_position(self.LEG_OFFSET_X, -self.LEG_OFFSET_Y)
            rotated_right_leg = pygame.transform.rotate(right_leg_scaled, -self.leg_angle + 180)

        # Draw left leg
        left_leg_rect = rotated_left_leg.get_rect(center=left_leg_pos)
        surface.blit(rotated_left_leg, left_leg_rect)

        # Draw right leg
        right_leg_rect = rotated_right_leg.get_rect(center=right_leg_pos)
        surface.blit(rotated_right_leg, right_leg_rect)

        # Draw left arm
        left_arm_angle = - self.left_arm_angle + self.direction + 90
        rotated_left_arm = pygame.transform.rotate(self.left_arm_image, -left_arm_angle)
        left_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, -self.ARM_OFFSET_Y)
        left_arm_rect = rotated_left_arm.get_rect(center=left_arm_pos)
        surface.blit(rotated_left_arm, left_arm_rect)

        # Draw right arm
        right_arm_angle = self.right_arm_angle + self.direction - 90
        rotated_right_arm = pygame.transform.rotate(self.right_arm_image, -right_arm_angle)
        right_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, self.ARM_OFFSET_Y)
        right_arm_rect = rotated_right_arm.get_rect(center=right_arm_pos)
        surface.blit(rotated_right_arm, right_arm_rect)

        # Rotate body based on direction
        rotated_body = pygame.transform.rotate(self.body_image, -self.direction)
        body_rect = rotated_body.get_rect(center=self.position)
        surface.blit(rotated_body, body_rect)

        # Update animations
        self.update_animation()
