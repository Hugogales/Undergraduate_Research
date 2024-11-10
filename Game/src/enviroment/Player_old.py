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

        # Movement keys
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key

        # Player attributes
        self.radius = ENV_PARAMS.PLAYER_RADIUS
        self.speed = ENV_PARAMS.PLAYER_SPEED
        self.velocity = [0, 0]  # (vx, vy)

        # Animation and Direction Attributes

        # Initialize rendering-specific attributes only if rendering is enabled
        if ENV_PARAMS.RENDER:
            self.color = VIS_PARAMS.BLUE if team_id == 1 else VIS_PARAMS.GREEN
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
            limb_size = (ENV_PARAMS.PLAYER_RADIUS *1.5, ENV_PARAMS.PLAYER_RADIUS * 0.75)

            self.body_image = pygame.transform.scale(self.body_image, body_size)
            self.arm = pygame.transform.scale(self.arm, limb_size)
            self.arm = pygame.transform.flip(self.arm, True, False)
            self.leg = pygame.transform.scale(self.leg, limb_size)

            # Initialize limb images for left and right arms and legs
            self.left_arm_image = self.arm
            self.right_arm_image = self.arm

            self.left_leg_image = self.leg
            self.right_leg_image = self.leg

            # Animation parameters for arms and legs
            self.arm_angle = 0
            self.arm_direction = 1  # 1 for forward, -1 for backward
            self.arm_min_angle = -30  # Minimum rotation angle for arms
            self.arm_max_angle = 30  # Maximum rotation angle for arms
            self.arm_speed = 3        # Degrees per frame for arms

            self.leg_angle = 0
            self.leg_direction = -1  # Opposite to arms for natural walking

            self.leg_size = 0
            self.leg_min_size = -30  # Minimum rotation angle for legs
            self.leg_max_size= 30  # Maximum rotation angle for legs
            self.leg_speed = 5        # Degrees per frame for legs

            # Constants to tweak limb positions relative to the body
            self.ARM_OFFSET_X = -ENV_PARAMS.PLAYER_RADIUS * 0.1  # Adjust as needed
            self.ARM_OFFSET_Y = ENV_PARAMS.PLAYER_RADIUS * 1.25   # Adjust as needed
            self.LEG_OFFSET_X = ENV_PARAMS.PLAYER_RADIUS /2 
            self.LEG_OFFSET_Y = ENV_PARAMS.PLAYER_RADIUS /2

    def handle_movement(self, keys):
        vx, vy = 0, 0

        if keys[self.up_key]:
            vy -= self.speed
        if keys[self.down_key]:
            vy += self.speed
        if keys[self.left_key]:
            vx -= self.speed
        if keys[self.right_key]:
            vx += self.speed

        self.velocity = [vx, vy]

        self.is_moving = (vx != 0 or vy != 0)

        if self.is_moving:
            self.direction = math.degrees(math.atan2(vy, vx))  # Negative vy because Pygame's y-axis is inverted
            if vy > 0 and vx == 0:
                angle =  0
                distance = self.leg_angle - angle
                self.leg_angle += distance 

            elif vy > 0 and vx > 0:
                angle = 45
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy == 0 and vx > 0:
                angle = 90
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy < 0 and vx > 0:
                angle = 135
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy < 0 and vx == 0:
                angle = 180
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy < 0 and vx < 0:
                angle = 225
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy == 0 and vx < 0:
                angle = 270
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            elif vy > 0 and vx< 0:
                angle = 315
                distance = self.leg_angle - angle
                self.leg_angle += distance 
            

        # Update position with boundary checks
        new_x = self.position[0] + vx
        new_y = self.position[1] + vy

        # Apply boundary checks here

        self.position = [new_x, new_y]

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
        if not ENV_PARAMS.RENDER:
            return

        if self.is_moving:
            # Update arm angles
            self.arm_angle += self.arm_direction * self.arm_speed
            if self.arm_angle > self.arm_max_angle or self.arm_angle < self.arm_min_angle:
                self.arm_direction *= -1  # Reverse direction

            # Update leg size for stretching
            self.leg_size += self.leg_direction * self.leg_speed
            if self.leg_size > self.leg_max_size or self.leg_size < self.leg_min_size:
                self.leg_direction *= -1  # Reverse direction
        else:
            # Reset angles when not moving
            self.arm_angle = 0
            self.leg_size = 0
    
    def calculate_arm_position(self, offset_x, offset_y):
        rad = math.radians(self.direction)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] + offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y
    
    def calculate_leg_position(self, offset_x, offset_y):   
        rad = math.radians(self.leg_angle)
        x = self.position[0] + offset_x * math.cos(rad) - offset_y * math.sin(rad)
        y = self.position[1] - offset_x * math.sin(rad) + offset_y * math.cos(rad)
        return x, y

    def draw(self, surface):
        if not ENV_PARAMS.RENDER:
            return

        # Rotate body based on direction
        rotated_body = pygame.transform.rotate(self.body_image, -self.direction)
        body_rect = rotated_body.get_rect(center=self.position)

        # Function to calculate limb positions (already updated)

        # Draw left arm
        left_arm_angle = self.arm_angle + self.direction + 90
        rotated_left_arm = pygame.transform.rotate(self.left_arm_image, -left_arm_angle)
        left_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, -self.ARM_OFFSET_Y)
        left_arm_rect = rotated_left_arm.get_rect(center=left_arm_pos)

        # Draw right arm
        right_arm_angle = -self.arm_angle + self.direction - 90
        rotated_right_arm = pygame.transform.rotate(self.right_arm_image, -right_arm_angle)
        right_arm_pos = self.calculate_arm_position(-self.ARM_OFFSET_X, self.ARM_OFFSET_Y)
        right_arm_rect = rotated_right_arm.get_rect(center=right_arm_pos)

        # Leg scaling factor based on leg_size
        leg_scale_factor = self.leg_size / 20 # Adjust as needed
        if leg_scale_factor < 0:
            leg_scale_factor = -leg_scale_factor
            new_direction = -self.direction - 180 
            new_offset_x = self.LEG_OFFSET_X
            #flip it 
            rotated_right_leg = pygame.transform.flip(self.right_leg_image, True, True)
            rotated_left_leg = self.left_leg_image
        else:
            new_direction = self.direction
            new_offset_x = -self.LEG_OFFSET_X

            rotated_right_leg = self.right_leg_image
            rotated_left_leg = pygame.transform.flip(self.left_leg_image, True, True)



        # Scale leg images (stretch along the y-axis)
        left_leg_scaled = pygame.transform.scale(
            rotated_left_leg,
            (self.left_leg_image.get_width()* leg_scale_factor, self.left_leg_image.get_height())
        )
        right_leg_scaled = pygame.transform.scale(
            rotated_right_leg,
            (self.right_leg_image.get_width() * leg_scale_factor, self.right_leg_image.get_height())
        )


        # Calculate leg positions
        left_leg_pos = self.calculate_leg_position(new_offset_x,- self.LEG_OFFSET_Y)
        right_leg_pos = self.calculate_leg_position(-new_offset_x, self.LEG_OFFSET_Y)

        pygame.transform.rotate(rotated_left_leg, self.leg_angle)
        pygame.transform.rotate(rotated_right_leg, self.leg_angle)

        # Draw left leg
        left_leg_rect = left_leg_scaled.get_rect(center=left_leg_pos)
        surface.blit(left_leg_scaled, left_leg_rect)
        # Draw right leg
        right_leg_rect = right_leg_scaled.get_rect(center=right_leg_pos)
        surface.blit(right_leg_scaled, right_leg_rect)


        surface.blit(rotated_left_arm, left_arm_rect)
        surface.blit(rotated_right_arm, right_arm_rect)

        surface.blit(rotated_body, body_rect)


        # Update animations
        self.update_animation()
