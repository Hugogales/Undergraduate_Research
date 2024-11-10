import pygame
import sys
import math
from params import EnvironmentHyperparameters, VisualHyperparametters
from enviroment.Ball import Ball
from enviroment.Player import Player
from enviroment.Goal import Goal

# Instantiate Pygame and initialize all imported Pygame modules

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()
if ENV_PARAMS.RENDER:
    VIS_PARAMS = VisualHyperparametters()

# =======================
# Game Class
# =======================

class Game:
    def __init__(self):
        """
        Initializes the Game object by setting up players, ball, goals, and scores.
        """
        self.screen = pygame.display.set_mode((ENV_PARAMS.WIDTH, ENV_PARAMS.HEIGHT))

        self.team_1 = []
        self.team_2 = []
        for i in range(ENV_PARAMS.NUMBER_OF_PLAYERS):
            if ENV_PARAMS.RENDER:
                #Wasd
                self.team_1.append(
                    Player(
                    team_id=1,
                    player_id=i,
                    up_key=pygame.K_w,
                    down_key=pygame.K_s,
                    left_key=pygame.K_a,
                    right_key=pygame.K_d
                ))
                #Arrow keys
                self.team_2.append(
                    Player(
                    team_id=2,
                    player_id=i,
                    up_key=pygame.K_UP,
                    down_key=pygame.K_DOWN,
                    left_key=pygame.K_LEFT,
                    right_key=pygame.K_RIGHT
                ))
            else:
                self.team_1.append(
                    Player(
                    team_id=1,
                    player_id=i
                ))
                self.team_2.append(
                    Player(
                    team_id=2,
                    player_id=i
                ))

        self.players = self.team_1 + self.team_2

        # Initialize Ball
        self.ball = Ball(
            position=[ENV_PARAMS.WIDTH // 2, ENV_PARAMS.HEIGHT // 2],
        )

        # Initialize Goals (only central part with inner boundaries)
        goal_height = ENV_PARAMS.GOAL_HEIGHT
        self.goal1 = Goal(
            position=[0, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Left side, centered vertically
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
            invert_image=True,
        )
        self.goal2 = Goal(
            position=[ENV_PARAMS.WIDTH - ENV_PARAMS.GOAL_WIDTH, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Right side, centered vertically
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
        )

        # Initialize Scores
        self.score_team1 = 0
        self.score_team2 = 0

        self.timer = ENV_PARAMS.GAME_DURATION
        self.simulation_time = 0
        self.delta_time = 1 / ENV_PARAMS.FPS

        if ENV_PARAMS.RENDER:
            self.clock = pygame.time.Clock()
            pygame.display.set_caption(VIS_PARAMS.TITLE)
            self.font = pygame.font.SysFont('Arial', 36)
            self.pitch_image = pygame.image.load(VIS_PARAMS.BACKGROUND).convert_alpha()
            self.pitch_image = pygame.transform.scale(
                self.pitch_image, 
                (ENV_PARAMS.PLAY_AREA_RIGHT - ENV_PARAMS.PLAY_AREA_LEFT, ENV_PARAMS.PLAY_AREA_BOTTOM - ENV_PARAMS.PLAY_AREA_TOP)
            )
        else:
            self.clock = None

    def handle_collisions(self):
        """
        Handles collision between players and the ball, and between players themselves.
        """
        for player in self.players:
            # Check collision between player and ball
            self.check_collision(player, self.ball)

            # Check collision between player and player
            for other_player in self.players:
                if player == other_player:
                    continue
                self.check_player_collision(player, other_player)

    def check_collision(self, player, ball):
        """
        Checks and handles collision between a player and the ball.

        :param player: Instance of Player.
        :param ball: Instance of Ball.
        """
        distance = math.hypot(ball.position[0] - player.position[0],
                              ball.position[1] - player.position[1])
        min_distance = player.radius + ball.radius

        if distance < min_distance:
            # Calculate the overlap
            overlap = min_distance - distance

            # Calculate the direction from player to ball
            if distance == 0:
                # Prevent division by zero; assign arbitrary direction
                direction_x, direction_y = 1, 0
            else:
                direction_x = (ball.position[0] - player.position[0]) / distance
                direction_y = (ball.position[1] - player.position[1]) / distance

            # Adjust ball position to prevent sticking
            ball.position[0] += direction_x * overlap
            ball.position[1] += direction_y * overlap

            # Calculate relative velocity
            relative_velocity_x = ball.velocity[0] - player.velocity[0]
            relative_velocity_y = ball.velocity[1] - player.velocity[1]

            # Calculate the velocity along the direction of collision
            velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y

            if velocity_along_normal > 0:
                return  # They are moving away from each other

            # Define restitution (elasticity)
            restitution = 0.7  # 0 = inelastic, 1 = perfectly elastic

            # Update ball's velocity based on collision
            ball.velocity[0] -= (1 + restitution) * velocity_along_normal * direction_x
            ball.velocity[1] -= (1 + restitution) * velocity_along_normal * direction_y

    def check_player_collision(self, player1, player2):
        """
        Checks and handles collision between two players.

        :param player1: Instance of Player.
        :param player2: Instance of Player.
        """
        distance = math.hypot(player2.position[0] - player1.position[0],
                              player2.position[1] - player1.position[1])
        min_distance = player1.radius + player2.radius

        if distance < min_distance:
            # Calculate the overlap
            overlap = min_distance - distance

            # Calculate the direction from player1 to player2
            if distance == 0:
                # Prevent division by zero; assign arbitrary direction
                direction_x, direction_y = 1, 0
            else:
                direction_x = (player2.position[0] - player1.position[0]) / distance
                direction_y = (player2.position[1] - player1.position[1]) / distance

            # Adjust positions to prevent sticking (push players apart equally)
            player1.position[0] -= direction_x * (overlap / 2)
            player1.position[1] -= direction_y * (overlap / 2)
            player2.position[0] += direction_x * (overlap / 2)
            player2.position[1] += direction_y * (overlap / 2)

            # Calculate relative velocity
            relative_velocity_x = player2.velocity[0] - player1.velocity[0]
            relative_velocity_y = player2.velocity[1] - player1.velocity[1]

            # Calculate the velocity along the direction of collision
            velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y

            if velocity_along_normal > 0:
                return  # They are moving away from each other

            # Define restitution (elasticity) less than ball's
            restitution = 0.7  # Players push each other less strongly

            # Update players' velocities based on collision
            player1.velocity[0] -= (1 + restitution) * velocity_along_normal * direction_x
            player1.velocity[1] -= (1 + restitution) * velocity_along_normal * direction_y
            player2.velocity[0] += (1 + restitution) * velocity_along_normal * direction_x
            player2.velocity[1] += (1 + restitution) * velocity_along_normal * direction_y

    def check_goals(self):
        """
        Checks if the ball has entirely entered either goal area and updates scores accordingly.
        """
        # Check Goal 1 (Left)
        if self.goal1.check_goal(self.ball):
            self.score_team2 += 1
            print(f"Goal for Player 2! Score: Player1 {self.score_team1} - Player2 {self.score_team2}")
            self.reset_game()

        # Check Goal 2 (Right)
        if self.goal2.check_goal(self.ball):
            self.score_team1 += 1
            print(f"Goal for Player 1! Score: Player1 {self.score_team1} - Player2 {self.score_team2}")
            self.reset_game()


    def reset_game(self):
        """
        Resets the ball  to the center of the screen with zero velocity.
        Resets players to their initial positions.  
        """
        self.ball.reset()
        for player in self.players:
            player.reset()
    

    def render_play_area(self):
        """
        Place the pitch image on the screen.
        """
        self.screen.blit(self.pitch_image, (ENV_PARAMS.PLAY_AREA_LEFT, ENV_PARAMS.PLAY_AREA_TOP))


    def render(self):
        """
        Renders all game elements on the screen.
        """
        if not ENV_PARAMS.RENDER:
            return

        self.screen.fill(ENV_PARAMS.BLACK)  # Clear the screen with white
        # Draw Play Area
        self.render_play_area()

        self.goal1.draw_grass(self.screen)
        self.goal2.draw_grass(self.screen)

        # Draw Players and Ball
        for player in self.players:
            player.draw(self.screen)
        self.ball.draw(self.screen)

        # Draw Goals
        self.goal1.draw(self.screen)
        self.goal2.draw(self.screen)

        # Define offsets
        offset_x = 100
        offset_y = 20
        
        # Draw Timer
        minutes = int(self.timer) // 60
        seconds = int(self.timer) % 60
        
        # Render Scores
        score_text = self.font.render(
            f"Team {VIS_PARAMS.TEAM_1_COLOR}: {self.score_team1} | Team {VIS_PARAMS.TEAM_2_COLOR} : {self.score_team2} | Time : {minutes:02}:{seconds:02}", 
            True, 
            ENV_PARAMS.BLACK
        )
        
        # Calculate the position and size of the box
        box_x = offset_x
        box_y = offset_y
        box_width = score_text.get_width() + 20  # Add some padding
        box_height = score_text.get_height() + 10  # Add some padding
        
        # Draw the box
        pygame.draw.rect(self.screen, ENV_PARAMS.WHITE, (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, ENV_PARAMS.BLACK, (box_x, box_y, box_width, box_height), 2)  # Border
        
        # Blit the text inside the box
        self.screen.blit(score_text, (box_x + 10, box_y + 5))


        pygame.display.flip()  # Update the full display surface to the screen

    def run(self):
        """
        Runs the main game loop.
        """
        running = True
        while running:
            if ENV_PARAMS.RENDER:
                self.clock.tick(ENV_PARAMS.FPS)  # Maintain the desired frame rate if rendering

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.simulation_time += self.delta_time
            self.timer = ENV_PARAMS.GAME_DURATION - self.simulation_time

            if self.timer <= 0:
                print(f"Game Over! Final Score: Player1 {self.score_team1} - Player2 {self.score_team2}")
                running = False

            # Get the current state of all keyboard buttons
            keys = pygame.key.get_pressed()

            # Handle players' movement
            for player in self.players:
                player.handle_movement(keys)

            # Update ball's movement
            self.ball.update_position()

            # Handle collisions between players and the ball, and between players themselves
            self.handle_collisions()

            # Check for goals
            self.check_goals()

            # Render everything
            self.render()

        # Clean up Pygame resources
        return self.score_team1, self.score_team2

    