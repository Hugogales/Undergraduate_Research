import pygame
import sys
import math
from src.enviroment import Ball, Goal
from src.params import EnvironmentHyperparameters
from enviroment import Player

# Instantiate Pygame and initialize all imported Pygame modules
pygame.init()

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()

# Instantiate CLOCK as a global constant
CLOCK = pygame.time.Clock()

# =======================
# Game Class
# =======================

class Game:
    def __init__(self):
        """
        Initializes the Game object by setting up players, ball, goals, and scores.
        """
        # Initialize Player 1 (Blue, W, A, S, D)
        self.player1 = Player(
            position=ENV_PARAMS.team_1_positions[0],
            color=ENV_PARAMS.BLUE,
            up_key=pygame.K_w,
            down_key=pygame.K_s,
            left_key=pygame.K_a,
            right_key=pygame.K_d
        )

        # Initialize Player 2 (Green, Arrow Keys)
        self.player2 = Player(
            position=ENV_PARAMS.team_2_positions[0],
            color=ENV_PARAMS.GREEN,
            up_key=pygame.K_UP,
            down_key=pygame.K_DOWN,
            left_key=pygame.K_LEFT,
            right_key=pygame.K_RIGHT
        )

        # Initialize Ball
        self.ball = Goal(
            position=[ENV_PARAMS.WIDTH // 2, ENV_PARAMS.HEIGHT // 2],
            color=ENV_PARAMS.BLACK
        )

        # Initialize Goals (only central part with inner boundaries)
        goal_height = ENV_PARAMS.GOAL_HEIGHT
        self.goal1 = Ball(
            position=[0, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Left side, centered vertically
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
            color=ENV_PARAMS.GOAL_COLOR
        )
        self.goal2 = Ball(
            position=[ENV_PARAMS.WIDTH - ENV_PARAMS.GOAL_WIDTH, (ENV_PARAMS.HEIGHT / 2) - (goal_height / 2)],  # Right side, centered vertically
            width=ENV_PARAMS.GOAL_WIDTH,
            height=goal_height,
            color=ENV_PARAMS.GOAL_COLOR
        )

        # Initialize Scores
        self.score_player1 = 0
        self.score_player2 = 0

        self.screen = pygame.display.set_mode((ENV_PARAMS.WIDTH, ENV_PARAMS.HEIGHT))
        pygame.display.set_caption(ENV_PARAMS.TITLE)

        self.font = pygame.font.SysFont('Arial', 36)

    def handle_collisions(self):
        """
        Handles collision between players and the ball, and between players themselves.
        """
        # Check collision between Player 1 and Ball
        self.check_collision(self.player1, self.ball)

        # Check collision between Player 2 and Ball
        self.check_collision(self.player2, self.ball)

        # Check collision between Player 1 and Player 2
        self.check_player_collision(self.player1, self.player2)

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
            restitution = 0.5  # Players push each other less strongly

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
            self.score_player2 += 1
            print(f"Goal for Player 2! Score: Player1 {self.score_player1} - Player2 {self.score_player2}")
            self.reset_game()

        # Check Goal 2 (Right)
        if self.goal2.check_goal(self.ball):
            self.score_player1 += 1
            print(f"Goal for Player 1! Score: Player1 {self.score_player1} - Player2 {self.score_player2}")
            self.reset_game()

    def reset_game(self):
        """
        Resets the ball  to the center of the screen with zero velocity.
        Resets players to their initial positions.  
        """
        self.ball.position = [ENV_PARAMS.WIDTH // 2, ENV_PARAMS.HEIGHT // 2]
        self.ball.velocity = [0, 0]
        self.player1.position = ENV_PARAMS.team_1_positions[0]
        self.player2.position = ENV_PARAMS.team_2_positions[0]
    

    def render_play_area(self):
        """
        Draws the play area boundaries on the screen.
        """
        # Highlight play area with a yellow rectangle
        pygame.draw.rect(self.screen, ENV_PARAMS.YELLOW, 
                         (ENV_PARAMS.PLAY_AREA_LEFT, ENV_PARAMS.PLAY_AREA_TOP,
                          ENV_PARAMS.PLAY_AREA_RIGHT - ENV_PARAMS.PLAY_AREA_LEFT,
                          ENV_PARAMS.PLAY_AREA_BOTTOM - ENV_PARAMS.PLAY_AREA_TOP), 2)

    def render(self):
        """
        Renders all game elements on the screen.
        """
        if not ENV_PARAMS.RENDER:
            return
        self.screen.fill(ENV_PARAMS.WHITE)  # Clear the screen with white

        # Draw Play Area
        self.render_play_area()

        # Draw Goals
        self.goal1.draw(self.screen)
        self.goal2.draw(self.screen)

        # Draw Players and Ball
        self.player1.draw(self.screen)
        self.player2.draw(self.screen)
        self.ball.draw(self.screen)

        # Draw Scores
        score_text = self.font.render(
            f"Player1: {self.score_player1}   Player2: {self.score_player2}", 
            True, 
            ENV_PARAMS.BLACK
        )
        self.screen.blit(score_text, (ENV_PARAMS.WIDTH//2 - score_text.get_width()//2, 20))

        pygame.display.flip()  # Update the full display surface to the screen

    def run(self):
        """
        Runs the main game loop.
        """
        running = True
        while running:
            CLOCK.tick(ENV_PARAMS.FPS)  # Maintain the desired frame rate

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the current state of all keyboard buttons
            keys = pygame.key.get_pressed()


            # Handle players' movement
            self.player1.handle_movement(keys)
            self.player2.handle_movement(keys)


            # Update ball's movement
            self.ball.update_position()

            # Handle collisions between players and the ball, and between players themselves
            self.handle_collisions()

            # Check for goals
            self.check_goals()

            # Render everything
            self.render()

        # Clean up Pygame resources
        pygame.quit()
        sys.exit()

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    game = Game()
    game.run()
