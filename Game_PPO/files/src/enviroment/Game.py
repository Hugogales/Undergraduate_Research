import pygame
import json
import math
import random
from params import EnvironmentHyperparameters, VisualHyperparametters, AIHyperparameters
from enviroment.Ball import Ball
from enviroment.Player import Player
from enviroment.Goal import Goal
from functions.Logger import Logger
from AI.StateParser import StateParser
from AI.RewardFunction import RewardFunction 
from AI.Memory import Memory


# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()
AI_PARAMS = AIHyperparameters()
if ENV_PARAMS.RENDER:
    VIS_PARAMS = VisualHyperparametters()


class Game:
    def __init__(self, log_name = None):
        """
        Initializes the Game object by setting up players, ball, goals, and scores.
        """
        self.screen = pygame.display.set_mode((ENV_PARAMS.WIDTH, ENV_PARAMS.HEIGHT))

        self.team_1 = []
        self.team_2 = []
        for i in range(ENV_PARAMS.NUMBER_OF_PLAYERS):
            if ENV_PARAMS.MODE == "play":
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
                    up_key=pygame.K_DOWN,
                    down_key=pygame.K_UP,
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
        self.ball_hits = 0

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

        self.randomize_players = ENV_PARAMS.RANDOMIZE_PLAYERS

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


        self.log_name = log_name
        if log_name is not None:
            self.logger = Logger(self.log_name)
            self.logger.write_parameters(ENV_PARAMS)

        
        # model stuff
        self.state_parser = StateParser(self)
        self.reward_function = RewardFunction(self)


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
            self.ball_hits += 1
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

#            # Calculate relative velocity
            #relative_velocity_x = player2.velocity[0] - player1.velocity[0]
            #relative_velocity_y = player2.velocity[1] - player1.velocity[1]

            ## Calculate the velocity along the direction of collision
            #velocity_along_normal = relative_velocity_x * direction_x + relative_velocity_y * direction_y

            #if velocity_along_normal > 0:
                #return  # They are moving away from each other

            ## Define restitution (elasticity) less than ball's
            #restitution = 0.7  # Players push each other less strongly

            ## Update players' velocities based on collision
            #player1.velocity[0] -= (1 + restitution) * velocity_along_normal * direction_x
            #player1.velocity[1] -= (1 + restitution) * velocity_along_normal * direction_y
            #player2.velocity[0] += (1 + restitution) * velocity_along_normal * direction_x
            #player2.velocity[1] += (1 + restitution) * velocity_along_normal * direction_y

    def check_goals(self):
        """
        Checks if the ball has entirely entered either goal area and updates scores accordingly.
        """
        # Check Goal 1 (Left)
        if self.goal1.check_goal(self.ball):
            self.score_team2 += 1
            self.reset_game()
            return (False, True)

        # Check Goal 2 (Right)
        if self.goal2.check_goal(self.ball):
            self.score_team1 += 1
            self.reset_game()
            return (True, False)
        
        return (False, False)

    def reset_game(self):
        """
        Resets the ball  to the center of the screen with zero velocity.
        Resets players to their initial positions.  
        """
        if self.randomize_players:
            # Reset the ball to a random position
            self.ball.position = [
                random.randint(ENV_PARAMS.PLAY_AREA_LEFT + self.ball.radius, ENV_PARAMS.PLAY_AREA_RIGHT - self.ball.radius),
                random.randint(self.ball.radius, ENV_PARAMS.HEIGHT - self.ball.radius)
            ]
            self.ball.velocity = [0, 0]

            # Reset players to random positions, ensuring no overlap
            for player in self.players:
                while True:
                    player.position = [
                        random.randint(ENV_PARAMS.PLAY_AREA_LEFT + player.radius, ENV_PARAMS.PLAY_AREA_RIGHT - player.radius),
                        random.randint(player.radius, ENV_PARAMS.HEIGHT - player.radius)
                    ]
                    # Check for overlap with other players
                    overlap = False
                    for other_player in self.players:
                        if other_player != player:
                            distance = math.hypot(player.position[0] - other_player.position[0], player.position[1] - other_player.position[1])
                            min_distance = player.radius + other_player.radius
                            if distance < min_distance:
                                overlap = True
                                break
                    # Check for overlap with the ball
                    distance_to_ball = math.hypot(player.position[0] - self.ball.position[0], player.position[1] - self.ball.position[1])
                    min_distance_to_ball = player.radius + self.ball.radius
                    if distance_to_ball < min_distance_to_ball:
                        overlap = True

                    if not overlap:
                        break

            # Reset player velocities
            for player in self.players:
                player.velocity = [0, 0]
        else:
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

        self.screen.fill(VIS_PARAMS.BLACK)  # Clear the screen with white
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
            VIS_PARAMS.BLACK
        )
        
        # Calculate the position and size of the box
        box_x = offset_x
        box_y = offset_y
        box_width = score_text.get_width() + 20  # Add some padding
        box_height = score_text.get_height() + 10  # Add some padding
        
        # Draw the box
        pygame.draw.rect(self.screen, VIS_PARAMS.WHITE, (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, VIS_PARAMS.BLACK, (box_x, box_y, box_width, box_height), 2)  # Border
        
        # Blit the text inside the box
        self.screen.blit(score_text, (box_x + 10, box_y + 5))


        pygame.display.flip()  # Update the full display surface to the screen


#  ----- GAME LOOPS FOR PLAYING, REPLAYING, AND RUN -------

    def run_play(self):
        """
        Runs the main game loop.
        Intende for human players.
        """
        running = True
        while running:
            if ENV_PARAMS.RENDER and ENV_PARAMS.CAP_FPS:
                self.clock.tick(ENV_PARAMS.FPS)  # Maintain the desired frame rate if rendering

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.simulation_time += self.delta_time
            self.timer = ENV_PARAMS.GAME_DURATION - self.simulation_time


            if self.timer <= 0:
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
            goal1, goal2 = self.check_goals()

            if self.log_name is not None:
                self.logger.log_state(self.players, self.ball, self.timer)

            # Render everything
            self.render()
        
        if self.log_name is not None:
            self.logger.close()

        # Clean up Pygame resources
        return self.score_team1, self.score_team2

    def replay(self, states):
        """
        Replays the game from the logged data.
        """
        running = True

        self.clock = pygame.time.Clock()

        current_state_index = 0

        while running and current_state_index < len(states):
            if ENV_PARAMS.RENDER and ENV_PARAMS.CAP_FPS:
                self.clock.tick(ENV_PARAMS.FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the current state
            state = states[current_state_index]
            self.timer = state['time']

            # Update Players
            for player, player_state in zip(self.players, state['players']):

                player.position = player_state['position']
                player.velocity = player_state['velocity']

                player.direction = math.degrees(math.atan2(
                    player.velocity[1],
                    player.velocity[0]
                ))
                player.leg_angle = math.degrees(math.atan2(
                    player.velocity[1],
                    player.velocity[0]
                ))
                if player.velocity[0] == 0 and player.velocity[1] == 0:
                    player.is_moving = False
                else:
                    player.is_moving = True
            
            # Update Ball
            self.ball.position = state['ball']['position']
            self.ball.velocity = state['ball']['velocity']

            # Render everything
            self.render()

            current_state_index += 1

    def run(self, model):
        """
        Runs the main game loop.
        """
        running = True
        total_rewards = []
        memories = []
        total_ball_distance = 0
        self.ball_hits = 0

        team_playing = [1,2]
        if AI_PARAMS.current_stage == 1: # only one team play
            team_playing = [random.randint(1,2)]
            self.randomize_players = False
            self.reset_game()
        elif AI_PARAMS.current_stage == 2: # one team play radomn locations
            team_playing = [random.randint(1,2)]
            self.randomize_players = False
            self.reset_game()
            self.randomize_players = True
        elif AI_PARAMS.current_stage == 3: # both teams play
            team_playing = [1,2]
            self.randomize_players = False
            self.reset_game()
        elif AI_PARAMS.current_stage == 4: # both teams play radomn locations
            team_playing = [1,2]
            self.randomize_players = False
            self.reset_game()
            self.randomize_players = True

        for player in self.players:
            memories.append(Memory())
            total_rewards.append(0)

        while running:
            if ENV_PARAMS.RENDER and ENV_PARAMS.CAP_FPS:
                self.clock.tick(ENV_PARAMS.FPS)  # Maintain the desired frame rate if rendering

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.simulation_time += self.delta_time
            self.timer = ENV_PARAMS.GAME_DURATION - self.simulation_time

            if self.timer <= 0:
                running = False

            states = self.state_parser.parse_state()


            # Handle players' movement
            for i, (player, state) in enumerate(zip(self.players, states)):
                if player.team_id not in team_playing:
                    continue
                action, log_prob, state_value = model.select_action(state)
                memories[i].states.append(state)
                memories[i].actions.append(action)
                memories[i].log_probs.append(log_prob)
                memories[i].state_values.append(state_value)
                move_input = model._action_to_input(action)
                player.move(move_input)

            ball_position = self.ball.position
            # Update ball's movement
            self.ball.update_position()

            # Handle collisions between players and the ball, and between players themselves
            self.handle_collisions()

            # Check for goals
            goal1, goal2 = self.check_goals()

            if not goal1 and not goal2:
                total_ball_distance += math.hypot(ball_position[0] - self.ball.position[0], ball_position[1] - self.ball.position[1])

            rewards = self.reward_function.calculate_rewards(goal1, goal2)
            for i, state in enumerate(states):
                if self.players[i].team_id not in team_playing:
                    continue
                reward = rewards[i]
                done = not running or goal1 or goal2
                memories[i].rewards.append(reward)
                memories[i].dones.append(done)
                total_rewards[i] += reward

            if self.log_name is not None:
                self.logger.log_state(self.players, self.ball, self.timer)

            # Render everything
            self.render()
        
        if self.log_name is not None:
            self.logger.close()

        total_rewards = [abs(reward) for reward in total_rewards]
        avg_reward = 100 * sum(total_rewards) / (ENV_PARAMS.NUMBER_OF_PLAYERS * 2 * ENV_PARAMS.FPS * ENV_PARAMS.GAME_DURATION)

        filtered_memories = []
        for i in range(len(total_rewards)):
            if self.players[i].team_id in team_playing:
                filtered_memories.append(memories[i])

        return self.score_team1, self.score_team2, avg_reward, total_ball_distance, self.ball_hits, memories