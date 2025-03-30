import math
from params import AIHyperparameters, EnvironmentHyperparameters

class RewardFunction:
    def __init__(self, game):
        """
        Initializes the RewardFunction with a reference to the game instance.

        :param game: An instance of the Game class.
        """
        self.game = game
        self.ai_params = AIHyperparameters()
        self.env_params = EnvironmentHyperparameters()

        # Store cumulative rewards for each player
        self.rewards = {player: 0 for player in self.game.players}

        # Precompute goal center points
        self.goal_centers = {
            1: [
                self.game.goal1.position[0] + self.game.goal1.width / 2,
                self.game.goal1.position[1] + self.game.goal1.height / 2
            ],
            2: [
                self.game.goal2.position[0] + self.game.goal2.width / 2,
                self.game.goal2.position[1] + self.game.goal2.height / 2
            ]
        }

        

    def calculate_player_to_ball_velocity_reward(self, player):
        """
        Calculates the reward based on how much the player's movement is directed towards the ball.

        :param player: The Player object for whom the reward is calculated.
        :return: The calculated reward.
        """
        # Vector from player to ball
        vector_to_ball = [
            self.game.ball.position[0] - player.position[0],
            self.game.ball.position[1] - player.position[1]
        ]

        # Normalize vector_to_ball
        distance_to_ball = math.hypot(vector_to_ball[0], vector_to_ball[1])
        if distance_to_ball == 0:
            return 0  # Avoid division by zero; no reward

        unit_vector_to_ball = [vector_to_ball[0] / distance_to_ball, vector_to_ball[1] / distance_to_ball]

        # Player's velocity vector
        player_velocity = player.velocity  # Assuming player.velocity is a list or tuple [vx, vy]

        # Compute dot product
        dot_product = player_velocity[0] * unit_vector_to_ball[0] + player_velocity[1] * unit_vector_to_ball[1]

        # Set the reward proportional to the dot product
        reward = dot_product * self.ai_params.PLAYER_TO_BALL_REWARD_COEFF


        return reward

    def calculate_ball_to_goal_velocity_reward(self, player):
        """
        Calculates the reward based on how much the ball's movement is directed towards the opponent's goal.

        :param player: The Player object for whom the reward is calculated.
        :return: The calculated reward.
        """
        team_id = player.team_id

        # Determine opponent's goal center
        opponent_team_id = 2 if team_id == 1 else 1
        opponent_goal_center = self.goal_centers[opponent_team_id]

        # Vector from ball to opponent's goal
        vector_to_goal = [
            opponent_goal_center[0] - self.game.ball.position[0],
            opponent_goal_center[1] - self.game.ball.position[1]
        ]

        # Normalize vector_to_goal
        distance_to_goal = math.hypot(vector_to_goal[0], vector_to_goal[1])
        if distance_to_goal == 0:
            return 0  # Ball is at the goal; no reward needed

        unit_vector_to_goal = [vector_to_goal[0] / distance_to_goal, vector_to_goal[1] / distance_to_goal]

        # Ball's velocity vector
        ball_velocity = self.game.ball.velocity  # Assuming ball.velocity is a list or tuple [vx, vy]

        # Compute dot product
        dot_product = ball_velocity[0] * unit_vector_to_goal[0] + ball_velocity[1] * unit_vector_to_goal[1]

        # Set the reward proportional to the dot product
        reward = dot_product * self.ai_params.BALL_TO_GOAL_REWARD_COEFF


        return reward


    def calculate_distance_to_players_reward(self, player):
        """
        Calculates the reward based on the distance to the other players.

        :param player: The Player object for whom the reward is calculated.
        :return: The calculated reward.
        """
        distance_reward = 0

        for other_player in self.game.players: # within the same team
            if other_player != player and other_player.team_id == player.team_id:
                distance = math.hypot(player.position[0] - other_player.position[0], player.position[1] - other_player.position[1])
                distance_reward += distance

        distance_reward /= len(self.game.players) - 1

        return self.ai_params.DISTANCE_REWARD_COEFF * min(distance_reward, self.ai_params.DISTANCE_REWARD_CAP)


    def calculate_rewards(self, goal1, goal2):
        """
        Calculates the rewards for each player based on the current game state.
        """

        rewards = []

        for player in self.game.players:
            total_reward = 0

            # Reward for player moving towards the ball
            total_reward += self.calculate_player_to_ball_velocity_reward(player)

            # Reward for ball moving towards the opponent's goal
            total_reward += self.calculate_ball_to_goal_velocity_reward(player)

            # Reward for player being faraway to teammates
            total_reward += self.calculate_distance_to_players_reward(player)

            # Goal rewards
            if goal1 and player.team_id == 1:
                # Reward team 1 for scoring
                total_reward += self.ai_params.GOAL_REWARD 
            if goal2 and player.team_id == 2:
                # Reward team 2 for scoring
                total_reward += self.ai_params.GOAL_REWARD
            if goal1 and player.team_id == 2:
                # Penalize team 2 for opponent scoring
                total_reward -= self.ai_params.GOAL_REWARD
            if goal2 and player.team_id == 1:
                # Penalize team 1 for opponent scoring
                total_reward -= self.ai_params.GOAL_REWARD

            rewards.append(total_reward)

        return rewards
