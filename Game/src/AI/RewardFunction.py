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

        self.distance_to_goal =  math.hypot(
            self.game.ball.position[0] - self.goal_centers[2][0],
            self.game.ball.position[1] - self.goal_centers[2][1]
        )


    def calculate_distance_reward(self, player):
        """
        Calculates the small reward based on the ball's distance to the opponent's goal.

        :param player: The Player object for whom the reward is calculated.
        :return: The calculated small reward.
        """
        team_id = player.team_id

        # Determine opponent's goal center
        opponent_team_id = 2 if team_id == 1 else 1
        opponent_goal_center = self.goal_centers[opponent_team_id]

        # Calculate distance between the ball and the opponent's goal
        distance = math.hypot(
            self.game.ball.position[0] - opponent_goal_center[0],
            self.game.ball.position[1] - opponent_goal_center[1]
        )

        # Normalize the distance
        normalized_distance = distance / self.distance_to_goal 

        # Calculate reward (closer to goal yields higher reward)
        reward = (1 - normalized_distance) * self.ai_params.DISTANCE_REWARD_GOAL

        return reward
    
    def calculate_distance_to_ball_reward(self, player):
        """
        Calculates the small reward based on the player's distance to the ball.

        :param player: The Player object for whom the reward is calculated.
        :return: The calculated small reward.
        """
        # Calculate distance between the player and the ball
        distance = math.hypot(
            player.position[0] - self.game.ball.position[0],
            player.position[1] - self.game.ball.position[1]
        )

        # Normalize the distance
        normalized_distance = distance / (self.env_params.PLAY_AREA_WIDTH / 3)

        # Calculate reward (closer to ball yields higher reward)
        reward = (1 - normalized_distance) * self.ai_params.DISTANCE_REWARD_BALL
        return reward
    
    def calculate_rewards(self, goal1, goal2):
        """
        Calculates the rewards for each player based on the current game state.
        """

        rewards = []

        for player in self.game.players:
            total_reward = self.calculate_distance_reward(player)
            total_reward += self.calculate_distance_to_ball_reward(player)

            if goal1 and player.team_id == 1:
                # Calculate the total reward for team 1
                total_reward += self.ai_params.GOAL_REWARD
            elif goal2 and player.team_id == 2:
                # Calculate the total reward for team 2
                total_reward += self.ai_params.GOAL_REWARD
            elif goal1 and player.team_id == 2:
                # Calculate the total reward for team 2
                total_reward -= self.ai_params.GOAL_REWARD
            elif goal2 and player.team_id == 1:
                # Calculate the total reward for team 1
                total_reward -= self.ai_params.GOAL_REWARD
            
            total_reward = total_reward / 100
            rewards.append(total_reward)

        return rewards