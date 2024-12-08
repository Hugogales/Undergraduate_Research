import math
from params import EnvironmentHyperparameters

class StateParser:
    def __init__(self, game):
        """
        Initializes the StateParser with a reference to the game instance.

        :param game: An instance of the Game class.
        """
        self.game = game
        self.env_params = EnvironmentHyperparameters()

        # Get field dimensions
        self.field_width = self.env_params.PLAY_AREA_WIDTH
        self.field_height = self.env_params.PLAY_AREA_HEIGHT

        # Precompute goal center points
        self.goal_centers = {
            1: [
                (self.game.goal1.position[0] + self.game.goal1.width / 2) / self.field_width,
                (self.game.goal1.position[1] + self.game.goal1.height / 2) / self.field_height
            ],
            2: [
                (self.game.goal2.position[0] + self.game.goal2.width / 2) / self.field_width,
                (self.game.goal2.position[1] + self.game.goal2.height / 2) / self.field_height
            ]
        }

    def get_state(self, player):
        """
        Generates the state vector for the given player.

        The state vector includes:
        - Vectors to players in own team (relative positions, normalized)
        - Vectors to players in opposing team (relative positions, normalized)
        - Vector to ball (relative position, normalized)
        - Velocity of ball (normalized)
        - Vectors to the two goals (opponent's goal first, then own goal, normalized)
        - Raycasts (north, south, east, west distances, normalized)

        Positions are mirrored/flipped for the opposing team to ensure consistent outputs.

        :param player: The Player object for whom the state is generated.
        :return: A list representing the state vector.
        """
        field_width = self.field_width
        field_height = self.field_height

        # Team ID
        team_id = player.team_id

        # Get own goal area and normalize
        own_goal = self.game.goal1 if team_id == 1 else self.game.goal2
        own_goal_x, own_goal_y = own_goal.position
        own_goal_width, own_goal_height = own_goal.width, own_goal.height
        own_goal_x /= field_width
        own_goal_y /= field_height
        own_goal_width /= field_width
        own_goal_height /= field_height

        # Get player's own position and normalize
        px, py = player.position
        px /= field_width
        py /= field_height

        # Determine if we need to flip positions for consistency
        flip = (team_id == 2)

        # Function to flip positions if necessary
        def flip_position(x, y):
            return 1.0 - x, y  # Flip horizontally in normalized coordinates

        # Flip player's position if necessary
        if flip:
            px, py = flip_position(px, py)

        # Vectors to teammates (relative positions, normalized)
        own_team_vectors = []
        for other_player in self.game.players:
            if other_player == player:
                continue  # Skip the player itself
            if other_player.team_id == team_id:
                ox, oy = other_player.position
                ox /= field_width
                oy /= field_height
                if flip:
                    ox, oy = flip_position(ox, oy)
                    rel_y = - oy + py
                else:
                    rel_y = oy - py

                # Compute relative position
                rel_x = ox - px
                own_team_vectors.extend([rel_x, rel_y])

        # Vectors to opponents (relative positions, normalized)
        opponent_team_vectors = []
        for other_player in self.game.players:
            if other_player.team_id != team_id:
                ox, oy = other_player.position
                ox /= field_width
                oy /= field_height
                if flip:
                    ox, oy = flip_position(ox, oy)
                    rel_y =  oy - py
                else:
                    rel_y = - oy + py

                rel_x = ox - px
                opponent_team_vectors.extend([rel_x, rel_y])

        # Vector to the ball (relative position, normalized)
        ball_x, ball_y = self.game.ball.position
        ball_x /= field_width
        ball_y /= field_height
        if flip:
            ball_x, ball_y = flip_position(ball_x, ball_y)
            rel_ball_y = - ball_y + py
        else :
            rel_ball_y = ball_y - py
        rel_ball_x = ball_x - px
        ball_vector = [rel_ball_x, rel_ball_y]

        # Velocity of the ball (normalized)
        max_ball_speed = self.env_params.BALL_MAX_SPEED  # You need to define this parameter
        ball_vx, ball_vy = self.game.ball.velocity
        ball_vx /= max_ball_speed
        ball_vy /= max_ball_speed
        if flip:
            ball_vx = -ball_vx  # Flip horizontal velocity
            ball_vy = -ball_vy  # Flip vertical velocity
        ball_velocity = [ball_vx, ball_vy]

        # Vectors to the two goals (opponent's goal first, then own goal, normalized)
        opponent_team_id = 2 if team_id == 1 else 1
        opponent_goal_center = self.goal_centers[opponent_team_id]
        own_goal_center = self.goal_centers[team_id]

        if flip:
            opponent_goal_center = flip_position(*opponent_goal_center)
            own_goal_center = flip_position(*own_goal_center)

        # Compute relative vectors to goals
        rel_opponent_goal_x = opponent_goal_center[0] - px
        rel_own_goal_x = own_goal_center[0] - px
        if flip:
            rel_opponent_goal_y = - opponent_goal_center[1] + py
            rel_own_goal_y = - own_goal_center[1] + py
        else:
            rel_opponent_goal_y = opponent_goal_center[1] - py
            rel_own_goal_y = own_goal_center[1] - py

        goal_vectors = [rel_opponent_goal_x, rel_opponent_goal_y, rel_own_goal_x, rel_own_goal_y]

        # Raycasts in north, south, east, west directions
        # Implementing raycasting logic as per your instructions

        # Flip player's y position if necessary

        if flip:
            px = 1.0 - px  # Since py is normalized

        # Determine if the player is in the goal area
        in_own_goal_area = False
        in_opponent_goal_area = False
        in_middle_strip = False

        if (own_goal_x <= px <= own_goal_x + own_goal_width and
            own_goal_y <= py <= own_goal_y + own_goal_height):
            in_own_goal_area = True

        # Get opponent goal area and normalize
        opponent_goal = self.game.goal2 if team_id == 1 else self.game.goal1
        opponent_goal_x, opponent_goal_y = opponent_goal.position
        opponent_goal_width, opponent_goal_height = opponent_goal.width, opponent_goal.height
        opponent_goal_x /= field_width
        opponent_goal_y /= field_height
        opponent_goal_width /= field_width
        opponent_goal_height /= field_height

        if (opponent_goal_x <= px <= opponent_goal_x + opponent_goal_width and
            opponent_goal_y <= py <= opponent_goal_y + opponent_goal_height):
            in_opponent_goal_area = True

        if not in_own_goal_area and not in_opponent_goal_area and own_goal_y <= py <= own_goal_y + own_goal_height:
            in_middle_strip = True

        # Check if player is in the middle strip

        # Initialize raycast distances
        north_distance = south_distance = east_distance = west_distance = 0.0

        if in_own_goal_area or in_opponent_goal_area:
            # Player is in the goal area
            # Calculate distances to top and bottom of the goal and distance from left goal to right goal
            # For the goal the player is in
            goal_x = own_goal_x if in_own_goal_area else opponent_goal_x
            goal_y = own_goal_y if in_own_goal_area else opponent_goal_y
            goal_height = own_goal_height if in_own_goal_area else opponent_goal_height

            # North (up): distance to top of goal
            north_distance = (goal_y + goal_height) - py
            # South (down): distance to bottom of goal
            south_distance = py - goal_y
            # West (left): distance to left side of goal
            east_distance = 1 - px + 2 *own_goal_width
            # East (right): distance to right side of goal
            west_distance = px 

        else:
            # Player is neither in goal area nor in middle strip
            # Use play area dimensions
            north_distance = 1.0 - py
            south_distance = py
            if in_middle_strip:
                east_distance = 1.0 - px + 2 * own_goal_width
                west_distance = px 
            else:
                east_distance = 1.0 - px  + own_goal_width
                west_distance = px  - own_goal_width

        # Raycasts are already normalized due to normalization of positions
        if flip:
            north_distance, south_distance = south_distance, north_distance
            east_distance, west_distance = west_distance, east_distance


        raycasts = [north_distance, south_distance, east_distance, west_distance]
        # print raycast with round to 2f

        # Assemble the state vector
        state_vector = (
            own_team_vectors +
            opponent_team_vectors +
            ball_vector +
            ball_velocity +
            goal_vectors +
            raycasts
        )

        return state_vector

    def parse_state(self):
        input_vector = []
        for player in self.game.players:
            state = self.get_state(player)
            input_vector.append(state)
        return input_vector
