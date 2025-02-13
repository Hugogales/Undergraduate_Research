from params import EnvironmentHyperparameters

class StateParser:
    def __init__(self):
        ENV_PARAMS = EnvironmentHyperparameters()
        self.height = ENV_PARAMS.HEIGHT
        self.width = ENV_PARAMS.WIDTH
        self.scale = ENV_PARAMS.HEIGHT  # Normalize positions by height
        self.max_player_speed = ENV_PARAMS.PLAYER_SPEED  # Maximum speed for players
        self.max_ball_speed = ENV_PARAMS.BALL_MAX_SPEED  # Assuming this exists in your parameters

    def parse_state(self, players, ball):
        """
        Parses the game state and returns standardized inputs for each player.

        :param players: List of all Player objects.
        :param ball: Ball object.
        :return: A dictionary with team IDs as keys and lists of input vectors as values.
        """
        inputs = []
        for player in players:
            input_vector = self.generate_input_vector(players, ball, player)
            inputs.append((input_vector))
        return inputs

    def generate_input_vector(self, players, ball, current_player):
        """
        Generates the input vector for a single player.

        :param players: List of all Player objects.
        :param ball: Ball object.
        :param current_player: The Player object for whom the input is generated.
        :return: A list representing the input vector.
        """
        input_vector = []

        # Center the positions and normalize
        def normalize_position(x, y):
            x = (x - self.width / 2) / self.scale
            y = (y - self.height / 2) / self.scale
            return x, y

        # Normalize velocities
        def normalize_velocity(vx, vy, max_speed):
            vx /= max_speed
            vy /= max_speed
            return vx, vy

        # Determine if we need to reverse coordinates
        reverse = current_player.team_id == 2

        # Function to reverse coordinates if necessary
        def maybe_reverse(x, y):
            if reverse:
                return -x, -y  # Reverse both x and y if needed
            return x, y

        # Function to reverse velocities if necessary
        def maybe_reverse_velocity(vx, vy):
            if reverse:
                return -vx, -vy  # Reverse both vx and vy if needed
            return vx, vy

        # Add current player's position first
        x, y = normalize_position(*current_player.position)
        x, y = maybe_reverse(x, y)
        input_vector.extend([x, y])

        # Add current player's velocity
        vx, vy = normalize_velocity(*current_player.velocity, self.max_player_speed)
        vx, vy = maybe_reverse_velocity(vx, vy)
        input_vector.extend([vx, vy])

        # Add other players' positions and velocities
        for player in players:
            if player != current_player and player.team_id == current_player.team_id: 
                x, y = normalize_position(*player.position)
                x, y = maybe_reverse(x, y)
                input_vector.extend([x, y])

                vx, vy = normalize_velocity(*player.velocity, self.max_player_speed)
                vx, vy = maybe_reverse_velocity(vx, vy)
                input_vector.extend([vx, vy])
        
        for player in players:
            if player != current_player and player.team_id != current_player.team_id: 
                x, y = normalize_position(*player.position)
                x, y = maybe_reverse(x, y)
                input_vector.extend([x, y])

                vx, vy = normalize_velocity(*player.velocity, self.max_player_speed)
                vx, vy = maybe_reverse_velocity(vx, vy)
                input_vector.extend([vx, vy])

        # Add ball position
        x, y = normalize_position(*ball.position)
        x, y = maybe_reverse(x, y)
        input_vector.extend([x, y])

        # Add ball velocity
        vx, vy = normalize_velocity(*ball.velocity, self.max_ball_speed)
        vx, vy = maybe_reverse_velocity(vx, vy)
        input_vector.extend([vx, vy])

        return input_vector
