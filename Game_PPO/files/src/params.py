import random 

class AIHyperparameters:
    _instance = None

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        ## rewards
        self._env = EnvironmentHyperparameters()


        self.PLAYER_TO_BALL_REWARD_COEFF = 0.002
        self.BALL_TO_GOAL_REWARD_COEFF = 0.2# expected max is 1 when bal
        self.GOAL_REWARD = 250
        self.positive_reward_coef = 1
    
        self.STATE_SIZE = 12 + 2 * (2* self._env.NUMBER_OF_PLAYERS - 1)
        self.ACTION_SIZE = 9

        self.gamma = 0.985 # discount rate
        self.learning_rate = 1.5e-4
        self.batch_size = 64
        self.c_entropy = 0.025
        self.temperature = 1
        self.max_grad_norm = 2.5
        self.lam = 0.965
        self.c_value = 0.5

        self.epsilon_clip = 0.12
        self.K_epochs = 7

        self.current_stage = 1
        self.stage1_steps = 6250 # one team plays 
        self.stage2_steps = 15000 # one team plays radomn locations
        self.stage3_steps = 50000 # both teams play 
        self.stage4_steps = 10000 # both teams play radomn locations

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AIHyperparameters, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

class EnvironmentHyperparameters:
    _instance = None

    def __init__(self):
        if self._initialized:
            return 
        self._initialized = True

        # Options: train, test, play, replay
        self.MODE = "train" # train or test

        if self.MODE == "play":
            self.NUMBER_OF_GAMES = 1
            self.FPS = 16
            self.NUMBER_OF_PLAYERS = 1
            self.GAME_DURATION = 90 #5* 60  # 5 minutes
            self.RENDER = True
            self.CAP_FPS = True

        elif self.MODE == "replay":
            self.FILE_NAME = "PPO_v3_game_1000"

            #params set automatically
            self.NUMBER_OF_GAMES = 0
            self.FPS = 0
            self.NUMBER_OF_PLAYERS = 0
            self.GAME_DURATION = 0
            self.RENDER = True
            self.CAP_FPS = True

        else: # train or test
            self.MODEL_NAME = "PPO_v6_1"
            self.Load_model = None
            self.log_name = "PPO_v6_1_game"
            self.log_interval = 250
            self.NUMBER_OF_GAMES = 1
            self.FPS = 16
            self.NUMBER_OF_PLAYERS = 2
            self.GAME_DURATION = 60 
            self.EPOCHS = 50000
            self.RENDER = False
            self.CAP_FPS = False

        self.RANDOMIZE_PLAYERS = True

        # Screen dimensions
        self.WIDTH = 1300  # Increased width for a wider field
        self.HEIGHT = 700

        # Player properties
        self.PLAYER_RADIUS = 20
        self.PLAYER_SPEED = 19
        self.PLAYER_HEIGHT = self.PLAYER_RADIUS * 2  # Diameter
        self.PLAYER_POWER = 2 # player vs player collision power 

        # Ball properties
        self.BALL_RADIUS = 12 
        self.BALL_FRICTION = 0.95
        self.BALL_MAX_SPEED = 13
        self.BALL_HEIGHT = self.BALL_RADIUS * 2  # Diameter
        self.BALL_POWER =  35  # player vs ball collision power

        # Goal properties
        self.GOAL_WIDTH = 0.05 * self.WIDTH  # 5% of the screen width (65 pixels)
        self.GOAL_HEIGHT = 0.26 * self.HEIGHT  # 25% of the screen height (175 pixels)

        # Play area vertical boundaries
        self.PLAY_AREA_TOP = 0
        self.PLAY_AREA_BOTTOM = self.HEIGHT

        # Play area horizontal boundaries
        self.PLAY_AREA_LEFT = self.GOAL_WIDTH  # 130
        self.PLAY_AREA_RIGHT = self.WIDTH - self.GOAL_WIDTH  # 1170

        self.PLAY_AREA_WIDTH = self.PLAY_AREA_RIGHT - self.PLAY_AREA_LEFT
        self.PLAY_AREA_HEIGHT = self.HEIGHT

        self.calculate_positions()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EnvironmentHyperparameters, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def calculate_positions(self):
        """
        Calculates and assigns player positions for both teams based on the number of players.
        Supports team sizes from 0 to 10 players.
        """
        # Initialize position lists for both teams
        self.team_1_positions = []
        self.team_2_positions = []
        
        # Calculate the number of players for each column
        back_column_players = min(3, self.NUMBER_OF_PLAYERS)
        middle_column_players = min(4, max(0, self.NUMBER_OF_PLAYERS - back_column_players))
        front_column_players = max(0, self.NUMBER_OF_PLAYERS - back_column_players - middle_column_players)

        # Calculate the vertical spacing between players
        vertical_spacing = self.HEIGHT / 5

        # Function to calculate positions for a column
        def calculate_column_positions(start_x, num_players, offset):
            positions = []
            for i in range(num_players):
                y_position = offset + (i + 1) * vertical_spacing
                positions.append([start_x, y_position])
            return positions

        # Calculate the offset to center the players vertically
        offset_1 = self.HEIGHT / 2 - (back_column_players + 1) * vertical_spacing / 2
        offset_2 = self.HEIGHT / 2 - (middle_column_players + 1) * vertical_spacing / 2
        offset_3 = self.HEIGHT / 2 - (front_column_players + 1) * vertical_spacing / 2

        # Calculate positions for each column
        back_column_x = self.PLAY_AREA_WIDTH / 8 + self.PLAY_AREA_LEFT
        middle_column_x = self. PLAY_AREA_WIDTH / 4  + self.PLAY_AREA_LEFT
        front_column_x = self. PLAY_AREA_WIDTH * 3 / 8 + self.PLAY_AREA_LEFT

        self.team_1_positions.extend(calculate_column_positions(back_column_x, back_column_players, offset_1))
        self.team_1_positions.extend(calculate_column_positions(middle_column_x, middle_column_players, offset_2))
        self.team_1_positions.extend(calculate_column_positions(front_column_x, front_column_players, offset_3))

        
        # Mirror positions for Team 2 across the vertical center line of the pitch
        self.team_2_positions = [
            [self.PLAY_AREA_LEFT + self.PLAY_AREA_WIDTH - (pos[0] - self.PLAY_AREA_LEFT), pos[1]]
            for pos in self.team_1_positions
        ]
    
class VisualHyperparametters:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VisualHyperparametters, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.update()

    def update(self):
        ENV = EnvironmentHyperparameters()
        self.TITLE = "2D Soccer Game"

        # options : blue, red, green, White
        self.TEAM_1_COLOR = "Blue"
        self.TEAM_2_COLOR = "Green"

        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)     # Player 1 color
        self.GREEN = (83, 160, 23)    # Player 2 color
        self.BLACK = (0, 0, 0)      # Ball color
        self.RED = (255, 0, 0)      # Goal color
        self.YELLOW = (255, 255, 0) # Play area boundary color


        self.TEAM1_SPRITES = []
        self.TEAM2_SPRITES = []
        self.TEAM1_ARMS = []
        self.TEAM1_LEGS = []
        self.TEAM2_ARMS = []
        self.TEAM2_LEGS = []

        for i in range(ENV.NUMBER_OF_PLAYERS):
            num1 = random.randint(1, 5) # number from 1 to 4 including 4
            self.TEAM1_SPRITES.append(f"files/Images/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} ({num1}).png")
            if num1 == 5:
                self.TEAM1_ARMS.append(f"files/Images/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (12).png")
                self.TEAM1_LEGS.append(f"files/Images/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (14).png")
            else:
                self.TEAM1_ARMS.append(f"files/Images/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (11).png")
                self.TEAM1_LEGS.append(f"files/Images/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (13).png")

            num2 = random.randint(1, 5) # number from 1 to 4 including 4
            self.TEAM2_SPRITES.append(f"files/Images/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} ({num2}).png")
            if num2 == 5:
                self.TEAM2_ARMS.append(f"files/Images/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (12).png")
                self.TEAM2_LEGS.append(f"files/Images/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (14).png")
            else:
                self.TEAM2_ARMS.append(f"files/Images/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (11).png")
                self.TEAM2_LEGS.append(f"files/Images/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (13).png")
        
        num = random.randint(1, 4)
        self.BALL_SPRITE = f"files/Images/Equipment/ball_soccer{num}.png"

        self.BACKGROUND = "files/Images/Backgrounds/pitch.png"

        self.GOAL_SPRITE = "files/Images/Backgrounds/goal.png"

        

        
    