import random 


class EnvironmentHyperparameters:
    def __init__(self):
        # Screen dimensions
        self.WIDTH = 1300  # Increased width for a wider field
        self.HEIGHT = 700

        # Colors (R, G, B) # TODO : REMOVE THIS
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)     # Player 1 color
        self.GREEN = (0, 255, 0)    # Player 2 color
        self.BLACK = (0, 0, 0)      # Ball color
        self.RED = (255, 0, 0)      # Goal color
        self.YELLOW = (255, 255, 0) # Play area boundary color

        # Frame rate
        self.FPS = 60

        # Player properties
        self.PLAYER_RADIUS = 25
        self.PLAYER_SPEED = 5
        self.PLAYER_HEIGHT = self.PLAYER_RADIUS * 2  # Diameter
        self.PLAYER_POWER = 1 # player vs player collision power 

        # Ball properties
        self.BALL_RADIUS = 15 
        self.BALL_FRICTION = 0.95
        self.BALL_MAX_SPEED = 13
        self.BALL_HEIGHT = self.BALL_RADIUS * 2  # Diameter
        self.BALL_POWER = 25 # player vs ball collision power

        # Goal properties
        self.GOAL_WIDTH = 0.05 * self.WIDTH  # 5% of the screen width (65 pixels)
        self.GOAL_HEIGHT = 0.26 * self.HEIGHT  # 25% of the screen height (175 pixels)
        self.GOAL_COLOR = self.RED

        # Play area vertical boundaries
        self.PLAY_AREA_TOP = 0
        self.PLAY_AREA_BOTTOM = self.HEIGHT

        # Play area horizontal boundaries
        self.PLAY_AREA_LEFT = self.GOAL_WIDTH  # 130
        self.PLAY_AREA_RIGHT = self.WIDTH - self.GOAL_WIDTH  # 1170

        self.PLAY_AREA_WIDTH = self.PLAY_AREA_RIGHT - self.PLAY_AREA_LEFT
        self.PLAY_AREA_HEIGHT = self.HEIGHT


        self.GAME_DURATION = 20 #5* 60  # 5 minutes

        # number of player
        self.NUMBER_OF_PLAYERS = 1

        self.calculate_positions()

        self.RENDER = True
    
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
    

ENV = EnvironmentHyperparameters()

class VisualHyperparametters:
    def __init__(self):

        self.TITLE = "2D Soccer Game"

        self.TEAM_1_COLOR = "Blue" #options : blue, red, green, White
        self.TEAM_2_COLOR = "Green"

        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)     # Player 1 color
        self.GREEN = (50, 200, 50)    # Player 2 color
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
            self.TEAM1_SPRITES.append(f"files/PNG/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} ({num1}).png")
            if num1 == 5:
                self.TEAM1_ARMS.append(f"files/PNG/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (12).png")
                self.TEAM1_LEGS.append(f"files/PNG/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (14).png")
            else:
                self.TEAM1_ARMS.append(f"files/PNG/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (11).png")
                self.TEAM1_LEGS.append(f"files/PNG/{self.TEAM_1_COLOR}/character{self.TEAM_1_COLOR} (13).png")

            num2 = random.randint(1, 5) # number from 1 to 4 including 4
            self.TEAM2_SPRITES.append(f"files/PNG/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} ({num2}).png")
            if num2 == 5:
                self.TEAM2_ARMS.append(f"files/PNG/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (12).png")
                self.TEAM2_LEGS.append(f"files/PNG/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (14).png")
            else:
                self.TEAM2_ARMS.append(f"files/PNG/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (11).png")
                self.TEAM2_LEGS.append(f"files/PNG/{self.TEAM_2_COLOR}/character{self.TEAM_2_COLOR} (13).png")
        
        num = random.randint(1, 4)
        self.BALL_SPRITE = f"files/PNG/Equipment/ball_soccer{num}.png"

        self.BACKGROUND = "files/PNG/Backgrounds/pitch.png"
        

        
    