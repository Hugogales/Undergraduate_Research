class EnvironmentHyperparameters:
    def __init__(self):
        # Screen dimensions
        self.WIDTH = 1300  # Increased width for a wider field
        self.HEIGHT = 700

        # Colors (R, G, B)
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

        # Ball properties
        self.BALL_RADIUS = 15 
        self.BALL_FRICTION = 0.95
        self.BALL_MAX_SPEED = 13
        self.BALL_HEIGHT = self.BALL_RADIUS * 2  # Diameter

        # Goal properties
        self.GOAL_WIDTH = 0.05 * self.WIDTH  # 5% of the screen width (65 pixels)
        self.GOAL_HEIGHT = 0.25 * self.HEIGHT  # 25% of the screen height (175 pixels)
        self.GOAL_COLOR = self.RED

        # Play area vertical boundaries
        self.PLAY_AREA_TOP = 0
        self.PLAY_AREA_BOTTOM = self.HEIGHT

        # Play area horizontal boundaries
        self.PLAY_AREA_LEFT = self.GOAL_WIDTH  # 130
        self.PLAY_AREA_RIGHT = self.WIDTH - self.GOAL_WIDTH  # 1170

        # Initialize Pygame display
        self.TITLE = "2D Soccer Game"

        self.team_1_positions = [[self.PLAY_AREA_LEFT + self.PLAYER_RADIUS + 100, self.HEIGHT / 2], [self.PLAY_AREA_LEFT + 100 + self.PLAYER_RADIUS, self.HEIGHT / 2 - 100], [self.PLAY_AREA_LEFT + 100 + self.PLAYER_RADIUS, self.HEIGHT / 2 + 100]]
        self.team_2_positions = [[self.PLAY_AREA_RIGHT - self.PLAYER_RADIUS - 100 , self.HEIGHT / 2], [self.PLAY_AREA_RIGHT - 100 - self.PLAYER_RADIUS, self.HEIGHT / 2 - 100], [self.PLAY_AREA_RIGHT - 100 - self.PLAYER_RADIUS, self.HEIGHT / 2 + 100]]

        self.RENDER = False