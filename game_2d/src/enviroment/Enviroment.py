from game_2d.src.params import EnvironmentHyperparameters
import pygame
import math
import os

# Instantiate Environment Hyperparameters as a global constant
ENV_PARAMS = EnvironmentHyperparameters()

if not ENV_PARAMS.RENDER:
    # Disable Pygame support if rendering is disabled
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# =======================
# Player Class
# =======================

# =======================
# Ball Class
# =======================

# =======================
# Goal Class
# =======================
