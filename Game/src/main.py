from enviroment.Game import Game
from functions.General import format_time, format_log_file
from AI.randmodel import RandomModel
import json
from functions.Logger import Logger, set_parameters
from params import EnvironmentHyperparameters, VisualHyperparametters
import multiprocessing
import time
import pygame
from tabulate import tabulate

pygame.init()

def run_game(queue):
    filename = format_log_file()
    game = Game(log_name=filename)
    score = game.run_play()
    queue.put(score)

def run_game(queue):
    filename = format_log_file("rand_log")
    game = Game(log_name=filename)
    rand = RandomModel()
    score = game.run(rand)
    queue.put(score)

def play_game():
    filename = format_log_file("last_game")
    game = Game(log_name=filename)
    score = game.run_play()

    if score[0] > score[1]:
        print("Team 1 wins!")
    elif score[0] < score[1]:
        print("Team 2 wins!")

    print(f"Final score: {score}")
    pygame.quit()

def replay_game():   
    ENV_PARAMS = EnvironmentHyperparameters()
    ENV_PARAMS.RENDER = True
    filename = format_log_file(ENV_PARAMS.FILE_NAME)
    with open(filename, "r") as file:
        log_data = json.load(file)
    
    set_parameters(log_data["parameters"])

    VIS_PARAMS = VisualHyperparametters()
    VIS_PARAMS.update()
    
    pygame.init()

    game = Game()
    game.replay(log_data["states"])

    pygame.quit()

def train():
    pass

def test():
    ENV_PARAMS = EnvironmentHyperparameters()
    start_time = time.time()
    
    processes = []
    queue = multiprocessing.Queue()

    for i in range(ENV_PARAMS.NUMBER_OF_GAMES):
        p = multiprocessing.Process(target=run_game, args=(queue,))
        processes.append(p)
        p.start()


    for p in processes:
        p.join()


    # Collect all scores
    scores = []
    while not queue.empty():
        scores.append(queue.get())

    pygame.quit()
    
    print(scores)

    # Calculate total goals
    goals = sum(score[0] + score[1] for score in scores)
    team_1_score = sum(score[0] for score in scores)
    team_2_score = sum(score[1] for score in scores)

    total_score = (team_1_score, team_2_score)
    diff = team_1_score - team_2_score
    average_score = (team_1_score / ENV_PARAMS.NUMBER_OF_GAMES, team_2_score / ENV_PARAMS.NUMBER_OF_GAMES)

    # Calculate times
    real_time_taken = time.time() - start_time
    real_time_per_game = real_time_taken / ENV_PARAMS.NUMBER_OF_GAMES
    ingame_time_played = ENV_PARAMS.NUMBER_OF_GAMES * ENV_PARAMS.GAME_DURATION
    ingame_time_per_game = ENV_PARAMS.GAME_DURATION

    # Prepare data for the table
    data = [
        ["REAL Time taken", format_time(real_time_taken)],
        ["REAL Time per game", format_time(real_time_per_game)],
        ["INGAME Time played", format_time(ingame_time_played)],
        ["INGAME Time played per game", format_time(ingame_time_per_game)],
    ]

    data2 = [
        ["Number of games", ENV_PARAMS.NUMBER_OF_GAMES],
        ["Goals scored", goals],
        ["Goals per game", f"{goals / ENV_PARAMS.NUMBER_OF_GAMES:.3f}"],
        ["Total score", total_score],
        ["Average score", str(average_score)],
        ["Goal Differential", diff]
    ]

    # Print the table
    print(tabulate(data, headers=["Metric", "Time"], tablefmt="pretty"))
    print(tabulate(data2, headers=["Metric", "Value"], tablefmt="pretty"))


def default():
    ENV_PARAMS = EnvironmentHyperparameters()
    start_time = time.time()
    
    processes = []
    queue = multiprocessing.Queue()

    for i in range(ENV_PARAMS.NUMBER_OF_GAMES):
        p = multiprocessing.Process(target=run_game, args=(queue,))
        processes.append(p)
        p.start()


    for p in processes:
        p.join()


    # Collect all scores
    scores = []
    while not queue.empty():
        scores.append(queue.get())

    pygame.quit()
    
    print(scores)

    # Calculate total goals
    goals = sum(score[0] + score[1] for score in scores)
    team_1_score = sum(score[0] for score in scores)
    team_2_score = sum(score[1] for score in scores)

    total_score = (team_1_score, team_2_score)
    diff = team_1_score - team_2_score
    average_score = (team_1_score / ENV_PARAMS.NUMBER_OF_GAMES, team_2_score / ENV_PARAMS.NUMBER_OF_GAMES)

    # Calculate times
    real_time_taken = time.time() - start_time
    real_time_per_game = real_time_taken / ENV_PARAMS.NUMBER_OF_GAMES
    ingame_time_played = ENV_PARAMS.NUMBER_OF_GAMES * ENV_PARAMS.GAME_DURATION
    ingame_time_per_game = ENV_PARAMS.GAME_DURATION

    # Prepare data for the table
    data = [
        ["REAL Time taken", format_time(real_time_taken)],
        ["REAL Time per game", format_time(real_time_per_game)],
        ["INGAME Time played", format_time(ingame_time_played)],
        ["INGAME Time played per game", format_time(ingame_time_per_game)],
    ]

    data2 = [
        ["Number of games", ENV_PARAMS.NUMBER_OF_GAMES],
        ["Goals scored", goals],
        ["Goals per game", f"{goals / ENV_PARAMS.NUMBER_OF_GAMES:.3f}"],
        ["Total score", total_score],
        ["Average score", str(average_score)],
        ["Goal Differential", diff]
    ]

    # Print the table
    print(tabulate(data, headers=["Metric", "Time"], tablefmt="pretty"))
    print(tabulate(data2, headers=["Metric", "Value"], tablefmt="pretty"))


if __name__ == "__main__":
    ENV_PARAMS = EnvironmentHyperparameters()

    if ENV_PARAMS.MODE == "play":
        play_game()
    elif ENV_PARAMS.MODE == "replay":
        replay_game()
    elif ENV_PARAMS.MODE == "test":
        test()
    else:
        default()