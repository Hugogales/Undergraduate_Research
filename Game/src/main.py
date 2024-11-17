from enviroment.Game import Game
from tqdm import tqdm
from functions.General import format_time, format_log_file
from AI.randmodel import RandomModel
from AI.DQN import DQNAgent
import json
from functions.Logger import Logger, set_parameters
from params import EnvironmentHyperparameters, VisualHyperparametters, AIHyperparameters
import multiprocessing
import time
import pygame
from tabulate import tabulate

pygame.init()


def run_game(model, filename):
    filename = format_log_file(filename)
    game = Game(log_name=filename)
    return game.run(model)

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
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()
    start_time = time.time()
    pygame.init()
    
    model = DQNAgent()
    scores = []

    for epoch in tqdm(range(ENV_PARAMS.EPOCHS)):
        if epoch == AI_PARAMS.stage2:
            AI_PARAMS.DISTANCE_REWARD_BALL = AI_PARAMS.DISTANCE_REWARD_BALL_matrix[1]
            AI_PARAMS.DISTANCE_REWARD_GOAL = AI_PARAMS.DISTANCE_REWARD_GOAL_matrix[1]
        elif epoch == AI_PARAMS.stage3:
            AI_PARAMS.DISTANCE_REWARD_BALL = AI_PARAMS.DISTANCE_REWARD_BALL_matrix[2]
            AI_PARAMS.DISTANCE_REWARD_GOAL = AI_PARAMS.DISTANCE_REWARD_GOAL_matrix[2]

        if epoch % ENV_PARAMS.log_interval == 0:
            filename = f"{ENV_PARAMS.log_name}_{epoch}"
        else:
            filename = None

        score1, score2, avg_reward = run_game(model, filename)
        scores.append((score1, score2))

        print(f"Epoch: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward}")
        if epoch % ENV_PARAMS.log_interval == 0:
            model.save_model()
    
    pygame.quit()

    
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
    elif ENV_PARAMS.MODE == "train":
        train()
    else:
        default()