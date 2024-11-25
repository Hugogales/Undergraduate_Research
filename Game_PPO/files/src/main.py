from enviroment.Game import Game
from tqdm import tqdm
from functions.General import format_time, format_log_file
from AI.randmodel import RandomModel
from AI.PPO import PPOAgent
import json
from functions.Logger import Logger, set_parameters
from params import EnvironmentHyperparameters, VisualHyperparametters, AIHyperparameters
import multiprocessing
import time
import pygame
from tabulate import tabulate
import torch
import torch.multiprocessing as mp

pygame.init()
print("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of CPUs: {mp.cpu_count()}")

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

def train_PPO():
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()
    start_time = time.time()
    pygame.init()
    
    model = PPOAgent()
    if ENV_PARAMS.Load_model:
        model.load_model(ENV_PARAMS.Load_model)
    scores = []
    
    stage_counter = 0
    for epoch in tqdm(range(ENV_PARAMS.EPOCHS)):
        if AI_PARAMS.current_stage == 1 and stage_counter >= AI_PARAMS.stage1_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            print("change to stage 2 - random locations")
        if AI_PARAMS.current_stage == 2 and stage_counter >= AI_PARAMS.stage2_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            print("change to stage 3 - both teams play")
        if AI_PARAMS.current_stage == 3 and stage_counter >= AI_PARAMS.stage3_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            print("change to stage 4 - both teams play random locations")
        stage_counter += 1

        # Adjust reward parameters based on training stage
        if epoch % ENV_PARAMS.log_interval == 0:
            filename = f"{ENV_PARAMS.log_name}_{epoch}"
        else:
            filename = None
        
        # Collect experiences
        score1, score2, avg_reward, ball_dist,ball_hits, memories = run_game(model, filename)
        scores.append((score1, score2))


        model.update(memories)

        # Log metrics
        print(f"Epoch: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}")
        for i in tqdm(range(1), desc=f"Epoch: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}"):
            print("")
        if epoch % ENV_PARAMS.log_interval == 0:
            model.save_model(ENV_PARAMS.MODEL_NAME)
    
    pygame.quit()
    
    # Calculate total goals
    goals = sum(score[0] + score[1] for score in scores)
    team_1_score = sum(score[0] for score in scores)
    team_2_score = sum(score[1] for score in scores)

    total_score = (team_1_score, team_2_score)
    diff = team_1_score - team_2_score
    average_score = (team_1_score / (ENV_PARAMS.NUMBER_OF_GAMES* ENV_PARAMS.EPOCHS)
                     , team_2_score / ENV_PARAMS.NUMBER_OF_GAMES * ENV_PARAMS.EPOCHS)

    # Calculate times
    real_time_taken = time.time() - start_time
    real_time_per_game = real_time_taken / (ENV_PARAMS.NUMBER_OF_GAMES * ENV_PARAMS.EPOCHS)
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

    model = PPOAgent()
    model.load_model(ENV_PARAMS.MODEL_NAME, test=True)
    scores = []
    for i in range(ENV_PARAMS.EPOCHS):
        score1, score2, reward  = run_game(model, filename=None)
        scores.append((score1, score2))

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
        train_PPO()
    else:
        default()