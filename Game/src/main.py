from enviroment.Game import Game
from params import EnvironmentHyperparameters
import multiprocessing
import time
import pygame
from tabulate import tabulate

pygame.init()


def run_game(queue):
    game = Game()
    score = game.run()
    print(score)
    queue.put(score)

def format_time(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {seconds:.3f}s"

if __name__ == "__main__":
    ENV_PARAMS = EnvironmentHyperparameters()
    start_time = time.time()
    
    processes = []
    queue = multiprocessing.Queue()

    # Start Game 1 with rendering
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