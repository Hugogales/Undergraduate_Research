from enviroment.Game import Game
from tqdm import tqdm
from functions.General import format_time, format_log_file
from AI.randmodel import RandomModel
from AI.OldPPO import OldPPOAgent
from AI.BadTransformer import BadTransformerPPOAgent
from AI.PPO import PPOAgent
import json
from functions.Logger import Logger, set_parameters
from params import EnvironmentHyperparameters, VisualHyperparametters, AIHyperparameters, print_hyper_params
import multiprocessing
import time
import math
import pygame
from tabulate import tabulate
import torch
import torch.multiprocessing as mp

pygame.init()
print("cuda" if torch.cuda.is_available() else "cpu")
print(f"Number of CPUs: {mp.cpu_count()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

def run_game(model, competing_model, filename, current_stage):
    filename = format_log_file(filename)
    game = Game(log_name=filename)
    return game.run(model, competing_model, current_stage)

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

    ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage1_time
    

    if ENV_PARAMS.model == "PPO_old":
        train_model = OldPPOAgent(mode="train")
        competing_model = OldPPOAgent(mode="test")
    elif ENV_PARAMS.model == "Bad_Transformer":
        train_model = BadTransformerPPOAgent(mode="train")
        competing_model = BadTransformerPPOAgent(mode="test")
    elif ENV_PARAMS.model == "PPO":
        train_model = PPOAgent(mode="train")
        competing_model = PPOAgent(mode="test")
    elif ENV_PARAMS.model == "A2C":
        # to be implemented
        pass
    elif ENV_PARAMS.model == "A3C":
        # to be implemented
        pass
    else:
        raise ValueError("Model not recognized")
    competing_model.policy.eval()
    competing_model.policy_old.eval()
    competing_model.policy.requires_grad = False
    competing_model.policy_old.requires_grad = False

    if ENV_PARAMS.Load_model:
        train_model.load_model(ENV_PARAMS.Load_model)
        competing_model.load_model(ENV_PARAMS.Load_model)

    scores = []
    
    stage_counter = 0
    for epoch in tqdm(range(AI_PARAMS.episodes), desc="Training PPO"):
        if AI_PARAMS.current_stage == 1 and stage_counter >= AI_PARAMS.stage1_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage2_time
            print("change to stage 2 - random locations ")
        if AI_PARAMS.current_stage == 2 and stage_counter >= AI_PARAMS.stage2_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage3_time
            print("change to stage 3 - both team plays - random positions")
        if AI_PARAMS.current_stage == 3 and stage_counter >= AI_PARAMS.stage3_steps:
            AI_PARAMS.current_stage += 1
            stage_counter = 0
            ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage4_time
            print("change to stage 4 - both teams plays - typical locations")
        stage_counter += 1

        # Adjust reward parameters based on training stage
        if epoch % ENV_PARAMS.log_interval == 0:
            filename = f"{ENV_PARAMS.log_name}_{epoch}"
            print_hyper_params()
        else:
            filename = None

        # Update competing model
        if epoch % AI_PARAMS.opposing_model_freeze_time == 0: 
            competing_model.policy.load_state_dict(train_model.policy.state_dict())
            competing_model.policy_old.load_state_dict(train_model.policy_old.state_dict())

        # Collect experiences
        score1, score2, avg_reward, ball_dist,ball_hits, team_playing, entropy = run_game(train_model, competing_model, filename, AI_PARAMS.current_stage)
        scores.append((score1, score2))

        # train the model on episode
        train_model.update()

        # Calculate entropy
        entropy_percent = entropy / math.log(AI_PARAMS.ACTION_SIZE)

        if epoch > 5:
            break

        # Log metrics
        print(f"Episode: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, team_playing: {team_playing}")
        for i in tqdm(range(1), desc=f"Episode: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, team_playing: {team_playing}"):
            a=1
        if epoch % ENV_PARAMS.log_interval == 0:
            train_model.save_model(ENV_PARAMS.MODEL_NAME)
    
    pygame.quit()
    
    # Calculate total goals
    goals = sum(score[0] + score[1] for score in scores)
    team_1_score = sum(score[0] for score in scores)
    team_2_score = sum(score[1] for score in scores)

    total_score = (team_1_score, team_2_score)
    diff = team_1_score - team_2_score
    average_score = (team_1_score / (ENV_PARAMS.NUMBER_OF_GAMES* AI_PARAMS.episodes)
                     , team_2_score / ENV_PARAMS.NUMBER_OF_GAMES * AI_PARAMS.episodes)

    # Calculate times
    real_time_taken = time.time() - start_time
    real_time_per_game = real_time_taken / (ENV_PARAMS.NUMBER_OF_GAMES * AI_PARAMS.episodes)
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

import multiprocessing as mp
import torch
from tqdm import tqdm
import pygame
import time
import math

def run_single_game(args):
    (model, competing_model, current_stage, filename, gpu_id) = args
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.assign_device(device)
    competing_model.assign_device(device)
    filename = format_log_file(filename)
    game = Game(log_name=filename)
    output = game.run(model, competing_model, current_stage)
    memories = model.memories
    return output, memories

def train_PPO_parralel():
    
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()

    pygame.init()
    ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage1_time
    num_games = ENV_PARAMS.NUMBER_OF_GAMES

    if ENV_PARAMS.model == "PPO_old":
        train_model = OldPPOAgent(mode="train")
        competing_model = OldPPOAgent(mode="test")
    elif ENV_PARAMS.model == "Bad_Transformer":
        train_model = BadTransformerPPOAgent(mode="train")
        competing_model = BadTransformerPPOAgent(mode="test")
    elif ENV_PARAMS.model == "PPO":
        train_model = PPOAgent(mode="train")
        competing_model = PPOAgent(mode="test")
    elif ENV_PARAMS.model == "A2C":
        # to be implemented
        pass
    elif ENV_PARAMS.model == "A3C":
        # to be implemented
        pass
    else:
        raise ValueError("Model not recognized")

    competing_model.policy.eval()
    competing_model.policy_old.eval()
    competing_model.policy.requires_grad = False
    competing_model.policy_old.requires_grad = False

    if ENV_PARAMS.Load_model:
        train_model.load_model(ENV_PARAMS.Load_model)
        competing_model.load_model(ENV_PARAMS.Load_model)
    
    stage_counter = 0

    # Create the pool once, outside the training loop
    with mp.Pool(processes=num_games) as pool:
        for epoch in tqdm(range(AI_PARAMS.episodes), desc="Training PPO"):
            if AI_PARAMS.current_stage == 1 and stage_counter >= AI_PARAMS.stage1_steps:
                AI_PARAMS.current_stage += 1
                stage_counter = 0
                ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage2_time
                print("change to stage 2 - random locations")

            if AI_PARAMS.current_stage == 2 and stage_counter >= AI_PARAMS.stage2_steps:
                AI_PARAMS.current_stage += 1
                stage_counter = 0
                ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage3_time
                print("change to stage 3 - both team plays - random positions")

            if AI_PARAMS.current_stage == 3 and stage_counter >= AI_PARAMS.stage3_steps:
                AI_PARAMS.current_stage += 1
                stage_counter = 0
                ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage4_time
                print("change to stage 4 - both teams plays - typical locations")

            stage_counter += 1

            if epoch % ENV_PARAMS.log_interval == 0:
                filename = f"{ENV_PARAMS.log_name}_{epoch}"
                print_hyper_params()
            else:
                filename = None

            if epoch % AI_PARAMS.opposing_model_freeze_time == 0:
                competing_model.policy.load_state_dict(train_model.policy.state_dict())
                competing_model.policy_old.load_state_dict(train_model.policy_old.state_dict())

            if torch.cuda.is_available():
                args_list = [(train_model, competing_model, AI_PARAMS.current_stage, filename, i % torch.cuda.device_count()) for i in range(num_games)]
            else:
                args_list = [(train_model, competing_model, AI_PARAMS.current_stage, filename, 0) for _ in range(num_games)]

            results = pool.map(run_single_game, args_list)

            # Combine results
            combined_memories = []
            score1 = score2 = avg_reward = ball_dist = ball_hits = entropy = 0
            team_playing = None

            for (metrics, memories) in results:
                s1, s2, ar, bd, bh, tp, en = metrics
                score1 += s1
                score2 += s2
                avg_reward += ar
                ball_dist += bd
                ball_hits += bh
                team_playing = tp
                entropy += en
                combined_memories.append(memories)

            score1 /= num_games
            score2 /= num_games
            avg_reward /= num_games
            ball_dist /= num_games
            entropy /= num_games

            combined_memories = [item for sublist in combined_memories for item in sublist]
            train_model.memories = combined_memories

            train_model.update()

            entropy_percent = entropy / math.log(AI_PARAMS.ACTION_SIZE) if AI_PARAMS.ACTION_SIZE > 1 else 0

            print(
                f"Episode: {epoch}, Score: {score1:.1f} - {score2:.1f}, "
                f"Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, "
                f"Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, "
                f"team_playing: {team_playing},"
                f"stage: {AI_PARAMS.current_stage}"
            )
            for i in tqdm(range(1), desc=f"Episode: {epoch}, Score: {score1:.1f} - {score2:.1f}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, team_playing: {team_playing}, stage: {AI_PARAMS.current_stage}"):
                break

            if epoch % ENV_PARAMS.log_interval == 0:
                train_model.save_model(ENV_PARAMS.MODEL_NAME)

    pygame.quit()

import cProfile

if __name__ == "__main__":
    ENV_PARAMS = EnvironmentHyperparameters()

    if ENV_PARAMS.MODE == "play":
        play_game()
    elif ENV_PARAMS.MODE == "replay":
        replay_game()
    elif ENV_PARAMS.MODE == "train":
        cProfile.run("train_PPO()", "profile1.log")    
    elif ENV_PARAMS.MODE == "train_parallel":
        mp.set_start_method("spawn", force=True)
        train_PPO_parralel()
    

