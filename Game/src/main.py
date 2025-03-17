from enviroment.Game import Game
from tqdm import tqdm
from functions.General import format_time, format_log_file
from functions.league import League
from AI.randmodel import RandomModel
from AI.OldPPO import OldPPOAgent
from AI.BadTransformer import BadTransformerPPOAgent
from AI.PPO import PPOAgent
from AI.MAAC import MAAC
from AI.HUGO import HUGO
import json
from functions.Logger import Logger, set_parameters
from functions.ELO import ELO
from functions.Statistics import StatsHistoryViewer
from params import EnvironmentHyperparameters, VisualHyperparametters, AIHyperparameters, print_hyper_params
import multiprocessing
import random
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
    stats = game.run(model, competing_model, current_stage)
    return stats.score[0], stats.score[1], stats.avg_reward, stats.ball_distance, stats.ball_hits, stats.avg_entropy, stats

def play_game():
    filename = format_log_file("last_game")
    game = Game(log_name=filename)
    stats = game.run_play()
    score = stats.score

    if score[0] > score[1]:
        print("Team 1 wins!")
    elif score[0] < score[1]:
        print("Team 2 wins!")

    print(f"Final score: {score}")
    stats.print()
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
    stats_history_viewer = StatsHistoryViewer(ENV_PARAMS.MODEL_NAME)

    ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage1_time
    elo_env = ELO()



    model1_rating = elo_env.init_rating()

    if ENV_PARAMS.model == "PPO_old":
        train_model = OldPPOAgent(mode="train")
        competing_model = OldPPOAgent(mode="test")
    elif ENV_PARAMS.model == "Bad_Transformer":
        train_model = BadTransformerPPOAgent(mode="train")
        competing_model = BadTransformerPPOAgent(mode="test")
    elif ENV_PARAMS.model == "PPO":
        train_model = PPOAgent(mode="train")
        competing_model = PPOAgent(mode="test")
    elif ENV_PARAMS.model == "MAAC":
        train_model = MAAC(mode="train")
        competing_model = MAAC(mode="test")
    elif ENV_PARAMS.model == "HUGO":
        train_model = HUGO(mode="train")
        competing_model = HUGO(mode="test")
    else:
        raise ValueError("Model not recognized")
    competing_model.policy.eval()
    competing_model.policy_old.eval()
    competing_model.policy.requires_grad = False
    competing_model.policy_old.requires_grad = False



    if ENV_PARAMS.Load_model:
        train_model.load_model(ENV_PARAMS.Load_model)

    league = League(elo_env, competing_model)
    league.add_player(train_model, elo=elo_env.init_rating())   

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
        competing_model, model2_rating = league.sample_player()

        # Collect experiences
        score1, score2, avg_reward, ball_dist,ball_hits, entropy, stats = run_game(train_model, competing_model, filename, AI_PARAMS.current_stage)
        stats_history_viewer.add(stats)

        if AI_PARAMS.current_stage >= 3 and model2_rating is not None:
            new_model1_rating, new_model2_rating = elo_env.calculate(model1_rating, model2_rating, score1, score2)
            model1_rating = new_model1_rating
        stats_history_viewer.add_elo(model1_rating)
            # model2_rating = new_model2_rating keep the same rating for the competing model
        
        league.update(train_model)

        # train the model on episode
        train_model.update()

        # Calculate entropy
        entropy_percent = entropy / math.log(AI_PARAMS.ACTION_SIZE)

        if epoch % ENV_PARAMS.STATS_UPDATE_INTERVAL == 0 and epoch > 0:
            stats_history_viewer.update()
        
        # Log metrics
        if model2_rating is None:
            model2_rating = elo_env.create_rating(1, 1)
        print(f"Episode: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, ELO: {model1_rating.mu:.2f} - {model2_rating.mu:.2f}")
        for i in tqdm(range(1), desc=f"Episode: {epoch}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, ELO {model1_rating.mu:.2f} - {model2_rating.mu:.2f}"):
            a=1
        if epoch % ENV_PARAMS.log_interval == 0:
            train_model.save_model(ENV_PARAMS.MODEL_NAME)
    
    pygame.quit()


import multiprocessing as mp
import torch
from tqdm import tqdm
import pygame
import time
import math

def run_single_game(args):
    (model, competing_model, current_stage, filename, gpu_id) = args
    AI_PARAMS = AIHyperparameters()
    ENV_PARAMS = EnvironmentHyperparameters()
    if current_stage == 1:
        ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage1_time
    if current_stage== 2:
        ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage2_time
    if current_stage == 3:
        ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage3_time
    if current_stage == 4:
        ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage4_time

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.assign_device(device)
    competing_model.assign_device(device)
    filename = format_log_file(filename)
    game = Game(log_name=filename)
    stats = game.run(model, competing_model, current_stage)
    memories = model.memories
    output = (stats.score[0], stats.score[1], stats.avg_reward, stats.ball_distance, stats.ball_hits, stats.avg_entropy, stats)
    return output, memories

def train_PPO_parralel():
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()

    pygame.init()
    ENV_PARAMS.GAME_DURATION = AI_PARAMS.stage1_time
    num_games = ENV_PARAMS.NUMBER_OF_GAMES

    stats_history_viewer = StatsHistoryViewer(ENV_PARAMS.MODEL_NAME)

    elo_env = ELO()
    model1_rating = elo_env.init_rating()

    if ENV_PARAMS.model == "PPO_old":
        train_model = OldPPOAgent(mode="train")
        competing_model = OldPPOAgent(mode="test")
    elif ENV_PARAMS.model == "Bad_Transformer":
        train_model = BadTransformerPPOAgent(mode="train")
        competing_model = BadTransformerPPOAgent(mode="test")
    elif ENV_PARAMS.model == "PPO":
        train_model = PPOAgent(mode="train")
        competing_model = PPOAgent(mode="test")
    elif ENV_PARAMS.model == "MAAC":
        train_model = MAAC(mode="train")
        competing_model = MAAC(mode="test")
    elif ENV_PARAMS.model == "HUGO":
        train_model = HUGO(mode="train")
        competing_model = HUGO(mode="test")
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

    league = League(elo_env, competing_model)
    league.add_player(train_model, elo=elo_env.init_rating())

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

            competing_model, model2_rating = league.sample_player()
            
            if torch.cuda.is_available():
                args_list = [(train_model, competing_model, AI_PARAMS.current_stage, filename if i == 0 else None, i % torch.cuda.device_count()) for i in range(num_games)]
            else:
                args_list = [(train_model, competing_model, AI_PARAMS.current_stage, filename if i == 0 else None, 0) for i in range(num_games)]


            results = pool.map(run_single_game, args_list)

            # Combine results
            combined_memories = []
            score1 = score2 = avg_reward = ball_dist = ball_hits = entropy = 0

            for (metrics, memories) in results:
                s1, s2, ar, bd, bh, en, stats = metrics
                score1 += s1
                score2 += s2
                avg_reward += ar
                ball_dist += bd
                ball_hits += bh
                entropy += en
                stats_history_viewer.add(stats)
                combined_memories.append(memories)

                if AI_PARAMS.current_stage >= 3 and model2_rating is not None:
                    new_model1_rating, new_model2_rating = elo_env.calculate(model1_rating, model2_rating, score1, score2)
                    model1_rating = new_model1_rating
                    # model2_rating = new_model2_rating keep the same rating for the competing model

            stats_history_viewer.combine_last_N(num_games)
            stats_history_viewer.add_elo(model1_rating)

            score1 /= num_games
            score2 /= num_games
            avg_reward /= num_games
            ball_dist /= num_games
            entropy /= num_games

            combined_memories = [item for sublist in combined_memories for item in sublist]
            train_model.memories = combined_memories

            similarity_loss = train_model.update()
            stats_history_viewer.add_similarity_loss(similarity_loss)

            entropy_percent = entropy / math.log(AI_PARAMS.ACTION_SIZE) if AI_PARAMS.ACTION_SIZE > 1 else 0
            if model2_rating is None:
                model2_rating = elo_env.create_rating(1, 1)
            print(
                f"Episode: {epoch}, Score: {score1:.1f} - {score2:.1f}, "
                f"Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, "
                f"Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, "
                f"stage: {AI_PARAMS.current_stage}"
                F"ELO: {model1_rating.mu:.2f} - {model2_rating.mu:.2f}"
                f"similarity loss: {similarity_loss:.2f}"
            )
            for i in tqdm(range(1), desc=f"Episode: {epoch}, Score: {score1:.1f} - {score2:.1f}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, entropy: {entropy_percent:.2f}, stage: {AI_PARAMS.current_stage}, ELO: {model1_rating.mu:.2f} - {model2_rating.mu:.2f}, similarity loss: {similarity_loss:.2f}"):
                a=1
                break

            if epoch % ENV_PARAMS.log_interval == 0:
                train_model.save_model(ENV_PARAMS.MODEL_NAME)

            if epoch % ENV_PARAMS.STATS_UPDATE_INTERVAL == 0 and epoch > 0:
                stats_history_viewer.update()

    pygame.quit()


def test_PPO():
    elo_env = ELO()

    models = []
    ratings = []
    names = []

    # Load models
    #maac = MAAC(mode="test")
    #maac.load_model("MAAC_v3_sub0")
    #names.append("MAAC")
    #ratings.append(elo_env.init_rating())
    #models.append(maac)

    ppo = PPOAgent(mode="test")
    ppo.load_model("PPO_v24_sub0")
    names.append("PPO")
    ratings.append(elo_env.init_rating())
    models.append(ppo)

    hugo = HUGO(mode="test")
    hugo.load_model("HUGO_v8_sub0")
    names.append("HUGO")
    ratings.append(elo_env.init_rating())
    models.append(hugo)

    #rand = RandomModel()
    #names.append("Random")
    #ratings.append(elo_env.init_rating())
    #models.append(rand)

    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()

    pygame.init()
    AI_PARAMS.current_stage = 4

    for epoch in tqdm(range(AI_PARAMS.episodes)):

        model1_idx = random.randint(0, len(models) - 1)
        while True:
            model2_idx = random.randint(0, len(models) - 1)
            if model2_idx != model1_idx:
                break
        
        model1 = models[model1_idx]
        model2 = models[model2_idx]

        model1_rating = ratings[model1_idx]
        model2_rating = ratings[model2_idx]

        score1, score2, avg_reward, ball_dist,ball_hits, entropy, stats = run_game(model1, model2, "testing", AI_PARAMS.current_stage)

        new_model1_rating, new_model2_rating = elo_env.calculate(model1_rating, model2_rating, score1, score2)
        ratings[model1_idx] = new_model1_rating
        ratings[model2_idx] = new_model2_rating

        print(f"Episode: {epoch},{names[model1_idx]} vs {names[model2_idx]}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, ELO {model1_rating.mu:.2f} - {model2_rating.mu:.2f}")
        for i in tqdm(range(1), desc=f"Episode: {epoch}, {names[model1_idx]} vs {names[model2_idx]}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, ELO {model1_rating.mu:.2f} - {model2_rating.mu:.2f}"):
            a=1

        if epoch % 20 == 0:
            print(tabulate([[names[i], ratings[i].mu] for i in range(len(models))], headers=["Model", "ELO"]))

import cProfile

if __name__ == "__main__":
    ENV_PARAMS = EnvironmentHyperparameters()

    if ENV_PARAMS.MODE == "play":
        play_game()
    elif ENV_PARAMS.MODE == "replay":
        replay_game()
    elif ENV_PARAMS.MODE == "train":
        #cProfile.run("train_PPO()", "train_stats.log")
        train_PPO()
    elif ENV_PARAMS.MODE == "train_parallel":
        mp.set_start_method("spawn", force=True)
        train_PPO_parralel()
    elif ENV_PARAMS.MODE == "test":
        test_PPO()
        pass
