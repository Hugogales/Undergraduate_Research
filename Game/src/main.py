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

    #new_model = HUGO(mode="test")
    #new_model.load_model("HUGO_v14_sub8")
    #league.add_player(new_model, elo=elo_env.init_rating())
    #new_model = HUGO(mode="test")
    #new_model.load_model("HUGO_v15_sub0")
    #league.add_player(new_model, elo=elo_env.init_rating())
    #new_model = HUGO(mode="test")
    #new_model.load_model("HUGO_v15_sub3")
    #league.add_player(new_model, elo=elo_env.init_rating())


    #new_model = PPOAgent(mode="test")
    #new_model.load_model("PPO_v29_sub3")
    #league.add_player(new_model, elo=elo_env.init_rating())
    new_model = PPOAgent(mode="test")
    new_model.load_model("PPO_v30_sub2")
    league.add_player(new_model, elo=elo_env.init_rating())

    #print(f"Number of players: {len(league.players)}")

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
                    model2_rating = new_model2_rating
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
    # Initialize Elo environment and containers for models, ratings, names, and per-copy stats.
    elo_env = ELO()

    models = []
    ratings = []
    names = []
    # Each entry will track: games, wins, ties, losses, and cumulative stats for team1
    model_stats = []
    
    groups = ["HUGO", "PPO", "MAAC", "Random"]
    # Create 5 copies for each model type.
    for _ in range(1):
        hugo = HUGO(mode="test")
        hugo.load_model("HUGO_v15_sub6")
        models.append(hugo)
        names.append("HUGO")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0  # New metric: cumulative goal differential (score1 - score2)
        })

    for _ in range(1):
        ppo = PPOAgent(mode="test")
        ppo.load_model("PPO_v30_sub4")
        models.append(ppo)
        names.append("PPO")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    for _ in range(1):
        maac = MAAC(mode="test")
        maac.load_model("MAAC_v5_sub20")
        models.append(maac)
        names.append("MAAC")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    for _ in range(1):
        rand = RandomModel()
        models.append(rand)
        names.append("Random")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })


    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()

    pygame.init()
    AI_PARAMS.current_stage = 4

    # We'll use a dictionary where key = (team1_group, team2_group)
    matchup_win = {}
    matchup_goal_diff = {}
    for g1 in groups:
        for g2 in groups:
            matchup_win[(g1, g2)] = {'wins': 0, 'games': 0}
            matchup_goal_diff[(g1, g2)] = {'total_goal_diff': 0.0, 'games': 0}

    # Main simulation loop.
    for epoch in tqdm(range(1)):
        # Randomly select two distinct models.
        model1_idx = random.randint(0, len(models) - 1)
        while True:
            model2_idx = random.randint(0, len(models) - 1)
            if model2_idx != model1_idx:
                break

        model1 = models[model1_idx]
        model2 = models[model2_idx]

        model1_rating = ratings[model1_idx]
        model2_rating = ratings[model2_idx]

        # Run the game between model1 (team1) and model2 (team2)
        score1, score2, avg_reward, ball_dist, ball_hits, entropy, stats = run_game(hugo, maac, "taac_vs_maac", AI_PARAMS.current_stage)

        # Determine outcome for team1
        if score1 > score2:
            outcome = "win"
        elif score1 == score2:
            outcome = "tie"
        else:
            outcome = "loss"

        # Update team1 stats for the model copy that played as team1.
        model_stats[model1_idx]['games'] += 1
        if outcome == "win":
            model_stats[model1_idx]['wins'] += 1
        elif outcome == "tie":
            model_stats[model1_idx]['ties'] += 1
        else:
            model_stats[model1_idx]['losses'] += 1

        model_stats[model1_idx]['total_connectivity'] += stats.avg_connectivity_team1
        model_stats[model1_idx]['total_frequency'] += stats.avg_frequency_possession_distance
        model_stats[model1_idx]['total_pairwise'] += stats.avg_pairwise_distance
        model_stats[model1_idx]['total_reward'] += stats.avg_reward
        model_stats[model1_idx]['total_goal_diff'] += (score1 - score2)

        # Update Elo ratings based on the game outcome.
        new_model1_rating, new_model2_rating = elo_env.calculate(model1_rating, model2_rating, score1, score2)
        ratings[model1_idx] = new_model1_rating
        ratings[model2_idx] = new_model2_rating

        # If both models belong to one of the three groups we track for the matchup matrices, update those stats.
        if names[model1_idx] in groups and names[model2_idx] in groups:
            key = (names[model1_idx], names[model2_idx])
            matchup_win[key]['games'] += 1
            if outcome == "win":
                matchup_win[key]['wins'] += 1

            matchup_goal_diff[key]['games'] += 1
            matchup_goal_diff[key]['total_goal_diff'] += (score1 - score2)

        print(f"Episode: {epoch}, {names[model1_idx]} vs {names[model2_idx]}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, ELO {model1_rating.mu:.2f} - {model2_rating.mu:.2f}")
        # Retaining the inner tqdm loop as in your original code.
        for i in tqdm(range(1), desc=f"Episode: {epoch}, {names[model1_idx]} vs {names[model2_idx]}, Score: {score1} - {score2}, Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, Ball hits: {ball_hits}, ELO {model1_rating.mu:.2f} - {model2_rating.mu:.2f}"):
            a = 1

        if epoch % 20 == 0:
            print(tabulate([[names[i], ratings[i].mu] for i in range(len(models))], headers=["Model", "ELO"]))

    # After all episodes, create two sets of reports:
    # (A) Individual stats table (per copy) grouped by model type (with the new goal differential column)
    # (B) Aggregated table per model group (weighted averages)
    from collections import defaultdict
    grouped_individual = defaultdict(list)
    aggregated = defaultdict(lambda: {
        'games': 0, 'wins': 0, 'ties': 0, 'losses': 0,
        'total_connectivity': 0.0, 'total_frequency': 0.0,
        'total_pairwise': 0.0, 'total_reward': 0.0, 'total_goal_diff': 0.0,
        'elo_sum': 0.0, 'copies': 0
    })

    for i, group in enumerate(names):
        games = model_stats[i]['games']
        wins = model_stats[i]['wins']
        ties = model_stats[i]['ties']
        losses = model_stats[i]['losses']
        if games > 0:
            win_rate = wins / games
            tie_rate = ties / games
            loss_rate = losses / games
            avg_conn = model_stats[i]['total_connectivity'] / games
            avg_freq = model_stats[i]['total_frequency'] / games
            avg_pairwise = model_stats[i]['total_pairwise'] / games
            avg_rwd = model_stats[i]['total_reward'] / games
            avg_goal_diff = model_stats[i]['total_goal_diff'] / games
        else:
            win_rate = tie_rate = loss_rate = avg_conn = avg_freq = avg_pairwise = avg_rwd = avg_goal_diff = 0.0

        # Append row for individual stats table.
        grouped_individual[group].append([
            i, 
            games, 
            f"{win_rate:.5f}", 
            f"{tie_rate:.5f}", 
            f"{loss_rate:.5f}", 
            f"{avg_conn:.5f}", 
            f"{avg_freq:.5f}", 
            f"{avg_pairwise:.5f}", 
            f"{avg_rwd:.5f}",
            f"{avg_goal_diff:.5f}",
            f"{ratings[i].mu:.5f}"
        ])

        # Update aggregated totals per group.
        aggregated[group]['games'] += games
        aggregated[group]['wins'] += wins
        aggregated[group]['ties'] += ties
        aggregated[group]['losses'] += losses
        aggregated[group]['total_connectivity'] += model_stats[i]['total_connectivity']
        aggregated[group]['total_frequency'] += model_stats[i]['total_frequency']
        aggregated[group]['total_pairwise'] += model_stats[i]['total_pairwise']
        aggregated[group]['total_reward'] += model_stats[i]['total_reward']
        aggregated[group]['total_goal_diff'] += model_stats[i]['total_goal_diff']
        aggregated[group]['elo_sum'] += ratings[i].mu
        aggregated[group]['copies'] += 1

    # Print individual stats table for each model group.
    for group, rows in grouped_individual.items():
        print(f"\nIndividual statistics for {group}:")
        print(tabulate(rows, headers=[
            "Copy", "Games", "Win Rate", "Tie Rate", "Loss Rate", 
            "Avg Connectivity", "Avg Freq Poss Dist", "Avg Pairwise Dist", "Avg Reward", "Avg Goal Diff", "ELO"
        ]))

    # Build and print aggregated table per model group.
    aggregated_table = []
    for group, data in aggregated.items():
        if data['games'] > 0:
            agg_win_rate = data['wins'] / data['games']
            agg_tie_rate = data['ties'] / data['games']
            agg_loss_rate = data['losses'] / data['games']
            agg_conn = data['total_connectivity'] / data['games']
            agg_freq = data['total_frequency'] / data['games']
            agg_pairwise = data['total_pairwise'] / data['games']
            agg_reward = data['total_reward'] / data['games']
            agg_goal_diff = data['total_goal_diff'] / data['games']
        else:
            agg_win_rate = agg_tie_rate = agg_loss_rate = agg_conn = agg_freq = agg_pairwise = agg_reward = agg_goal_diff = 0.0

        # Average Elo across copies.
        avg_elo = data['elo_sum'] / data['copies'] if data['copies'] > 0 else 0.0

        aggregated_table.append([
            group, data['games'], f"{agg_win_rate:.5f}", f"{agg_tie_rate:.5f}", f"{agg_loss_rate:.5f}",
            f"{agg_conn:.5f}", f"{agg_freq:.5f}", f"{agg_pairwise:.5f}", f"{agg_reward:.5f}", f"{agg_goal_diff:.5f}",
            f"{avg_elo:.5f}"
        ])

    print("\nAggregated statistics per model group:")
    print(tabulate(aggregated_table, headers=[
        "Model", "Total Games", "Win Rate", "Tie Rate", "Loss Rate",
        "Avg Connectivity", "Avg Freq Poss Dist", "Avg Pairwise Dist", "Avg Reward", "Avg Goal Diff", "Avg ELO"
    ]))

    # Build matchup matrices for the three groups: win rate and goal differential.
    # The matrix keys are tuples (team1_group, team2_group)
    win_matrix = []
    goal_diff_matrix = []
    # Header row for matrices.
    header = [""] + groups

    for g1 in groups:
        win_row = [g1]
        goal_diff_row = [g1]
        for g2 in groups:
            games = matchup_win[(g1, g2)]['games']
            if games > 0:
                win_rate_val = matchup_win[(g1, g2)]['wins'] / games
                goal_diff_val = matchup_goal_diff[(g1, g2)]['total_goal_diff'] / matchup_goal_diff[(g1, g2)]['games']
                win_row.append(f"{win_rate_val:.5f}")
                goal_diff_row.append(f"{goal_diff_val:.5f}")
            else:
                win_row.append("N/A")
                goal_diff_row.append("N/A")
        win_matrix.append(win_row)
        goal_diff_matrix.append(goal_diff_row)

    print("\nMatchup Win Rate Matrix (rows: team1, columns: team2):")
    print(tabulate(win_matrix, headers=header))
    
    print("\nMatchup Average Goal Differential Matrix (rows: team1, columns: team2):")
    print(tabulate(goal_diff_matrix, headers=header))


import multiprocessing as mp
import torch
from tqdm import tqdm
import pygame
import math
import random
from tabulate import tabulate
from collections import defaultdict

def run_single_test_game(args):
    """
    Worker function that runs a single test game in a separate process.

    Args:
        args: tuple of
          (model1, model2, model1_idx, model2_idx, stage, gpu_id)

    Returns:
        A tuple containing:
          (model1_idx, model2_idx, score1, score2, avg_reward, ball_dist, ball_hits, stats)
    """
    (model1, model2, model1_idx, model2_idx, current_stage, gpu_id) = args

    # Assign device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model1.assign_device(device)
    model2.assign_device(device)

    # Create the environment and run the game
    # If you want to pass a filename or do logging, adapt as needed
    game = Game(log_name=None)  # or pass an appropriate filename
    stats = game.run(model1, model2, current_stage)

    return (
        model1_idx,
        model2_idx,
        stats.score[0],
        stats.score[1],
        stats.avg_reward,
        stats.ball_distance,
        stats.ball_hits,
        stats
    )


def test_PPO_parallel():
    """
    Adapted version of test_PPO that runs games in parallel for faster evaluation.
    Reuses as much code as possible from the original test_PPO.
    """
    # Initialize Elo environment and containers for models, ratings, names, and per-copy stats.
    elo_env = ELO()

    models = []
    ratings = []
    names = []
    # Each entry tracks: games, wins, ties, losses, cumulative stats for team1
    model_stats = []

    groups = ["HUGO", "PPO", "MAAC", "Random"]
    num_copies_per_model = 5  # We'll create 5 copies for each group, as in the original

    # ---- Create / Load Model Copies ----
    for _ in range(num_copies_per_model):
        hugo = HUGO(mode="test")
        hugo.load_model("HUGO_v15_sub6")
        models.append(hugo)
        names.append("HUGO")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    for _ in range(num_copies_per_model):
        ppo = PPOAgent(mode="test")
        ppo.load_model("PPO_v30_sub4")
        models.append(ppo)
        names.append("PPO")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    for _ in range(num_copies_per_model):
        maac = MAAC(mode="test")
        maac.load_model("MAAC_v5_sub20")
        models.append(maac)
        names.append("MAAC")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    for _ in range(num_copies_per_model):
        rand = RandomModel()
        models.append(rand)
        names.append("Random")
        ratings.append(elo_env.init_rating())
        model_stats.append({
            'games': 0,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'total_connectivity': 0.0,
            'total_frequency': 0.0,
            'total_pairwise': 0.0,
            'total_reward': 0.0,
            'total_goal_diff': 0.0
        })

    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()
    pygame.init()

    # We set stage to 4 as in your original test_PPO
    AI_PARAMS.current_stage = 4

    # Prepare matchup tracking
    matchup_win = {}
    matchup_goal_diff = {}
    for g1 in groups:
        for g2 in groups:
            matchup_win[(g1, g2)] = {'wins': 0, 'games': 0}
            matchup_goal_diff[(g1, g2)] = {'total_goal_diff': 0.0, 'games': 0}

    # Total number of games to run
    total_games = 10000
    batch_size = 8
    num_batches = math.ceil(total_games / batch_size)  # ADDED: Compute total number of batches

    # ADDED: Wrap the outer loop with tqdm for overall progress.
    with mp.Pool(processes=min(batch_size, mp.cpu_count())) as pool:
        epoch = 0
        for batch in tqdm(range(num_batches), desc="Testing PPO in Parallel"):  # ADDED: tqdm progress bar
            tasks = []
            current_batch_size = min(batch_size, total_games - epoch)
            for _ in range(current_batch_size):
                # Randomly select two distinct models.
                model1_idx = random.randint(0, len(models) - 1)
                while True:
                    model2_idx = random.randint(0, len(models) - 1)
                    if model2_idx != model1_idx:
                        break

                # GPU assignment.
                if torch.cuda.is_available():
                    gpu_id = epoch % torch.cuda.device_count()
                else:
                    gpu_id = 0

                args = (
                    models[model1_idx],
                    models[model2_idx],
                    model1_idx,
                    model2_idx,
                    AI_PARAMS.current_stage,
                    gpu_id
                )
                tasks.append(args)

            results = pool.map(run_single_test_game, tasks)

            for result in results:
                (m1_idx, m2_idx,
                 score1, score2,
                 avg_reward, ball_dist,
                 ball_hits, stats) = result

                if score1 > score2:
                    outcome = "win"
                elif score1 == score2:
                    outcome = "tie"
                else:
                    outcome = "loss"

                model_stats[m1_idx]['games'] += 1
                if outcome == "win":
                    model_stats[m1_idx]['wins'] += 1
                elif outcome == "tie":
                    model_stats[m1_idx]['ties'] += 1
                else:
                    model_stats[m1_idx]['losses'] += 1

                model_stats[m1_idx]['total_connectivity'] += stats.avg_connectivity_team1
                model_stats[m1_idx]['total_frequency'] += stats.avg_frequency_possession_distance
                model_stats[m1_idx]['total_pairwise'] += stats.avg_pairwise_distance
                model_stats[m1_idx]['total_reward'] += stats.avg_reward
                model_stats[m1_idx]['total_goal_diff'] += (score1 - score2)

                new_model1_rating, new_model2_rating = elo_env.calculate(
                    ratings[m1_idx],
                    ratings[m2_idx],
                    score1,
                    score2
                )
                ratings[m1_idx] = new_model1_rating
                ratings[m2_idx] = new_model2_rating

                group1 = names[m1_idx]
                group2 = names[m2_idx]
                if group1 in groups and group2 in groups:
                    matchup_win[(group1, group2)]['games'] += 1
                    if outcome == "win":
                        matchup_win[(group1, group2)]['wins'] += 1
                    matchup_goal_diff[(group1, group2)]['games'] += 1
                    matchup_goal_diff[(group1, group2)]['total_goal_diff'] += (score1 - score2)

                print(
                    f"Game: {epoch}, {names[m1_idx]} vs {names[m2_idx]}, "
                    f"Score: {score1} - {score2}, "
                    f"Avg Reward: {avg_reward:.2f}, Ball dist: {ball_dist:.2f}, "
                    f"Ball hits: {ball_hits}, "
                    f"ELO {ratings[m1_idx].mu:.2f} - {ratings[m2_idx].mu:.2f}"
                )

                epoch += 1
                if epoch >= total_games:
                    break

            if epoch % 20 == 0:
                print(f"--- ELO table after {epoch} games ---")
                table_data = [[names[i], f"{ratings[i].mu:.2f}"] for i in range(len(models))]
                print(tabulate(table_data, headers=["Model", "ELO"]))

            if epoch >= total_games:
                break

    # --------------------------
    # After all games, build final reports (same as original test_PPO)
    # --------------------------
    
    # A) Individual stats table (per copy)
    grouped_individual = defaultdict(list)
    aggregated = defaultdict(lambda: {
        'games': 0, 'wins': 0, 'ties': 0, 'losses': 0,
        'total_connectivity': 0.0, 'total_frequency': 0.0,
        'total_pairwise': 0.0, 'total_reward': 0.0, 'total_goal_diff': 0.0,
        'elo_sum': 0.0, 'copies': 0
    })

    for i, group in enumerate(names):
        games = model_stats[i]['games']
        wins = model_stats[i]['wins']
        ties = model_stats[i]['ties']
        losses = model_stats[i]['losses']
        if games > 0:
            win_rate = wins / games
            tie_rate = ties / games
            loss_rate = losses / games
            avg_conn = model_stats[i]['total_connectivity'] / games
            avg_freq = model_stats[i]['total_frequency'] / games
            avg_pairwise = model_stats[i]['total_pairwise'] / games
            avg_rwd = model_stats[i]['total_reward'] / games
            avg_goal_diff = model_stats[i]['total_goal_diff'] / games
        else:
            win_rate = tie_rate = loss_rate = avg_conn = avg_freq = avg_pairwise = avg_rwd = avg_goal_diff = 0.0

        grouped_individual[group].append([
            i, games, f"{win_rate:.5f}", f"{tie_rate:.5f}", f"{loss_rate:.5f}",
            f"{avg_conn:.5f}", f"{avg_freq:.5f}", f"{avg_pairwise:.5f}",
            f"{avg_rwd:.5f}", f"{avg_goal_diff:.5f}", f"{ratings[i].mu:.5f}"
        ])

        aggregated[group]['games'] += games
        aggregated[group]['wins'] += wins
        aggregated[group]['ties'] += ties
        aggregated[group]['losses'] += losses
        aggregated[group]['total_connectivity'] += model_stats[i]['total_connectivity']
        aggregated[group]['total_frequency'] += model_stats[i]['total_frequency']
        aggregated[group]['total_pairwise'] += model_stats[i]['total_pairwise']
        aggregated[group]['total_reward'] += model_stats[i]['total_reward']
        aggregated[group]['total_goal_diff'] += model_stats[i]['total_goal_diff']
        aggregated[group]['elo_sum'] += ratings[i].mu
        aggregated[group]['copies'] += 1

    # Print individual stats table for each model group
    for group, rows in grouped_individual.items():
        print(f"\nIndividual statistics for {group}:")
        print(tabulate(rows, headers=[
            "Copy", "Games", "Win Rate", "Tie Rate", "Loss Rate", 
            "Avg Connectivity", "Avg Freq Poss Dist", "Avg Pairwise Dist", 
            "Avg Reward", "Avg Goal Diff", "ELO"
        ]))

    # B) Aggregated table per model group
    aggregated_table = []
    for group, data in aggregated.items():
        if data['games'] > 0:
            agg_win_rate = data['wins'] / data['games']
            agg_tie_rate = data['ties'] / data['games']
            agg_loss_rate = data['losses'] / data['games']
            agg_conn = data['total_connectivity'] / data['games']
            agg_freq = data['total_frequency'] / data['games']
            agg_pairwise = data['total_pairwise'] / data['games']
            agg_reward = data['total_reward'] / data['games']
            agg_goal_diff = data['total_goal_diff'] / data['games']
        else:
            agg_win_rate = agg_tie_rate = agg_loss_rate = 0.0
            agg_conn = agg_freq = agg_pairwise = agg_reward = agg_goal_diff = 0.0

        avg_elo = data['elo_sum'] / data['copies'] if data['copies'] > 0 else 0.0

        aggregated_table.append([
            group, data['games'], f"{agg_win_rate:.5f}", f"{agg_tie_rate:.5f}", f"{agg_loss_rate:.5f}",
            f"{agg_conn:.5f}", f"{agg_freq:.5f}", f"{agg_pairwise:.5f}", f"{agg_reward:.5f}",
            f"{agg_goal_diff:.5f}", f"{avg_elo:.5f}"
        ])

    print("\nAggregated statistics per model group:")
    print(tabulate(aggregated_table, headers=[
        "Model", "Total Games", "Win Rate", "Tie Rate", "Loss Rate",
        "Avg Connectivity", "Avg Freq Poss Dist", "Avg Pairwise Dist", 
        "Avg Reward", "Avg Goal Diff", "Avg ELO"
    ]))

    # C) Build matchup matrices for the three groups (win rate, goal differential)
    win_matrix = []
    goal_diff_matrix = []
    header = [""] + groups

    for g1 in groups:
        win_row = [g1]
        goal_diff_row = [g1]
        for g2 in groups:
            games = matchup_win[(g1, g2)]['games']
            if games > 0:
                wr = matchup_win[(g1, g2)]['wins'] / games
                gd = matchup_goal_diff[(g1, g2)]['total_goal_diff'] / matchup_goal_diff[(g1, g2)]['games']
                win_row.append(f"{wr:.5f}")
                goal_diff_row.append(f"{gd:.5f}")
            else:
                win_row.append("N/A")
                goal_diff_row.append("N/A")

        win_matrix.append(win_row)
        goal_diff_matrix.append(goal_diff_row)

    print("\nMatchup Win Rate Matrix (rows: team1, columns: team2):")
    print(tabulate(win_matrix, headers=header))

    print("\nMatchup Average Goal Differential Matrix (rows: team1, columns: team2):")
    print(tabulate(goal_diff_matrix, headers=header))

    pygame.quit()


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
    elif ENV_PARAMS.MODE == "test_parallel":
        mp.set_start_method("spawn", force=True)
        test_PPO_parallel()
