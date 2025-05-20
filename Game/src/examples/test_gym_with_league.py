import os
import sys
import time

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pygame
from functions.ELO import ELO
from functions.league import League
from enviroment.GymAdapter import SoccerEnv
from params import EnvironmentHyperparameters, AIHyperparameters
from AI.ModelAdapter import ModelAdapter
from AI.HUGO import HUGO
from AI.PPO import PPOAgent
from AI.randmodel import RandomModel

def test_with_rendering():
    """
    Test the SoccerEnv with rendering enabled and League integration.
    This demonstrates how to use the gym adapter with different stages and models.
    """
    # Initialize pygame (needed for rendering)
    pygame.init()
    
    # Initialize parameters
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()
    
    # Enable rendering
    ENV_PARAMS.RENDER = True
    
    # Create ELO environment
    elo_env = ELO()
    
    # Create a model that will play as team 1
    player_model = HUGO(mode="test")
    if os.path.exists("HUGO_v100_sub0"):
        player_model.load_model("HUGO_v100_sub0")
    
    # Create a model adapter for the player model
    player_adapter = ModelAdapter(player_model)
    
    # Create a base model for opponents
    base_opponent = HUGO(mode="test")
    
    # Create a league with the base opponent
    league = League(elo_env, base_opponent)
    
    # Add some models to the league
    model1 = HUGO(mode="test")
    if os.path.exists("HUGO_v100_sub0"):
        model1.load_model("HUGO_v100_sub0")
    league.add_player(model1, elo_env.init_rating())
    
    model2 = PPOAgent(mode="test")
    if os.path.exists("PPO_v30_sub4"):
        model2.load_model("PPO_v30_sub4")
    league.add_player(model2, elo_env.init_rating())
    
    # Add a random model as well
    random_model = RandomModel()
    league.add_player(random_model, elo_env.init_rating())
    
    # Test different stages
    stages = [1, 2, 3, 4]
    
    for stage in stages:
        print(f"Testing Stage {stage}")
        AI_PARAMS.current_stage = stage
        
        # Configure environment for the current stage
        if stage <= 2:
            # Stages 1-2: Random opponent
            ENV_PARAMS.RANDOMIZE_PLAYERS = (stage == 2)
            env = SoccerEnv(current_stage=stage)
        else:
            # Stages 3-4: League opponent
            ENV_PARAMS.RANDOMIZE_PLAYERS = (stage == 3)
            env = SoccerEnv(current_stage=stage, league=league)
        
        # Run a few episodes
        for episode in range(3):
            obs = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 2000:  # Limit steps to avoid infinite loops
                # Get actions from the player's model through the adapter
                action, _ = player_adapter.act(obs)
                
                # Step the environment
                next_obs, reward, done, info = env.step(action)
                
                # Update observation
                obs = next_obs
                
                # Accumulate reward
                episode_reward += reward
                steps += 1
                
                # Show info every 100 steps
                if steps % 100 == 0:
                    print(f"Stage {stage}, Episode {episode}, Step {steps}, Score: {info['score_team1']} - {info['score_team2']}")
                
                # Slow down a bit for visualization
                time.sleep(0.01)
            
            # Episode finished
            print(f"Stage {stage}, Episode {episode} finished. Score: {info['score_team1']} - {info['score_team2']}, Reward: {episode_reward:.2f}")
            
            # If using league, update player model's rating based on outcome
            if stage >= 3 and info['competing_model_rating'] is not None:
                player_rating = elo_env.init_rating()  # This would normally track across episodes
                new_player_rating, _ = elo_env.calculate(
                    player_rating, 
                    info['competing_model_rating'],
                    info['score_team1'],
                    info['score_team2']
                )
                print(f"Player rating updated: {player_rating.mu:.2f} -> {new_player_rating.mu:.2f}")
        
        # Close the environment
        env.close()
    
    pygame.quit()
    print("Testing complete!")

def test_dynamic_model_changes():
    """
    Test changing competing models during training.
    """
    # Initialize pygame
    pygame.init()
    
    # Initialize parameters
    ENV_PARAMS = EnvironmentHyperparameters()
    AI_PARAMS = AIHyperparameters()
    
    # Enable rendering
    ENV_PARAMS.RENDER = True
    
    # Create a model that will play as team 1
    player_model = HUGO(mode="test")
    if os.path.exists("HUGO_v100_sub0"):
        player_model.load_model("HUGO_v100_sub0")
    
    # Create a model adapter for the player model
    player_adapter = ModelAdapter(player_model)
    
    # Create environment in stage 4 (competing model)
    env = SoccerEnv(current_stage=4)
    
    # Create some models to switch between
    models = []
    
    model1 = HUGO(mode="test")
    if os.path.exists("HUGO_v100_sub0"):
        model1.load_model("HUGO_v100_sub0")
    models.append(("HUGO", model1))
    
    model2 = PPOAgent(mode="test")
    if os.path.exists("PPO_v30_sub4"):
        model2.load_model("PPO_v30_sub4")
    models.append(("PPO", model2))
    
    models.append(("Random", RandomModel()))
    
    # Run episodes with different competing models
    for model_name, model in models:
        print(f"Testing against {model_name}")
        
        # Change the competing model
        env.set_competing_model(model)
        
        # Run an episode
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < 1000:
            # Get actions from the player's model through the adapter
            action, _ = player_adapter.act(obs)
            
            # Step the environment
            next_obs, reward, done, info = env.step(action)
            
            # Update observation
            obs = next_obs
            
            # Accumulate reward
            episode_reward += reward
            steps += 1
            
            # Show info every 100 steps
            if steps % 100 == 0:
                print(f"Competing model: {model_name}, Step {steps}, Score: {info['score_team1']} - {info['score_team2']}")
            
            # Slow down a bit for visualization
            time.sleep(0.01)
        
        # Episode finished
        print(f"Episode against {model_name} finished. Score: {info['score_team1']} - {info['score_team2']}, Reward: {episode_reward:.2f}")
    
    # Close the environment
    env.close()
    pygame.quit()
    print("Dynamic model changes testing complete!")

if __name__ == "__main__":
    print("Running test with rendering and league...")
    test_with_rendering()
    
    print("\nRunning test with dynamic model changes...")
    test_dynamic_model_changes() 