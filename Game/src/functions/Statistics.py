
import numpy as np
import os

from params import EnvironmentHyperparameters

class GameStatsOutput:
    def __init__(self, ball_hits, ball_distance, avg_reward,
                 avg_entropy, avg_connectivity_team1, avg_connectivity_team2,
                 avg_pairwise_distance, avg_frequency_possession_hit,
                avg_frequency_possession_distance, score):
        self.ball_hits = ball_hits
        self.ball_distance = ball_distance
        self.avg_reward = avg_reward
        self.avg_entropy = avg_entropy
        self.avg_connectivity_team1 = avg_connectivity_team1
        self.avg_connectivity_team2 = avg_connectivity_team2
        self.avg_pairwise_distance = avg_pairwise_distance
        self.avg_frequency_possession_hit = avg_frequency_possession_hit
        self.avg_frequency_possession_distance = avg_frequency_possession_distance
        self.score = score
    
    def print(self):
        print("Ball hits: ", self.ball_hits)
        print("Ball distance: ", self.ball_distance)
        print("Average reward: ", self.avg_reward)
        print("Average entropy: ", self.avg_entropy)
        print("Average connectivity team 1: ", self.avg_connectivity_team1)
        print("Average connectivity team 2: ", self.avg_connectivity_team2)
        print("Average pairwise distance: ", self.avg_pairwise_distance)
        print("Average frequency possession hit: ", self.avg_frequency_possession_hit)
        print("Average frequency possession distance: ", self.avg_frequency_possession_distance)
        print("Score: ", self.score)

class GameStats:
    """Track statistics for Alien Invasion."""

    def __init__(self, game):
        """Initialize statistics."""
        self.game = game
        self.ENV_PARAMS = EnvironmentHyperparameters()

        self.avg_pairwise_distances = []
        self.connectivity_team1 = []
        self.connectivity_team2 = []
        self.posession_hit = []
        self.posession_distance = []
        self.rewards = []
        self.ball_distances = []
        self.entropies = []
    
    def calculate_pairwise_distances(self, positions):
        """Calculate the pairwise distances between players."""
        distances = []
        positions = np.array(positions)
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
        
        distance = sum(distances) / len(distances)
        return distance
    
    def calculate_connections(self, positions_team1, positions_team2):
        """
        Calculate the number of unbroken connections between players on the same team.
        A connection exists when players are within a certain distance range and no 
        opponent or teammate is positioned in between.
        """
        import numpy as np
    
        min_distance = 50
        max_distance = 700
        blocking_threshold = 40
        
        positions_team1 = np.array(positions_team1)
        positions_team2 = np.array(positions_team2)
    
        def count_team_connections(team_positions, opponent_positions):
            connections = 0
            for i in range(len(team_positions)):
                for j in range(i + 1, len(team_positions)):
                    p1 = team_positions[i]
                    p2 = team_positions[j]
    
                    # Distance must be within the valid range
                    dist_p1_p2 = np.linalg.norm(p1 - p2)
                    if not (min_distance <= dist_p1_p2 <= max_distance):
                        continue
    
                    # Combine the entire pitch’s positions from both teams
                    # so a teammate in the path also blocks the connection
                    all_positions = np.concatenate((team_positions, opponent_positions), axis=0)
    
                    connection_blocked = False
                    line_length = np.linalg.norm(p2 - p1)
                    if line_length == 0:
                        # p1 and p2 are the exact same position; skip
                        continue
    
                    line_direction = (p2 - p1) / line_length
    
                    for p3 in all_positions:
                        # Skip if p3 is exactly one of the endpoints
                        if np.array_equal(p3, p1) or np.array_equal(p3, p2):
                            continue
    
                        v = p3 - p1
                        projection_length = np.dot(v, line_direction)
    
                        # Check only if p3 lies between p1 and p2
                        if 0 <= projection_length <= line_length:
                            projection_point = p1 + projection_length * line_direction
                            perp_distance = np.linalg.norm(p3 - projection_point)
                            if perp_distance < blocking_threshold:
                                # Something in the path blocks the connection
                                connection_blocked = True
                                break
    
                    if not connection_blocked:
                        connections += 1
    
            return connections
    
        # Count within-team connections, still blocked by anyone in the path
        team1_connections = count_team_connections(positions_team1, positions_team2)
        team2_connections = count_team_connections(positions_team2, positions_team1)
    
        max_connections = len(positions_team1) * (len(positions_team1) - 1) / 2
        team1_connectivity = team1_connections / max_connections if max_connections else 0
    
        max_connections = len(positions_team2) * (len(positions_team2) - 1) / 2
        team2_connectivity = team2_connections / max_connections if max_connections else 0
    
        return team1_connectivity, team2_connectivity


    def calculate_closest_player_to_ball(self, positions_team1, positions_team2, ball_positions):
        """
        Calculate the player closest to the ball.
        """
        # Calculate the distance between each player and the ball
        distances_team1 = [np.linalg.norm(np.array(ball_positions[0]) - np.array(pos)) for pos in positions_team1]
        distances_team2 = [np.linalg.norm(np.array(ball_positions[0]) - np.array(pos)) for pos in positions_team2]

        # Find the player with the minimum distance to the ball
        min_distance_team1 = min(distances_team1)
        min_distance_team2 = min(distances_team2)
    
        team_id = 0
        player_id = 0   
        # Determine the closest player
        if min_distance_team1 < min_distance_team2:
            closest_player = np.argmin(distances_team1)
            team_id = 1
            player_id = closest_player

        
    
        return (team_id, player_id)

    def calculate_stats(self, reward, entropy, ball_dist):
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.ball_distances.append(ball_dist)

        positions_team1 = []
        positions_team2 = []
        ball_positions = []

        for player in self.game.team_1:
            positions_team1.append(player.position)

        for player in self.game.team_2:
            positions_team2.append(player.position)

        ball_positions.append(self.game.ball.position)

        # Calculate avg pairwise distances between players in team 1
        if len(positions_team1) > 1:
            self.avg_pairwise_distances.append(self.calculate_pairwise_distances(positions_team1))

            # calculate number of connections between players in team 1 
            connectivy_team1, connectivity_team2 = self.calculate_connections(positions_team1, positions_team2)
            self.connectivity_team1.append(connectivy_team1)
            self.connectivity_team2.append(connectivity_team2)

        # Calculate the hit possession
        self.last_hit_player_id = self.game.ball.last_hit_player_id
        self.posession_hit.append(self.last_hit_player_id)

        # Calculate the closest player to the ball
        self.posession_distance.append(self.calculate_closest_player_to_ball(positions_team1, positions_team2, ball_positions))

    def final(self):
        ball_hits = self.game.ball_hits

        ball_distance = sum(self.ball_distances)
        avg_reward = 100 * sum(self.rewards) / (self.ENV_PARAMS.NUMBER_OF_PLAYERS * 2 * self.ENV_PARAMS.FPS * self.ENV_PARAMS.GAME_DURATION)
        avg_entropy = np.mean(np.array(self.entropies))
        if len(self.avg_pairwise_distances) == 0:
            avg_pairwise_distance = 0
            avg_connectivity_team1 = 0
            avg_connectivity_team2 = 0
        else:
            avg_connectivity_team1 = sum(self.connectivity_team1) / len(self.connectivity_team1)
            avg_connectivity_team2 = sum(self.connectivity_team2) / len(self.connectivity_team2)
            avg_pairwise_distance = sum(self.avg_pairwise_distances) / len(self.avg_pairwise_distances)
        
        frequency_possession_hit = 0
        frequency_possession_distance = 0

        # count number of times possesion changes
        for i in range(1, len(self.posession_hit)):
            if self.posession_hit[i] != self.posession_hit[i - 1]:
                frequency_possession_hit += 1
            
            if self.posession_distance[i] != self.posession_distance[i - 1]:
                frequency_possession_distance += 1
        
        avg_frequency_possession_hit = frequency_possession_hit / len(self.posession_hit)
        avg_frequency_possession_distance = frequency_possession_distance / len(self.posession_distance)

        return GameStatsOutput(
                ball_hits=ball_hits,
                ball_distance=ball_distance,
                avg_reward=avg_reward,
                avg_entropy=avg_entropy,
                avg_connectivity_team1=avg_connectivity_team1,  
                avg_connectivity_team2=avg_connectivity_team2,
                avg_pairwise_distance=avg_pairwise_distance,
                avg_frequency_possession_hit=avg_frequency_possession_hit,
                avg_frequency_possession_distance=avg_frequency_possession_distance,
                score=[self.game.score_team1, self.game.score_team2]
            )

# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import json
from datetime import datetime
import numpy as np
        
class StatsHistoryViewer:
    def __init__(self, model_name):
        self.model_name = model_name
        
        # Create a new folder for the model
        self.folder = "files/stats/" + model_name + "/"
        os.makedirs(self.folder, exist_ok=True)
        self.stats = []
        self.elos = []
        
        
        # Store libraries as instance attributes
        self.plt = plt
        self.json = json
        self.datetime = datetime
        
        # Set the style for plots
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'team1': '#FF0000',  # Red
            'team2': '#FFD700',  # Yellow
            'background': '#FFFFFF',
            'text': '#333333',
            'grid': '#DDDDDD'
        }

    def add(self, stats):
        self.stats.append(stats)
    
    def add_elo(self, elo):
        self.elos.append(elo)

    def combine_last_N(self, N):
        """Combine the last N episodes into a single GameStatsOutput object."""
        last_N = self.stats[-N:]
        
        # avg all the stats
        ball_hits = sum(s.ball_hits for s in last_N) / N
        ball_distance = sum(s.ball_distance for s in last_N) / N
        avg_reward = sum(s.avg_reward for s in last_N) / N
        avg_entropy = sum(s.avg_entropy for s in last_N) / N
        avg_connectivity_team1 = sum(s.avg_connectivity_team1 for s in last_N) / N
        avg_connectivity_team2 = sum(s.avg_connectivity_team2 for s in last_N) / N
        avg_pairwise_distance = sum(s.avg_pairwise_distance for s in last_N) / N
        avg_frequency_possession_hit = sum(s.avg_frequency_possession_hit for s in last_N) / N
        avg_frequency_possession_distance = sum(s.avg_frequency_possession_distance for s in last_N) / N
        score = [sum(s.score[i] for s in last_N) / N for i in range(2)]

        self.stats = self.stats[:-N]
        self.stats.append(GameStatsOutput(
            ball_hits=ball_hits,
            ball_distance=ball_distance,
            avg_reward=avg_reward,
            avg_entropy=avg_entropy,
            avg_connectivity_team1=avg_connectivity_team1,
            avg_connectivity_team2=avg_connectivity_team2,
            avg_pairwise_distance=avg_pairwise_distance,
            avg_frequency_possession_hit=avg_frequency_possession_hit,
            avg_frequency_possession_distance=avg_frequency_possession_distance,
            score=score
        ))
        
    def _to_serializable(self, obj):
        """Convert GameStatsOutput objects and NumPy types to JSON serializable objects."""
        
        if isinstance(obj, GameStatsOutput):
            return {
                'ball_hits': self._to_serializable(obj.ball_hits),
                'ball_distance': self._to_serializable(obj.ball_distance),
                'avg_reward': self._to_serializable(obj.avg_reward),
                'avg_entropy': self._to_serializable(obj.avg_entropy),
                'avg_connectivity_team1': self._to_serializable(obj.avg_connectivity_team1),
                'avg_connectivity_team2': self._to_serializable(obj.avg_connectivity_team2),
                'avg_pairwise_distance': self._to_serializable(obj.avg_pairwise_distance),
                'avg_frequency_possession_hit': self._to_serializable(obj.avg_frequency_possession_hit),
                'avg_frequency_possession_distance': self._to_serializable(obj.avg_frequency_possession_distance),
                'score': [self._to_serializable(s) for s in obj.score]
            }
        # Convert NumPy types to standard Python types
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def _calculate_summary_stats(self):
        """Calculate summary statistics across all episodes."""
        if not self.stats:
            return {}
                    
        summary = {
            'total_episodes': len(self.stats),
            'total_goals_team1': sum(s.score[0] for s in self.stats),
            'total_goals_team2': sum(s.score[1] for s in self.stats),
            'wins_team1': sum(1 for s in self.stats if s.score[0] > s.score[1]),
            'wins_team2': sum(1 for s in self.stats if s.score[1] > s.score[0]),
            'draws': sum(1 for s in self.stats if s.score[0] == s.score[1]),
            'metrics': {}
        }
                
        # Win percentage
        summary['win_rate_team1'] = summary['wins_team1'] / summary['total_episodes'] * 100
        summary['win_rate_team2'] = summary['wins_team2'] / summary['total_episodes'] * 100
                
        # Calculate min, max, avg for each numerical metric
        metrics = [
            'ball_hits', 'ball_distance', 'avg_reward', 'avg_entropy',
            'avg_connectivity_team1', 'avg_connectivity_team2', 'avg_pairwise_distance',
            'avg_frequency_possession_hit', 'avg_frequency_possession_distance'
        ]
                
        for metric in metrics:
            values = [getattr(s, metric) for s in self.stats]
            summary['metrics'][metric] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'trend': 'improving' if len(values) > 5 and sum(values[-5:]) / 5 > sum(values[:5]) / 5 else 
                            'declining' if len(values) > 5 and sum(values[-5:]) / 5 < sum(values[:5]) / 5 else 'stable'
            }
                
        # Add ELO metrics if available
        if self.elos:
            elo_metrics = {}
            if len(self.elos) > 0:
                # Extract ELO values for team1 and team2
                elos_team1 = [e[0].mu for e in self.elos]
                elos_team2 = [e[1].mu for e in self.elos]
                        
                # Calculate uncertainty (sigma)
                sigmas_team1 = [e[0].sigma for e in self.elos]
                sigmas_team2 = [e[1].sigma for e in self.elos]
                        
                elo_metrics['team1'] = {
                    'start': elos_team1[0] if elos_team1 else 0,
                    'current': elos_team1[-1] if elos_team1 else 0,
                    'min': min(elos_team1) if elos_team1 else 0,
                    'max': max(elos_team1) if elos_team1 else 0,
                    'change': elos_team1[-1] - elos_team1[0] if len(elos_team1) > 1 else 0,
                    'uncertainty': sigmas_team1[-1] if sigmas_team1 else 0
                }
                        
                elo_metrics['team2'] = {
                    'start': elos_team2[0] if elos_team2 else 0,
                    'current': elos_team2[-1] if elos_team2 else 0,
                    'min': min(elos_team2) if elos_team2 else 0,
                    'max': max(elos_team2) if elos_team2 else 0,
                    'change': elos_team2[-1] - elos_team2[0] if len(elos_team2) > 1 else 0,
                    'uncertainty': sigmas_team2[-1] if sigmas_team2 else 0
                }
                        
                # ELO trend
                elo_metrics['team1']['trend'] = 'improving' if elo_metrics['team1']['change'] > 0 else 'declining' if elo_metrics['team1']['change'] < 0 else 'stable'
                elo_metrics['team2']['trend'] = 'improving' if elo_metrics['team2']['change'] > 0 else 'declining' if elo_metrics['team2']['change'] < 0 else 'stable'
                        
                summary['elo'] = elo_metrics
                    
        return summary
            
    def _create_plots(self):
        """Generate all plots from the stats history."""
        if not self.stats:
            return {}
                    
        plots = {}
        episodes = range(1, len(self.stats) + 1)
                
        # Create figure directory
        fig_dir = os.path.join(self.folder, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
                
        # 1. Team connectivity plot
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.plot(episodes, [s.avg_connectivity_team1 for s in self.stats], 
                color=self.colors['team1'], label='Team 1', linewidth=2)
        ax.plot(episodes, [s.avg_connectivity_team2 for s in self.stats], 
                color=self.colors['team2'], label='Team 2', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Connectivity')
        ax.set_title('Team Connectivity Over Time')
        ax.legend()
        ax.grid(True, color=self.colors['grid'])
        connectivity_path = os.path.join(fig_dir, 'connectivity.png')
        fig.savefig(connectivity_path)
        plots['connectivity'] = os.path.basename(connectivity_path)
        self.plt.close(fig)
                
        # 2. Score plot
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.bar(episodes, [s.score[0] for s in self.stats], 
                color=self.colors['team1'], label='Team 1', alpha=0.7, width=0.4)
        ax.bar([e + 0.4 for e in episodes], [s.score[1] for s in self.stats], 
                color=self.colors['team2'], label='Team 2', alpha=0.7, width=0.4)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Goals')
        ax.set_title('Match Scores by Episode')
        ax.legend()
        score_path = os.path.join(fig_dir, 'score.png')
        fig.savefig(score_path)
        plots['score'] = os.path.basename(score_path)
        self.plt.close(fig)
                
        # 3. Ball metrics plot
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.plot(episodes, [s.ball_hits for s in self.stats], 
                color='blue', label='Ball Hits', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Ball Hits')
        ax2 = ax.twinx()
        ax2.plot(episodes, [s.ball_distance for s in self.stats], 
                    color='green', label='Ball Distance', linewidth=2, linestyle='--')
        ax2.set_ylabel('Ball Distance')
        ax.set_title('Ball Metrics Over Time')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ball_metrics_path = os.path.join(fig_dir, 'ball_metrics.png')
        fig.savefig(ball_metrics_path)
        plots['ball_metrics'] = os.path.basename(ball_metrics_path)
        self.plt.close(fig)
                
        # 4. Reward and Entropy plot
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.plot(episodes, [s.avg_reward for s in self.stats], 
                color='purple', label='Average Reward', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax2 = ax.twinx()
        ax2.plot(episodes, [s.avg_entropy for s in self.stats], 
                    color='orange', label='Average Entropy', linewidth=2, linestyle='--')
        ax2.set_ylabel('Average Entropy')
        ax.set_title('Training Metrics Over Time')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        training_metrics_path = os.path.join(fig_dir, 'training_metrics.png')
        fig.savefig(training_metrics_path)
        plots['training_metrics'] = os.path.basename(training_metrics_path)
        self.plt.close(fig)
                
        # 5. Possession metrics
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.plot(episodes, [s.avg_frequency_possession_hit for s in self.stats], 
                color=self.colors['team1'], label='Hit-based Possession Changes', linewidth=2)
        ax.plot(episodes, [s.avg_frequency_possession_distance for s in self.stats], 
                color=self.colors['team2'], label='Distance-based Possession Changes', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Frequency')
        ax.set_title('Possession Changes Over Time')
        ax.legend()
        ax.grid(True, color=self.colors['grid'])
        possession_path = os.path.join(fig_dir, 'possession.png')
        fig.savefig(possession_path)
        plots['possession'] = os.path.basename(possession_path)
        self.plt.close(fig)
                
        # 6. Pairwise distance
        fig, ax = self.plt.subplots(figsize=(10, 6))
        ax.plot(episodes, [s.avg_pairwise_distance for s in self.stats], 
                color='teal', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Pairwise Distance')
        ax.set_title('Team Formation Spread Over Time')
        ax.grid(True, color=self.colors['grid'])
        distance_path = os.path.join(fig_dir, 'pairwise_distance.png')
        fig.savefig(distance_path)
        plots['pairwise_distance'] = os.path.basename(distance_path)
        self.plt.close(fig)
                
        # 7. NEW: ELO Rating plot
        if self.elos and len(self.elos) > 0:
            elo_episodes = range(1, len(self.elos) + 1)
            fig, ax = self.plt.subplots(figsize=(10, 6))
                    
            if isinstance(self.elos[0], tuple) and len(self.elos[0]) == 2:
                # Extract ratings
                elos_team1 = [e[0].mu for e in self.elos]
                elos_team2 = [e[1].mu for e in self.elos]
                        
                # Plot with confidence intervals (shaded regions)
                sigma_team1 = [e[0].sigma for e in self.elos]
                sigma_team2 = [e[1].sigma for e in self.elos]
                        
                # Plot the ratings
                ax.plot(elo_episodes, elos_team1, color=self.colors['team1'], 
                        label='Team 1 Rating', linewidth=2)
                ax.plot(elo_episodes, elos_team2, color=self.colors['team2'], 
                        label='Team 2 Rating', linewidth=2)
                        
                # Add confidence intervals (±3 sigma)
                ax.fill_between(elo_episodes, 
                                [mu - 3*sigma for mu, sigma in zip(elos_team1, sigma_team1)],
                                [mu + 3*sigma for mu, sigma in zip(elos_team1, sigma_team1)],
                                color=self.colors['team1'], alpha=0.2)
                ax.fill_between(elo_episodes, 
                                [mu - 3*sigma for mu, sigma in zip(elos_team2, sigma_team2)],
                                [mu + 3*sigma for mu, sigma in zip(elos_team2, sigma_team2)],
                                color=self.colors['team2'], alpha=0.2)
                        
            ax.set_xlabel('Episode')
            ax.set_ylabel('ELO Rating')
            ax.set_title('Team ELO Ratings Over Time')
            ax.legend()
            ax.grid(True, color=self.colors['grid'])
            elo_path = os.path.join(fig_dir, 'elo_ratings.png')
            fig.savefig(elo_path)
            plots['elo_ratings'] = os.path.basename(elo_path)
            self.plt.close(fig)
                
        return plots
            
    def _generate_html(self, summary, plots):
        """Generate a beautiful HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{model_name}} - Training Statistics</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f9f9f9;
                }
                .header {
                    background: linear-gradient(135deg, #FF0000, #FFD700);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                }
                h1, h2, h3 {
                    margin-top: 0;
                }
                .card {
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric {
                    display: inline-block;
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px;
                    min-width: 150px;
                    text-align: center;
                }
                .metric .value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #FF0000;
                }
                .metric .label {
                    font-size: 14px;
                    color: #666;
                }
                .plot {
                    margin-top: 20px;
                    text-align: center;
                }
                .plot img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                table, th, td {
                    border: 1px solid #ddd;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                .trend-improving {
                    color: green;
                }
                .trend-declining {
                    color: red;
                }
                .trend-stable {
                    color: blue;
                }
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    color: #666;
                    font-size: 14px;
                }
                .team1 {
                    color: #FF0000;
                }
                .team2 {
                    color: #FFD700;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{model_name}} - Training Statistics</h1>
                <p>Generated on {{timestamp}}</p>
            </div>
                    
            <div class="card">
                <h2>Summary</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="value">{{summary.total_episodes}}</div>
                        <div class="label">Total Episodes</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.wins_team1}}</div>
                        <div class="label">Team 1 Wins</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.wins_team2}}</div>
                        <div class="label">Team 2 Wins</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.draws}}</div>
                        <div class="label">Draws</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.win_rate_team1|round(1)}}%</div>
                        <div class="label">Team 1 Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.win_rate_team2|round(1)}}%</div>
                        <div class="label">Team 2 Win Rate</div>
                    </div>
                            
                    {% if summary.elo %}
                    <div class="metric">
                        <div class="value">{{summary.elo.team1.current|round(1)}}</div>
                        <div class="label">Team 1 ELO Rating</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{summary.elo.team2.current|round(1)}}</div>
                        <div class="label">Team 2 ELO Rating</div>
                    </div>
                    {% endif %}
                </div>
            </div>
                    
            <div class="card">
                <h2>Key Metrics Overview</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Average</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Trend</th>
                    </tr>
                    {% for metric, data in summary.metrics.items() %}
                    <tr>
                        <td>{{ metric|replace('_', ' ')|title }}</td>
                        <td>{{ data.avg|round(3) }}</td>
                        <td>{{ data.min|round(3) }}</td>
                        <td>{{ data.max|round(3) }}</td>
                        <td class="trend-{{ data.trend }}">{{ data.trend|title }}</td>
                    </tr>
                    {% endfor %}
                            
                    {% if summary.elo %}
                    <tr>
                        <td>Team 1 ELO Rating</td>
                        <td>{{ summary.elo.team1.current|round(1) }}</td>
                        <td>{{ summary.elo.team1.min|round(1) }}</td>
                        <td>{{ summary.elo.team1.max|round(1) }}</td>
                        <td class="trend-{{ summary.elo.team1.trend }}">{{ summary.elo.team1.trend|title }}</td>
                    </tr>
                    <tr>
                        <td>Team 2 ELO Rating</td>
                        <td>{{ summary.elo.team2.current|round(1) }}</td>
                        <td>{{ summary.elo.team2.min|round(1) }}</td>
                        <td>{{ summary.elo.team2.max|round(1) }}</td>
                        <td class="trend-{{ summary.elo.team2.trend }}">{{ summary.elo.team2.trend|title }}</td>
                    </tr>
                    {% endif %}
                </table>
            </div>
                    
            {% if latest %}
            <div class="card">
                <h2>Latest Game Results</h2>
                <h3>Score: <span class="team1">Team 1: {{ latest.score[0] }}</span> - <span class="team2">Team 2: {{ latest.score[1] }}</span></h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="value">{{ latest.ball_hits }}</div>
                        <div class="label">Ball Hits</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ latest.ball_distance|round(1) }}</div>
                        <div class="label">Ball Distance</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ latest.avg_reward|round(3) }}</div>
                        <div class="label">Avg Reward</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ latest.avg_entropy|round(3) }}</div>
                        <div class="label">Avg Entropy</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ latest.avg_connectivity_team1|round(3) }}</div>
                        <div class="label">Team 1 Connectivity</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ latest.avg_connectivity_team2|round(3) }}</div>
                        <div class="label">Team 2 Connectivity</div>
                    </div>
                            
                    {% if summary.elo %}
                    <div class="metric">
                        <div class="value">{{ summary.elo.team1.current|round(1) }}</div>
                        <div class="label">Team 1 ELO</div>
                    </div>
                    <div class="metric">
                        <div class="value">{{ summary.elo.team2.current|round(1) }}</div>
                        <div class="label">Team 2 ELO</div>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endif %}
                    
            <div class="card">
                <h2>Team Performance</h2>
                <div class="plot">
                    <h3>Team Score Comparison</h3>
                    <img src="figures/{{ plots.score }}" alt="Scores">
                </div>
                <div class="plot">
                    <h3>Team Connectivity</h3>
                    <img src="figures/{{ plots.connectivity }}" alt="Team Connectivity">
                </div>
                        
                {% if plots.elo_ratings %}
                <div class="plot">
                    <h3>ELO Ratings</h3>
                    <img src="figures/{{ plots.elo_ratings }}" alt="ELO Ratings">
                </div>
                {% endif %}
            </div>
                    
            <div class="card">
                <h2>Ball Metrics</h2>
                <div class="plot">
                    <h3>Ball Hits and Distance</h3>
                    <img src="figures/{{ plots.ball_metrics }}" alt="Ball Metrics">
                </div>
                <div class="plot">
                    <h3>Possession Changes</h3>
                    <img src="figures/{{ plots.possession }}" alt="Possession Changes">
                </div>
            </div>
                    
            <div class="card">
                <h2>Training Progress</h2>
                <div class="plot">
                    <h3>Reward and Entropy</h3>
                    <img src="figures/{{ plots.training_metrics }}" alt="Training Metrics">
                </div>
                <div class="plot">
                    <h3>Team Formation (Pairwise Distance)</h3>
                    <img src="figures/{{ plots.pairwise_distance }}" alt="Pairwise Distance">
                </div>
            </div>
                    
            <div class="footer">
                <p>Generated for {{model_name}} using UR Game Stats Viewer | {{timestamp}}</p>
            </div>
        </body>
        </html>
        """
                
        # Use Jinja2 for templating
        from jinja2 import Template
        template = Template(html_template)
                
        latest = self.stats[-1] if self.stats else None
        timestamp = self.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
        html_content = template.render(
            model_name=self.model_name,
            summary=summary,
            plots=plots,
            latest=latest,
            timestamp=timestamp
        )
                
        return html_content
    def update(self):
        """Generate updated statistics, plots, and HTML report."""
        if not self.stats:
            print("No stats available to update.")
            return
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats()
        
        # Create plots
        plots = self._create_plots()
        
        # Generate HTML
        html_content = self._generate_html(summary, plots)
        
        # Save HTML to file
        html_path = os.path.join(self.folder, 'stats_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save stats to JSON
        json_data = [self._to_serializable(stat) for stat in self.stats]
        json_path = os.path.join(self.folder, 'stats_history.json')
        with open(json_path, 'w') as f:
            self.json.dump(json_data, f, indent=2)
        
        print(f"Stats updated. Report available at: {html_path}")