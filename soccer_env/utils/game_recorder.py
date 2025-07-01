"""
Game Recorder for Soccer Environment

This module handles recording game states during play and saving them
for later replay. It's compatible with both training and human play.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..core.entities import GameState, Player, Ball, Goal


class GameRecorder:
    """
    Records game states for later replay.
    """
    
    def __init__(self, record_dir: str = "files/Recordings"):
        """
        Initialize the game recorder.
        
        Args:
            record_dir: Directory to save recordings
        """
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        
        self.recording = False
        self.game_data = {}
        self.states = []
        self.metadata = {}
        
    def start_recording(
        self, 
        game_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Start recording a new game.
        
        Args:
            game_name: Optional name for the game recording
            metadata: Optional metadata about the game
            
        Returns:
            The filename of the recording
        """
        if game_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            game_name = f"game_{timestamp}"
        
        self.recording = True
        self.states = []
        self.metadata = metadata or {}
        self.metadata.update({
            "start_time": datetime.now().isoformat(),
            "version": "1.0",
            "format": "soccer_env_recording"
        })
        
        # Create the full recording data structure
        self.game_data = {
            "metadata": self.metadata,
            "states": self.states
        }
        
        self.filename = self.record_dir / f"{game_name}.json"
        
        print(f"🎬 Started recording game: {self.filename}")
        return str(self.filename)
    
    def record_state(self, game_state: GameState, actions: Optional[Dict] = None):
        """
        Record a single game state.
        
        Args:
            game_state: Current game state to record
            actions: Optional actions taken by agents this step
        """
        if not self.recording or game_state is None:
            return
        
        # Convert game state to serializable format
        state_data = {
            "episode_step": game_state.episode_step,
            "time_remaining": getattr(game_state, 'time_remaining', 0),
            "team_0_score": game_state.team_0_score,
            "team_1_score": game_state.team_1_score,
            "goal_scored": getattr(game_state, 'goal_scored', False),
            "ball_possession": getattr(game_state, 'ball_possession', None),
            
            # Ball data
            "ball": {
                "x": game_state.ball.x,
                "y": game_state.ball.y,
                "vx": game_state.ball.vx,
                "vy": game_state.ball.vy,
                "radius": getattr(game_state.ball, 'radius', 3),
            },
            
            # Players data
            "players": [],
            
            # Goals data
            "goals": []
        }
        
        # Record players
        for player in game_state.players:
            player_data = {
                "x": player.x,
                "y": player.y,
                "vx": player.vx,
                "vy": player.vy,
                "team_id": player.team_id,
                "player_id": getattr(player, 'player_id', 0),
                "radius": getattr(player, 'radius', 8),
            }
            state_data["players"].append(player_data)
        
        # Record goals
        for goal in game_state.goals:
            goal_data = {
                "x": goal.x,
                "y": goal.y,
                "width": goal.width,
                "height": goal.height,
                "team_id": getattr(goal, 'team_id', 0),
            }
            state_data["goals"].append(goal_data)
        
        # Add actions if provided
        if actions:
            state_data["actions"] = actions
        
        self.states.append(state_data)
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save the game to file.
        
        Returns:
            Path to saved file, or None if not recording
        """
        if not self.recording:
            return None
        
        self.recording = False
        
        # Add final metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_states"] = len(self.states)
        self.metadata["duration_states"] = len(self.states)
        
        # Final score
        if self.states:
            final_state = self.states[-1]
            self.metadata["final_score"] = [
                final_state["team_0_score"],
                final_state["team_1_score"]
            ]
        
        # Save to file
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.game_data, f, indent=2)
            
            print(f"💾 Game recording saved: {self.filename}")
            print(f"   States recorded: {len(self.states)}")
            print(f"   Final score: {self.metadata.get('final_score', 'N/A')}")
            
            return str(self.filename)
        except Exception as e:
            print(f"❌ Error saving recording: {e}")
            return None
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording
    
    def get_recording_info(self) -> Dict:
        """Get information about current recording."""
        return {
            "recording": self.recording,
            "states_recorded": len(self.states),
            "filename": str(self.filename) if hasattr(self, 'filename') else None,
            "metadata": self.metadata.copy()
        }


class GameReplayer:
    """
    Replays recorded games.
    """
    
    def __init__(self):
        """Initialize the game replayer."""
        pass
    
    def load_recording(self, recording_path: str) -> Dict:
        """
        Load a game recording from file.
        
        Args:
            recording_path: Path to the recording file
            
        Returns:
            Game recording data
        """
        try:
            with open(recording_path, 'r') as f:
                data = json.load(f)
            
            print(f"📂 Loaded recording: {recording_path}")
            if "metadata" in data:
                metadata = data["metadata"]
                print(f"   Game info: {metadata.get('start_time', 'Unknown time')}")
                print(f"   States: {metadata.get('total_states', len(data.get('states', [])))}")
                print(f"   Final score: {metadata.get('final_score', 'N/A')}")
            
            return data
        except FileNotFoundError:
            print(f"❌ Recording file not found: {recording_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing recording file: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error loading recording: {e}")
            return {}
    
    def list_recordings(self, record_dir: str = "files/Recordings") -> List[str]:
        """
        List available recordings.
        
        Args:
            record_dir: Directory to search for recordings
            
        Returns:
            List of recording file paths
        """
        record_path = Path(record_dir)
        if not record_path.exists():
            return []
        
        recordings = []
        for file in record_path.glob("*.json"):
            recordings.append(str(file))
        
        recordings.sort()  # Sort by filename (includes timestamp)
        return recordings
    
    def get_recording_info(self, recording_path: str) -> Dict:
        """
        Get information about a recording without loading all states.
        
        Args:
            recording_path: Path to the recording file
            
        Returns:
            Recording metadata
        """
        try:
            with open(recording_path, 'r') as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            states_count = len(data.get("states", []))
            
            return {
                "file_path": recording_path,
                "file_name": Path(recording_path).name,
                "metadata": metadata,
                "states_count": states_count,
                "start_time": metadata.get("start_time", "Unknown"),
                "final_score": metadata.get("final_score", [0, 0]),
                "duration": metadata.get("duration_states", states_count),
            }
        except Exception as e:
            return {
                "file_path": recording_path,
                "error": str(e)
            }
    
    def create_game_state_from_recorded_state(self, state_data: Dict) -> GameState:
        """
        Create a GameState object from recorded state data.
        
        Args:
            state_data: Recorded state data
            
        Returns:
            GameState object
        """
        # Create ball
        ball_data = state_data["ball"]
        ball = Ball(
            x=ball_data["x"],
            y=ball_data["y"],
            radius=ball_data.get("radius", 3)
        )
        ball.vx = ball_data["vx"]
        ball.vy = ball_data["vy"]
        
        # Create players
        players = []
        for player_data in state_data["players"]:
            player = Player(
                x=player_data["x"],
                y=player_data["y"],
                team_id=player_data["team_id"],
                radius=player_data.get("radius", 8)
            )
            player.vx = player_data["vx"]
            player.vy = player_data["vy"]
            player.player_id = player_data.get("player_id", 0)
            players.append(player)
        
        # Create goals
        goals = []
        for goal_data in state_data["goals"]:
            goal = Goal(
                x=goal_data["x"],
                y=goal_data["y"],
                width=goal_data["width"],
                height=goal_data["height"],
                team_id=goal_data.get("team_id", 0)
            )
            goals.append(goal)
        
        # Create game state
        game_state = GameState(
            ball=ball,
            players=players,
            goals=goals,
            episode_step=state_data["episode_step"],
            team_0_score=state_data["team_0_score"],
            team_1_score=state_data["team_1_score"]
        )
        
        # Set additional attributes
        game_state.time_remaining = state_data.get("time_remaining", 0)
        game_state.goal_scored = state_data.get("goal_scored", False)
        game_state.ball_possession = state_data.get("ball_possession", None)
        
        return game_state


def find_latest_recording(record_dir: str = "files/Recordings") -> Optional[str]:
    """
    Find the most recent recording.
    
    Args:
        record_dir: Directory to search for recordings
        
    Returns:
        Path to the latest recording, or None if none found
    """
    replayer = GameReplayer()
    recordings = replayer.list_recordings(record_dir)
    
    if recordings:
        # Sort by modification time (most recent first)
        recordings.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return recordings[0]
    
    return None


def quick_record_example():
    """
    Example of how to use the recording system.
    This is for documentation purposes.
    """
    # This would typically be done within the environment
    recorder = GameRecorder()
    
    # Start recording
    filename = recorder.start_recording(
        game_name="example_game",
        metadata={
            "players_per_team": 2,
            "game_mode": "default",
            "human_players": True
        }
    )
    
    # During game loop (this would be in the environment):
    # recorder.record_state(current_game_state, current_actions)
    
    # Stop recording
    saved_file = recorder.stop_recording()
    
    return saved_file 