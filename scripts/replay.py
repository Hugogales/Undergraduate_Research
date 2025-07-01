#!/usr/bin/env python3

"""
Game Replay Script for Soccer Environment

This script allows replaying recorded games with various playback options.
Supports loading specific recordings or finding the latest one automatically.
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

# ======== STANDARD IMPORT METHOD: sys.path manipulation ========
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from soccer_env.envs.soccer_env import env
from soccer_env.utils.game_recorder import GameReplayer, find_latest_recording
from soccer_env.core.renderer import SoccerRenderer


class GameReplayPlayer:
    """
    Handles replaying recorded games with various playback controls.
    """
    
    def __init__(self, field_width: float = 100.0, field_height: float = 60.0):
        """
        Initialize the replay player.
        
        Args:
            field_width: Field width for rendering
            field_height: Field height for rendering
        """
        self.field_width = field_width
        self.field_height = field_height
        self.renderer = None
        self.replayer = GameReplayer()
        
    def play_recording(
        self, 
        recording_path: str,
        playback_speed: float = 1.0,
        start_state: int = 0,
        end_state: Optional[int] = None,
        loop: bool = False,
        show_controls: bool = True
    ):
        """
        Play a recorded game.
        
        Args:
            recording_path: Path to the recording file
            playback_speed: Playback speed multiplier (1.0 = normal speed)
            start_state: State index to start from
            end_state: State index to end at (None = play to end)
            loop: Whether to loop the recording
            show_controls: Whether to show playback controls
        """
        # Load recording
        recording_data = self.replayer.load_recording(recording_path)
        if not recording_data:
            print("❌ Failed to load recording")
            return False
        
        states = recording_data.get("states", [])
        if not states:
            print("❌ No states found in recording")
            return False
        
        print(f"🎬 Playing recording with {len(states)} states")
        if show_controls:
            self._print_controls()
        
        # Initialize renderer
        self.renderer = SoccerRenderer(
            field_width=self.field_width,
            field_height=self.field_height
        )
        self.renderer.show_instructions = show_controls
        
        # Calculate playback timing
        base_frame_time = 1.0 / 60.0  # 60 FPS base
        frame_time = base_frame_time / playback_speed
        
        # Determine playback range
        if end_state is None:
            end_state = len(states)
        else:
            end_state = min(end_state, len(states))
        
        start_state = max(0, start_state)
        
        # Playback loop
        playing = True
        paused = False
        current_index = start_state
        
        try:
            while playing and current_index < end_state:
                loop_start_time = time.time()
                
                # Handle events
                events = self.renderer.handle_events()
                if events['quit']:
                    break
                
                # Handle keyboard controls
                keys = events.get('keys', {})
                if keys.get('space'):
                    paused = not paused
                    print(f"{'⏸️  Paused' if paused else '▶️  Playing'}")
                elif keys.get('left'):
                    current_index = max(start_state, current_index - 10)
                    print(f"⏪ Seek backward: {current_index}")
                elif keys.get('right'):
                    current_index = min(end_state - 1, current_index + 10)
                    print(f"⏩ Seek forward: {current_index}")
                elif keys.get('up'):
                    playback_speed = min(5.0, playback_speed * 1.5)
                    frame_time = base_frame_time / playback_speed
                    print(f"⏫ Speed up: {playback_speed:.1f}x")
                elif keys.get('down'):
                    playback_speed = max(0.1, playback_speed / 1.5)
                    frame_time = base_frame_time / playback_speed
                    print(f"⏬ Slow down: {playback_speed:.1f}x")
                elif keys.get('r'):
                    current_index = start_state
                    print("🔄 Restart")
                elif keys.get('escape'):
                    playing = False
                    break
                
                if not paused:
                    # Get current state
                    state_data = states[current_index]
                    
                    # Convert to game state object
                    game_state = self.replayer.create_game_state_from_recorded_state(state_data)
                    
                    # Render the state
                    self.renderer.render(game_state, throttle_fps=False)
                    
                    # Show state info
                    if show_controls and current_index % 60 == 0:  # Every second at 60fps
                        self._show_state_info(current_index, len(states), state_data, playback_speed)
                    
                    current_index += 1
                    
                    # Check for end of recording
                    if current_index >= end_state:
                        if loop:
                            current_index = start_state
                            print("🔄 Looping recording")
                        else:
                            print("✅ Recording finished")
                            break
                
                # Timing control
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n⏹️  Playback interrupted")
        finally:
            if self.renderer:
                self.renderer.close()
                self.renderer = None
        
        return True
    
    def _print_controls(self):
        """Print playback controls."""
        print("\n🎮 Playback Controls:")
        print("  SPACE     - Pause/Resume")
        print("  ←/→       - Seek backward/forward (10 frames)")
        print("  ↑/↓       - Speed up/slow down")
        print("  R         - Restart")
        print("  ESC       - Exit")
        print("  CTRL+C    - Force quit")
        print()
    
    def _show_state_info(self, current: int, total: int, state_data: dict, speed: float):
        """Show current playback information."""
        progress = (current / total) * 100
        score = f"{state_data['team_0_score']}-{state_data['team_1_score']}"
        time_remaining = state_data.get('time_remaining', 0)
        
        print(f"\r🎬 [{progress:5.1f}%] State {current:4d}/{total} | "
              f"Score: {score} | Time: {time_remaining:5.1f}s | "
              f"Speed: {speed:.1f}x", end="", flush=True)


def list_recordings(record_dir: str = "files/Recordings"):
    """List available recordings with details."""
    replayer = GameReplayer()
    recordings = replayer.list_recordings(record_dir)
    
    if not recordings:
        print(f"📁 No recordings found in {record_dir}")
        return
    
    print(f"📁 Available recordings in {record_dir}:")
    print("-" * 80)
    
    for i, recording_path in enumerate(recordings, 1):
        info = replayer.get_recording_info(recording_path)
        
        if "error" in info:
            print(f"{i:2d}. {info['file_name']} - ❌ Error: {info['error']}")
        else:
            score = info.get('final_score', [0, 0])
            start_time = info.get('start_time', 'Unknown')[:19].replace('T', ' ')
            states = info.get('states_count', 0)
            
            print(f"{i:2d}. {info['file_name']}")
            print(f"    📅 {start_time} | 🥅 {score[0]}-{score[1]} | 🎬 {states} states")
    
    print("-" * 80)


def main():
    """Main function for console entry point."""
    parser = argparse.ArgumentParser(description="Replay Soccer Environment Games")
    
    # Input options
    parser.add_argument("recording", nargs="?", help="Path to recording file (default: latest)")
    parser.add_argument("--list", "-l", action="store_true", 
                        help="List available recordings")
    parser.add_argument("--record-dir", type=str, default="files/Recordings",
                        help="Directory containing recordings (default: files/Recordings)")
    
    # Playback options
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start state index (default: 0)")
    parser.add_argument("--end", type=int, help="End state index (default: play to end)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the recording")
    parser.add_argument("--no-controls", action="store_true",
                        help="Hide playback controls")
    
    # Field dimensions
    parser.add_argument("--field-width", type=float, default=100.0,
                        help="Field width (default: 100.0)")
    parser.add_argument("--field-height", type=float, default=60.0,
                        help="Field height (default: 60.0)")
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_recordings(args.record_dir)
        return
    
    # Determine recording to play
    recording_path = args.recording
    if recording_path is None:
        # Find latest recording
        recording_path = find_latest_recording(args.record_dir)
        if recording_path is None:
            print(f"❌ No recordings found in {args.record_dir}")
            print(f"💡 Try: python scripts/replay.py --list")
            return
        print(f"🎬 Playing latest recording: {recording_path}")
    else:
        # Use provided path
        if not Path(recording_path).exists():
            print(f"❌ Recording not found: {recording_path}")
            return
    
    # Create replay player
    player = GameReplayPlayer(
        field_width=args.field_width,
        field_height=args.field_height
    )
    
    # Play the recording
    success = player.play_recording(
        recording_path=recording_path,
        playback_speed=args.speed,
        start_state=args.start,
        end_state=args.end,
        loop=args.loop,
        show_controls=not args.no_controls
    )
    
    if success:
        print("\n✅ Replay completed")
    else:
        print("\n❌ Replay failed")


if __name__ == "__main__":
    main() 