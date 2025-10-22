"""
Inference module for playing 2048 with trained model.

Uses Tinker's sampling client for inference with synchronous API.
"""

import os
import time
from pathlib import Path
from typing import Optional

try:
    import tinker
    from tinker import ServiceClient
    from tinker.types import SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: required packages not installed")
    tinker = None
    ServiceClient = None
    SamplingParams = None
    AutoTokenizer = None

from simple_agent_example.envs import Game2048Env
from simple_agent_example.models import ActionParser, TextStateEncoder
from simple_agent_example.training.config import TrainingConfig


class GamePlayer:
    """
    Play 2048 using a trained LORA model.

    This class handles inference and visualization of the trained agent.
    Uses Tinker's SamplingClient with synchronous API.
    """

    def __init__(
        self,
        checkpoint_name: Optional[str] = None,
        config: Optional[TrainingConfig] = None,
        render: bool = True,
    ):
        """
        Initialize game player.

        Args:
            checkpoint_name: Name/label of trained model checkpoint (not a file path)
            config: Training config (uses defaults if None)
            render: Whether to render the game
        """
        self.config = config or TrainingConfig()
        self.checkpoint_name = checkpoint_name
        self.render = render

        # Initialize environment and utilities
        self.env = Game2048Env(render_mode="human" if render else None)
        self.encoder = TextStateEncoder()
        self.parser = ActionParser()

        # Tinker clients
        self.service_client = None
        self.sampling_client = None
        self.tokenizer = None

        # Setup Tinker
        self._setup_tinker()

    def _setup_tinker(self):
        """Initialize Tinker client."""
        if tinker is None:
            raise ImportError("Tinker not installed. Install with: pip install tinker")

        # Get API key
        api_key = self.config.tinker_api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError("TINKER_API_KEY not found")

        # Create service client
        self.service_client = ServiceClient(api_key=api_key)

        # Load tokenizer
        print(f"Loading tokenizer for {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        # Create sampling client
        if self.checkpoint_name:
            # Load from checkpoint
            print(f"Loading model from checkpoint: {self.checkpoint_name}")
            # Create training client
            training_client = self.service_client.create_lora_training_client(
                self.config.base_model,
                rank=self.config.rank,
            )

            # Load the checkpoint
            training_client.load_state(self.checkpoint_name).result()

            # Save weights and get sampling client
            self.sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"{self.checkpoint_name}-sampler"
            )
        else:
            # Use base model
            print(f"Using base model: {self.config.base_model}")
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=self.config.base_model
            )

    def play_game(
        self,
        max_moves: int = 1000,
        delay: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """
        Play one complete game.

        Args:
            max_moves: Maximum number of moves
            delay: Delay between moves (seconds) for visualization
            verbose: Print detailed information

        Returns:
            Dictionary with game statistics
        """
        obs, info = self.env.reset()
        done = False
        move_count = 0

        if verbose:
            print("\n" + "=" * 50)
            print("Starting new game!")
            print("=" * 50)

        while not done and move_count < max_moves:
            # Render current state
            if self.render:
                self.env.render()
                if delay > 0:
                    time.sleep(delay)

            # Get state text
            state_text = self.encoder.encode_state(
                grid=obs,
                score=info["score"],
                move_count=move_count,
            )

            # Sample action from model
            try:
                # Tokenize the state text
                tokens = self.tokenizer.encode(state_text, return_tensors=None)

                # Create ModelInput from tokens
                model_input = tinker.types.ModelInput.from_ints(tokens)

                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                )

                # Sample from the model
                result = self.sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                ).result()

                # Get the output tokens
                output_tokens = result.sequences[0].tokens

                # Decode tokens to text
                model_output = self.tokenizer.decode(
                    output_tokens, skip_special_tokens=True
                )
                action = self.parser.parse_action(model_output)

                if verbose:
                    print(f"\nMove {move_count + 1}:")
                    print(f"  Model output: {model_output[:100]}...")
                    print(f"  Parsed action: {self.parser.action_to_text(action)}")

            except Exception as e:
                if verbose:
                    print(f"Error getting model action: {e}")
                action = -1

            # Fallback to random if parsing failed
            if action == -1:
                action = self.env.action_space.sample()
                if verbose:
                    print(
                        f"  Using random action: {self.parser.action_to_text(action)}"
                    )

            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if verbose and reward != 0:
                print(f"  Reward: {reward:.2f}")

            move_count += 1

        # Final render
        if self.render:
            self.env.render()

        # Game summary
        game_won = info["max_tile"] >= 2048

        stats = {
            "final_score": info["score"],
            "max_tile": info["max_tile"],
            "moves": move_count,
            "won": game_won,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("Game Over!")
            print(f"Final Score: {stats['final_score']}")
            print(f"Max Tile: {stats['max_tile']}")
            print(f"Total Moves: {stats['moves']}")
            print(f"Won (reached 2048): {'Yes' if game_won else 'No'}")
            print("=" * 50 + "\n")

        return stats

    def play_multiple_games(
        self,
        num_games: int = 10,
        max_moves: int = 1000,
        delay: float = 0.0,
        verbose: bool = False,
    ) -> dict:
        """
        Play multiple games and return aggregate statistics.

        Args:
            num_games: Number of games to play
            max_moves: Max moves per game
            delay: Delay between moves
            verbose: Print detailed info for each game

        Returns:
            Dictionary with aggregate statistics
        """
        print(f"\nPlaying {num_games} games...")

        all_stats = []
        for i in range(num_games):
            print(f"\n--- Game {i + 1}/{num_games} ---")
            stats = self.play_game(
                max_moves=max_moves,
                delay=delay,
                verbose=verbose,
            )
            all_stats.append(stats)

        # Compute aggregate statistics
        scores = [s["final_score"] for s in all_stats]
        max_tiles = [s["max_tile"] for s in all_stats]
        moves = [s["moves"] for s in all_stats]
        wins = sum(1 for s in all_stats if s["won"])

        aggregate = {
            "num_games": num_games,
            "avg_score": sum(scores) / num_games,
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_max_tile": sum(max_tiles) / num_games,
            "best_tile": max(max_tiles),
            "avg_moves": sum(moves) / num_games,
            "win_rate": wins / num_games,
            "wins": wins,
        }

        print("\n" + "=" * 50)
        print("AGGREGATE STATISTICS")
        print("=" * 50)
        print(f"Games played: {aggregate['num_games']}")
        print(f"Average score: {aggregate['avg_score']:.2f}")
        print(f"Best score: {aggregate['max_score']}")
        print(f"Average max tile: {aggregate['avg_max_tile']:.2f}")
        print(f"Best tile achieved: {aggregate['best_tile']}")
        print(f"Average moves: {aggregate['avg_moves']:.2f}")
        print(f"Win rate: {aggregate['win_rate']:.2%} ({wins}/{num_games})")
        print("=" * 50 + "\n")

        return aggregate


def main():
    """Demo function to play games with trained model."""
    import argparse

    parser = argparse.ArgumentParser(description="Play 2048 with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint name/label (e.g., 'checkpoint-ep000100'), not a file path",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=1000,
        help="Maximum moves per game",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between moves (seconds)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information",
    )

    args = parser.parse_args()

    # Create player
    player = GamePlayer(
        checkpoint_name=args.checkpoint,
        render=not args.no_render,
    )

    # Play games
    if args.num_games == 1:
        player.play_game(
            max_moves=args.max_moves,
            delay=args.delay,
            verbose=True,
        )
    else:
        player.play_multiple_games(
            num_games=args.num_games,
            max_moves=args.max_moves,
            delay=args.delay,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
