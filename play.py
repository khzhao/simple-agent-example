#!/usr/bin/env python3
"""
Play 2048 with a trained model.

Example usage:
    python play.py --checkpoint checkpoints/checkpoint_ep100.pt
    python play.py --checkpoint checkpoints/checkpoint_ep100.pt --num-games 10
"""

import argparse
from pathlib import Path

from simple_agent_example.inference import GamePlayer
from simple_agent_example.training import TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play 2048 with trained model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=1000,
        help="Maximum moves per game (default: 1000)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between moves in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visual rendering"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model (should match training config)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print("\n" + "=" * 60)
    print("2048 GAME PLAYER")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Number of games: {args.num_games}")
    print(f"Max moves per game: {args.max_moves}")
    print("=" * 60 + "\n")

    # Create config
    config = TrainingConfig(base_model=args.base_model)

    # Create player
    try:
        player = GamePlayer(
            checkpoint_path=args.checkpoint,
            config=config,
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

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
