#!/usr/bin/env python3
"""
Main training script for 2048 RL agent with Tinker API.

Example usage:
    python train.py --episodes 1000 --batch-size 8
    python train.py --config config.json
"""

import argparse
import json
from pathlib import Path

from simple_agent_example.training import TrainingConfig, RLTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 2048 RL agent with Tinker LORA API"
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training updates (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )

    # Model parameters
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model for LORA training"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LORA rank (default: 8)"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="2048-rl",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints"
    )

    # WandB settings
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="2048-tinker-rl",
        help="WandB project name"
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Evaluate every N episodes (default: 25)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save checkpoint every N episodes (default: 50)"
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (overrides other args)"
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> TrainingConfig:
    """Load training config from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return TrainingConfig(**config_dict)


def create_config_from_args(args) -> TrainingConfig:
    """Create training config from command line arguments."""
    return TrainingConfig(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_model=args.base_model,
        rank=args.lora_rank,  # Command line arg is still --lora-rank for compatibility
        experiment_name=args.experiment_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )


def main():
    """Main training function."""
    args = parse_args()

    # Load or create config
    if args.config:
        print(f"Loading config from {args.config}")
        config = load_config_from_file(args.config)
    else:
        config = create_config_from_args(args)

    # Print configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA rank: {config.rank}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"WandB enabled: {config.use_wandb}")
    print("=" * 60 + "\n")

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_save_path = Path(config.checkpoint_dir) / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    print(f"Saved config to {config_save_path}\n")

    try:
        # Create trainer (handles its own logging internally)
        trainer = RLTrainer(config)

        # Start training
        print("Starting training...\n")
        trainer.train()

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
