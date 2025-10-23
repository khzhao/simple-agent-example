from dataclasses import dataclass


@dataclass
class TrainingConfigV2:
    """Configuration for RL training with Tinker API."""

    # Tinker API settings
    base_model: str
    sft_model_path: str
    rank: int = 32  # LoRA rank

    # Training hyperparameters
    learning_rate: float = 1e-4
    num_episodes: int = 2000
    # max_steps_per_episode: int = 1000
    max_steps_per_episode: int = 500
    batch_size: int = 16  # Number of episodes before update
    gamma: float = 0.99  # Discount factor for future rewards

    # wandb settings
    wandb_project: str = "2048-rl-v2"
    wandb_tags: tuple[str] = ("2048", "rl", "tinker", "lora")

    # Checkpointing settings
    save_interval: int = 5  # Save every N episodes
