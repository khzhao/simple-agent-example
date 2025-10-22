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
    max_steps_per_episode: int = 25
    batch_size: int = 16  # Number of episodes before update
    gamma: float = 0.99  # Discount factor for future rewards

    # Reward function parameters
    max_tile_reward_weight: float = 1.0  # Weight for max tile increase reward
    valid_moves_reward_weight: float = 2.0  # Weight for number of valid moves bonus
    invalid_move_penalty: float = -10.0  # Penalty for invalid moves
    terminal_penalty: float = -50.0  # Additional penalty when game ends

    # wandb settings
    wandb_project: str = "2048-rl-v2"
    wandb_tags: tuple[str] = ("2048", "rl", "tinker", "lora")

    # Checkpointing settings
    save_interval: int = 10  # Save every N episodes
