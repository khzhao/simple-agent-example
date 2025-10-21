"""
Training configuration for 2048 RL agent.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for RL training with Tinker API."""

    # Tinker API settings
    tinker_api_key: Optional[str] = None  # Will use env var TINKER_API_KEY if None
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for fast iteration
    rank: int = 8  # LoRA rank

    # Training hyperparameters
    learning_rate: float = 1e-4
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 8  # Number of episodes before update
    gamma: float = 0.99  # Discount factor for future rewards

    # Reward shaping
    score_reward_scale: float = 1.0
    invalid_move_penalty: float = -10.0
    game_over_penalty: float = -50.0
    valid_move_bonus: float = 1.0

    # Checkpointing
    save_interval: int = 50  # Save every N episodes
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "2048-rl"

    # Evaluation
    eval_interval: int = 25  # Evaluate every N episodes
    eval_episodes: int = 5  # Number of episodes for evaluation

    # Logging
    use_wandb: bool = True
    wandb_project: str = "2048-tinker-rl"
    wandb_entity: Optional[str] = None
    log_interval: int = 1  # Log every N episodes

    # Sampling parameters for model
    temperature: float = 0.7
    max_tokens: int = 50
    top_p: float = 0.9

    # RL algorithm settings
    use_baseline: bool = True  # Use value baseline for variance reduction
    entropy_coef: float = 0.01  # Entropy regularization
    clip_grad_norm: float = 1.0  # Gradient clipping

    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    episode_num: int
    total_reward: float
    final_score: int
    max_tile: int
    num_moves: int
    game_won: bool  # Whether reached 2048 tile
    invalid_moves: int = 0
    average_reward: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.num_moves > 0:
            self.average_reward = self.total_reward / self.num_moves


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""

    episodes_completed: int = 0
    total_steps: int = 0

    # Running averages
    avg_episode_reward: float = 0.0
    avg_episode_score: float = 0.0
    avg_max_tile: float = 0.0
    avg_moves_per_episode: float = 0.0

    # Best performance
    best_score: int = 0
    best_max_tile: int = 0
    best_episode_reward: float = float('-inf')

    # Success metrics
    win_rate: float = 0.0  # Percentage reaching 2048
    games_won: int = 0

    # Recent episode history (for moving averages)
    recent_rewards: list = field(default_factory=list)
    recent_scores: list = field(default_factory=list)
    window_size: int = 100

    def update(self, episode_metrics: EpisodeMetrics):
        """Update training metrics with new episode results."""
        self.episodes_completed += 1
        self.total_steps += episode_metrics.num_moves

        # Update recent history
        self.recent_rewards.append(episode_metrics.total_reward)
        self.recent_scores.append(episode_metrics.final_score)

        # Keep only recent window
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            self.recent_scores.pop(0)

        # Update moving averages
        self.avg_episode_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        self.avg_episode_score = sum(self.recent_scores) / len(self.recent_scores)

        # Update best metrics
        if episode_metrics.total_reward > self.best_episode_reward:
            self.best_episode_reward = episode_metrics.total_reward

        if episode_metrics.final_score > self.best_score:
            self.best_score = episode_metrics.final_score

        if episode_metrics.max_tile > self.best_max_tile:
            self.best_max_tile = episode_metrics.max_tile

        # Update win rate
        if episode_metrics.game_won:
            self.games_won += 1
        self.win_rate = self.games_won / self.episodes_completed

    def get_summary(self) -> dict:
        """Get summary of current training metrics."""
        return {
            "episodes": self.episodes_completed,
            "total_steps": self.total_steps,
            "avg_reward": self.avg_episode_reward,
            "avg_score": self.avg_episode_score,
            "best_score": self.best_score,
            "best_tile": self.best_max_tile,
            "win_rate": self.win_rate,
            "games_won": self.games_won,
        }
