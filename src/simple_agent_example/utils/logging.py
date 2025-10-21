"""
Logging utilities for training, including WandB integration.
"""

from typing import Dict, Any, Optional
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from simple_agent_example.training.config import TrainingConfig, EpisodeMetrics


class WandBLogger:
    """
    Weights & Biases logger for training metrics.

    Handles logging of training metrics, episode statistics, and model performance.
    """

    def __init__(self, config: TrainingConfig, enabled: bool = None):
        """
        Initialize WandB logger.

        Args:
            config: Training configuration
            enabled: Override config.use_wandb setting
        """
        self.config = config
        self.enabled = enabled if enabled is not None else config.use_wandb

        if self.enabled and not WANDB_AVAILABLE:
            print("WandB requested but not available. Logging disabled.")
            self.enabled = False

        self.run = None

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB run."""
        try:
            # Initialize wandb
            self.run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config={
                    "base_model": self.config.base_model,
                    "rank": self.config.rank,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "gamma": self.config.gamma,
                    "num_episodes": self.config.num_episodes,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                tags=["2048", "rl", "tinker", "lora"],
            )

            print(f"WandB initialized: {self.run.url}")

        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            self.enabled = False

    def log_episode(self, metrics: EpisodeMetrics):
        """
        Log metrics from a single episode.

        Args:
            metrics: Episode metrics to log
        """
        if not self.enabled:
            return

        try:
            wandb.log({
                "episode": metrics.episode_num,
                "episode/reward": metrics.total_reward,
                "episode/score": metrics.final_score,
                "episode/max_tile": metrics.max_tile,
                "episode/num_moves": metrics.num_moves,
                "episode/avg_reward": metrics.average_reward,
                "episode/invalid_moves": metrics.invalid_moves,
                "episode/won": int(metrics.game_won),
            }, step=metrics.episode_num)

        except Exception as e:
            print(f"Error logging episode to WandB: {e}")

    def log_training_metrics(self, metrics_dict: Dict[str, Any], step: int):
        """
        Log training metrics.

        Args:
            metrics_dict: Dictionary of metrics to log
            step: Training step/episode number
        """
        if not self.enabled:
            return

        try:
            wandb.log(metrics_dict, step=step)
        except Exception as e:
            print(f"Error logging training metrics: {e}")

    def log_aggregate_metrics(
        self,
        avg_reward: float,
        avg_score: float,
        best_score: int,
        win_rate: float,
        step: int,
    ):
        """
        Log aggregate training metrics.

        Args:
            avg_reward: Average episode reward
            avg_score: Average game score
            best_score: Best score achieved
            win_rate: Win rate (reaching 2048)
            step: Current episode/step
        """
        if not self.enabled:
            return

        try:
            wandb.log({
                "train/avg_reward": avg_reward,
                "train/avg_score": avg_score,
                "train/best_score": best_score,
                "train/win_rate": win_rate,
            }, step=step)
        except Exception as e:
            print(f"Error logging aggregate metrics: {e}")

    def log_evaluation(self, eval_metrics: Dict[str, Any], step: int):
        """
        Log evaluation metrics.

        Args:
            eval_metrics: Dictionary of evaluation metrics
            step: Current episode/step
        """
        if not self.enabled:
            return

        try:
            wandb.log(eval_metrics, step=step)
        except Exception as e:
            print(f"Error logging evaluation metrics: {e}")

    def log_model_output(
        self,
        state_text: str,
        model_output: str,
        action: int,
        reward: float,
        step: int,
    ):
        """
        Log model inputs and outputs for debugging.

        Args:
            state_text: Input state text
            model_output: Raw model output
            action: Parsed action
            reward: Reward received
            step: Current step
        """
        if not self.enabled:
            return

        try:
            # Log as a table for easy inspection
            table = wandb.Table(
                columns=["state", "model_output", "action", "reward"],
                data=[[state_text[:200], model_output[:200], action, reward]]
            )

            wandb.log({"model_samples": table}, step=step)

        except Exception as e:
            print(f"Error logging model output: {e}")

    def watch_model(self, model):
        """
        Watch model for gradient and parameter tracking.

        Args:
            model: PyTorch model to watch
        """
        if not self.enabled:
            return

        try:
            wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"Error watching model: {e}")

    def save_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "model"):
        """
        Save file as WandB artifact.

        Args:
            file_path: Path to file to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (e.g., 'model', 'dataset')
        """
        if not self.enabled:
            return

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            print(f"Saved artifact: {artifact_name}")

        except Exception as e:
            print(f"Error saving artifact: {e}")

    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
                print("WandB run finished")
            except Exception as e:
                print(f"Error finishing WandB run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class ConsoleLogger:
    """
    Simple console logger as fallback when WandB is not available.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize console logger."""
        self.config = config
        print("Using console logging (WandB not available)")

    def log_episode(self, metrics: EpisodeMetrics):
        """Log episode to console."""
        print(f"Episode {metrics.episode_num}: "
              f"Score={metrics.final_score}, "
              f"Tile={metrics.max_tile}, "
              f"Reward={metrics.total_reward:.2f}, "
              f"Moves={metrics.num_moves}")

    def log_training_metrics(self, metrics_dict: Dict[str, Any], step: int):
        """Log training metrics to console."""
        print(f"Step {step}: {metrics_dict}")

    def log_aggregate_metrics(self, avg_reward, avg_score, best_score, win_rate, step):
        """Log aggregate metrics to console."""
        print(f"Step {step}: AvgReward={avg_reward:.2f}, "
              f"AvgScore={avg_score:.2f}, "
              f"BestScore={best_score}, "
              f"WinRate={win_rate:.2%}")

    def log_evaluation(self, eval_metrics: Dict[str, Any], step: int):
        """Log evaluation to console."""
        print(f"Evaluation at step {step}: {eval_metrics}")

    def log_model_output(self, state_text, model_output, action, reward, step):
        """Log model output to console."""
        pass  # Too verbose for console

    def watch_model(self, model):
        """No-op for console logger."""
        pass

    def save_artifact(self, file_path, artifact_name, artifact_type="model"):
        """No-op for console logger."""
        pass

    def finish(self):
        """No-op for console logger."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_logger(config: TrainingConfig) -> WandBLogger:
    """
    Get appropriate logger based on config and availability.

    Args:
        config: Training configuration

    Returns:
        WandBLogger or ConsoleLogger
    """
    if config.use_wandb and WANDB_AVAILABLE:
        return WandBLogger(config)
    else:
        return ConsoleLogger(config)
