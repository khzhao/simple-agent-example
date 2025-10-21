"""
Reinforcement Learning trainer for 2048 using Tinker API for LORA fine-tuning.

This implementation uses Tinker's synchronous API based on official documentation.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

try:
    import tinker
    from tinker import ServiceClient
    from tinker.types import SamplingParams
    import torch
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: required packages not installed")
    tinker = None
    ServiceClient = None
    SamplingParams = None
    torch = None
    AutoTokenizer = None

from simple_agent_example.envs import Game2048Env
from simple_agent_example.models import TextStateEncoder, ActionParser
from simple_agent_example.training.config import (
    TrainingConfig,
    EpisodeMetrics,
    TrainingMetrics,
)
from simple_agent_example.utils import WandBLogger


class RLTrainer:
    """
    RL Trainer for 2048 using Tinker's LORA API.

    This implements a policy gradient approach where the model learns
    to play 2048 through self-play and reward feedback.

    Uses synchronous Tinker API.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the trainer."""
        self.config = config
        self.env = Game2048Env()
        self.encoder = TextStateEncoder()
        self.parser = ActionParser()

        # Tinker clients
        self.service_client = None
        self.training_client = None
        self.sampling_client = None
        self.tokenizer = None

        # Metrics tracking
        self.metrics = TrainingMetrics()

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Episode buffer for batch updates
        self.episode_buffer = []

        # Batch counter for checkpointing
        self.batch_counter = 0

        # WandB logger
        self.logger = None

        # Initialize Tinker API
        self._setup_tinker()

    def _setup_tinker(self):
        """Initialize Tinker service and training client."""
        if tinker is None:
            raise ImportError("Tinker is not installed. Install with: pip install tinker")

        # Get API key from config or environment
        api_key = self.config.tinker_api_key or os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError(
                "TINKER_API_KEY not found. Set it in config or environment."
            )

        # Create service client
        self.service_client = ServiceClient(api_key=api_key)

        # Create LORA training client
        print(f"Creating LORA training client with base model: {self.config.base_model}")
        self.training_client = self.service_client.create_lora_training_client(
            self.config.base_model,
            rank=self.config.rank,
        )

        # Load tokenizer
        print(f"Loading tokenizer for {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        print("Tinker LORA training client initialized successfully!")

    def _get_sampling_client(self, checkpoint_name: Optional[str] = None):
        """
        Get or create a sampling client.

        Args:
            checkpoint_name: Name/label of checkpoint to load. If None, uses current state.
        """
        if checkpoint_name:
            # Load the checkpoint first
            self.training_client.load_state(checkpoint_name).result()
            print(f"Loaded checkpoint: {checkpoint_name}")

        # Save weights and get sampling client
        # Use a unique name each time to avoid conflicts
        sampler_name = f"sampler-{self.batch_counter}" if checkpoint_name is None else f"{checkpoint_name}-sampler"
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client(name=sampler_name)

        return self.sampling_client

    def _sample_action(self, state_text: str) -> Tuple[int, str]:
        """
        Sample action from the model given state text.

        Returns:
            Tuple of (action_int, model_output_text)
        """
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
            model_output = self.tokenizer.decode(output_tokens, skip_special_tokens=True)

            # Parse action from output
            action = self.parser.parse_action(model_output)

            return action, model_output

        except Exception as e:
            print(f"Error sampling from model: {e}")
            import traceback
            traceback.print_exc()
            # Return random action on error
            return self.env.action_space.sample(), ""

    def _collect_episode(self, episode_num: int) -> EpisodeMetrics:
        """
        Collect one episode of experience.

        Returns:
            Episode metrics and trajectory
        """
        obs, info = self.env.reset()
        done = False
        trajectory = []

        episode_reward = 0.0
        invalid_moves = 0
        move_count = 0

        while not done and move_count < self.config.max_steps_per_episode:
            # Encode current state as text
            state_text = self.encoder.encode_state(
                grid=obs,
                score=info["score"],
                move_count=move_count,
            )

            # Sample action from model
            action, model_output = self._sample_action(state_text)

            # Handle invalid action parsing
            if action == -1:
                # Random fallback action
                action = self.env.action_space.sample()
                invalid_moves += 1

            # Get action text for training
            action_text = self.parser.action_to_text(action)

            # Take action in environment
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated

            # Store transition with all info needed for training
            trajectory.append({
                "state_text": state_text,
                "action": action,
                "action_text": action_text,
                "reward": reward,
                "done": done,
                "model_output": model_output,
            })

            episode_reward += reward
            move_count += 1
            obs = next_obs
            info = next_info

        # Create episode metrics
        game_won = info["max_tile"] >= 2048

        metrics = EpisodeMetrics(
            episode_num=episode_num,
            total_reward=episode_reward,
            final_score=info["score"],
            max_tile=info["max_tile"],
            num_moves=move_count,
            game_won=game_won,
            invalid_moves=invalid_moves,
        )

        # Store trajectory for training
        self.episode_buffer.append({
            "trajectory": trajectory,
            "metrics": metrics,
        })

        return metrics

    def _compute_returns(self, trajectory: List[Dict]) -> List[float]:
        """
        Compute discounted returns for a trajectory.

        Args:
            trajectory: List of transitions

        Returns:
            List of discounted returns
        """
        returns = []
        G = 0.0

        # Compute returns backward
        for transition in reversed(trajectory):
            G = transition["reward"] + self.config.gamma * G
            returns.insert(0, G)

        return returns

    def _create_datum(self, state_text: str, action_text: str, advantage: float) -> tinker.Datum:
        """
        Create a Tinker Datum object for training.

        Args:
            state_text: The state prompt
            action_text: The action completion
            advantage: The advantage value (for weighting)

        Returns:
            A Tinker Datum object
        """
        # Tokenize state and action
        state_tokens = self.tokenizer.encode(state_text, add_special_tokens=True)
        action_tokens = self.tokenizer.encode(action_text, add_special_tokens=False)

        # Create model input from state tokens
        model_input = tinker.types.ModelInput.from_ints(state_tokens)

        # Create datum with action tokens as target and advantage as weight
        return tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": action_tokens,
                "advantage": float(advantage),
            }
        )

    def _train_on_batch(self):
        """
        Perform training update on collected batch of episodes.

        This implements policy gradient update using Tinker's API.
        """
        if len(self.episode_buffer) == 0:
            return

        print(f"\nTraining on batch of {len(self.episode_buffer)} episodes...")

        all_data = []

        # Process all episodes in buffer
        for episode_data in self.episode_buffer:
            trajectory = episode_data["trajectory"]

            # Compute returns (advantages)
            returns = self._compute_returns(trajectory)

            # Collect training data
            for transition, advantage in zip(trajectory, returns):
                all_data.append({
                    "state_text": transition["state_text"],
                    "action_text": transition["action_text"],
                    "advantage": advantage,
                })

        # Normalize advantages for stability
        advantages = np.array([d["advantage"] for d in all_data])
        if len(advantages) > 1 and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for i, d in enumerate(all_data):
                d["advantage"] = advantages[i]

        try:
            # Create Datum objects for training
            datum_list = []
            for data in all_data:
                datum = self._create_datum(
                    state_text=data["state_text"],
                    action_text=data["action_text"],
                    advantage=data["advantage"],
                )
                datum_list.append(datum)

            # Forward-backward pass
            self.training_client.forward_backward(
                datum_list,
                loss_fn="importance_sampling",
            ).result()

            # Optimizer step
            adam_params = tinker.AdamParams(
                learning_rate=self.config.learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )
            self.training_client.optim_step(adam_params).result()

            print(f"Training complete. Processed {len(datum_list)} transitions.")

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

        # Clear buffer
        self.episode_buffer = []
        self.batch_counter += 1

    def _save_checkpoint(self, episode_num: int):
        """Save model checkpoint."""
        # Tinker expects a simple label name
        checkpoint_name = f"checkpoint-ep{episode_num:06d}"

        try:
            # Save using Tinker API
            self.training_client.save_state(name=checkpoint_name).result()

            print(f"Saved checkpoint: {checkpoint_name}")

            # Also save metrics locally
            metrics_path = self.checkpoint_dir / f"metrics_ep{episode_num}.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics.get_summary(), f, indent=2)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def _evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating for {num_episodes} episodes...")

        # Refresh sampling client with current state
        self._get_sampling_client()

        eval_scores = []
        eval_max_tiles = []
        eval_wins = 0

        for i in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            move_count = 0

            while not done and move_count < self.config.max_steps_per_episode:
                state_text = self.encoder.encode_state(obs, info["score"], move_count)

                # Sample action
                action, _ = self._sample_action(state_text)

                if action == -1:
                    action = self.env.action_space.sample()

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                move_count += 1

            eval_scores.append(info["score"])
            eval_max_tiles.append(info["max_tile"])
            if info["max_tile"] >= 2048:
                eval_wins += 1

        eval_metrics = {
            "eval/avg_score": np.mean(eval_scores),
            "eval/avg_max_tile": np.mean(eval_max_tiles),
            "eval/win_rate": eval_wins / num_episodes,
            "eval/best_score": max(eval_scores),
            "eval/best_tile": max(eval_max_tiles),
        }

        print(f"Evaluation results: {eval_metrics}")
        return eval_metrics

    def train(self):
        """Main training loop."""
        # Initialize WandB logger
        if self.config.use_wandb:
            self.logger = WandBLogger(self.config)

        try:
            # Get initial sampling client
            self._get_sampling_client()

            print(f"Starting training for {self.config.num_episodes} episodes...")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Base model: {self.config.base_model}")
            print(f"Rank: {self.config.rank}")
            print(f"Experiment: {self.config.experiment_name}\n")

            for episode in range(1, self.config.num_episodes + 1):
                # Collect episode
                episode_metrics = self._collect_episode(episode)

                # Update metrics
                self.metrics.update(episode_metrics)

                # Log to WandB
                if self.logger:
                    self.logger.log_episode(episode_metrics)

                # Log progress
                if episode % self.config.log_interval == 0:
                    print(f"Episode {episode}/{self.config.num_episodes}")
                    print(f"  Score: {episode_metrics.final_score}")
                    print(f"  Max Tile: {episode_metrics.max_tile}")
                    print(f"  Reward: {episode_metrics.total_reward:.2f}")
                    print(f"  Moves: {episode_metrics.num_moves}")
                    print(f"  Avg Score (last 100): {self.metrics.avg_episode_score:.2f}")
                    print(f"  Win Rate: {self.metrics.win_rate:.2%}\n")

                    # Log aggregate metrics to WandB
                    if self.logger:
                        self.logger.log_aggregate_metrics(
                            avg_reward=self.metrics.avg_episode_reward,
                            avg_score=self.metrics.avg_episode_score,
                            best_score=self.metrics.best_score,
                            win_rate=self.metrics.win_rate,
                            step=episode,
                        )

                # Train when batch is full
                if len(self.episode_buffer) >= self.config.batch_size:
                    self._train_on_batch()

                    # Refresh sampling client after training
                    self._get_sampling_client()

                # Evaluate
                if episode % self.config.eval_interval == 0:
                    eval_metrics = self._evaluate(self.config.eval_episodes)

                    # Log evaluation to WandB
                    if self.logger:
                        self.logger.log_evaluation(eval_metrics, step=episode)

                # Save checkpoint
                if episode % self.config.save_interval == 0:
                    self._save_checkpoint(episode)

            # Final checkpoint
            self._save_checkpoint(self.config.num_episodes)

            print("\nTraining complete!")
            print(f"Final metrics: {self.metrics.get_summary()}")

        finally:
            # Clean up WandB
            if self.logger:
                self.logger.finish()

    def load_checkpoint(self, checkpoint_name: str):
        """Load model from checkpoint.

        Args:
            checkpoint_name: The checkpoint label/name (not a file path)
        """
        try:
            self.training_client.load_state(checkpoint_name).result()
            print(f"Loaded checkpoint: {checkpoint_name}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
