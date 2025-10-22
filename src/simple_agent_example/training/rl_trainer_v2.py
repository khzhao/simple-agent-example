import asyncio
import uuid

import numpy as np
import tinker
import torch
from tinker import ServiceClient, types

import wandb
from simple_agent_example.envs import Game2048Env
from simple_agent_example.models import ActionParser, TextStateEncoder
from simple_agent_example.training.config_v2 import TrainingConfigV2


class RLTrainerV2:
    """
    RL Trainer for 2048 using Tinker API for LORA fine-tuning.
    """

    def __init__(self, config: TrainingConfigV2, model_path: str):
        self.config = config
        self.model_path = model_path

        self._setup(model_path)

    def _setup(self, model_path: str):
        """Initialize Tinker service and training client."""
        if tinker is None:
            raise ImportError(
                "Tinker is not installed. Install with: pip install tinker"
            )

        wandb.init(
            project=self.config.wandb_project,
            name=f"trainer-{str(uuid.uuid4())}",
            config={
                **self.config.__dict__,
            },
            tags=self.config.wandb_tags,
        )

        # Get API key from config or environment
        # Create service client
        self.service_client = ServiceClient()

        # Create LORA training client
        print(
            f"Creating LORA training client with base model: {self.config.base_model}"
        )
        self.training_client = self.service_client.create_lora_training_client(
            self.config.base_model,
            rank=self.config.rank,
        )

        if model_path is not None:
            print(f"Loading model from {model_path}")
            self.training_client.load_state(model_path).result()

        self.tokenizer = self.training_client.get_tokenizer()

        self.current_model_path = model_path
        self.current_sampler_path = None
        self.sampling_client = self._save_weights_and_get_sampling_client()

    def _save_weights_and_get_sampling_client(self):
        """Save weights and get a sampling client."""
        sampling_path = (
            self.training_client.save_weights_for_sampler(
                name=f"sampler-{str(uuid.uuid4())}"
            )
            .result()
            .path
        )
        self.current_sampler_path = sampling_path
        return self.service_client.create_sampling_client(model_path=sampling_path)

    def _save_model_weights(self):
        """Save model weights."""
        self.current_model_path = (
            self.training_client.save_state(name=f"checkpoint-{str(uuid.uuid4())}")
            .result()
            .path
        )

    async def generate_episode(self):
        """Generate an episode."""
        env = Game2048Env()
        obs, info = env.reset()
        move_count = 0
        done = False
        sampling_params = types.SamplingParams(
            max_tokens=20, top_p=0.9, temperature=1.0, stop=["\n"]
        )

        max_tile = 0

        episode = []

        while not done and move_count < self.config.max_steps_per_episode:
            state_text = TextStateEncoder.encode_state(obs, info["score"], move_count)
            prompt = types.ModelInput.from_ints(
                self.tokenizer.encode(state_text, add_special_tokens=False)
            )
            result = await self.sampling_client.sample_async(
                prompt=prompt, sampling_params=sampling_params, num_samples=1
            )

            sequence = result.sequences[0]
            output_tokens = sequence.tokens
            logprobs = sequence.logprobs
            assert logprobs is not None

            action_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            action = ActionParser.parse_action(action_text)
            if action == -1:
                print(f"Invalid action: {action_text}, using random action...")
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            max_tile = max(max_tile, next_info["max_tile"])
            obs = next_obs
            info = next_info
            move_count += 1

            episode.append(
                {
                    "state_text": state_text,
                    "action": action,
                    "action_text": action_text,
                    "logprobs": logprobs,
                    "reward": reward,
                    "done": done,
                    "move_count": move_count,
                    "max_tile": max_tile,
                }
            )
        return episode

    async def generate_batch(self):
        """Generate a batch of episodes."""
        episode_batch = []
        for _ in range(self.config.batch_size):
            episode_task = asyncio.create_task(self.generate_episode())
            episode_batch.append(episode_task)
        return await asyncio.gather(*episode_batch)

    async def train_on_new_batch(self):
        """Train the model on a new batch of episodes."""
        episode_batch = await self.generate_batch()
        return await self.train_on_batch(episode_batch)

    async def train_on_batch(self, episode_batch):
        """Train the model."""
        episode_batch_processed = []
        all_rewards = []
        for episode in episode_batch:
            processed_episode = []
            all_rewards.append([transition["reward"] for transition in episode])
            for i, transition in enumerate(episode):
                state_text = transition["state_text"]
                action_text = transition["action_text"]
                logprobs = transition["logprobs"]
                reward = transition["reward"]
                max_tile = transition["max_tile"]

                processed_episode.append(
                    {
                        "state_text": state_text,
                        "action_text": action_text,
                        "logprobs": logprobs,
                        "reward": reward,  # Use raw reward, no discounting
                        "max_tile": max_tile,
                    }
                )
            episode_batch_processed.append(processed_episode)

        tinker_datums = []

        # nan pad at the end of all_rewards to the same length as the longest episode
        max_length = max([len(episode) for episode in episode_batch_processed])
        all_rewards = [
            episode + [np.nan] * (max_length - len(episode)) for episode in all_rewards
        ]
        all_rewards_mean = np.nanmean(all_rewards, axis=0)
        for episode in episode_batch_processed:
            for i, transition in enumerate(episode):
                state_tokens = self.tokenizer.encode(
                    transition["state_text"], add_special_tokens=False
                )
                action_tokens = self.tokenizer.encode(
                    transition["action_text"], add_special_tokens=False
                )

                tokens = state_tokens + action_tokens
                logprobs = [-10] * len(state_tokens) + transition["logprobs"]
                rewards = [0] * len(state_tokens) + [
                    transition["reward"] - all_rewards_mean[i]
                ] * len(transition["logprobs"])
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]

                tinker_datums.append(
                    tinker.Datum(
                        model_input=types.ModelInput.from_ints(input_tokens),
                        loss_fn_inputs={
                            "target_tokens": types.TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": types.TensorData.from_torch(
                                torch.tensor(logprobs[1:])
                            ),
                            "advantages": types.TensorData.from_torch(
                                torch.tensor(rewards[1:])
                            ),
                        },
                    )
                )

        fwd_bwd_result = await self.training_client.forward_backward_async(
            tinker_datums, "ppo"
        )
        fb_result = await fwd_bwd_result

        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        optim_result = await self.training_client.optim_step_async(adam_params)
        opt_result = await optim_result

        all_rewards = []
        all_max_tiles = [
            max([transition["max_tile"] for transition in episode])
            for episode in episode_batch
        ]
        total_moves = [len(transition) for transition in episode_batch]
        for episode in episode_batch:
            all_rewards.extend([transition["reward"] for transition in episode])

        wandb.log(
            {
                "batch_loss": fb_result.metrics["loss:sum"],
                "average_reward": np.mean(all_rewards),
                "average_max_tile": np.mean(all_max_tiles),
                "average_move_count": np.mean(total_moves),
            }
        )
        return fb_result, opt_result

    async def train(self):
        for episode_num in range(self.config.num_episodes):
            print(f"Training on episode {episode_num}")
            fb_result, opt_result = await self.train_on_new_batch()
            print(f"Episode {episode_num}: {fb_result.metrics}, {opt_result}")

            if episode_num % self.config.save_interval == 0 and episode_num > 0:
                self._save_model_weights()
                print(f"Saved model weights to {self.current_model_path}")

            # Update sampling client with the new trained weights
            self.sampling_client = self._save_weights_and_get_sampling_client()
            print(
                f"Updated sampling client with new trained weights: {self.current_sampler_path}"
            )
