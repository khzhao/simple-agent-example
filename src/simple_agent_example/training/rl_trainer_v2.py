import asyncio
import os
import uuid

import tinker
from tinker import AdamParams, ServiceClient, types

from simple_agent_example.envs import Game2048Env
from simple_agent_example.models import ActionParser, TextStateEncoder
from simple_agent_example.training.config_v2 import TrainingConfigV2


class RLTrainerV2:
    """
    RL Trainer for 2048 using Tinker API for LORA fine-tuning.
    """

    def __init__(self, config: TrainingConfigV2, model_path: str):
        self.config = config
        self.env = Game2048Env()
        self.model_path = model_path

        self._setup_tinker(model_path)

    def _setup_tinker(self, model_path: str):
        """Initialize Tinker service and training client."""
        if tinker is None:
            raise ImportError(
                "Tinker is not installed. Install with: pip install tinker"
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
        self.training_client.save_state(name=f"checkpoint-{str(uuid.uuid4())}").result()
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
                continue

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
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
