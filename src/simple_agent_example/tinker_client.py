"""Thin wrappers around the official Tinker SDK for PPO-style training."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Literal, Optional

import numpy as np
import torch
import tinker
from tinker import ServiceClient
from tinker import types as t_types

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    """Conversation turn."""

    role: Role
    content: str


@dataclass
class TrajectoryStep:
    """Single model decision within a trajectory."""

    prompt_text: str
    prompt_tokens: list[int]
    response_text: str
    response_tokens: list[int]
    logprobs: list[float]
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    weight: float = 1.0


@dataclass
class Trajectory:
    """Single self-contained rollout."""

    messages: list[ChatMessage] = field(default_factory=list)
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    steps: list[TrajectoryStep] = field(default_factory=list)


TrajectoryGroup = list[Trajectory]


class TinkerTrainableModel:
    """Wrapper around Tinker's ServiceClient with PPO helpers."""

    def __init__(
        self,
        *,
        name: str,
        project: str,
        base_model: str,
        api_key: Optional[str] = None,
        rank: int = 8,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 16,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        self.name = name
        self.project = project
        self.base_model = base_model

        key = api_key or os.getenv("TINKER_API_KEY")
        if not key:
            raise ValueError("TINKER_API_KEY not set. Export it or pass api_key.")

        self.service_client: ServiceClient = ServiceClient(api_key=key)
        self.training_client = self.service_client.create_lora_training_client(
            base_model=base_model,
            rank=rank,
        )
        self.tokenizer = self.training_client.get_tokenizer()

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._sampling_client = None
        self._sampler_counter = 0

    # ------------------------------------------------------------------
    # Sampling utilities

    def _encode_prompt(self, prompt_text: str) -> list[int]:
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        if bos is not None and (len(tokens) == 0 or tokens[0] != bos):
            return [bos] + tokens
        return tokens

    def _ensure_sampling_client(self) -> None:
        if self._sampling_client is None:
            self._sampling_client = self.training_client.save_weights_and_get_sampling_client(
                name=f"sampler-{self._sampler_counter}"
            )
            self._sampler_counter += 1

    def _refresh_sampling_client(self) -> None:
        self._sampling_client = self.training_client.save_weights_and_get_sampling_client(
            name=f"sampler-{self._sampler_counter}"
        )
        self._sampler_counter += 1

    def _sample_action_sync(
        self,
        prompt_text: str,
        max_tokens: Optional[int] = None,
    ) -> tuple[ChatMessage, TrajectoryStep]:
        self._ensure_sampling_client()

        prompt_tokens = self._encode_prompt(prompt_text)
        model_input = t_types.ModelInput.from_ints(prompt_tokens)

        sampling_params = t_types.SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            logprobs=True,
        )

        result = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        sequence = result.sequences[0]
        response_tokens = list(sequence.tokens)
        logprobs = list(getattr(sequence, "logprobs", [0.0] * len(response_tokens)))
        response_text = self.tokenizer.decode(
            response_tokens, skip_special_tokens=True
        ).strip()

        message = ChatMessage(role="assistant", content=response_text)
        step = TrajectoryStep(
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            response_text=response_text,
            response_tokens=response_tokens,
            logprobs=logprobs,
        )
        return message, step

    async def sample_action(
        self,
        prompt_text: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> tuple[ChatMessage, TrajectoryStep]:
        return await asyncio.to_thread(
            self._sample_action_sync,
            prompt_text,
            max_tokens,
        )

    async def refresh_sampling_client(self) -> None:
        await asyncio.to_thread(self._refresh_sampling_client)

    # ------------------------------------------------------------------
    # Training utilities

    def _create_datum(
        self,
        prompt_tokens: list[int],
        response_tokens: list[int],
        advantage: float,
        weight: float,
        logprobs: list[float],
    ) -> tinker.Datum:
        full_tokens = prompt_tokens + response_tokens

        model_input = t_types.ModelInput.from_ints(full_tokens)
        target_tokens_tensor = torch.tensor(full_tokens, dtype=torch.long)
        target_tokens = tinker.TensorData.from_torch(target_tokens_tensor)

        scaled_advantage = advantage * weight
        advantages = [0.0] * len(prompt_tokens) + [scaled_advantage] * len(response_tokens)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        advantages_data = tinker.TensorData.from_torch(advantages_tensor)

        logprobs_full = [0.0] * len(prompt_tokens) + logprobs
        logprobs_tensor = torch.tensor(logprobs_full, dtype=torch.float32)
        logprobs_data = tinker.TensorData.from_torch(logprobs_tensor)

        return tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "advantages": advantages_data,
                "logprobs": logprobs_data,
            },
        )

    def _train_sync(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        learning_rate: float,
    ) -> None:
        datums: list[tinker.Datum] = []

        for group in trajectory_groups:
            rewards = [
                trajectory.reward
                for trajectory in group
                if trajectory.steps
            ]
            if len(rewards) < 2:
                continue
            reward_mean = float(np.mean(rewards))
            reward_std = float(np.std(rewards))
            normalize = reward_std if reward_std > 1e-6 else 1.0

            for trajectory in group:
                if not trajectory.steps:
                    continue
                advantage = (trajectory.reward - reward_mean) / (normalize + 1e-6)
                if abs(advantage) < 1e-9:
                    continue

                total_assistant_tokens = sum(
                    len(step.response_tokens) for step in trajectory.steps if step.response_tokens
                )
                if total_assistant_tokens == 0:
                    continue

                weight_per_step = 1.0 / total_assistant_tokens

                for step in trajectory.steps:
                    if not step.response_tokens:
                        continue
                    step.advantage = advantage
                    step.weight = weight_per_step
                    datums.append(
                        self._create_datum(
                            prompt_tokens=step.prompt_tokens,
                            response_tokens=step.response_tokens,
                            advantage=float(advantage),
                            weight=weight_per_step,
                            logprobs=step.logprobs,
                        )
                    )

        if not datums:
            return

        self.training_client.forward_backward(
            datums,
            loss_fn="importance_sampling",
        ).result()

        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
        )
        self.training_client.optim_step(adam_params).result()

        self._refresh_sampling_client()

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float,
    ) -> None:
        await asyncio.to_thread(
            self._train_sync,
            trajectory_groups,
            learning_rate,
        )

    async def save_checkpoint(self, name: Optional[str] = None) -> None:
        label = name or f"checkpoint-step-{self._sampler_counter:04d}"
        await asyncio.to_thread(
            lambda: self.training_client.save_state(name=label).result()
        )

    async def load_checkpoint(self, name: str) -> None:
        await asyncio.to_thread(lambda: self.training_client.load_state(name).result())
        await self.refresh_sampling_client()

    async def score_group(self, group: TrajectoryGroup) -> Any:
        """Placeholder for optional external scoring."""
        _ = group  # Explicitly unused
        return None


async def gather_trajectory_groups(
    generators: Iterable[Iterable[Awaitable[Trajectory]]],
    *,
    after_each: Optional[Callable[[TrajectoryGroup], Awaitable[Any]]] = None,
) -> list[TrajectoryGroup]:
    """Utility that mirrors ART's gather_trajectory_groups helper."""

    results: list[TrajectoryGroup] = []

    for group_coroutines in generators:
        trajectories = await asyncio.gather(*group_coroutines)
        group: TrajectoryGroup = [
            trajectory for trajectory in trajectories if trajectory is not None
        ]
        if after_each is not None and group:
            await after_each(group)
        if group:
            results.append(group)

    return results


def environment_seed(seed: Optional[int] = None) -> None:
    """Optionally control global randomness for reproducibility."""
    if seed is None:
        return
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
