"""Thin wrappers around the official Tinker SDK for PPO-style training."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Iterable, Literal, Optional

import math
import numpy as np
import tinker
import torch
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
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    teacher_response_text: Optional[str] = None
    reward_reason: Optional[str] = None
    raw_reward: float = 0.0
    scaled_reward: float = 0.0


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
        temperature_max: Optional[float] = None,
        top_p: float = 0.95,
        top_p_min: Optional[float] = None,
        top_p_max: Optional[float] = None,
        max_tokens: int = 200,
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
        default_temperature_max = min(1.5, max(temperature, temperature + 0.3))
        self.temperature_max = (
            temperature_max if temperature_max is not None else default_temperature_max
        )
        self.top_p = top_p
        default_top_p_max = min(0.99, max(top_p, top_p + 0.05))
        self.top_p_min = top_p_min if top_p_min is not None else top_p
        self.top_p_max = top_p_max if top_p_max is not None else default_top_p_max
        self.max_tokens = max_tokens
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._sampling_client = None
        self._sampler_counter = 0
        self._last_train_stats: dict[str, float] = {}
        self._sample_counter = 0

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
            self._sampling_client = (
                self.training_client.save_weights_and_get_sampling_client(
                    name=f"sampler-{self._sampler_counter}"
                )
            )
            self._sampler_counter += 1

    def _refresh_sampling_client(self) -> None:
        self._sampling_client = (
            self.training_client.save_weights_and_get_sampling_client(
                name=f"sampler-{self._sampler_counter}"
            )
        )
        self._sampler_counter += 1

    def _temperature_for_iteration(self, iteration: int) -> float:
        low = min(self.temperature, self.temperature_max)
        high = max(self.temperature, self.temperature_max)
        if math.isclose(low, high, rel_tol=1e-6, abs_tol=1e-9):
            return high
        span = high - low
        return low + span / float(iteration + 1)

    def _top_p_for_iteration(self, iteration: int) -> float:
        min_top = max(0.0, min(1.0, min(self.top_p_min, self.top_p_max)))
        max_top = max(0.0, min(1.0, max(self.top_p_min, self.top_p_max)))
        if math.isclose(min_top, max_top, rel_tol=1e-6, abs_tol=1e-9):
            return max_top
        span = max_top - min_top
        return min_top + span / float(iteration + 1)

    def _sample_action_sync(
        self,
        prompt_text: str,
        max_tokens: Optional[int] = None,
    ) -> tuple[ChatMessage, TrajectoryStep]:
        self._ensure_sampling_client()

        prompt_tokens = self._encode_prompt(prompt_text)
        model_input = t_types.ModelInput.from_ints(prompt_tokens)

        iteration = self._sample_counter
        temperature_value = self._temperature_for_iteration(iteration)
        top_p_value = self._top_p_for_iteration(iteration)

        sampling_params = t_types.SamplingParams(
            temperature=temperature_value,
            top_p=top_p_value,
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
        response_text = self.tokenizer.decode(
            response_tokens, skip_special_tokens=True
        ).strip()
        logprobs = list(getattr(sequence, "logprobs", [0.0] * len(response_tokens)))

        message = ChatMessage(role="assistant", content=response_text)
        step = TrajectoryStep(
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            response_text=response_text,
            response_tokens=response_tokens,
            logprobs=logprobs,
            temperature=temperature_value,
            top_p=top_p_value,
        )
        self._sample_counter += 1
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
        advantages = [0.0] * len(prompt_tokens) + [scaled_advantage] * len(
            response_tokens
        )
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
        submitted_groups = 0
        trainable_groups = 0

        for group in trajectory_groups:
            if not group:
                continue
            submitted_groups += 1
            rewards = [trajectory.reward for trajectory in group if trajectory.steps]
            if not rewards:
                continue
            trainable_groups += 1
            if len(rewards) > 1:
                reward_mean = float(np.mean(rewards))
                reward_std = float(np.std(rewards))
            else:
                reward_mean = 0.0
                reward_std = 0.0
            normalize = reward_std if reward_std > 1e-6 else 1.0

            for trajectory in group:
                if not trajectory.steps:
                    continue
                advantage = (trajectory.reward - reward_mean) / (normalize + 1e-6)
                if abs(advantage) < 1e-9:
                    continue

                total_assistant_tokens = sum(
                    len(step.response_tokens)
                    for step in trajectory.steps
                    if step.response_tokens
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
            self._last_train_stats = {
                "submitted_groups": float(submitted_groups),
                "trainable_groups": float(trainable_groups),
                "num_datums": 0.0,
            }
            return self._last_train_stats

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
        self._last_train_stats = {
            "submitted_groups": float(submitted_groups),
            "trainable_groups": float(trainable_groups),
            "num_datums": float(len(datums)),
        }
        return self._last_train_stats

    async def train(
        self,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float,
    ) -> dict[str, float]:
        return await asyncio.to_thread(
            self._train_sync,
            trajectory_groups,
            learning_rate,
        )

    async def save_checkpoint(self, name: Optional[str] = None) -> str:
        label = name or f"checkpoint-step-{self._sampler_counter:04d}"
        checkpoint = await asyncio.to_thread(
            lambda: self.training_client.save_state(name=label).result()
        )
        return getattr(checkpoint, "path", label)

    async def load_checkpoint(self, name: str) -> str:
        checkpoint = await asyncio.to_thread(
            lambda: self.training_client.load_state(name).result()
        )
        await self.refresh_sampling_client()
        return getattr(checkpoint, "path", name)

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
