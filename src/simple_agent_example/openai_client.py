"""OpenAI helpers for teacher rollouts and reward grading."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from .tinker_client import ChatMessage

logger = logging.getLogger(__name__)

load_dotenv()


def _ensure_openai_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be provided for OpenAI integration.")
    return key


def _extract_json_object(text: str) -> Optional[dict]:
    """Attempt to parse a JSON object from the model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class RewardResult:
    score: float
    reasoning: str


class OpenAIChatModel:
    """Thin wrapper around OpenAI chat completions for teacher rollouts using native async."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 10000,
        system_prompt: Optional[str] = None,
    ) -> None:
        key = _ensure_openai_key(api_key)
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt

    def _convert_messages(
        self, messages: Sequence[ChatMessage]
    ) -> list[dict[str, str]]:
        converted: list[dict[str, str]] = []
        if self.system_prompt:
            converted.append({"role": "system", "content": self.system_prompt})
        for message in messages:
            converted.append({"role": message.role, "content": message.content})
        return converted

    async def sample_action(
        self,
        messages: Sequence[ChatMessage],
    ) -> ChatMessage:
        payload = self._convert_messages(messages)

        # Build API parameters - some models don't support temperature/max_completion_tokens
        api_params = {
            "model": self.model,
            "messages": payload,
        }

        # Only add temperature if not default (some models like o1 don't support it)
        if self.temperature != 1.0:
            api_params["temperature"] = self.temperature

        # Add max_completion_tokens if specified
        if self.max_output_tokens:
            api_params["max_tokens"] = 10000

        print(api_params)

        response = await self._client.chat.completions.create(**api_params)
        choice = response.choices[0]
        content = choice.message.content or ""
        return ChatMessage(role="assistant", content=content.strip())


class OpenAIRewardModel:
    """Use an OpenAI model to score student responses against a teacher using native async."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 10000,
        system_prompt: Optional[str] = None,
    ) -> None:
        key = _ensure_openai_key(api_key)
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_prompt = system_prompt or (
            "You are a strict grader for the 2048 puzzle game. "
            "Return a JSON object with keys 'score' (float between 0 and 1) "
            "and 'reasoning' (short string) evaluating how well the student "
            "response matches the teacher response and follows formatting requirements."
        )

    def _build_messages(
        self,
        *,
        board_view: str,
        teacher_response: str,
        student_response: str,
    ) -> list[dict[str, str]]:
        user_prompt = (
            "Evaluate the student's answer for the 2048 board below.\n"
            "Board:\n"
            f"{board_view}\n\n"
            "Teacher response:\n"
            f"{teacher_response}\n\n"
            "Student response:\n"
            f"{student_response}\n\n"
            "Rules:\n"
            "- The student should exactly match the teacher's <move> direction.\n"
            "- Penalize invalid formatting or missing tags.\n"
            '- Return JSON like {"score": 1.0, "reasoning": "..."}.\n'
            "- Clamp score to [0, 1]."
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def score(
        self,
        *,
        board_view: str,
        teacher_response: str,
        student_response: str,
    ) -> RewardResult:
        messages = self._build_messages(
            board_view=board_view,
            teacher_response=teacher_response,
            student_response=student_response,
        )

        # Build API parameters - some models don't support temperature/max_completion_tokens
        api_params = {
            "model": self.model,
            "messages": messages,
        }

        # Only add temperature if not default (some models like o1 don't support it)
        if self.temperature != 1.0:
            api_params["temperature"] = self.temperature

        # Add max_completion_tokens if specified
        if self.max_output_tokens:
            api_params["max_completion_tokens"] = self.max_output_tokens

        response = await self._client.chat.completions.create(**api_params)
        content = (response.choices[0].message.content or "").strip()

        parsed = _extract_json_object(content)
        if not parsed:
            logger.warning(
                "Failed to parse reward JSON. Content: %s", content.replace("\n", " ")
            )
            return RewardResult(score=0.0, reasoning="Failed to parse reward output.")

        score = parsed.get("score", 0.0)
        reasoning = parsed.get("reasoning", content)
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        score_value = max(0.0, min(1.0, score_value))
        return RewardResult(score=score_value, reasoning=str(reasoning))
