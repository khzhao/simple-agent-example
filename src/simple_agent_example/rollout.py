"""Rollout logic for on-policy PPO data collection."""

from __future__ import annotations

import asyncio
import logging
import random
import re

from .env import (WINNING_VALUE, apply_agent_move, check_game_finished,
                  generate_game, max_cell_value, render_board,
                  total_board_value)
from .openai_client import (OpenAIChatModel, OpenAIRewardModel, RewardResult)
from .tinker_client import ChatMessage, TinkerTrainableModel, Trajectory

logger = logging.getLogger(__name__)


def _format_board(board_view: str) -> str:
    lines: list[str] = []
    for raw_line in board_view.splitlines():
        cells = [cell.strip() for cell in raw_line.split("|")]
        pretty = " | ".join(f"{(cell if cell != '_' else '.'):>3}" for cell in cells)
        lines.append(pretty)
    return "\n".join(lines)


def _format_response(content: str) -> str:
    return "\n".join(f"    {line}" for line in content.splitlines())

RESPONSE_PATTERN = re.compile(
    r"^<think>.*?</think>\s*<move>(up|down|left|right)</move>$",
    re.IGNORECASE | re.DOTALL,
)

SYSTEM_PROMPT = (
    "You are an excellent 2048 player."
    " Think silently inside a <think>...</think> block."
    " After the reasoning, output exactly one <move>...</move> tag with left/right/up/down."
    " Do not emit tool calls, JSON, explanations, or additional text."
    " Your entire reply must consist of the <think> block immediately followed by the <move> block."
    " You must absolutely follow the instructions above or else something very bad will happen to you."
)


def build_prompt(board_view: str) -> str:
    """Create the textual prompt sent to the language model."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Current board:\n"
        f"{board_view}\n"
        "Return your answer using exactly:\n"
        "<think>...</think>\n"
        "<move>direction</move>"
    )


def extract_move_xml(model_output: str) -> str:
    """Ensure the response matches the required format and extract the move tag."""
    cleaned = model_output.strip()
    match = RESPONSE_PATTERN.fullmatch(cleaned)
    if not match:
        raise ValueError("No valid <move> tag found")
    direction = match.group(1).lower()
    return f"<move>{direction}</move>"


def _teacher_prompt(board_view: str) -> str:
    return (
        "Current board:\n"
        f"{board_view}\n"
        "Return your answer using exactly:\n"
        "<think>...</think>\n"
        "<move>direction</move>"
    )


class TeacherFormatError(RuntimeError):
    """Raised when the teacher response does not follow the required format."""


async def _sample_teacher_with_retry(
    teacher: OpenAIChatModel,
    prompt_text: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> tuple[ChatMessage, int, int, str]:
    last_error: Exception | None = None
    format_failures = 0
    current_prompt = prompt_text
    for attempt in range(1, max_retries + 1):
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=current_prompt),
        ]
        try:
            reply = await teacher.sample_action(messages)
        except Exception as exc:  # pragma: no cover - network failure path
            last_error = exc
            logger.warning(
                "Teacher sampling failed on attempt %d/%d: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff_seconds)
                continue
            break

        content = reply.content.strip()
        if not RESPONSE_PATTERN.fullmatch(content):
            format_failures += 1
            last_error = TeacherFormatError(
                "Teacher response missing required <think>/<move> format."
            )
            logger.warning(
                "Teacher produced invalid format on attempt %d/%d",
                attempt,
                max_retries,
            )
            current_prompt = (
                f"{current_prompt}\n\nThe previous response was invalid: {content!r}"
            )
            if attempt < max_retries:
                await asyncio.sleep(backoff_seconds)
                continue
            break

        return reply, attempt, format_failures, current_prompt
    if last_error is None:
        raise RuntimeError("Teacher sampling failed without exception.")
    raise last_error


async def rollout(
    model: TinkerTrainableModel,
    teacher: OpenAIChatModel,
    reward_model: OpenAIRewardModel,
    step: int,
    *,
    is_validation: bool = False,
    verbose: bool = False,
) -> Trajectory:
    """Generate a single trajectory using an OpenAI teacher for rollouts."""
    game = generate_game()
    move_count = 0

    trajectory = Trajectory(
        messages=[ChatMessage(role="system", content=SYSTEM_PROMPT)],
        metadata={"game_id": game["id"], "step": step, "validation": is_validation},
        reward=0.0,
        metrics={},
    )

    trajectory.metadata["teacher_responses"] = []
    raw_reward_values: list[float] = []
    scaled_reward_values: list[float] = []

    while True:
        board_view = render_board(game)
        user_message = ChatMessage(role="user", content=board_view)
        trajectory.messages.append(user_message)
        if verbose:
            print(f"\n=== Step {move_count:02d} ===")
            print(_format_board(board_view))

        prompt_text = build_prompt(board_view)
        teacher_prompt = _teacher_prompt(board_view)
        total_attempts = 0
        total_format_retries = 0
        noop_retries = 0
        invalid_move_retries = 0
        teacher_reply: ChatMessage | None = None
        teacher_move_xml: str | None = None
        teacher_valid = False

        while True:
            try:
                (
                    teacher_reply_candidate,
                    attempts_used,
                    format_retries,
                    teacher_prompt,
                ) = await _sample_teacher_with_retry(
                    teacher,
                    teacher_prompt,
                )
            except TeacherFormatError as exc:
                trajectory.metrics["teacher_invalid_format"] = 1
                trajectory.metadata["teacher_sampling_error"] = str(exc)
                if verbose:
                    print(f"Teacher returned invalid format after retries: {exc}")
                if trajectory.steps:
                    trajectory.steps[-1].done = True
                break
            except Exception as exc:  # pragma: no cover - network failure path
                trajectory.metrics["teacher_sampling_failure"] = 1
                trajectory.metadata["teacher_sampling_error"] = str(exc)
                if verbose:
                    print(f"Teacher sampling failed after retries: {exc}")
                if trajectory.steps:
                    trajectory.steps[-1].done = True
                break

            total_attempts += attempts_used
            total_format_retries += format_retries

            content = teacher_reply_candidate.content.strip()
            try:
                move_xml = extract_move_xml(content)
            except ValueError:
                invalid_move_retries += 1
                teacher_prompt = (
                    f"{teacher_prompt}\n\nThe previous response was invalid: {content!r}"
                )
                continue

            rng_state = random.getstate()
            preview_game = {
                "id": game["id"],
                "board": [row[:] for row in game["board"]],
            }
            try:
                apply_agent_move(preview_game, move_xml)
            except ValueError as exc:
                random.setstate(rng_state)
                message_lower = str(exc).lower()
                if "did not change board" in message_lower:
                    noop_retries += 1
                    teacher_prompt = (
                        f"{teacher_prompt}\n\nYour last move '{move_xml}' did not change the board. Try a different direction."
                    )
                    continue
                invalid_move_retries += 1
                teacher_prompt = (
                    f"{teacher_prompt}\n\nThe previous move was invalid: {exc}"
                )
                continue
            random.setstate(rng_state)

            teacher_reply = teacher_reply_candidate
            teacher_move_xml = move_xml
            teacher_valid = True
            break

        if not teacher_valid or teacher_reply is None or teacher_move_xml is None:
            break

        if total_attempts > 1:
            trajectory.metrics["teacher_retry_attempts"] = (
                trajectory.metrics.get("teacher_retry_attempts", 0.0)
                + float(total_attempts - 1)
            )
        if total_format_retries:
            trajectory.metrics["teacher_invalid_format_retries"] = (
                trajectory.metrics.get("teacher_invalid_format_retries", 0.0)
                + float(total_format_retries)
            )
        if noop_retries:
            trajectory.metrics["teacher_noop_retries"] = (
                trajectory.metrics.get("teacher_noop_retries", 0.0) + float(noop_retries)
            )
        if invalid_move_retries:
            trajectory.metrics["teacher_invalid_move_retries"] = (
                trajectory.metrics.get("teacher_invalid_move_retries", 0.0)
                + float(invalid_move_retries)
            )

        trajectory.metadata["teacher_responses"].append(teacher_reply.content)

        assistant_message, step_info = await model.sample_action(prompt_text)
        trajectory.messages.append(assistant_message)
        step_info.teacher_response_text = teacher_reply.content
        trajectory.steps.append(step_info)

        reward_result: RewardResult = await reward_model.score(
            board_view=board_view,
            teacher_response=teacher_reply.content,
            student_response=assistant_message.content,
        )
        step_info.raw_reward = reward_result.score
        step_info.reward_reason = reward_result.reasoning
        scaled_reward = (2.0 * reward_result.score) - 1.0
        step_info.scaled_reward = scaled_reward
        raw_reward_values.append(reward_result.score)
        scaled_reward_values.append(scaled_reward)

        if verbose:
            print("Teacher response:")
            print(_format_response(teacher_reply.content))
            print("Student response:")
            print(_format_response(assistant_message.content))
            print(
                f"Reward raw: {reward_result.score:.3f} | scaled: {scaled_reward:.3f} | {reward_result.reasoning}"
            )

        apply_agent_move(game, teacher_move_xml)
        move_count += 1
        trajectory.metrics.setdefault("teacher_valid_actions", 0)
        trajectory.metrics["teacher_valid_actions"] += 1

        if check_game_finished(game):
            trajectory.metrics["teacher_invalid_move"] = 0
            if trajectory.steps:
                trajectory.steps[-1].done = True
            break

    if verbose:
        print("\n=== Final Board ===")
        print(_format_board(render_board(game)))

    max_value = max_cell_value(game)
    board_value = total_board_value(game)
    agent_won = max_value == WINNING_VALUE

    trajectory.metrics.update(
        {
            "max_value": max_value,
            "board_value": board_value,
            "num_moves": move_count,
            "win": agent_won,
        }
    )

    if scaled_reward_values:
        trajectory.reward = float(sum(scaled_reward_values) / len(scaled_reward_values))
        trajectory.metrics["teacher_reward_mean"] = trajectory.reward
        trajectory.metrics["teacher_reward_min"] = float(min(scaled_reward_values))
        trajectory.metrics["teacher_reward_max"] = float(max(scaled_reward_values))
        trajectory.metrics["teacher_reward_count"] = float(len(scaled_reward_values))
        trajectory.metrics["teacher_reward_raw_mean"] = float(
            sum(raw_reward_values) / len(raw_reward_values)
        )
        trajectory.metrics["teacher_reward_raw_min"] = float(min(raw_reward_values))
        trajectory.metrics["teacher_reward_raw_max"] = float(max(raw_reward_values))
    else:
        trajectory.reward = 0.0
        trajectory.metrics["teacher_reward_mean"] = 0.0
        trajectory.metrics["teacher_reward_min"] = 0.0
        trajectory.metrics["teacher_reward_max"] = 0.0
        trajectory.metrics["teacher_reward_count"] = 0.0
        trajectory.metrics["teacher_reward_raw_mean"] = 0.0
        trajectory.metrics["teacher_reward_raw_min"] = 0.0
        trajectory.metrics["teacher_reward_raw_max"] = 0.0

    trajectory.metrics["invalid_move"] = trajectory.metrics.get("teacher_invalid_move", 0)

    if trajectory.steps:
        for step_info in trajectory.steps:
            step_info.reward = trajectory.reward

    if verbose:
        print(
            f"Reward: {trajectory.reward:.3f} | Max tile: {max_value} | Board value: {board_value} | Moves: {move_count}"
        )

    return trajectory
