"""Rollout logic for on-policy REINFORCE-style data collection."""

from __future__ import annotations

import asyncio
import logging
import random
import re

from .env import (WINNING_VALUE, apply_agent_move, check_game_finished,
                  generate_game, max_cell_value, render_board,
                  total_board_value)
from .openai_client import OpenAIChatModel, OpenAIRewardModel, RewardResult
from .tinker_client import ChatMessage, TinkerTrainableModel, Trajectory

logger = logging.getLogger(__name__)

DISCOUNT_FACTOR = 0.99
INVALID_ACTION_PENALTY = -1.0
MAX_STEPS_PER_EPISODE = 200


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
MOVE_ONLY_PATTERN = re.compile(
    r"<move>\s*(up|down|left|right)\s*</move>", re.IGNORECASE
)

SYSTEM_PROMPT = (
    "You are an expert 2048 player. Your goal is to merge tiles to create higher values and reach the 2048 tile (or beyond).\n\n"
    "GAME RULES:\n"
    "- The board is 4x4 with numbered tiles (2, 4, 8, 16, 32, 64, 128, 256, 512, ..., 2048.)\n"
    "- When you move, all tiles slide in that direction\n"
    "- Tiles with the same number merge when they touch: 2+2=4, 4+4=8, 8+8=16, etc.\n"
    "- After each move, a new tile (2 or 4) spawns in an empty cell\n"
    "- The game ends when the board is full and no moves can merge tiles\n\n"
    "WINNING STRATEGY:\n"
    "- Keep your highest tile in a corner (preferably bottom-right or bottom-left)\n"
    "- Build tiles in descending order from the corner: e.g., 128→64→32→16→8\n"
    "- Avoid moving away from your chosen corner unless absolutely necessary\n"
    "- Always consider which direction will create the most merges\n"
    "- Plan ahead: think about where the new tile will spawn and how it affects your strategy\n"
    "- Avoid random moves that break your tile ordering\n\n"
    "OUTPUT FORMAT (CRITICAL):\n"
    "Your response must contain exactly two XML tags in this order:\n"
    "1. <think>your reasoning here</think>\n"
    "2. <move>direction</move> where direction is one of: up, down, left, right\n\n"
    "In your <think> block, analyze:\n"
    "- Current board state and highest tile location\n"
    "- Which moves will create merges\n"
    "- How each move affects your corner strategy\n"
    "- Where a new tile might spawn and its impact\n\n"
    "CORRECT OUTPUT EXAMPLES:\n\n"
    "Example response 1:\n"
    "<think>The highest tile is 64 in the bottom-left corner. I have a 32 above it and a 16 to its right, "
    "creating a good descending chain. Moving left will slide the 4 and 4 together on the top row to create "
    "an 8, and keep my corner strategy intact. Moving down would also work but left gives me an extra merge. "
    "I'll avoid moving up or right as that would disrupt my corner positioning.</think>\n"
    "<move>left</move>\n\n"
    "Example response 2:\n"
    "<think>I see my highest tile is 128 in the bottom-right. The board has two pairs of 2s that can merge. "
    "If I move right, both pairs will slide and merge: top row 2+2→4 and middle row 2+2→4. This creates space "
    "and maintains my corner strategy with the 128. Down would also keep the corner but wouldn't merge as many tiles. "
    "Right is the optimal move here.</think>\n"
    "<move>right</move>\n\n"
    "Example response 3:\n"
    "<think>The board has a 256 in the top-left corner with 128, 64, 32 descending to the right. This is a strong "
    "position. I need to avoid moving down as it would bury my highest tiles. Moving left keeps everything aligned. "
    "I see a 4 and 4 on the bottom row that will merge if I move left, which is an added bonus. Left maintains "
    "structure and creates a merge.</think>\n"
    "<move>left</move>\n\n"
    "Do not output anything else - no explanations, no JSON, no tool calls, just <think> followed by <move>."
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


def extract_move_xml(model_output: str, *, strict: bool = True) -> str:
    """Extract the move tag, optionally requiring the full structured format."""
    cleaned = model_output.strip()
    if strict:
        match = RESPONSE_PATTERN.fullmatch(cleaned)
    else:
        match = MOVE_ONLY_PATTERN.search(cleaned)
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
                "Teacher produced invalid format on attempt %d/%d : %s",
                attempt,
                max_retries,
                content,
            )
            current_prompt = f"""{current_prompt}\n\nThe previous response was invalid because it did not follow the required format:\n
                The required format is:
                <think>your reasoning here</think>
                <move>direction</move>
                where direction is one of: up, down, left, right
                The previous response was: {content!r}"""
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
    consecutive_noop_moves = 0

    while True:
        if move_count >= MAX_STEPS_PER_EPISODE:
            trajectory.metrics["max_step_limit_reached"] = 1
            if trajectory.steps:
                trajectory.steps[-1].done = True
            break

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
                teacher_prompt = f"{teacher_prompt}\n\nThe previous response was invalid: {content!r}"
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
                    teacher_prompt = f"{teacher_prompt}\n\nYour last move '{move_xml}' did not change the board. Try a different direction."
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
            trajectory.metrics["teacher_retry_attempts"] = trajectory.metrics.get(
                "teacher_retry_attempts", 0.0
            ) + float(total_attempts - 1)
        if total_format_retries:
            trajectory.metrics["teacher_invalid_format_retries"] = (
                trajectory.metrics.get("teacher_invalid_format_retries", 0.0)
                + float(total_format_retries)
            )
        if noop_retries:
            trajectory.metrics["teacher_noop_retries"] = trajectory.metrics.get(
                "teacher_noop_retries", 0.0
            ) + float(noop_retries)
        if invalid_move_retries:
            trajectory.metrics["teacher_invalid_move_retries"] = trajectory.metrics.get(
                "teacher_invalid_move_retries", 0.0
            ) + float(invalid_move_retries)

        trajectory.metadata["teacher_responses"].append(teacher_reply.content)

        assistant_message, step_info = await model.sample_action(prompt_text)
        trajectory.messages.append(assistant_message)
        step_info.teacher_response_text = teacher_reply.content
        trajectory.steps.append(step_info)
        student_content = assistant_message.content.strip()
        try:
            student_move_xml = extract_move_xml(student_content, strict=False)
        except ValueError:
            step_info.raw_reward = 0.0
            step_info.scaled_reward = INVALID_ACTION_PENALTY
            step_info.reward_reason = "Student response missing required <move> tag."
            raw_reward_values.append(0.0)
            scaled_reward_values.append(step_info.scaled_reward)
            trajectory.metrics["student_invalid_format"] = (
                trajectory.metrics.get("student_invalid_format", 0.0) + 1.0
            )
            if verbose:
                print("Student response:")
                print(_format_response(assistant_message.content))
                print(
                    "Penalty applied: response missing <move> direction; reward scaled to "
                    f"{step_info.scaled_reward:.3f}"
                )
            move_count += 1
            continue

        rng_state = random.getstate()
        preview_game = {
            "id": game["id"],
            "board": [row[:] for row in game["board"]],
        }
        try:
            apply_agent_move(preview_game, student_move_xml)
        except ValueError as exc:
            random.setstate(rng_state)
            step_info.raw_reward = 0.0
            step_info.scaled_reward = INVALID_ACTION_PENALTY
            step_info.reward_reason = f"Invalid student move: {exc}"
            raw_reward_values.append(0.0)
            scaled_reward_values.append(step_info.scaled_reward)
            message_lower = str(exc).lower()
            metric_key = (
                "student_noop_moves"
                if "did not change board" in message_lower
                else "student_invalid_move"
            )
            trajectory.metrics[metric_key] = (
                trajectory.metrics.get(metric_key, 0.0) + 1.0
            )
            if metric_key == "student_noop_moves":
                consecutive_noop_moves += 1
            else:
                consecutive_noop_moves = 0
            if verbose:
                print("Student response:")
                print(_format_response(assistant_message.content))
                print(
                    "Penalty applied: invalid student move "
                    f"({exc}); reward scaled to {step_info.scaled_reward:.3f}"
                )
            if consecutive_noop_moves >= 3:
                trajectory.metrics["student_noop_termination"] = (
                    trajectory.metrics.get("student_noop_termination", 0.0) + 1.0
                )
                step_info.done = True
                break
            move_count += 1
            continue
        random.setstate(rng_state)

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

        apply_agent_move(game, student_move_xml)
        move_count += 1
        trajectory.metrics.setdefault("student_valid_actions", 0)
        trajectory.metrics["student_valid_actions"] += 1
        consecutive_noop_moves = 0

        if check_game_finished(game):
            step_info.done = True
            trajectory.metrics["student_invalid_move"] = trajectory.metrics.get(
                "student_invalid_move", 0.0
            )
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

    if trajectory.steps and not trajectory.steps[-1].done:
        trajectory.steps[-1].done = True

    return_values: list[float] = []
    future_return = 0.0
    for step_info in reversed(trajectory.steps):
        reward_value = step_info.scaled_reward
        if step_info.done:
            future_return = reward_value
        else:
            future_return = reward_value + (DISCOUNT_FACTOR * future_return)
        step_info.reward = float(future_return)
        return_values.append(step_info.reward)
    return_values.reverse()

    if return_values:
        trajectory.reward = float(return_values[0])
    else:
        trajectory.reward = 0.0

    if scaled_reward_values:
        trajectory.metrics["student_step_reward_mean"] = float(
            sum(scaled_reward_values) / len(scaled_reward_values)
        )
        trajectory.metrics["student_step_reward_min"] = float(
            min(scaled_reward_values)
        )
        trajectory.metrics["student_step_reward_max"] = float(
            max(scaled_reward_values)
        )
        trajectory.metrics["student_step_reward_count"] = float(
            len(scaled_reward_values)
        )
        trajectory.metrics["student_step_reward_raw_mean"] = float(
            sum(raw_reward_values) / len(raw_reward_values)
        )
        trajectory.metrics["student_step_reward_raw_min"] = float(
            min(raw_reward_values)
        )
        trajectory.metrics["student_step_reward_raw_max"] = float(
            max(raw_reward_values)
        )
    else:
        trajectory.metrics["student_step_reward_mean"] = 0.0
        trajectory.metrics["student_step_reward_min"] = 0.0
        trajectory.metrics["student_step_reward_max"] = 0.0
        trajectory.metrics["student_step_reward_count"] = 0.0
        trajectory.metrics["student_step_reward_raw_mean"] = 0.0
        trajectory.metrics["student_step_reward_raw_min"] = 0.0
        trajectory.metrics["student_step_reward_raw_max"] = 0.0

    if return_values:
        trajectory.metrics["student_return_mean"] = float(
            sum(return_values) / len(return_values)
        )
        trajectory.metrics["student_return_min"] = float(min(return_values))
        trajectory.metrics["student_return_max"] = float(max(return_values))
    else:
        trajectory.metrics["student_return_mean"] = 0.0
        trajectory.metrics["student_return_min"] = 0.0
        trajectory.metrics["student_return_max"] = 0.0

    trajectory.metrics["invalid_move"] = trajectory.metrics.get(
        "student_invalid_move", 0.0
    )

    if verbose:
        print(
            f"Reward: {trajectory.reward:.3f} | Max tile: {max_value} | Board value: {board_value} | Moves: {move_count}"
        )

    return trajectory
