"""Rollout logic for on-policy PPO data collection."""

from __future__ import annotations

import difflib
import math
import re

from .env import (WINNING_VALUE, apply_agent_move, check_game_finished,
                  generate_game, max_cell_value, render_board,
                  total_board_value)
from .tinker_client import ChatMessage, TinkerTrainableModel, Trajectory


def _format_board(board_view: str) -> str:
    lines: list[str] = []
    for raw_line in board_view.splitlines():
        cells = [cell.strip() for cell in raw_line.split("|")]
        pretty = " | ".join(f"{(cell if cell != '_' else '.'):>3}" for cell in cells)
        lines.append(pretty)
    return "\n".join(lines)


def _format_response(content: str) -> str:
    return "\n".join(f"    {line}" for line in content.splitlines())

VALID_ACTION_BONUS = 0.01
INVALID_BASE_PENALTY = -5.0
MINIMUM_MOVES_FOR_REWARD = 3
SHORT_GAME_BASE_PENALTY = -2.0
WIN_REWARD = 2.0
FORMAT_ACCEPT_THRESHOLD = 0.95
INVALID_PENALTY_SCALE = 3.0
EXPECTED_FORMAT_CANONICAL = "<think></think><move>direction</move>"
RESPONSE_PATTERN = re.compile(
    r"^<think>.*?</think>\s*<move>(up|down|left|right)</move>$",
    re.IGNORECASE | re.DOTALL,
)


def _canonicalize_for_similarity(raw_text: str) -> str:
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "<think></think>", cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        r"<move>\s*(up|down|left|right)\s*</move>",
        "<move>direction</move>",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _format_similarity(raw_text: str) -> float:
    canonical = _canonicalize_for_similarity(raw_text)
    if not canonical:
        return 0.0
    ratio = difflib.SequenceMatcher(None, canonical, EXPECTED_FORMAT_CANONICAL).ratio()
    return float(max(0.0, min(1.0, ratio)))


def _invalid_reward(valid_actions: int, format_similarity: float) -> float:
    severity = 1.0 + (1.0 - format_similarity) * INVALID_PENALTY_SCALE
    return severity * INVALID_BASE_PENALTY + VALID_ACTION_BONUS * valid_actions


def _compute_reward(
    *,
    invalid_move: bool,
    move_count: int,
    valid_actions: int,
    agent_won: bool,
    max_value: int,
    board_value: int,
    format_similarity: float,
) -> float:
    """Score trajectories with format-aware rewards."""
    negative_reward = _invalid_reward(valid_actions, format_similarity)

    if invalid_move:
        return negative_reward

    if format_similarity < FORMAT_ACCEPT_THRESHOLD:
        return negative_reward

    if move_count < MINIMUM_MOVES_FOR_REWARD:
        return SHORT_GAME_BASE_PENALTY + VALID_ACTION_BONUS * valid_actions

    if agent_won:
        base_reward = WIN_REWARD
    else:
        max_value_reward = (math.log(max_value, 2) - 1) / (math.log(WINNING_VALUE, 2) - 1)
        board_value_reward = (math.log(board_value, 2) - 1) / (
            math.log(WINNING_VALUE * 16, 2) - 1
        )
        base_reward = max_value_reward + (board_value_reward * 0.2)

    base_reward += VALID_ACTION_BONUS * valid_actions
    return base_reward * format_similarity

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


async def rollout(
    model: TinkerTrainableModel,
    step: int,
    *,
    is_validation: bool = False,
    verbose: bool = False,
) -> Trajectory:
    """Generate a single trajectory by playing a game to completion."""
    game = generate_game()
    move_count = 0

    trajectory = Trajectory(
        messages=[ChatMessage(role="system", content=SYSTEM_PROMPT)],
        metadata={"game_id": game["id"], "step": step, "validation": is_validation},
        reward=0.0,
        metrics={},
    )

    invalid_move = False

    while True:
        board_view = render_board(game)
        trajectory.messages.append(ChatMessage(role="user", content=board_view))
        if verbose:
            print(f"\n=== Step {move_count:02d} ===")
            print(_format_board(board_view))

        prompt_text = build_prompt(board_view)
        assistant_message, step_info = await model.sample_action(prompt_text)
        trajectory.messages.append(assistant_message)
        trajectory.steps.append(step_info)

        format_similarity = _format_similarity(assistant_message.content)
        trajectory.metrics["format_similarity_sum"] = (
            trajectory.metrics.get("format_similarity_sum", 0.0) + format_similarity
        )
        trajectory.metrics["format_similarity_count"] = (
            trajectory.metrics.get("format_similarity_count", 0) + 1
        )

        try:
            move_xml = extract_move_xml(assistant_message.content)
            apply_agent_move(game, move_xml)
            move_count += 1
            trajectory.metrics.setdefault("valid_actions", 0)
            trajectory.metrics["valid_actions"] += 1
            if verbose:
                print("Assistant response:")
                print(_format_response(assistant_message.content))
        except ValueError:
            trajectory.metrics["invalid_move"] = 1
            invalid_move = True
            if trajectory.steps:
                trajectory.steps[-1].done = True
            break

        if check_game_finished(game):
            trajectory.metrics["invalid_move"] = 0
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

    valid_actions = trajectory.metrics.get("valid_actions", 0)

    format_similarity_sum = trajectory.metrics.pop("format_similarity_sum", 0.0)
    format_similarity_count = trajectory.metrics.pop("format_similarity_count", 0)
    format_similarity = (
        float(format_similarity_sum / format_similarity_count)
        if format_similarity_count
        else 0.0
    )
    trajectory.metrics["format_similarity"] = format_similarity

    trajectory.reward = _compute_reward(
        invalid_move=invalid_move,
        move_count=move_count,
        valid_actions=valid_actions,
        agent_won=agent_won,
        max_value=max_value,
        board_value=board_value,
        format_similarity=format_similarity,
    )
    if trajectory.steps:
        for step_info in trajectory.steps:
            step_info.reward = trajectory.reward

    if verbose:
        print(
            f"Reward: {trajectory.reward:.3f} | Max tile: {max_value} | Board value: {board_value} | Valid actions: {valid_actions}"
        )

    return trajectory
