"""Rollout logic for on-policy PPO data collection."""

from __future__ import annotations

import math

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

SYSTEM_PROMPT = (
    "You are an excellent 2048 player. Think through the board inside a <think>...</think> block, "
    "then give the final move as <move>left</move>, <move>right</move>, <move>up</move>, or <move>down</move>. "
    "Do not call tools. Do not emit <tool_call> or any metadata—respond only with the <think> and <move> tags."
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
    """Extract <move>...</move> XML from the model output."""
    import re

    match = re.search(r"<move>\s*(up|down|left|right)\s*</move>", model_output, re.IGNORECASE)
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

        try:
            move_xml = extract_move_xml(assistant_message.content)
            apply_agent_move(game, move_xml)
            move_count += 1
            if verbose:
                print("Assistant response:")
                print(_format_response(assistant_message.content))
        except ValueError:
            trajectory.metrics["invalid_move"] = 1
            trajectory.reward = -1.0
            invalid_move = True
            if trajectory.steps:
                trajectory.steps[-1].reward = -1.0
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

    if not invalid_move:
        if agent_won:
            trajectory.reward = 2.0
        else:
            max_value_reward = (math.log(max_value, 2) - 1) / (
                math.log(WINNING_VALUE, 2) - 1
            )
            board_value_reward = (math.log(board_value, 2) - 1) / (
                math.log(WINNING_VALUE * 16, 2) - 1
            )
            trajectory.reward = max_value_reward + (board_value_reward * 0.2)

    if trajectory.steps:
        for step_info in trajectory.steps:
            step_info.reward = trajectory.reward

    if verbose:
        print(
            f"Reward: {trajectory.reward:.3f} | Max tile: {max_value} | Board value: {board_value}"
        )

    return trajectory
