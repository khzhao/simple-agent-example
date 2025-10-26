"""Rollout logic for on-policy PPO data collection."""

from __future__ import annotations

import math

from .env import (WINNING_VALUE, apply_agent_move, check_game_finished,
                  generate_game, max_cell_value, render_board,
                  total_board_value)
from .tinker_client import ChatMessage, TinkerTrainableModel, Trajectory

SYSTEM_PROMPT = (
    "You are an excellent 2048 player. Always choose the move most likely to combine tiles and "
    "eventually reach 2048. Valid moves: left, right, up, down. Respond ONLY with an XML element "
    "holding the move, e.g. <move>left</move>."
)


def build_prompt(board_view: str) -> str:
    """Create the textual prompt sent to the language model."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Current board:\n"
        f"{board_view}\n"
        "Return your move exactly as an XML element, e.g. <move>left</move>."
    )


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
            print(board_view)

        prompt_text = build_prompt(board_view)
        assistant_message, step_info = await model.sample_action(prompt_text)
        trajectory.messages.append(assistant_message)
        trajectory.steps.append(step_info)

        try:
            apply_agent_move(game, assistant_message.content)
            move_count += 1
            if verbose:
                print(assistant_message.content)
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

    return trajectory
