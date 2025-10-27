"""Replay the recommended 2048 actions from a JSONL SFT dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable

from simple_agent_example.env.game import (Direction, TwentyFortyEightGame,
                                           apply_agent_move, render_board)

BOARD_MARKER = "Current board:\n"
RETURN_MARKER = "Return your answer"
MOVE_REGEX = re.compile(r"<move>\s*(left|right|up|down)\s*</move>", re.IGNORECASE)


def _parse_board(prompt: str) -> list[list[int | None]]:
    """Extract the 4x4 board state from the prompt."""
    try:
        start = prompt.index(BOARD_MARKER) + len(BOARD_MARKER)
    except ValueError as exc:
        raise ValueError("Prompt missing board marker") from exc

    board_lines: list[str] = []
    for line in prompt[start:].splitlines():
        if line.startswith(RETURN_MARKER):
            break
        if not line.strip():
            continue
        board_lines.append(line)

    if len(board_lines) != 4:
        raise ValueError(f"Expected 4 board rows, found {len(board_lines)}")

    board: list[list[int | None]] = []
    for row_text in board_lines:
        cells = []
        for cell_text in row_text.split("|"):
            cell = cell_text.strip()
            if not cell or cell == "_":
                cells.append(None)
            else:
                try:
                    cells.append(int(cell))
                except ValueError as exc:
                    raise ValueError(f"Unable to parse cell value '{cell}'") from exc
        if len(cells) != 4:
            raise ValueError(
                f"Expected 4 columns, found {len(cells)} in row '{row_text}'"
            )
        board.append(cells)

    return board


def _parse_move(completion: str) -> Direction:
    """Extract the recommended move from the completion string."""
    match = MOVE_REGEX.search(completion)
    if not match:
        raise ValueError("Completion missing <move> tag")
    return Direction(match.group(1).lower())


def _render_cells(cells: list[list[int | None]]) -> str:
    """Render a board without mutating the provided cells."""
    snapshot = [row[:] for row in cells]
    game_snapshot: TwentyFortyEightGame = TwentyFortyEightGame(
        id="snapshot",
        board=snapshot,
    )
    return render_board(game_snapshot)


def _load_entries(
    dataset_path: Path,
) -> Iterable[tuple[list[list[int | None]], Direction]]:
    """Yield (board, move) tuples from the dataset."""
    with dataset_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            try:
                board = _parse_board(record["prompt"])
            except KeyError as exc:
                raise ValueError(f"Missing prompt on line {line_number}") from exc
            try:
                move = _parse_move(record["completion"])
            except KeyError as exc:
                raise ValueError(f"Missing completion on line {line_number}") from exc

            yield board, move


def replay_dataset(
    dataset_path: Path,
    limit: int | None = None,
    seed: int | None = None,
) -> None:
    """Replay recommended actions sequentially and print the outcomes."""
    successes = 0
    failures = 0

    for index, (board, move) in enumerate(_load_entries(dataset_path)):
        if limit is not None and index >= limit:
            break

        if seed is not None:
            random.seed(seed + index)

        initial_board_text = _render_cells(board)
        game: TwentyFortyEightGame = TwentyFortyEightGame(
            id=f"replay-{index}",
            board=[row[:] for row in board],
        )

        try:
            apply_agent_move(game, f"<move>{move.value}</move>")
        except ValueError as exc:
            failures += 1
            print(f"[{index + 1}] Move '{move.value}' failed: {exc}")
            continue

        successes += 1
        print(f"[{index + 1}] Recommended move: {move.value}")
        print("Before:")
        print(initial_board_text)
        print("After:")
        print(render_board(game))
        print("-" * 32)

    print("Replay complete.")
    print(f"Successful moves: {successes}")
    print(f"Failed moves: {failures}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the JSONL dataset (e.g. data/sft/large_2048_sft.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to replay",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for deterministic tile spawns",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    replay_dataset(args.dataset, args.limit, args.seed)


if __name__ == "__main__":
    main()
