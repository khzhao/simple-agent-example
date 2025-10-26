"""Core 2048 environment logic mirroring the ART example."""

from __future__ import annotations

import random
import string
import xml.etree.ElementTree as ET
from enum import Enum
from typing import TypedDict

WINNING_VALUE = 128


class TwentyFortyEightGame(TypedDict):
    """Game state container."""

    id: str
    board: list[list[int | None]]


class Direction(str, Enum):
    """Valid move directions."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


def populate_random_cell(game: TwentyFortyEightGame) -> None:
    """Randomly populate an empty cell with a 2 (90%) or 4 (10%)."""
    empties = [
        (row_idx, col_idx)
        for row_idx, row in enumerate(game["board"])
        for col_idx, value in enumerate(row)
        if value is None
    ]
    if not empties:
        return

    row_idx, col_idx = random.choice(empties)
    game["board"][row_idx][col_idx] = 2 if random.random() < 0.9 else 4


def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    """Create a fresh 2048 board with two populated cells."""
    game_id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    board = [[None for _ in range(board_length)] for _ in range(board_length)]
    game = TwentyFortyEightGame(id=game_id, board=board)

    populate_random_cell(game)
    populate_random_cell(game)
    return game


def render_board(game: TwentyFortyEightGame) -> str:
    """Render the board as a fixed-width grid string."""
    board = game["board"]
    max_cell_width = max(
        (len(str(cell)) for row in board for cell in row if cell is not None),
        default=1,
    )

    lines: list[str] = []
    for row in board:
        cells = [
            (
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
            )
            for cell in row
        ]
        lines.append("|".join(cells))
    return "\n".join(lines)


def _condense_sequence(sequence: list[int | None]) -> list[int | None]:
    """Slide and merge a one-dimensional slice of the board."""
    gapless = [cell for cell in sequence if cell is not None]
    condensed: list[int | None] = []

    idx = 0
    while idx < len(gapless):
        if idx + 1 < len(gapless) and gapless[idx] == gapless[idx + 1]:
            condensed.append(gapless[idx] * 2)
            idx += 2
        else:
            condensed.append(gapless[idx])
            idx += 1

    return condensed + [None] * (4 - len(condensed))


def _condense_in_place(game: TwentyFortyEightGame, direction: Direction) -> None:
    """Apply merges/slides to the board for a given direction."""
    board = game["board"]
    if direction == Direction.LEFT:
        for row_idx, row in enumerate(board):
            condensed = _condense_sequence(row)
            board[row_idx] = condensed
    elif direction == Direction.RIGHT:
        for row_idx, row in enumerate(board):
            condensed = list(reversed(_condense_sequence(list(reversed(row)))))
            board[row_idx] = condensed
    elif direction == Direction.UP:
        for col_idx in range(len(board[0])):
            column = [board[row_idx][col_idx] for row_idx in range(len(board))]
            condensed = _condense_sequence(column)
            for row_idx in range(len(board)):
                board[row_idx][col_idx] = condensed[row_idx]
    elif direction == Direction.DOWN:
        for col_idx in range(len(board[0])):
            column = [board[row_idx][col_idx] for row_idx in range(len(board))]
            condensed = list(reversed(_condense_sequence(list(reversed(column)))))
            for row_idx in range(len(board)):
                board[row_idx][col_idx] = condensed[row_idx]


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    """Parse XML move and update board state."""
    try:
        root = ET.fromstring(move_xml)
        direction_text = root.text or ""
    except ET.ParseError as exc:
        raise ValueError("Invalid XML payload") from exc

    try:
        direction = Direction(direction_text)
    except ValueError as exc:
        raise ValueError("Invalid direction") from exc

    _condense_in_place(game, direction)
    populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    """Return the largest tile value on the board."""
    return max(
        (cell for row in game["board"] for cell in row if cell is not None), default=0
    )


def total_board_value(game: TwentyFortyEightGame) -> int:
    """Return the sum of all tile values."""
    return sum(cell for row in game["board"] for cell in row if cell is not None)


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    """Check win condition or no-move condition."""
    if max_cell_value(game) >= WINNING_VALUE:
        return True

    has_empty = any(cell is None for row in game["board"] for cell in row)
    return not has_empty
