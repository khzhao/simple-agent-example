import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class Game2048Env(gym.Env):
    """
    2048 game environment following Gymnasium interface.

    Action space: Discrete(4) - 0: Up, 1: Down, 2: Left, 3: Right
    Observation space: Box(4, 4) - 4x4 grid with tile values
    """

    metadata = {"render_modes": ["human", "text"]}

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.render_mode = render_mode
        self.grid_size = 4

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: 4x4 grid with values from 0 to 2^17 (131072)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=131072,
            shape=(self.grid_size, self.grid_size),
            dtype=np.int32
        )

        self.grid = None
        self.score = 0
        self.max_tile = 0
        self.move_count = 0

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the game to initial state."""
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.move_count = 0

        # Add two initial tiles
        self._add_random_tile()
        self._add_random_tile()

        info = self._get_info()

        return self.grid.copy(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right

        Returns:
            observation, reward, terminated, truncated, info
        """
        previous_score = self.score
        previous_grid = self.grid.copy()

        # Execute move
        moved = self._move(action)

        # Calculate reward
        reward = 0.0
        if moved:
            # Add random tile after successful move
            self._add_random_tile()

            # Reward is the score increase from the move
            score_increase = self.score - previous_score
            reward = float(score_increase)

            # Small bonus for making valid moves
            reward += 1.0

            self.move_count += 1
        else:
            # Penalty for invalid moves
            reward = -10.0

        # Check if game is over (no valid moves)
        terminated = not self._has_valid_moves()
        truncated = False

        # Additional penalty for losing
        if terminated:
            reward -= 50.0

        # Update max tile
        self.max_tile = int(np.max(self.grid))

        info = self._get_info()

        return self.grid.copy(), reward, terminated, truncated, info

    def _move(self, action: int) -> bool:
        """
        Execute a move in the specified direction.

        Returns:
            True if the move changed the board, False otherwise
        """
        original_grid = self.grid.copy()

        if action == 0:  # Up
            self.grid, score_delta = self._move_up()
        elif action == 1:  # Down
            self.grid, score_delta = self._move_down()
        elif action == 2:  # Left
            self.grid, score_delta = self._move_left()
        elif action == 3:  # Right
            self.grid, score_delta = self._move_right()
        else:
            raise ValueError(f"Invalid action: {action}")

        self.score += score_delta

        # Check if board changed
        return not np.array_equal(original_grid, self.grid)

    def _move_left(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles left."""
        new_grid = np.zeros_like(self.grid)
        score_delta = 0

        for row_idx in range(self.grid_size):
            # Get non-zero values
            row = self.grid[row_idx, :]
            non_zero = row[row != 0]

            # Merge tiles
            merged = []
            skip = False
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue

                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    # Merge
                    merged_value = non_zero[i] * 2
                    merged.append(merged_value)
                    score_delta += merged_value
                    skip = True
                else:
                    merged.append(non_zero[i])

            # Place merged tiles
            for i, val in enumerate(merged):
                new_grid[row_idx, i] = val

        return new_grid, score_delta

    def _move_right(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles right."""
        # Flip, move left, flip back
        self.grid = np.fliplr(self.grid)
        new_grid, score_delta = self._move_left()
        return np.fliplr(new_grid), score_delta

    def _move_up(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles up."""
        # Transpose, move left, transpose back
        self.grid = self.grid.T
        new_grid, score_delta = self._move_left()
        return new_grid.T, score_delta

    def _move_down(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles down."""
        # Transpose, move right, transpose back
        self.grid = self.grid.T
        new_grid, score_delta = self._move_right()
        return new_grid.T, score_delta

    def _add_random_tile(self) -> None:
        """Add a random tile (2 or 4) to an empty cell."""
        # Find empty cells
        empty_cells = np.argwhere(self.grid == 0)

        if len(empty_cells) > 0:
            # Choose random empty cell
            idx = self.np_random.integers(0, len(empty_cells))
            row, col = empty_cells[idx]

            # 90% chance of 2, 10% chance of 4
            value = 2 if self.np_random.random() < 0.9 else 4
            self.grid[row, col] = value

    def _has_valid_moves(self) -> bool:
        """Check if any valid moves are available."""
        # Check for empty cells
        if np.any(self.grid == 0):
            return True

        # Check for possible merges horizontally
        for row in range(self.grid_size):
            for col in range(self.grid_size - 1):
                if self.grid[row, col] == self.grid[row, col + 1]:
                    return True

        # Check for possible merges vertically
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                if self.grid[row, col] == self.grid[row + 1, col]:
                    return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            "score": self.score,
            "max_tile": int(np.max(self.grid)) if self.grid is not None else 0,
            "move_count": self.move_count,
            "empty_cells": int(np.sum(self.grid == 0)) if self.grid is not None else 0,
        }

    def render(self):
        """Render the game state."""
        if self.render_mode == "human" or self.render_mode == "text":
            print(f"\nScore: {self.score} | Max Tile: {self.max_tile} | Moves: {self.move_count}")
            print("-" * 25)
            for row in self.grid:
                print("|" + "|".join(f"{int(val):5}" if val != 0 else "     " for val in row) + "|")
                print("-" * 25)
            print()

    def get_text_state(self) -> str:
        """
        Get text representation of the game state for LLM input.

        Returns:
            String describing the current board state
        """
        text = f"Current 2048 game state (Score: {self.score}):\n"

        # Add grid representation
        for row_idx, row in enumerate(self.grid):
            row_text = " | ".join(str(int(val)) if val != 0 else "." for val in row)
            text += f"Row {row_idx + 1}: {row_text}\n"

        text += f"\nEmpty cells: {np.sum(self.grid == 0)}"
        text += f"\nMax tile: {np.max(self.grid)}"
        text += "\nAvailable actions: up, down, left, right"

        return text
