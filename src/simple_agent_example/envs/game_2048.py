from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np


class Game2048Env(gym.Env):
    """
    2048 game environment following Gymnasium interface.

    Action space: Discrete(4) - 0: Up, 1: Down, 2: Left, 3: Right
    Observation space: Box(4, 4) - 4x4 grid with tile values
    """

    def __init__(self):
        """
        Initialize the 2048 environment with ART-style reward function.

        Reward function returns the max tile value on the board, encouraging
        the agent to create higher tiles.
        """
        super().__init__()

        self.grid_size = 4

        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: 4x4 grid with values from 0 to 2^17 (131072)
        self.observation_space = gym.spaces.Box(
            low=0, high=131072, shape=(self.grid_size, self.grid_size), dtype=np.int32
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
        Execute one step in the environment using ART-style reward function.

        Reward = max_tile_value on the board after the action.
        This simple reward function encourages the agent to create higher tiles.

        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute move
        moved = self._move(action)

        if moved:
            # Add random tile after successful move
            self._add_random_tile()
            self.move_count += 1

        # Update max tile
        self.max_tile = int(np.max(self.grid))

        # ART-style reward: simply the max tile value
        reward = float(self.max_tile)

        # Check if game is over (no valid moves)
        terminated = not self._has_valid_moves()
        truncated = False

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
        return self._move_left_grid(self.grid)

    def _move_left_grid(self, grid: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Move/merge tiles left on a given grid.

        Args:
            grid: The grid to process

        Returns:
            Tuple of (new_grid, score_delta)
        """
        new_grid = np.zeros_like(grid)
        score_delta = 0

        for row_idx in range(self.grid_size):
            # Get non-zero values
            row = grid[row_idx, :]
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
        # Flip, move left, flip back (without modifying self.grid)
        flipped_grid = np.fliplr(self.grid)
        new_grid, score_delta = self._move_left_grid(flipped_grid)
        return np.fliplr(new_grid), score_delta

    def _move_up(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles up."""
        # Transpose, move left, transpose back (without modifying self.grid)
        transposed_grid = self.grid.T
        new_grid, score_delta = self._move_left_grid(transposed_grid)
        return new_grid.T, score_delta

    def _move_down(self) -> Tuple[np.ndarray, int]:
        """Move/merge tiles down."""
        # Transpose, flip, move left, flip back, transpose back (without modifying self.grid)
        transposed_grid = self.grid.T
        flipped_grid = np.fliplr(transposed_grid)
        new_grid, score_delta = self._move_left_grid(flipped_grid)
        unflipped_grid = np.fliplr(new_grid)
        return unflipped_grid.T, score_delta

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

    def get_valid_actions(self) -> list[int]:
        """
        Get list of valid actions that would change the board state.

        Returns:
            List of valid action indices (0=Up, 1=Down, 2=Left, 3=Right)
        """
        valid_actions = []

        for action in range(4):
            # Simulate the move without modifying the actual grid
            if action == 0:  # Up
                transposed_grid = self.grid.T
                new_grid, _ = self._move_left_grid(transposed_grid)
                new_grid = new_grid.T
            elif action == 1:  # Down
                transposed_grid = self.grid.T
                flipped_grid = np.fliplr(transposed_grid)
                new_grid, _ = self._move_left_grid(flipped_grid)
                new_grid = np.fliplr(new_grid).T
            elif action == 2:  # Left
                new_grid, _ = self._move_left_grid(self.grid)
            else:  # Right (action == 3)
                flipped_grid = np.fliplr(self.grid)
                new_grid, _ = self._move_left_grid(flipped_grid)
                new_grid = np.fliplr(new_grid)

            # Check if the move would change the board
            if not np.array_equal(self.grid, new_grid):
                valid_actions.append(action)

        return valid_actions

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            "score": self.score,
            "max_tile": int(np.max(self.grid)) if self.grid is not None else 0,
            "move_count": self.move_count,
            "empty_cells": int(np.sum(self.grid == 0)) if self.grid is not None else 0,
            "valid_actions": self.get_valid_actions() if self.grid is not None else [],
        }

    def render(self):
        """Render the game state."""
        print(
            f"\nScore: {self.score} | Max Tile: {self.max_tile} | Moves: {self.move_count}"
        )
        print("-" * 25)
        for row in self.grid:
            print(
                "|"
                + "|".join(f"{int(val):5}" if val != 0 else "     " for val in row)
                + "|"
            )
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

        # Show only valid actions
        valid_actions = self.get_valid_actions()
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        valid_action_names = [action_names[action] for action in valid_actions]

        if valid_action_names:
            text += f"\nAvailable actions: {', '.join(valid_action_names)}"
        else:
            text += "\nNo valid actions available (game over)"

        return text
