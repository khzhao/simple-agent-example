"""
Text-based state encoders for game environments.
Converts game states to natural language and parses model outputs.
"""

import re
from typing import Dict, List

import numpy as np


class TextStateEncoder:
    """Encodes 2048 game state as text for LLM input."""

    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
    ACTION_TO_INT = {"up": 0, "down": 1, "left": 2, "right": 3}

    @staticmethod
    def encode_state(grid: np.ndarray, score: int, move_count: int = 0) -> str:
        """
        Convert game state to text description.

        Args:
            grid: 4x4 numpy array of tile values
            score: Current game score
            move_count: Number of moves made

        Returns:
            Text description of the game state
        """
        text_parts = [
            "You are playing the 2048 game.",
            f"Current score: {score}",
            f"Moves made: {move_count}",
            "\nBoard state:",
        ]

        # Describe the board row by row
        for row_idx, row in enumerate(grid):
            row_values = [str(int(val)) if val != 0 else "empty" for val in row]
            text_parts.append(f"  Row {row_idx + 1}: [{', '.join(row_values)}]")

        # Add game statistics
        max_tile = int(np.max(grid))
        empty_cells = int(np.sum(grid == 0))

        text_parts.extend(
            [
                f"\nLargest tile: {max_tile}",
                f"Empty cells: {empty_cells}",
                "\nChoose your next move from: up, down, left, right",
                "Respond with only the direction (e.g., 'up' or 'down' or 'left' or 'right').",
            ]
        )

        return "\n".join(text_parts)

    @staticmethod
    def encode_state_compact(grid: np.ndarray, score: int) -> str:
        """
        Compact version of state encoding.

        Args:
            grid: 4x4 numpy array
            score: Current score

        Returns:
            Compact text representation
        """
        lines = ["2048 Game | Score: " + str(score)]

        for row in grid:
            row_str = " | ".join(f"{int(v):4}" if v != 0 else "   ." for v in row)
            lines.append(row_str)

        lines.append("Action (up/down/left/right):")

        return "\n".join(lines)

    @staticmethod
    def create_prompt_with_state(
        grid: np.ndarray, score: int, move_count: int = 0
    ) -> str:
        """
        Create a complete prompt for the model including game state.

        This is the main method to use for generating model inputs.
        """
        return TextStateEncoder.encode_state(grid, score, move_count)

    @staticmethod
    def create_few_shot_examples() -> str:
        """
        Create few-shot examples for in-context learning.

        Returns:
            String with example state-action pairs
        """
        examples = """Here are some example moves:

Example 1:
Row 1: [2, 4, 8, 16]
Row 2: [empty, 2, 4, 8]
Row 3: [empty, empty, 2, 4]
Row 4: [empty, empty, empty, 2]
Best move: left (to merge tiles and consolidate)

Example 2:
Row 1: [2, empty, empty, empty]
Row 2: [4, empty, empty, empty]
Row 3: [8, empty, empty, empty]
Row 4: [16, empty, empty, empty]
Best move: down (to merge in the corner strategy)

Example 3:
Row 1: [2, 4, 2, 4]
Row 2: [4, 2, 4, 2]
Row 3: [2, 4, 2, 4]
Row 4: [4, 2, 4, 2]
Best move: left (to create merge opportunities)
"""
        return examples


class ActionParser:
    """Parses model output to extract actions."""

    VALID_ACTIONS = ["up", "down", "left", "right"]
    ACTION_TO_INT = {"up": 0, "down": 1, "left": 2, "right": 3}

    @staticmethod
    def parse_action(model_output: str) -> int:
        """
        Parse model output to extract action.

        Args:
            model_output: Raw text output from model

        Returns:
            Action integer (0-3) or -1 if parsing fails
        """
        # Clean and lowercase the output
        output_lower = model_output.lower().strip()

        # Direct match
        for action in ActionParser.VALID_ACTIONS:
            if action in output_lower:
                return ActionParser.ACTION_TO_INT[action]

        # Try to extract from common patterns
        # Pattern: "I will move [direction]" or "Move: [direction]"
        patterns = [
            r"move\s+(up|down|left|right)",
            r"go\s+(up|down|left|right)",
            r"direction:\s*(up|down|left|right)",
            r"action:\s*(up|down|left|right)",
            r"^(up|down|left|right)",  # Action at start
        ]

        for pattern in patterns:
            match = re.search(pattern, output_lower)
            if match:
                action_str = match.group(1)
                return ActionParser.ACTION_TO_INT.get(action_str, -1)

        # If no valid action found, return -1 (invalid)
        return -1

    @staticmethod
    def parse_action_with_confidence(model_output: str) -> tuple[int, float]:
        """
        Parse action and estimate confidence.

        Returns:
            Tuple of (action_int, confidence_score)
        """
        action = ActionParser.parse_action(model_output)

        # Simple confidence heuristic: based on clarity of response
        confidence = 1.0 if action != -1 else 0.0

        # Reduce confidence if output is very long (might be uncertain)
        if len(model_output) > 100:
            confidence *= 0.7

        return action, confidence

    @staticmethod
    def action_to_text(action_int: int) -> str:
        """Convert action integer to text."""
        action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        return action_map.get(action_int, "invalid")


class TicTacToeEncoder:
    """Encodes Tic Tac Toe game state as text for LLM input."""
    
    @staticmethod
    def encode_state(board: np.ndarray, move_count: int = 0) -> str:
        """
        Convert Tic Tac Toe state to text description.
        
        Args:
            board: 3x3 numpy array (-1=O, 0=empty, 1=X)
            move_count: Number of moves made
        
        Returns:
            Text description of the game state
        """
        symbols = {-1: "O", 0: ".", 1: "X"}
        
        text_parts = [
            "You are playing Tic Tac Toe as X.",
            "Your goal is to get three X's in a row (horizontal, vertical, or diagonal).",
            f"Moves made: {move_count}",
            "\nPosition numbers:",
            "0 | 1 | 2",
            "3 | 4 | 5",
            "6 | 7 | 8",
            "\nCurrent board:",
        ]
        
        # Show current board
        for row in board:
            text_parts.append(" | ".join(symbols[val] for val in row))
        
        # List valid moves
        valid_moves = [i for i in range(9) if board[i // 3, i % 3] == 0]
        if valid_moves:
            text_parts.append(f"\nValid moves: {', '.join(map(str, valid_moves))}")
            text_parts.append("\nPick a position number from the valid moves:")
        
        return "\n".join(text_parts)


class TicTacToeActionParser:
    """Parses LLM output to extract Tic Tac Toe move (0-8)."""
    
    @staticmethod
    def parse_action(model_output: str) -> int:
        """
        Parse model output to extract move position.
        
        Args:
            model_output: Raw text output from model
        
        Returns:
            Position 0-8, or -1 if parsing failed
        """
        if not model_output:
            return -1
        
        # Clean the output
        output_clean = model_output.strip().lower()
        
        # Try to find a number 0-8
        # Look for explicit numbers
        number_patterns = [
            r'\b([0-8])\b',  # Single digit 0-8
            r'position\s*([0-8])',
            r'move\s*([0-8])',
            r'pick\s*([0-8])',
            r'choose\s*([0-8])',
            r'^([0-8])',  # Number at start
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, output_clean)
            if match:
                try:
                    position = int(match.group(1))
                    if 0 <= position <= 8:
                        return position
                except (ValueError, IndexError):
                    continue
        
        # Look for any digit 0-8 in the output
        digits = re.findall(r'[0-8]', output_clean)
        if digits:
            return int(digits[0])
        
        return -1
    
    @staticmethod
    def action_to_text(action_int: int) -> str:
        """Convert action integer to text."""
        if 0 <= action_int <= 8:
            return f"position {action_int}"
        return "invalid"
