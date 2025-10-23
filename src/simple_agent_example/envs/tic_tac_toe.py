"""
Tic Tac Toe environment for reinforcement learning.

The agent plays as X against a simple opponent playing as O.
"""
from typing import Any, Dict, Tuple
import random

import gymnasium as gym
import numpy as np


class TicTacToeEnv(gym.Env):
    """
    Tic Tac Toe environment following Gymnasium interface.
    
    Agent plays as X (1), opponent plays as O (-1).
    Empty cells are 0.
    
    Action space: Discrete(9) - positions 0-8 in row-major order:
        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
    
    Observation space: Box(3, 3) - 3x3 grid with values {-1, 0, 1}
    """
    
    def __init__(self, opponent_strategy: str = "random"):
        """
        Initialize Tic Tac Toe environment.
        
        Args:
            opponent_strategy: Strategy for opponent
                - "random": Random valid moves
                - "smart": Blocks wins and takes winning moves
                - "minimax": Optimal play (very hard to beat)
        """
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.int32
        )
        
        self.opponent_strategy = opponent_strategy
        self.board = None
        self.move_count = 0
        self.winner = None
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the game to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((3, 3), dtype=np.int32)
        self.move_count = 0
        self.winner = None
        
        info = self._get_info()
        return self.board.copy(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Agent (X) makes a move, then opponent (O) responds.
        
        Args:
            action: Position 0-8 to place X
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        terminated = False
        
        # Agent's move (X)
        row, col = action // 3, action % 3
        
        # Check if move is valid
        if self.board[row, col] != 0:
            # Invalid move - penalty and game over
            reward = -10.0
            terminated = True
            info = self._get_info()
            return self.board.copy(), reward, terminated, False, info
        
        # Make the move
        self.board[row, col] = 1  # X
        self.move_count += 1
        
        # Check if agent won
        if self._check_winner(1):
            self.winner = 1
            reward = 10.0  # Win reward
            terminated = True
            info = self._get_info()
            return self.board.copy(), reward, terminated, False, info
        
        # Check if board is full (draw)
        if self._is_board_full():
            self.winner = 0
            reward = 0.0  # Draw
            terminated = True
            info = self._get_info()
            return self.board.copy(), reward, terminated, False, info
        
        # Opponent's move (O)
        opponent_action = self._get_opponent_move()
        if opponent_action is not None:
            o_row, o_col = opponent_action // 3, opponent_action % 3
            self.board[o_row, o_col] = -1  # O
            self.move_count += 1
            
            # Check if opponent won
            if self._check_winner(-1):
                self.winner = -1
                reward = -10.0  # Loss penalty
                terminated = True
            
            # Check if board is full (draw)
            elif self._is_board_full():
                self.winner = 0
                reward = 0.0  # Draw
                terminated = True
        
        info = self._get_info()
        return self.board.copy(), reward, terminated, False, info
    
    def _get_opponent_move(self) -> int:
        """Get opponent's move based on strategy."""
        if self.opponent_strategy == "random":
            return self._random_move()
        elif self.opponent_strategy == "smart":
            return self._smart_move()
        elif self.opponent_strategy == "minimax":
            return self._minimax_move()
        else:
            return self._random_move()
    
    def _random_move(self) -> int:
        """Random valid move."""
        valid_moves = self._get_valid_moves()
        if valid_moves:
            return self.np_random.choice(valid_moves)
        return None
    
    def _smart_move(self) -> int:
        """
        Smart strategy: 
        1. Take winning move if available
        2. Block opponent's winning move
        3. Take center if available
        4. Take corner
        5. Random otherwise
        """
        valid_moves = self._get_valid_moves()
        if not valid_moves:
            return None
        
        # Check for winning move
        for move in valid_moves:
            row, col = move // 3, move % 3
            self.board[row, col] = -1
            if self._check_winner(-1):
                self.board[row, col] = 0  # Undo
                return move
            self.board[row, col] = 0  # Undo
        
        # Check for blocking move
        for move in valid_moves:
            row, col = move // 3, move % 3
            self.board[row, col] = 1
            if self._check_winner(1):
                self.board[row, col] = 0  # Undo
                return move
            self.board[row, col] = 0  # Undo
        
        # Take center
        if 4 in valid_moves:
            return 4
        
        # Take corners
        corners = [0, 2, 6, 8]
        corner_moves = [m for m in valid_moves if m in corners]
        if corner_moves:
            return self.np_random.choice(corner_moves)
        
        # Random
        return self.np_random.choice(valid_moves)
    
    def _minimax_move(self) -> int:
        """Minimax algorithm for optimal play."""
        best_score = float('-inf')
        best_move = None
        
        for move in self._get_valid_moves():
            row, col = move // 3, move % 3
            self.board[row, col] = -1
            score = self._minimax(False)
            self.board[row, col] = 0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, is_maximizing: bool) -> float:
        """Minimax recursive helper."""
        # Check terminal states
        if self._check_winner(-1):
            return 1  # Opponent wins (good for minimizer)
        if self._check_winner(1):
            return -1  # Agent wins (bad for minimizer)
        if self._is_board_full():
            return 0  # Draw
        
        if is_maximizing:
            # Opponent's turn
            best_score = float('-inf')
            for move in self._get_valid_moves():
                row, col = move // 3, move % 3
                self.board[row, col] = -1
                score = self._minimax(False)
                self.board[row, col] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            # Agent's turn
            best_score = float('inf')
            for move in self._get_valid_moves():
                row, col = move // 3, move % 3
                self.board[row, col] = 1
                score = self._minimax(True)
                self.board[row, col] = 0
                best_score = min(score, best_score)
            return best_score
    
    def _get_valid_moves(self) -> list:
        """Get list of valid move positions."""
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]
    
    def _is_board_full(self) -> bool:
        """Check if board is full."""
        return np.all(self.board != 0)
    
    def _check_winner(self, player: int) -> bool:
        """
        Check if player has won.
        
        Args:
            player: 1 for X, -1 for O
        
        Returns:
            True if player has won
        """
        # Check rows
        for row in range(3):
            if np.all(self.board[row, :] == player):
                return True
        
        # Check columns
        for col in range(3):
            if np.all(self.board[:, col] == player):
                return True
        
        # Check diagonals
        if np.all(np.diag(self.board) == player):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        return {
            "move_count": self.move_count,
            "valid_moves": self._get_valid_moves(),
            "winner": self.winner,  # 1=X wins, -1=O wins, 0=draw, None=ongoing
            "board_full": self._is_board_full(),
        }
    
    def render(self):
        """Render the game state."""
        symbols = {-1: "O", 0: " ", 1: "X"}
        
        print("\n" + "=" * 13)
        for i, row in enumerate(self.board):
            print("|", end="")
            for val in row:
                print(f" {symbols[val]} |", end="")
            print()
            if i < 2:
                print("-" * 13)
        print("=" * 13)
        
        if self.winner is not None:
            if self.winner == 1:
                print("🎉 X (Agent) wins!")
            elif self.winner == -1:
                print("💔 O (Opponent) wins!")
            else:
                print("🤝 Draw!")
        print()
    
    def get_text_state(self) -> str:
        """
        Get text representation of the game state for LLM input.
        
        Returns:
            String describing the current board state
        """
        symbols = {-1: "O", 0: ".", 1: "X"}
        
        text = "Tic Tac Toe board (you are X):\n"
        text += "Position numbers:\n"
        text += "0 | 1 | 2\n"
        text += "---------\n"
        text += "3 | 4 | 5\n"
        text += "---------\n"
        text += "6 | 7 | 8\n\n"
        
        text += "Current board:\n"
        for i, row in enumerate(self.board):
            text += " | ".join(symbols[val] for val in row)
            if i < 2:
                text += "\n---------\n"
        
        text += f"\n\nMove count: {self.move_count}\n"
        
        valid_moves = self._get_valid_moves()
        if valid_moves:
            text += f"Valid moves: {', '.join(map(str, valid_moves))}\n"
        
        text += "\nYour move (pick a number from valid moves):"
        
        return text

