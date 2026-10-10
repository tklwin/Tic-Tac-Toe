"""
Tic-Tac-Toe Game Engine

This module defines the core TicTacToe class representing the game board,
move validation, state updates, win/draw detection, and backtracking (undo).
"""

from typing import List, Optional, Tuple

PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '

WINNING_COMBINATIONS: List[Tuple[int, int, int]] = [
    # Rows
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    # Columns
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    # Diagonals
    (0, 4, 8),
    (2, 4, 6)
]


class TicTacToe:
    """
    A representation of the 3x3 Tic-Tac-Toe game.
    
    Board indices:
      0 | 1 | 2
     ---+---+---
      3 | 4 | 5
     ---+---+---
      6 | 7 | 8
    """

    def __init__(self, board: Optional[List[str]] = None, current_player: Optional[str] = None):
        """
        Initialize the game board and current turn.
        
        Args:
            board: Optional list of 9 characters ('X', 'O', ' ').
            current_player: Optional starting player ('X' or 'O').
        """
        if board is not None:
            if len(board) != 9:
                raise ValueError("Board must contain exactly 9 cells.")
            self.board = list(board)
        else:
            self.board = [EMPTY] * 9

        if current_player is not None:
            self.current_player = current_player
        else:
            x_count = self.board.count(PLAYER_X)
            o_count = self.board.count(PLAYER_O)
            self.current_player = PLAYER_O if x_count > o_count else PLAYER_X

    def get_available_moves(self) -> List[int]:
        """Return a list of integer indices (0-8) for empty cells."""
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def is_valid_move(self, pos: int) -> bool:
        """Check whether placing a mark at `pos` is valid."""
        return 0 <= pos < 9 and self.board[pos] == EMPTY

    def make_move(self, pos: int, player: Optional[str] = None) -> bool:
        """
        Execute a move on the board for the specified player (or current_player).
        Switches the current_player upon successful move execution.
        
        Returns:
            True if the move succeeded, False otherwise.
        """
        if not self.is_valid_move(pos):
            return False

        p = player if player is not None else self.current_player
        self.board[pos] = p
        self.current_player = PLAYER_O if p == PLAYER_X else PLAYER_X
        return True

    def undo_move(self, pos: int) -> bool:
        """
        Revert a move at `pos`, restoring it to EMPTY and flipping turn back.
        Crucial for recursive game tree backtracking in Minimax.
        """
        if 0 <= pos < 9 and self.board[pos] != EMPTY:
            self.board[pos] = EMPTY
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
            return True
        return False

    def get_winning_line(self) -> Optional[Tuple[int, int, int]]:
        """Return the tuple of 3 indices forming a winning line, or None."""
        for combo in WINNING_COMBINATIONS:
            a, b, c = combo
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return combo
        return None

    def check_winner(self) -> Optional[str]:
        """Return 'X' or 'O' if a player has won, otherwise None."""
        line = self.get_winning_line()
        if line is not None:
            return self.board[line[0]]
        return None

    def is_draw(self) -> bool:
        """Check if game ended in a draw (no winner and no empty cells)."""
        return self.check_winner() is None and len(self.get_available_moves()) == 0

    def is_game_over(self) -> bool:
        """Check if game has reached a terminal state (win or draw)."""
        return self.check_winner() is not None or self.is_draw()

    def clone(self) -> "TicTacToe":
        """Return a deep copy of the current game state."""
        return TicTacToe(board=list(self.board), current_player=self.current_player)

    def reset(self) -> None:
        """Reset board to empty initial state."""
        self.board = [EMPTY] * 9
        self.current_player = PLAYER_X

    def to_dict(self) -> dict:
        """Serialize current game state for JSON APIs."""
        return {
            "board": list(self.board),
            "current_player": self.current_player,
            "winner": self.check_winner(),
            "winning_line": self.get_winning_line(),
            "is_draw": self.is_draw(),
            "is_game_over": self.is_game_over(),
            "available_moves": self.get_available_moves()
        }

