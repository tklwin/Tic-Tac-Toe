"""
AI Search Agents for Tic-Tac-Toe

This module implements three distinct AI decision strategies:
1. RandomAgent: Baseline agent making uniformly random valid moves.
2. HeuristicAgent: Rule-based agent following a priority hierarchy:
   Immediate Win > Immediate Block > Center > Corners > Edges.
3. MinimaxAgent: Optimal Adversarial Search agent with Alpha-Beta pruning
   and terminal depth scoring.
4. simulate_match: Multi-game benchmarking utility.
"""

import random
from typing import Dict, List, Optional
from game import TicTacToe, PLAYER_X, PLAYER_O, EMPTY


class RandomAgent:
    """Baseline AI agent that chooses randomly among available legal moves."""

    def __init__(self, player: str = PLAYER_X):
        self.player = player

    @property
    def opponent(self) -> str:
        return PLAYER_O if self.player == PLAYER_X else PLAYER_X

    def get_move(self, game: TicTacToe) -> Optional[int]:
        """Select a random legal move from available board positions."""
        moves = game.get_available_moves()
        if not moves:
            return None
        return random.choice(moves)


class HeuristicAgent:
    """
    Rule-based AI agent using priority hierarchy:
    1. Immediate Win: Take any move that completes 3-in-a-row for self.
    2. Immediate Block: Block any opponent move that would complete 3-in-a-row.
    3. Center: Take position 4 if available.
    4. Corners: Take positions 0, 2, 6, 8.
    5. Edges: Take positions 1, 3, 5, 7.
    """

    def __init__(self, player: str = PLAYER_X):
        self.player = player

    @property
    def opponent(self) -> str:
        return PLAYER_O if self.player == PLAYER_X else PLAYER_X

    def get_move(self, game: TicTacToe) -> Optional[int]:
        available_moves = game.get_available_moves()
        if not available_moves:
            return None

        orig_player = game.current_player
        try:
            # 1. Immediate Win: Can we win in one move?
            for move in available_moves:
                game.board[move] = self.player
                if game.check_winner() == self.player:
                    game.board[move] = EMPTY
                    return move
                game.board[move] = EMPTY

            # 2. Immediate Block: Would the opponent win on their next move?
            opp = self.opponent
            for move in available_moves:
                game.board[move] = opp
                if game.check_winner() == opp:
                    game.board[move] = EMPTY
                    return move
                game.board[move] = EMPTY

            # 3. Center Priority: Position 4
            if 4 in available_moves:
                return 4

            # 4. Corners Priority: 0, 2, 6, 8
            corners = [0, 2, 6, 8]
            available_corners = [c for c in corners if c in available_moves]
            if available_corners:
                return available_corners[0]

            # 5. Edges Priority: 1, 3, 5, 7
            edges = [1, 3, 5, 7]
            available_edges = [e for e in edges if e in available_moves]
            if available_edges:
                return available_edges[0]

            return available_moves[0]
        finally:
            game.current_player = orig_player



class MinimaxAgent:
    """
    Minimax Adversarial Search agent with Alpha-Beta pruning.
    
    Guarantees mathematically unbeatable play (0 losses) in Tic-Tac-Toe.
    Uses depth-adjusted terminal payoffs:
    - Win: +10 - depth (favors fast wins)
    - Loss: depth - 10 (favors delaying losses)
    - Draw: 0
    """

    def __init__(self, player: str = PLAYER_X, use_alpha_beta: bool = True):
        self.player = player
        self.use_alpha_beta = use_alpha_beta
        self.nodes_explored: int = 0
        self.candidate_scores: Dict[int, int] = {}

    @property
    def opponent(self) -> str:
        return PLAYER_O if self.player == PLAYER_X else PLAYER_X

    def get_move(self, game: TicTacToe) -> Optional[int]:
        """
        Compute optimal move for current game state using Minimax
        with Alpha-Beta pruning. Populates candidate_scores and nodes_explored.
        """
        available_moves = game.get_available_moves()
        if not available_moves:
            return None

        self.nodes_explored = 0
        self.candidate_scores = {}

        best_score = float('-inf')
        best_moves: List[int] = []

        orig_player = game.current_player
        try:
            # Evaluate every available move at root
            for move in available_moves:
                self.nodes_explored += 1
                game.make_move(move, self.player)

                if game.check_winner() == self.player:
                    score = 10 - 1
                elif game.is_draw():
                    score = 0
                else:
                    score = self._minimax(
                        game,
                        depth=1,
                        is_maximizing=False,
                        alpha=float('-inf'),
                        beta=float('inf')
                    )

                game.undo_move(move)

                self.candidate_scores[move] = score

                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

            # Tie-breaking heuristic: Prefer center (4), then corners, then first
            if 4 in best_moves:
                return 4
            for corner in [0, 2, 6, 8]:
                if corner in best_moves:
                    return corner

            return best_moves[0]
        finally:
            game.current_player = orig_player


    def _minimax(
        self,
        game: TicTacToe,
        depth: int,
        is_maximizing: bool,
        alpha: float,
        beta: float
    ) -> int:
        """
        Recursive Minimax search with terminal depth scoring and Alpha-Beta pruning.
        """
        self.nodes_explored += 1

        winner = game.check_winner()
        if winner == self.player:
            return 10 - depth
        if winner == self.opponent:
            return depth - 10
        if game.is_draw():
            return 0

        available_moves = game.get_available_moves()

        if is_maximizing:
            max_eval = float('-inf')
            for move in available_moves:
                game.make_move(move, self.player)
                eval_score = self._minimax(game, depth + 1, False, alpha, beta)
                game.undo_move(move)
                max_eval = max(max_eval, eval_score)
                if self.use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in available_moves:
                game.make_move(move, self.opponent)
                eval_score = self._minimax(game, depth + 1, True, alpha, beta)
                game.undo_move(move)
                min_eval = min(min_eval, eval_score)
                if self.use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            return min_eval


def simulate_match(agent1, agent2, num_games: int = 50) -> Dict[str, int]:
    """
    Simulate a series of matches between agent1 and agent2.
    
    Args:
        agent1: First agent instance.
        agent2: Second agent instance.
        num_games: Total number of games to simulate.
        
    Returns:
        Dict with keys: 'agent1_wins', 'agent2_wins', 'draws', 'total_games'.
    """
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "total_games": num_games
    }

    # Ensure distinct player symbols if identical
    orig_p1 = agent1.player
    orig_p2 = agent2.player
    if agent1.player == agent2.player:
        agent2.player = PLAYER_O if agent1.player == PLAYER_X else PLAYER_X

    try:
        for _ in range(num_games):
            game = TicTacToe()
            while not game.is_game_over():
                if game.current_player == agent1.player:
                    move = agent1.get_move(game)
                else:
                    move = agent2.get_move(game)

                if move is None:
                    break
                game.make_move(move)

            winner = game.check_winner()
            if winner == agent1.player:
                results["agent1_wins"] += 1
            elif winner == agent2.player:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
    finally:
        agent1.player = orig_p1
        agent2.player = orig_p2

    return results

