"""
Flask Web Application for MMDT Tic-Tac-Toe AI Visualizer

Features:
- Multiple Game Modes: Human vs AI (Easy, Medium, Unbeatable), 2-Player Local, AI vs AI.
- Real-time Minimax decision evaluation matrix & search node counter.
- Automated Bot vs Bot simulation benchmark runner.
"""

import time
from flask import Flask, render_template, request, jsonify
from game import TicTacToe, PLAYER_X, PLAYER_O, EMPTY
from agent import RandomAgent, HeuristicAgent, MinimaxAgent, simulate_match

app = Flask(__name__)


def create_agent(agent_type: str, player: str):
    """Instantiate an agent given a string type ('random', 'heuristic', 'minimax')."""
    agent_type = (agent_type or "minimax").lower()
    if agent_type in ("random", "easy"):
        return RandomAgent(player=player)
    elif agent_type in ("heuristic", "medium"):
        return HeuristicAgent(player=player)
    else:
        return MinimaxAgent(player=player, use_alpha_beta=True)


def format_candidate_analysis(scores):
    """Format numeric minimax scores into human-readable tactical interpretations."""
    analysis = {}
    for move, sc in scores.items():
        if sc > 0:
            turns = max(1, round((10 - sc) / 2))
            label = f"+{sc} (Win in {turns} Turn{'s' if turns > 1 else ''})"
            status = "win"
        elif sc < 0:
            turns = max(1, round((10 + sc) / 2))
            label = f"{sc} (Lose in {turns} Turn{'s' if turns > 1 else ''})"
            status = "loss"
        else:
            label = "0 (Draw Equilibrium)"
            status = "draw"
        analysis[str(move)] = {
            "score": sc,
            "label": label,
            "status": status
        }
    return analysis


@app.route("/")
def index():
    """Render the interactive game visualizer frontend."""
    return render_template("index.html")


@app.route("/api/evaluate", methods=["POST"])
def evaluate_board():
    """
    Analyze the current board state with Minimax to return
    candidate move scores and explored node count for visualization.
    """
    data = request.get_json() or {}
    board = data.get("board", [EMPTY] * 9)
    current_player = data.get("current_player", PLAYER_X)

    game = TicTacToe(board=board, current_player=current_player)

    use_alpha_beta = bool(data.get("use_alpha_beta", True))

    if game.is_game_over():
        return jsonify({
            "candidate_scores": {},
            "candidate_details": {},
            "nodes_explored": 0,
            "best_move": None,
            "eval_time_ms": 0.0,
            "use_alpha_beta": use_alpha_beta
        })

    minimax = MinimaxAgent(player=current_player, use_alpha_beta=use_alpha_beta)
    t0 = time.perf_counter()
    best_move = minimax.get_move(game)
    eval_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "candidate_scores": minimax.candidate_scores,
        "candidate_details": format_candidate_analysis(minimax.candidate_scores),
        "nodes_explored": minimax.nodes_explored,
        "best_move": best_move,
        "eval_time_ms": eval_time_ms,
        "use_alpha_beta": use_alpha_beta
    })


@app.route("/api/move", methods=["POST"])
def make_move():
    """
    Process a move from Human or request AI move.
    """
    data = request.get_json() or {}
    board = data.get("board", [EMPTY] * 9)
    current_player = data.get("current_player", PLAYER_X)
    difficulty = data.get("difficulty", "unbeatable")
    player_move = data.get("player_move")  # Optional index if human moved
    use_alpha_beta = bool(data.get("use_alpha_beta", True))

    game = TicTacToe(board=board, current_player=current_player)

    # 1. If a human move is provided, execute it first
    if player_move is not None:
        if not game.is_valid_move(int(player_move)):
            return jsonify({"error": "Invalid human move"}), 400
        game.make_move(int(player_move))

    # 2. Check if game ended after human move
    if game.is_game_over():
        state = game.to_dict()
        state.update({
            "ai_move": None,
            "candidate_scores": {},
            "candidate_details": {},
            "nodes_explored": 0,
            "eval_time_ms": 0,
            "use_alpha_beta": use_alpha_beta
        })
        return jsonify(state)

    # 3. If AI's turn is requested, compute AI move
    is_ai_turn = data.get("is_ai_turn", True)
    ai_move = None
    candidate_scores = {}
    nodes_explored = 0
    eval_time_ms = 0

    difficulty_x = data.get("difficulty_x", difficulty)
    difficulty_o = data.get("difficulty_o", difficulty)
    active_diff = difficulty_x if game.current_player == PLAYER_X else difficulty_o

    if is_ai_turn and not game.is_game_over():
        if active_diff in ("unbeatable", "hard", "minimax"):
            agent = MinimaxAgent(player=game.current_player, use_alpha_beta=use_alpha_beta)
        else:
            agent = create_agent(active_diff, game.current_player)


        t0 = time.perf_counter()
        ai_move = agent.get_move(game)
        eval_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        if isinstance(agent, MinimaxAgent):
            candidate_scores = agent.candidate_scores
            nodes_explored = agent.nodes_explored

        if ai_move is not None:
            game.make_move(ai_move)

    state = game.to_dict()
    state.update({
        "ai_move": ai_move,
        "candidate_scores": candidate_scores,
        "candidate_details": format_candidate_analysis(candidate_scores),
        "nodes_explored": nodes_explored,
        "eval_time_ms": eval_time_ms,
        "use_alpha_beta": use_alpha_beta
    })
    return jsonify(state)




@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """
    Run automated multi-game matches between two chosen agent strategies.
    """
    data = request.get_json() or {}
    agent1_type = data.get("agent1", "minimax")
    agent2_type = data.get("agent2", "random")
    num_games = int(data.get("num_games", 50))
    num_games = max(1, min(num_games, 200))  # Bounds for responsiveness

    agent1 = create_agent(agent1_type, PLAYER_X)
    agent2 = create_agent(agent2_type, PLAYER_O)

    t0 = time.perf_counter()
    results = simulate_match(agent1, agent2, num_games=num_games)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    total = results["total_games"]
    results.update({
        "agent1_type": agent1_type.capitalize(),
        "agent2_type": agent2_type.capitalize(),
        "agent1_win_pct": round(results["agent1_wins"] / total * 100, 1),
        "agent2_win_pct": round(results["agent2_wins"] / total * 100, 1),
        "draw_pct": round(results["draws"] / total * 100, 1),
        "elapsed_ms": elapsed_ms
    })

    return jsonify(results)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)


