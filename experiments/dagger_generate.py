"""experiments/dagger_generate.py — DAgger data collection.

Core idea (Ross et al. 2011): the STUDENT plays games (greedy policy), and at
every position it encounters, the EXPERT (alpha-beta) provides the correct move
as the training label. This fixes Behavioral Cloning's covariate shift: the
training distribution matches what the student actually sees at test time.

Flow per game:
  1. Student (greedy argmax of policy logits) picks moves for BOTH sides.
  2. At each position before the student moves, AB searches and we record
     (planes, AB_best_move) as a supervised sample.
  3. The game continues with the STUDENT's move (not AB's) — this is what
     creates the on-policy trajectory.
  4. After the game ends, value = outcome from mover's perspective.

Output format matches sl_teacher.jsonl so supervised.py can load it directly.

Usage:
    python experiments/dagger_generate.py --model models/sl_net.pt.best \
        --games 500 --ab-depth 3 --out data/expert/dagger_iter1.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from engine.board import Board
from engine.piece import Color
from engine.rules import game_result
from engine.move_generator import legal_moves
from nn.network import create_network, move_to_index
from nn.dataset import encode_planes
from search.alphabeta import alphabeta


def greedy_move(board, net, device):
    """Student's move: argmax of policy logits over legal moves."""
    moves = legal_moves(board)
    if not moves:
        return None, moves
    planes = torch.from_numpy(board.to_planes()).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = net(planes)
    logits_np = logits.squeeze(0).cpu().numpy()
    idxs = [move_to_index(m) for m in moves]
    best_i = int(np.argmax([logits_np[i] for i in idxs]))
    return moves[best_i], moves


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Horizontal mirror augmentation (flip columns)."""
    return planes[:, :, ::-1].copy()


def mirror_move_str(move_str: str) -> str:
    """Mirror a move string like 'a1b2' -> 'g1f2' (flip file: a<->g, b<->f, c<->e, d<->d)."""
    file_map = {'a': 'g', 'b': 'f', 'c': 'e', 'd': 'd', 'e': 'c', 'f': 'b', 'g': 'a'}
    if len(move_str) != 4:
        return move_str
    return file_map[move_str[0]] + move_str[1] + file_map[move_str[2]] + move_str[3]


def play_one_dagger_game(net, device, ab_depth, max_plies=200, augment=True,
                         student_color=None):
    """Play one DAgger game: student plays one side, AB plays the other.

    The student's moves determine the trajectory on its turn; AB responds on
    the other turn (acting as the "environment"). Labels (AB best move) are
    recorded at EVERY position so the model learns correct play for both
    sides from positions it actually encounters during real play.

    Parameters
    ----------
    student_color : Color or None
        Which side the student plays. If None, defaults to RED.
    """
    if student_color is None:
        student_color = Color.RED

    board = Board()
    samples = []
    ply = 0

    while ply < max_plies:
        result = game_result(board)
        if result is not None:
            break

        moves = legal_moves(board)
        if not moves:
            break

        # --- AB labels the current position (always) ---
        _score, ab_mv = alphabeta(board, depth=ab_depth)
        if ab_mv is None:
            ab_mv = moves[0]

        # Build one-hot policy target for AB's move
        ab_idx = move_to_index(ab_mv)
        policy_onehot = [0.0] * 2401
        policy_onehot[ab_idx] = 1.0

        # Planes
        planes = board.to_planes()  # numpy (C, 7, 7)

        # Record sample (value filled later)
        sample = {
            "planes": encode_planes(planes),
            "policy": policy_onehot,
            "value": 0.0,  # placeholder
            "move": str(ab_mv),
            "teacher": True,
        }
        samples.append(sample)

        # Mirror augmentation
        if augment:
            m_planes = mirror_planes(planes)
            m_move = mirror_move_str(str(ab_mv))
            m_idx = move_to_index_from_str(m_move)
            if m_idx is not None:
                m_policy = [0.0] * 2401
                m_policy[m_idx] = 1.0
                samples.append({
                    "planes": encode_planes(m_planes),
                    "policy": m_policy,
                    "value": 0.0,
                    "move": m_move,
                    "teacher": True,
                })

        # --- Determine who moves ---
        mover = board.side_to_move
        if mover == student_color:
            # Student moves (determines on-policy trajectory)
            student_mv, _ = greedy_move(board, net, device)
            if student_mv is None:
                break
            board.make_move(student_mv)
        else:
            # AB moves (the "environment" responds)
            board.make_move(ab_mv)
        ply += 1

    # Determine outcome
    result = game_result(board)
    if result is None:
        outcome_val = 0.0
        reason = "max_plies"
    else:
        reason = result.reason
        ov = result.outcome.value
        if ov == "red_wins":
            outcome_val = 1.0
        elif ov == "black_wins":
            outcome_val = -1.0
        else:
            outcome_val = 0.0

    # Fill value for each sample: outcome from mover's perspective
    # Sample i was at ply i (before mirror doubling). Mover alternates.
    # With augmentation, pairs are (original, mirror) at same ply.
    idx = 0
    for p in range(ply):
        # mover at ply p: RED if p even, BLACK if p odd
        # outcome_val is red-positive: +1 = red wins
        if p % 2 == 0:  # RED to move
            val = outcome_val
        else:  # BLACK to move
            val = -outcome_val
        if idx < len(samples):
            samples[idx]["value"] = val
            idx += 1
        if augment and idx < len(samples):
            samples[idx]["value"] = val
            idx += 1

    return {
        "samples": samples,
        "outcome": outcome_val,
        "plies": ply,
        "reason": reason,
    }


# --- helper: move_to_index from string ---
_MOVE_TO_INDEX_CACHE = {}

def move_to_index_from_str(move_str: str):
    """Convert a move string like 'a1b2' to a move index, or None if invalid."""
    if move_str in _MOVE_TO_INDEX_CACHE:
        return _MOVE_TO_INDEX_CACHE[move_str]
    try:
        from engine.move import Move
        mv = Move.from_uci(move_str)
        idx = move_to_index(mv)
        _MOVE_TO_INDEX_CACHE[move_str] = idx
        return idx
    except Exception:
        _MOVE_TO_INDEX_CACHE[move_str] = None
        return None


def main():
    ap = argparse.ArgumentParser(description="DAgger data generation")
    ap.add_argument("--model", required=True, help="Current SL model path")
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--ab-depth", type=int, default=3)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--out", default="data/expert/dagger_iter1.jsonl")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_network().to(device)
    st = torch.load(args.model, map_location=device, weights_only=False)
    net.load_state_dict(st)
    net.eval()
    print(f"[dagger] model loaded from {args.model} on {device}")
    print(f"[dagger] generating {args.games} games, AB depth={args.ab_depth}, "
          f"augment={not args.no_augment}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    n_samples = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for gi in range(args.games):
            # Alternate student color for balanced training
            student_color = Color.RED if gi % 2 == 0 else Color.BLACK
            record = play_one_dagger_game(
                net, device, args.ab_depth, args.max_plies,
                augment=not args.no_augment,
                student_color=student_color,
            )
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_samples += len(record["samples"])

            if (gi + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (gi + 1) / elapsed * 60
                print(f"  [{gi+1}/{args.games}] samples={n_samples} "
                      f"elapsed={elapsed:.0f}s rate={rate:.1f} games/min")

    elapsed = time.time() - t0
    print(f"\n[dagger] done: {args.games} games, {n_samples} samples, "
          f"{elapsed:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
