"""Self-play game generation for Mini Xiangqi AlphaZero training.

This module implements the core self-play loop:
- Play full games using MCTS + neural network guidance
- Record training samples (planes, policy targets, value targets)
- Support data augmentation via horizontal board flipping
- Batch game generation with dataset persistence
"""

import random
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for the selfplay module: pip install numpy")

from engine.board import Board
from engine.move import Move
from engine.move_generator import legal_moves
from engine.rules import game_result, GameOutcome
from engine.piece import Color
from search.mcts import MCTS
from nn.network import PolicyValueNet, move_to_index, NUM_MOVE_ACTIONS
from nn.dataset import SelfPlaySample, SelfPlayDataset
from utils.logger import logger


def _mirror_planes(planes: "np.ndarray") -> "np.ndarray":
    """Apply horizontal flip (mirror columns) to board planes.

    For a 7x7 board, column c maps to column (6 - c).
    This doubles training data by exploiting left-right symmetry.

    Args:
        planes: Board representation as numpy array with shape
                (num_channels, 7, 7) or similar spatial layout.

    Returns:
        Mirrored planes with columns reversed along the last axis.
    """
    return np.flip(planes, axis=-1).copy()


def _mirror_move_index(move_index: int, board_width: int = 7) -> int:
    """Compute the mirrored move index for horizontal flip.

    Move encoding: index = from_sq * board_width + to_sq (or similar).
    Under horizontal mirror, square (row, col) -> (row, width-1-col).

    Args:
        move_index: Original move action index.
        board_width: Width of the board (default 7 for Mini Xiangqi).

    Returns:
        Mirrored move action index.
    """
    total_squares = board_width * board_width  # 49 for 7x7
    from_sq = move_index // total_squares
    to_sq = move_index % total_squares

    from_row, from_col = divmod(from_sq, board_width)
    to_row, to_col = divmod(to_sq, board_width)

    mirrored_from = from_row * board_width + (board_width - 1 - from_col)
    mirrored_to = to_row * board_width + (board_width - 1 - to_col)

    return mirrored_from * total_squares + mirrored_to


def _mirror_policy(policy: "np.ndarray", board_width: int = 7) -> "np.ndarray":
    """Mirror a policy vector under horizontal board flip.

    Args:
        policy: Policy array of shape (NUM_MOVE_ACTIONS,).
        board_width: Width of the board.

    Returns:
        Mirrored policy array with remapped move indices.
    """
    mirrored = np.zeros_like(policy)
    for idx in range(len(policy)):
        if policy[idx] > 0:
            mirrored_idx = _mirror_move_index(idx, board_width)
            if mirrored_idx < len(mirrored):
                mirrored[mirrored_idx] += policy[idx]
    # Re-normalize to handle any numerical drift
    total = mirrored.sum()
    if total > 0:
        mirrored /= total
    return mirrored


def _mcts_visits_to_policy(
    visit_counts: Dict[Move, float],
) -> "np.ndarray":
    """Convert MCTS visit-count dictionary to a policy vector.

    Args:
        visit_counts: Mapping from Move to visit count (or proportion).

    Returns:
        Numpy array of shape (NUM_MOVE_ACTIONS,) with normalized
        visit-count distribution.
    """
    policy = np.zeros(NUM_MOVE_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total <= 0:
        # Fallback: uniform over legal moves
        n = len(visit_counts)
        if n > 0:
            for move in visit_counts:
                idx = move_to_index(move)
                if 0 <= idx < NUM_MOVE_ACTIONS:
                    policy[idx] = 1.0 / n
        return policy

    for move, count in visit_counts.items():
        idx = move_to_index(move)
        if 0 <= idx < NUM_MOVE_ACTIONS:
            policy[idx] = count / total
    return policy


def _select_move_from_net(
    board: Board,
    net: PolicyValueNet,
    temperature: float,
) -> Tuple[Dict[Move, float], Move]:
    """Select a move directly from the network's policy (no MCTS).

    Fast path for random/untrained networks or quick smoke tests.
    Returns (visit_counts_as_probs, chosen_move) to match MCTS interface.
    """
    import math

    moves = legal_moves(board)
    if not moves:
        raise RuntimeError("no legal moves available")

    logits, _value = net.predict(board, moves)
    scores = [logits.get(move_to_index(m), 0.0) for m in moves]

    if temperature <= 1e-3:
        # Greedy: pick the highest-scoring move (ties broken randomly)
        best = max(scores)
        candidates = [m for m, s in zip(moves, scores) if abs(s - best) < 1e-9]
        chosen = random.choice(candidates)
    else:
        # Softmax with temperature, then sample
        scaled = [s / temperature for s in scores]
        mx = max(scaled)
        exps = [math.exp(s - mx) for s in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]
        r = random.random()
        cum = 0.0
        chosen = moves[-1]
        for m, p in zip(moves, probs):
            cum += p
            if r <= cum:
                chosen = m
                break

    # Build a "visit counts" dict (uniform-ish for interface compatibility)
    visit_counts = {m: (1.0 if m == chosen else 0.0) for m in moves}
    return visit_counts, chosen


def play_game(
    net: PolicyValueNet,
    mcts: Optional[MCTS] = None,
    max_plies: int = 200,
    temperature: float = 1.0,
    temp_drop_after: int = 30,
    seed: Optional[int] = None,
) -> dict:
    """Play a single self-play game and collect training samples.

    If ``mcts`` is provided, uses MCTS + network for move selection (strong
    play, slow). If ``mcts`` is None, samples directly from the network's
    policy output (fast, suitable for random/untrained nets and smoke tests).

    Early moves use temperature > 0 for exploration; after temp_drop_after
    plies the agent plays greedily (temperature -> 0).

    Args:
        net: Neural network for policy/value evaluation.
        mcts: MCTS instance. If None, uses direct network sampling (no MCTS).
        max_plies: Maximum number of half-moves before declaring a draw.
        temperature: Initial exploration temperature for move sampling.
        temp_drop_after: After this many plies, switch to greedy play.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - "samples": List[SelfPlaySample] — one per ply played.
            - "outcome": int — +1 if Red wins, -1 if Black wins, 0 draw.
            - "plies": int — number of half-moves played.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    board = Board()
    samples: List[SelfPlaySample] = []
    history: List[Tuple["np.ndarray", "np.ndarray", str, Color]] = []

    ply = 0
    first_rep_ply: Optional[int] = None
    while ply < max_plies:
        # Check for terminal state
        result = game_result(board)
        if result is not None:
            break

        # Determine temperature for this ply
        current_temp = temperature if ply < temp_drop_after else 0.0

        # Move selection: MCTS if available, otherwise direct net sampling
        if mcts is not None:
            if current_temp > 0:
                visit_counts, best_move = mcts.search_with_temperature(
                    board, net, current_temp
                )
            else:
                visit_counts, best_move = mcts.search(board, net)
        else:
            visit_counts, best_move = _select_move_from_net(
                board, net, current_temp
            )

        # Build policy target from visit counts
        policy = _mcts_visits_to_policy(visit_counts)

        # Record position data
        planes = board.to_planes()
        move_uci = best_move.uci()
        mover = board.side_to_move

        history.append((planes, policy, move_uci, mover))

        # Execute the move
        board.make_move(best_move)
        ply += 1

        # Track the first time a position recurs (diagnostic: how late does the
        # game first start repeating itself). A higher value means the agent is
        # playing more purposefully before falling into a repetition loop.
        if first_rep_ply is None and board.repetition_count() >= 2:
            first_rep_ply = ply

    # Determine game outcome
    result = game_result(board)
    if result is None:
        # Max plies reached without terminal state — draw
        outcome = 0
        reason = "max_plies"
    elif result.outcome == GameOutcome.RED_WINS:
        outcome = 1
        reason = result.reason
    elif result.outcome == GameOutcome.BLACK_WINS:
        outcome = -1
        reason = result.reason
    else:
        outcome = 0
        reason = result.reason

    # Build samples with value from mover's perspective
    for planes, policy, move_uci, mover in history:
        # Value from the perspective of the player who made this move:
        # If Red made the move and Red wins -> +1
        # If Red made the move and Black wins -> -1
        if mover == Color.RED:
            value = float(outcome)
        else:
            value = float(-outcome)

        sample = SelfPlaySample(
            planes=planes,
            policy=policy,
            value=value,
            move=move_uci,
        )
        samples.append(sample)

    logger.debug(
        f"Self-play game complete: {ply} plies, outcome={outcome}"
    )

    return {
        "samples": samples,
        "outcome": outcome,
        "plies": ply,
        "reason": reason,
        "first_rep_ply": first_rep_ply,
    }


def _augment_samples(samples: List[SelfPlaySample]) -> List[SelfPlaySample]:
    """Generate augmented samples via horizontal board flip.

    Each original sample produces one mirrored counterpart, effectively
    doubling the training data.

    Args:
        samples: Original list of self-play samples.

    Returns:
        New list containing both original and mirrored samples.
    """
    augmented = list(samples)
    for sample in samples:
        planes = np.asarray(sample.planes)
        policy = np.asarray(sample.policy, dtype=np.float32)

        mirrored_planes = _mirror_planes(planes)
        mirrored_policy = _mirror_policy(policy)

        aug_sample = SelfPlaySample(
            planes=mirrored_planes,
            policy=mirrored_policy,
            value=sample.value,
            move=sample.move,  # UCI string kept as-is; consumer can remap
        )
        augmented.append(aug_sample)

    return augmented


def generate_games(
    dataset: SelfPlayDataset,
    n_games: int,
    net: PolicyValueNet,
    mcts: Optional[MCTS] = None,
    seed: Optional[int] = None,
    augment: bool = True,
    **kwargs,
) -> List[int]:
    """Play multiple self-play games and store samples in a dataset.

    Args:
        dataset: SelfPlayDataset instance for persisting samples.
        n_games: Number of games to play.
        net: Neural network for evaluation.
        mcts: Optional shared MCTS instance (reused across games).
        seed: Base random seed. Each game gets seed+i for variety.
        augment: If True, apply horizontal-flip augmentation to double data.
        **kwargs: Additional keyword arguments passed to play_game
                  (e.g., max_plies, temperature, temp_drop_after).

    Returns:
        List of game outcomes (+1 Red wins, -1 Black wins, 0 draw).
    """
    outcomes: List[int] = []

    for i in range(n_games):
        game_seed = (seed + i) if seed is not None else None
        logger.info(f"  self-play game {i+1}/{n_games} ...")

        result = play_game(net=net, mcts=mcts, seed=game_seed, **kwargs)
        samples = result["samples"]
        outcome = result["outcome"]

        if augment:
            samples = _augment_samples(samples)

        # Append each sample to the dataset
        for sample in samples:
            dataset.add(sample)

        outcomes.append(outcome)
        outcome_str = {1: "Red wins", -1: "Black wins", 0: "Draw"}[outcome]
        logger.info(f"  game {i+1}/{n_games} done: {outcome_str} "
                    f"({len(samples)} samples)")

        if (i + 1) % 10 == 0 or (i + 1) == n_games:
            logger.info(
                f"Self-play progress: {i + 1}/{n_games} games "
                f"(last outcome={outcome}, plies={result['plies']})"
            )

    # Persist dataset to storage
    dataset.save()

    red_wins = outcomes.count(1)
    black_wins = outcomes.count(-1)
    draws = outcomes.count(0)
    logger.info(
        f"Generated {n_games} games: "
        f"Red={red_wins}, Black={black_wins}, Draw={draws}"
    )

    return outcomes


# --------------------------------------------------------------------------- #
# Parallel self-play (multi-process)
# --------------------------------------------------------------------------- #
# Persistent per-worker network. Each worker process builds its network once
# (first game) and reuses it for every subsequent game/iteration, reloading
# only the updated weights. Combined with the persistent pool below, this
# eliminates the per-iteration process-spawn + torch-import + model-build cost.
_WORKER_NET = None


def _play_one_game_worker(args: dict) -> dict:
    """Worker function for parallel self-play. Runs in a separate process.

    Receives a dict with all info needed to play one game, returns a dict
    with samples, outcome, and plies. All data is serializable (numpy arrays,
    plain dicts, ints, floats).
    """
    global _WORKER_NET
    state_dict = args["state_dict"]
    net_kwargs = args["net_kwargs"]
    device = args.get("device", "cpu")
    mcts_kwargs = args["mcts_kwargs"]
    game_seed = args["game_seed"]
    play_kwargs = args["play_kwargs"]

    # Build the network once per worker process ON THE TARGET DEVICE (GPU);
    # afterwards just load the (iteration-updated) CPU weights into it. Building
    # on the device is essential — a CPU network here makes self-play thousands of
    # times slower (every MCTS evaluation runs on the CPU instead of the GPU).
    if _WORKER_NET is None:
        from nn.network import create_network
        _WORKER_NET = create_network(**net_kwargs).to(device)
    _WORKER_NET.load_state_dict(state_dict)
    _WORKER_NET.eval()

    mcts = None
    if mcts_kwargs is not None:
        mcts = MCTS(**mcts_kwargs)

    result = play_game(net=_WORKER_NET, mcts=mcts, seed=game_seed, **play_kwargs)
    return {
        "samples": result["samples"],
        "outcome": result["outcome"],
        "plies": result["plies"],
        "reason": result["reason"],
        "first_rep_ply": result["first_rep_ply"],
    }


# Persistent process pool, reused across generate_games_parallel calls so the
# worker processes (and their imported modules / built networks) stay alive
# between training iterations instead of being respawned every time.
_POOL = None
_POOL_WORKERS = None


def _get_pool(num_workers: int):
    """Return a persistent ProcessPoolExecutor, creating it on first use."""
    global _POOL, _POOL_WORKERS
    import atexit
    from concurrent.futures import ProcessPoolExecutor
    if _POOL is None or _POOL_WORKERS != num_workers:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
        _POOL = ProcessPoolExecutor(max_workers=num_workers)
        _POOL_WORKERS = num_workers
        atexit.register(shutdown_worker_pool)
    return _POOL


def shutdown_worker_pool() -> None:
    """Shut down the persistent worker pool (registered for clean exit)."""
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None
        _POOL_WORKERS = None


def generate_games_parallel(
    dataset: SelfPlayDataset,
    n_games: int,
    net: "PolicyValueNet",
    mcts: Optional[MCTS] = None,
    seed: Optional[int] = None,
    augment: bool = True,
    num_workers: Optional[int] = None,
    net_kwargs: Optional[dict] = None,
    **kwargs,
) -> List[int]:
    """Play self-play games in parallel across multiple CPU cores.

    This is the recommended way to generate training data. Each worker process
    gets its own copy of the network and MCTS, plays one game independently,
    and returns the results.

    Args:
        dataset: SelfPlayDataset instance for persisting samples.
        n_games: Number of games to play.
        net: Neural network (its state_dict is shared with workers).
        mcts: Optional MCTS instance (config is shared with workers).
        seed: Base random seed.
        augment: If True, apply horizontal-flip augmentation.
        num_workers: Number of parallel processes. Defaults to CPU count.
        net_kwargs: Dict of kwargs for create_network (hidden, num_res_blocks).
        **kwargs: Passed to play_game (max_plies, temperature, etc.).

    Returns:
        A tuple ``(outcomes, plies, reasons, first_reps)`` where ``outcomes`` is
        a list of game results (+1 Red wins, -1 Black wins, 0 draw), ``plies``
        is the number of half-moves each game lasted, ``reasons`` is the
        terminal reason per game (``"checkmate"``/``"no_legal_moves"``/
        ``"repetition"``/``"king_captured"``/``"max_plies"``), and ``first_reps``
        is the ply at which each game first repeated a position (or ``None``).
    """
    import os
    from concurrent.futures import as_completed

    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, n_games)

    # Prepare serializable args for workers.
    try:
        raw_state_dict = net.state_dict()
    except AttributeError:
        # RandomPolicyValueNet has no state_dict; fall back to serial.
        logger.info("Network has no state_dict; falling back to serial play.")
        serial_outcomes = generate_games(dataset, n_games, net, mcts, seed, augment, **kwargs)
        return serial_outcomes, [], [], []

    # Detect the device the training net lives on, and ship the weights as CPU
    # tensors. Each worker builds its own network on that device and loads the
    # CPU weights there (avoids passing CUDA tensors across process boundaries).
    try:
        device = str(next(net.parameters()).device)
    except StopIteration:
        device = "cpu"
    state_dict = {k: v.cpu() for k, v in raw_state_dict.items()}

    if net_kwargs is None:
        net_kwargs = {
            "hidden": getattr(net, "hidden", 128),
            "num_res_blocks": getattr(net, "num_res_blocks", 4),
        }

    mcts_kwargs = None
    if mcts is not None:
        mcts_kwargs = {
            "num_simulations": mcts.num_simulations,
            "c_puct": mcts.c_puct,
            "dirichlet_alpha": mcts.dirichlet_alpha,
            "dirichlet_epsilon": mcts.dirichlet_epsilon,
            "add_noise": mcts.add_noise,
        }

    # Build task list.
    tasks = []
    for i in range(n_games):
        tasks.append({
            "state_dict": state_dict,
            "net_kwargs": net_kwargs,
            "device": device,
            "mcts_kwargs": mcts_kwargs,
            "game_seed": (seed + i) if seed is not None else None,
            "play_kwargs": kwargs,
        })

    logger.info(f"  self-play: {n_games} games on {num_workers} workers ...")

    outcomes: List[int] = []
    all_plies: List[int] = []
    all_reasons: List[str] = []
    all_first_reps: List[Optional[int]] = []
    all_samples: List = []

    executor = _get_pool(num_workers)
    futures = {executor.submit(_play_one_game_worker, t): i
               for i, t in enumerate(tasks)}

    done_count = 0
    for future in as_completed(futures):
        done_count += 1
        result = future.result()
        samples = result["samples"]
        outcome = result["outcome"]

        if augment:
            samples = _augment_samples(samples)

        for sample in samples:
            dataset.add(sample)

        outcomes.append(outcome)
        all_plies.append(result["plies"])
        all_reasons.append(result["reason"])
        all_first_reps.append(result["first_rep_ply"])
        if done_count % max(1, n_games // 5) == 0 or done_count == n_games:
            logger.info(f"  self-play progress: {done_count}/{n_games} games done")

    dataset.save()

    red_wins = outcomes.count(1)
    black_wins = outcomes.count(-1)
    draws = outcomes.count(0)
    avg_plies = sum(all_plies) / len(all_plies) if all_plies else 0.0
    logger.info(
        f"Generated {n_games} games (parallel): "
        f"Red={red_wins}, Black={black_wins}, Draw={draws} "
        f"avg_plies={avg_plies:.0f}"
    )

    return outcomes, all_plies, all_reasons, all_first_reps
