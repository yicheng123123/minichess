"""train/trainer.py — AlphaZero-style training loop for Mini Xiangqi.

Implements the classic self-play -> learn -> evaluate -> checkpoint loop:

  For each iteration:
    1. **Self-play** — generate games with the current network + MCTS and push
       the resulting positions into a :class:`~train.replay_buffer.ReplayBuffer`
       (and persist them to a :class:`nn.dataset.SelfPlayDataset` JSONL file).
    2. **Train** — sample mini-batches from the buffer and run a number of
       gradient steps on the combined policy + value loss.
    3. **Evaluate** — optionally match the freshly trained network against the
       current best (arena match) to decide whether it improved.
    4. **Checkpoint** — save an iteration checkpoint, and promote the network to
       ``best.pt`` when it passes the evaluation gate.

The module degrades gracefully without PyTorch: it imports cleanly, but
constructing a :class:`Trainer` (which needs to build and train a real network)
raises a clear ``RuntimeError``.

CLI::

    python -m train.trainer --iterations 10 --games 20 --epochs 5 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from nn.dataset import SelfPlayDataset, SelfPlaySample
from search.mcts import MCTS
from selfplay.player import generate_games, generate_games_parallel
from selfplay.arena import evaluate_match
from utils.config import Config, get_config
from utils.logger import logger
from utils.seed import set_seed

from train.checkpoint import CheckpointManager
from train.replay_buffer import ReplayBuffer

try:
    import torch
    from nn.network import create_network
    from nn import loss as nn_loss

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required for train.trainer; install torch to run the "
            "AlphaZero training loop."
        )


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 23m' or '4m 12s'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


def _print_training_progress(
    current: int, total: int, loss: float, sims: int,
    elapsed: float, buffer_size: int,
) -> None:
    """Print a single-line progress bar for the training loop."""
    pct = (current + 1) / total
    bar_len = 25
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)

    eta = elapsed / (current + 1) * (total - current - 1) if current > 0 else 0
    print(
        f"\r  [{bar}] {current+1}/{total} "
        f"| loss={loss:.4f} sims={sims} buf={buffer_size} "
        f"| {_format_time(elapsed)}<{_format_time(eta)}",
        end="", flush=True,
    )


def _save_loss_history(history: List[Dict], path: str, quiet: bool = False) -> None:
    """Save training loss history to a JSON file.

    When ``quiet`` is True the confirmation log line is suppressed, which is
    useful when saving incrementally once per iteration.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if not quiet:
        logger.info(f"Loss history saved -> {path}")


def _plot_loss_curve(history: List[Dict], path: str, quiet: bool = False) -> None:
    """Plot policy_loss and value_loss curves and save as PNG.

    When ``quiet`` is True log messages are suppressed, which is useful when
    re-plotting incrementally once per iteration.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        if not quiet:
            logger.info("matplotlib not installed; skipping loss curve plot. "
                        "Install with: pip install matplotlib")
        return

    iters = [h["iteration"] for h in history]
    policy = [h["policy_loss"] for h in history]
    value = [h["value_loss"] for h in history]
    total = [h["total_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, total, label="Total Loss", linewidth=1.5)
    ax.plot(iters, policy, label="Policy Loss", linewidth=1, alpha=0.7)
    ax.plot(iters, value, label="Value Loss", linewidth=1, alpha=0.7)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve (Mini Xiangqi AlphaZero)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    if not quiet:
        logger.info(f"Loss curve saved -> {path}")


class _GameCollectingDataset:
    """Adapter bridging :func:`selfplay.player.generate_games` to a real dataset.

    ``generate_games`` appends samples one at a time via ``dataset.add(sample)``
    and finalizes with ``dataset.save()``, whereas
    :class:`nn.dataset.SelfPlayDataset` stores whole games via ``append_game``.
    This adapter buffers the samples and, on ``save()``, forwards them to the
    wrapped dataset and records them under :attr:`games` so the trainer can feed
    the same samples into the replay buffer.

    Note:
        ``generate_games`` invokes ``save()`` once per call (after all games),
        so a single call's samples are persisted to the dataset as one entry.
        Every sample is still captured and pushed to the replay buffer, which is
        what training consumes; the dataset here serves as a persistent record.
    """

    def __init__(self, dataset: SelfPlayDataset) -> None:
        self._dataset = dataset
        self._current: List[SelfPlaySample] = []
        self.games: List[List[SelfPlaySample]] = []

    def add(self, sample: SelfPlaySample) -> None:
        """Buffer a single sample produced during self-play."""
        self._current.append(sample)

    def save(self) -> None:
        """Flush the buffered samples to the wrapped dataset and record them."""
        if self._current:
            self._dataset.append_game(self._current)
            self.games.append(self._current)
            self._current = []


class Trainer:
    """AlphaZero-style self-play training driver.

    Wires together the network, MCTS, replay buffer, checkpoint manager and the
    on-disk self-play dataset, and exposes :meth:`train` to run the full loop.

    Args:
        config: Optional :class:`utils.config.Config`. Defaults to
            :func:`utils.config.get_config`.
        net: Optional pre-built network (must be a ``torch.nn.Module`` with the
            ``forward(planes) -> (policy_logits, value)`` contract). Created via
            :func:`nn.network.create_network` when omitted.
        replay_buffer_size: Capacity of the experience replay buffer.
        accept_threshold: Minimum arena score rate (wins + 0.5*draws)/games for
            the new network to be promoted to ``best.pt``.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        net: Optional[Any] = None,
        replay_buffer_size: int = 100_000,
        accept_threshold: float = 0.55,
    ) -> None:
        _require_torch()
        self.config = config or get_config()
        cfg = self.config

        # Device: use GPU if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network: residual policy-value net sized from the config.
        if net is None:
            net = create_network(
                hidden=cfg.hidden_channels,
                num_res_blocks=cfg.num_res_blocks,
            )
        self.net = net.to(self.device)

        # MCTS used for self-play (exploration noise on by default).
        self.mcts = MCTS(
            num_simulations=cfg.num_simulations,
            c_puct=cfg.c_puct,
            dirichlet_alpha=cfg.dirichlet_alpha,
        )

        # Experience replay buffer.
        self.buffer = ReplayBuffer(max_size=replay_buffer_size)

        # Checkpoint manager + on-disk self-play dataset.
        self.checkpoints = CheckpointManager(checkpoint_dir=cfg.checkpoint_dir)
        data_path = os.path.join(cfg.data_dir, "games", "games.jsonl")
        self.dataset = SelfPlayDataset(data_path)

        self.accept_threshold = accept_threshold
        self._best_score_rate: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Optimizer / single gradient step
    # ------------------------------------------------------------------ #
    def _create_optimizer(self, net: Any, lr: float) -> "torch.optim.Optimizer":
        """Create an Adam optimizer for ``net`` at learning rate ``lr``."""
        return torch.optim.Adam(net.parameters(), lr=lr)

    def train_batch(
        self,
        samples: List[SelfPlaySample],
        optimizer: "torch.optim.Optimizer",
        value_weight: float = 1.0,
        l2_weight: float = 0.0,
    ) -> Dict[str, float]:
        """Run one gradient step on a mini-batch of samples.

        Args:
            samples: Mini-batch of :class:`SelfPlaySample`.
            optimizer: The optimizer to step.
            value_weight: Weight of the value loss in the combined loss.
            l2_weight: Optional L2 regularization weight (0 disables it).

        Returns:
            A dict of scalar metrics: ``total_loss``, ``policy_loss``,
            ``value_loss`` and ``batch_size``.
        """
        if not samples:
            return {"total_loss": 0.0, "policy_loss": 0.0,
                    "value_loss": 0.0, "batch_size": 0}

        self.net.train()

        planes = torch.stack(
            [torch.from_numpy(_as_numpy(s.planes)).float() for s in samples]
        ).to(self.device)
        policy_target = torch.stack(
            [torch.from_numpy(_as_numpy(s.policy)).float() for s in samples]
        ).to(self.device)
        value_target = torch.tensor(
            [s.value for s in samples], dtype=torch.float32
        ).to(self.device)

        optimizer.zero_grad()
        logits, value = self.net(planes)

        policy_loss = nn_loss.policy_loss(logits, policy_target)
        value_loss = nn_loss.value_loss(value, value_target)
        total = nn_loss.combined_loss(
            logits, value, policy_target, value_target, value_weight=value_weight
        )
        if l2_weight > 0:
            total = total + nn_loss.l2_regularization(self.net, l2_weight)

        total.backward()
        optimizer.step()

        return {
            "total_loss": float(total.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "batch_size": len(samples),
        }

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def train(
        self,
        iterations: int = 100,
        games_per_iter: int = 20,
        epochs_per_iter: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        evaluate_games: int = 20,
        evaluate_simulations: Optional[int] = None,
        eval_every: int = 1,
        seed: Optional[int] = None,
        resume: bool = True,
        num_workers: Optional[int] = None,
        curriculum: bool = True,
        warm_start: Optional[str] = None,
        warm_start_epochs: int = 2,
        fresh_buffer: bool = False,
    ) -> Any:
        """Run the AlphaZero self-play training loop.

        Args:
            iterations: Number of outer self-play/train iterations.
            games_per_iter: Self-play games generated per iteration.
            epochs_per_iter: Gradient steps (mini-batches) per iteration.
            batch_size: Mini-batch size sampled from the replay buffer.
            lr: Adam learning rate.
            evaluate_games: Games per arena match against the current best.
            evaluate_simulations: MCTS simulations for arena play (defaults to
                the configured self-play simulation count).
            eval_every: Run the arena evaluation once every this many iterations
                (1 = every iteration, the original behaviour). The first
                iteration (to establish a baseline best.pt) and the final
                iteration are always evaluated regardless.
            seed: Optional base seed for reproducibility.
            resume: If True, resume from the latest checkpoint when present.
            warm_start: Optional path to an expert JSONL file (see
                selfplay/expert.py). When given, the network is pretrained on
                it (value on all positions, policy on teacher positions only)
                and the expert samples are seeded into the replay buffer before
                the self-play loop begins.
            warm_start_epochs: Number of pretraining passes over the expert
                data (keep small, e.g. 2-3, to avoid imitation learning).
            fresh_buffer: If True (and resuming), load the network weights from
                the checkpoint but do NOT warm the replay buffer from the
                on-disk dataset. Useful for a warm-start run that should begin
                from a clean buffer (expert data + new self-play only) instead
                of re-reading a large, draw-heavy dataset.

        Returns:
            The trained network.
        """
        if seed is not None:
            set_seed(seed)

        if evaluate_simulations is None:
            evaluate_simulations = self.config.num_simulations

        start_iter = 0
        if resume:
            latest = self.checkpoints.load_latest(self.net)
            if latest is not None:
                start_iter = latest + 1
                logger.info(f"Resumed from checkpoint at iteration {latest}")
                if fresh_buffer:
                    logger.info("Fresh buffer requested; skipping replay warm-up "
                                "from the on-disk dataset")
                else:
                    # Warm the buffer from persisted self-play data, if any.
                    loaded = self.buffer.load_from_dataset(self.dataset)
                    logger.info(f"Replay buffer warmed with {loaded} games "
                                f"({len(self.buffer)} samples)")

        # Optional warm-start: pretrain on expert data to give the value head
        # its first real win/loss signal, then seed the buffer with it.
        if warm_start:
            from train.warmstart import (
                pretrain_from_expert, load_expert_samples,
                expert_to_selfplay_samples,
            )
            pretrain_from_expert(
                self.net, warm_start, epochs=warm_start_epochs,
                batch_size=batch_size, lr=lr, device=self.device,
            )
            expert_raw = load_expert_samples(warm_start)
            expert_samples = expert_to_selfplay_samples(expert_raw)
            self.buffer.add_game(expert_samples)
            logger.info(f"Seeded replay buffer with {len(expert_samples)} expert "
                        f"samples; buffer={len(self.buffer)} samples")

        optimizer = self._create_optimizer(self.net, lr)
        logger.info(
            f"Starting training: iterations={iterations} "
            f"games/iter={games_per_iter} epochs/iter={epochs_per_iter} "
            f"batch_size={batch_size} lr={lr}"
        )
        logger.info(f"Device: {self.device}"
                    + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))

        train_start_time = time.time()
        loss_history: List[Dict] = []

        for it in range(start_iter, start_iter + iterations):
            iter_seed = None if seed is None else seed + it

            # Curriculum learning: ramp up simulations over the first half.
            if curriculum and iterations > 1:
                progress = min(1.0, (it - start_iter) / max(1, iterations // 2))
                cur_sims = max(25, int(self.config.num_simulations * (0.25 + 0.75 * progress)))
            else:
                cur_sims = self.config.num_simulations
            self.mcts.num_simulations = cur_sims

            # 1. Self-play: generate games in parallel, push into buffer + dataset.
            adapter = _GameCollectingDataset(self.dataset)
            outcomes, plies = generate_games_parallel(
                adapter,
                games_per_iter,
                net=self.net,
                mcts=self.mcts,
                seed=iter_seed,
                num_workers=num_workers,
                net_kwargs={
                    "hidden": self.config.hidden_channels,
                    "num_res_blocks": self.config.num_res_blocks,
                },
            )
            for game in adapter.games:
                self.buffer.add_game(game)

            red = outcomes.count(1)
            black = outcomes.count(-1)
            draws = outcomes.count(0)
            n_games_done = len(outcomes)
            win_rate = (red + black) / n_games_done if n_games_done else 0.0
            draw_rate = draws / n_games_done if n_games_done else 0.0
            avg_plies = sum(plies) / len(plies) if plies else 0.0
            logger.info(
                f"[iter {it}] self-play: {n_games_done} games "
                f"(red={red} black={black} draw={draws}) "
                f"win_rate={win_rate:.2f} draw_rate={draw_rate:.2f} "
                f"avg_plies={avg_plies:.0f} "
                f"sims={cur_sims}; buffer={len(self.buffer)} samples"
            )

            # Replay buffer outcome composition (is decisive data accumulating?).
            comp = self.buffer.value_composition()
            tot = comp["total"] or 1
            logger.info(
                f"[iter {it}] buffer mix: "
                f"win={100*comp['win']/tot:.1f}% "
                f"loss={100*comp['loss']/tot:.1f}% "
                f"draw={100*comp['draw']/tot:.1f}% "
                f"(of {comp['total']} samples)"
            )

            # 2. Train: run epochs_per_iter gradient steps on sampled batches.
            if len(self.buffer) == 0:
                logger.warning(f"[iter {it}] replay buffer empty; skipping training")
                continue

            epoch_stats: List[Dict[str, float]] = []
            for _ in range(epochs_per_iter):
                batch = self.buffer.sample(batch_size)
                stats = self.train_batch(batch, optimizer)
                epoch_stats.append(stats)

            avg = _average_stats(epoch_stats)
            logger.info(
                f"[iter {it}] train: {epochs_per_iter} steps "
                f"total_loss={avg['total_loss']:.4f} "
                f"policy_loss={avg['policy_loss']:.4f} "
                f"value_loss={avg['value_loss']:.4f}"
            )

            # Visual progress bar.
            elapsed = time.time() - train_start_time
            _print_training_progress(
                current=it - start_iter, total=iterations,
                loss=avg["total_loss"], sims=cur_sims,
                elapsed=elapsed, buffer_size=len(self.buffer),
            )

            # Record loss history.
            loss_history.append({
                "iteration": it,
                "total_loss": avg["total_loss"],
                "policy_loss": avg["policy_loss"],
                "value_loss": avg["value_loss"],
                "sims": cur_sims,
                "buffer_size": len(self.buffer),
            })

            # 3. Checkpoint the iteration.
            metadata = {
                "iteration": it,
                "config": self.config.to_dict(),
                "train_stats": avg,
                "buffer_size": len(self.buffer),
            }
            ckpt_path = self.checkpoints.save(self.net, it, metadata=metadata)
            logger.info(f"[iter {it}] saved checkpoint -> {ckpt_path}")

            # Incrementally persist the loss record + curve so an interrupted
            # run still keeps its progress. Negligible cost (a few KB JSON and
            # a small PNG) next to the minutes spent on self-play/evaluation.
            history_path = os.path.join(self.config.checkpoint_dir, "loss_history.json")
            _save_loss_history(loss_history, history_path, quiet=True)
            curve_path = os.path.join(self.config.checkpoint_dir, "loss_curve.png")
            _plot_loss_curve(loss_history, curve_path, quiet=True)

            # 4. Evaluate against the current best and maybe promote.
            #    Honour eval_every: skip the (slow) arena match on most
            #    iterations, but always evaluate on the first iteration (to
            #    establish a baseline best.pt), on schedule, and on the last one.
            is_last = (it == start_iter + iterations - 1)
            no_best_yet = not self.checkpoints.has_best()
            on_schedule = (eval_every <= 1) or ((it - start_iter) % eval_every == 0)
            if on_schedule or is_last or no_best_yet:
                self._maybe_promote_best(it, evaluate_games, evaluate_simulations, iter_seed)
            else:
                remaining = eval_every - ((it - start_iter) % eval_every)
                logger.info(f"[iter {it}] skipping arena eval "
                            f"(next eval in {remaining} iter)")

        print()  # newline after progress bar

        # Save loss history and plot curve.
        if loss_history:
            history_path = os.path.join(self.config.checkpoint_dir, "loss_history.json")
            _save_loss_history(loss_history, history_path)
            curve_path = os.path.join(self.config.checkpoint_dir, "loss_curve.png")
            _plot_loss_curve(loss_history, curve_path)

        logger.info(f"Training complete. Total time: {_format_time(time.time() - train_start_time)}")
        return self.net

    # ------------------------------------------------------------------ #
    # Evaluation / promotion
    # ------------------------------------------------------------------ #
    def _maybe_promote_best(
        self,
        iteration: int,
        evaluate_games: int,
        evaluate_simulations: int,
        seed: Optional[int],
    ) -> None:
        """Compare the current net to the best model and promote it if better.

        On the very first promotion (no ``best.pt`` yet) the current network is
        saved as best unconditionally. Afterwards an arena match gates promotion
        on :attr:`accept_threshold`.
        """
        if not self.checkpoints.has_best():
            self.checkpoints.save_best(
                self.net,
                metadata={"iteration": iteration, "config": self.config.to_dict(),
                          "reason": "initial"},
            )
            self._best_score_rate = None
            logger.info(f"[iter {iteration}] no prior best model; "
                        f"current net promoted to best.pt")
            return

        # Load the reigning best model into a fresh network for the match.
        best_net = create_network(
            hidden=self.config.hidden_channels,
            num_res_blocks=self.config.num_res_blocks,
        )
        self.checkpoints.load_best(best_net)

        result = evaluate_match(
            self.net,
            best_net,
            n_games=evaluate_games,
            mcts_simulations=evaluate_simulations,
            seed=seed,
            c_puct=self.config.c_puct,
        )
        total = result["wins_a"] + result["wins_b"] + result["draws"]
        score_rate = (result["wins_a"] + 0.5 * result["draws"]) / total if total else 0.0
        logger.info(
            f"[iter {iteration}] arena vs best: "
            f"new={result['wins_a']} best={result['wins_b']} "
            f"draw={result['draws']} score_rate={score_rate:.3f}"
        )

        if score_rate >= self.accept_threshold:
            self.checkpoints.save_best(
                self.net,
                metadata={"iteration": iteration, "config": self.config.to_dict(),
                          "score_rate": score_rate, "match": result},
            )
            self._best_score_rate = score_rate
            logger.info(f"[iter {iteration}] new model ACCEPTED as best "
                        f"(score_rate={score_rate:.3f})")
        else:
            logger.info(f"[iter {iteration}] new model REJECTED; keeping best "
                        f"(score_rate={score_rate:.3f} < {self.accept_threshold})")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_numpy(array: Any) -> Any:
    """Ensure ``array`` is a numpy ndarray (no-op if it already is)."""
    import numpy as np

    return np.asarray(array)


def _average_stats(stats: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dicts key-wise."""
    if not stats:
        return {"total_loss": 0.0, "policy_loss": 0.0,
                "value_loss": 0.0, "batch_size": 0}
    keys = stats[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        values = [s[key] for s in stats]
        out[key] = sum(values) / len(values)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    """Command-line entry point: ``python -m train.trainer``."""
    parser = argparse.ArgumentParser(
        description="Mini Xiangqi AlphaZero-style training loop"
    )
    parser.add_argument("--iterations", type=int, default=100,
                        help="number of self-play/train iterations")
    parser.add_argument("--games", type=int, default=20,
                        help="self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=5,
                        help="gradient steps per iteration")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="mini-batch size (default: config.batch_size)")
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate (default: config.learning_rate)")
    parser.add_argument("--evaluate-games", type=int, default=20,
                        help="games per arena match against the best model")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="run the arena match once every N iterations "
                             "(default: 1 = every iteration)")
    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="replay buffer capacity")
    parser.add_argument("--accept-threshold", type=float, default=0.55,
                        help="arena score rate needed to promote a new best model")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="checkpoint directory (default: config.checkpoint_dir)")
    parser.add_argument("--data-dir", default=None,
                        help="self-play data directory (default: config.data_dir)")
    parser.add_argument("--seed", type=int, default=None,
                        help="base random seed for reproducibility")
    parser.add_argument("--no-resume", action="store_true",
                        help="start from scratch instead of resuming")
    args = parser.parse_args()

    config = get_config()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.data_dir is not None:
        config.data_dir = args.data_dir

    trainer = Trainer(
        config=config,
        replay_buffer_size=args.buffer_size,
        accept_threshold=args.accept_threshold,
    )
    trainer.train(
        iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        evaluate_games=args.evaluate_games,
        eval_every=args.eval_every,
        seed=args.seed,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
