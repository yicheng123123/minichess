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
import os
from typing import Any, Dict, List, Optional

from nn.dataset import SelfPlayDataset, SelfPlaySample
from search.mcts import MCTS
from selfplay.player import generate_games
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

        # Network: residual policy-value net sized from the config.
        if net is None:
            net = create_network(
                hidden=cfg.hidden_channels,
                num_res_blocks=cfg.num_res_blocks,
            )
        self.net = net

        # MCTS used for self-play (exploration noise on by default).
        self.mcts = MCTS(num_simulations=cfg.num_simulations, c_puct=cfg.c_puct)

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
        )
        policy_target = torch.stack(
            [torch.from_numpy(_as_numpy(s.policy)).float() for s in samples]
        )
        value_target = torch.tensor(
            [s.value for s in samples], dtype=torch.float32
        )

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
        seed: Optional[int] = None,
        resume: bool = True,
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
            seed: Optional base seed for reproducibility.
            resume: If True, resume from the latest checkpoint when present.

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
                # Warm the buffer from persisted self-play data, if any.
                loaded = self.buffer.load_from_dataset(self.dataset)
                logger.info(f"Replay buffer warmed with {loaded} games "
                            f"({len(self.buffer)} samples)")

        optimizer = self._create_optimizer(self.net, lr)
        logger.info(
            f"Starting training: iterations={iterations} "
            f"games/iter={games_per_iter} epochs/iter={epochs_per_iter} "
            f"batch_size={batch_size} lr={lr}"
        )

        for it in range(start_iter, start_iter + iterations):
            iter_seed = None if seed is None else seed + it

            # 1. Self-play: generate games, push into buffer + dataset.
            adapter = _GameCollectingDataset(self.dataset)
            outcomes = generate_games(
                adapter,
                games_per_iter,
                net=self.net,
                mcts=self.mcts,
                seed=iter_seed,
            )
            for game in adapter.games:
                self.buffer.add_game(game)

            red = outcomes.count(1)
            black = outcomes.count(-1)
            draws = outcomes.count(0)
            logger.info(
                f"[iter {it}] self-play: {len(outcomes)} games "
                f"(red={red} black={black} draw={draws}); "
                f"buffer={len(self.buffer)} samples"
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

            # 3. Checkpoint the iteration.
            metadata = {
                "iteration": it,
                "config": self.config.to_dict(),
                "train_stats": avg,
                "buffer_size": len(self.buffer),
            }
            ckpt_path = self.checkpoints.save(self.net, it, metadata=metadata)
            logger.info(f"[iter {it}] saved checkpoint -> {ckpt_path}")

            # 4. Evaluate against the current best and maybe promote.
            self._maybe_promote_best(it, evaluate_games, evaluate_simulations, iter_seed)

        logger.info("Training complete.")
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
        seed=args.seed,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
