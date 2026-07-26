"""search/mcts.py — Monte Carlo Tree Search with neural-network guidance for Mini Xiangqi.

Implements the AlphaZero-style MCTS used by the self-play and inference
pipelines. The search is guided by a :class:`nn.network.PolicyValueNet`
that supplies:

  * **policy priors** P(s, a) — initial move probabilities, and
  * a scalar **value** v(s) in [-1, 1] from the perspective of the side to move.

The tree uses the PUCT formula for node selection:

    U(s, a) = c_puct * P(s, a) * sqrt(N_parent) / (1 + N_child)
    score   = Q(s, a) + U(s, a)

where Q(s, a) = W(s, a) / N(s, a) is the mean action value.

Key classes:
  * :class:`MCTSNode` — a single tree node storing visit counts, values, priors.
  * :class:`MCTS` — the search driver that runs N simulations and returns a
    move-probability distribution and the best move.

Usage example::

    from engine.board import Board
    from nn.network import RandomPolicyValueNet
    from search.mcts import MCTS

    board = Board()
    net = RandomPolicyValueNet(seed=42)
    searcher = MCTS(num_simulations=400, c_puct=2.5)
    move_probs, best_move = searcher.search(board, net)
    print(f"Best move: {best_move.uci()}")
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from engine.board import Board
from engine.move import Move
from engine.move_generator import legal_moves
from engine.piece import Color
from engine.rules import GameOutcome, GameResult, game_result
from nn.network import PolicyValueNet, move_to_index


# --------------------------------------------------------------------------- #
# MCTSNode
# --------------------------------------------------------------------------- #
class MCTSNode:
    """A node in the Monte Carlo search tree.

    Each node corresponds to a board position reached by a sequence of moves
    from the root. It stores statistics accumulated over simulations.

    Attributes
    ----------
    parent : Optional[MCTSNode]
        The parent node (None for the root).
    children : Dict[Move, MCTSNode]
        Mapping from the move that leads to a child, to the child node.
    move : Optional[Move]
        The move that was played from the parent to reach this node.
        None for the root.
    prior : float
        The prior probability P(s, a) assigned by the neural network policy
        for the move leading to this node.
    visit_count : int
        Number of times this node has been visited (N).
    value_sum : float
        Sum of backed-up values through this node (W).
    """

    __slots__ = (
        "parent",
        "children",
        "move",
        "prior",
        "visit_count",
        "value_sum",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Move] = None,
        prior: float = 0.0,
    ) -> None:
        self.parent = parent
        self.children: Dict[Move, "MCTSNode"] = {}
        self.move = move
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    @property
    def value(self) -> float:
        """Mean action value Q(s, a) = W / N. Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """A node is a leaf if it has no expanded children."""
        return len(self.children) == 0

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (children created)."""
        return len(self.children) > 0

    def __repr__(self) -> str:
        move_str = self.move.uci() if self.move else "root"
        return (
            f"MCTSNode(move={move_str}, N={self.visit_count}, "
            f"Q={self.value:.3f}, P={self.prior:.3f})"
        )


# --------------------------------------------------------------------------- #
# MCTS
# --------------------------------------------------------------------------- #
class MCTS:
    """Monte Carlo Tree Search with PUCT selection and neural-network guidance.

    Parameters
    ----------
    num_simulations : int
        Number of MCTS simulations (playouts) to run per search. Default 400.
    c_puct : float
        Exploration constant in the PUCT formula. Higher values encourage
        more exploration of less-visited moves. Default 2.5.
    dirichlet_alpha : float
        Concentration parameter for Dirichlet noise added at the root.
        Default 0.15 (suits a chess-like game with ~20-40 legal moves; the
        smaller 0.03 used for Go is too sparse here).
    dirichlet_epsilon : float
        Mixing weight for Dirichlet noise at the root:
        P_noisy = (1 - eps) * P + eps * Dir(alpha). Default 0.25.
    add_noise : bool
        Whether to add Dirichlet noise at the root for exploration.
        Enable during self-play training; disable during evaluation/inference.
        Default True.
    """

    def __init__(
        self,
        num_simulations: int = 400,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.15,
        dirichlet_epsilon: float = 0.25,
        add_noise: bool = True,
    ) -> None:
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.add_noise = add_noise

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(
        self,
        board: Board,
        net: PolicyValueNet,
    ) -> Tuple[Dict[Move, float], Move]:
        """Run MCTS from the given board position and return results.

        Parameters
        ----------
        board : Board
            The current board state to search from.
        net : PolicyValueNet
            The neural network providing policy priors and value estimates.

        Returns
        -------
        move_probs : Dict[Move, float]
            Visit-count distribution over root moves (sums to 1.0).
            This is used as the training policy target in self-play.
        best_move : Move
            The move with the highest visit count (most robust choice).
        """
        root = MCTSNode(parent=None, move=None, prior=0.0)

        # Expand the root node with legal moves and network priors.
        self._expand_node(root, board, net)

        # Add Dirichlet noise at the root for exploration.
        if self.add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations.
        for _ in range(self.num_simulations):
            node = root
            # We need to track the board state as we descend the tree.
            # Use make_move/undo_move for efficiency.
            search_path: List[MCTSNode] = [node]
            moves_played: List[Move] = []

            # --- SELECT ---
            # Descend the tree until we reach a leaf or terminal node.
            while not node.is_leaf:
                move, child = self._select_child(node)
                board.make_move(move)
                moves_played.append(move)
                node = child
                search_path.append(node)

            # --- EVALUATE ---
            # Check for terminal state; if not terminal, use the network.
            result = game_result(board)
            if result is not None:
                # Terminal node: compute value from the perspective of the
                # side to move at this node.
                value = self._terminal_value(result, board.side_to_move)
            else:
                # --- EXPAND ---
                if node.is_leaf:
                    self._expand_node(node, board, net)
                # Get the network value for this position.
                _, value = net.predict(board)

            # --- BACKPROPAGATE ---
            self._backpropagate(search_path, value)

            # Undo all moves to restore the board.
            for move in reversed(moves_played):
                board.undo_move()

        # Extract visit-count distribution from the root.
        move_probs = self._get_policy(root)
        best_move = self._get_best_move(root)

        return move_probs, best_move

    def search_with_temperature(
        self,
        board: Board,
        net: PolicyValueNet,
        temperature: float = 1.0,
    ) -> Tuple[Dict[Move, float], Move]:
        """Run MCTS and select a move using a temperature-scaled distribution.

        During early self-play games, a higher temperature encourages diverse
        openings. At temperature -> 0, the most-visited move is chosen
        deterministically.

        Parameters
        ----------
        board : Board
            The current board state.
        net : PolicyValueNet
            The neural network.
        temperature : float
            Temperature for move selection. T=1 uses visit counts directly;
            T<1 sharpens toward the best move; T>1 flattens the distribution.
            Default 1.0.

        Returns
        -------
        move_probs : Dict[Move, float]
            The temperature-adjusted move probability distribution.
        selected_move : Move
            A move sampled from the temperature-adjusted distribution.
        """
        raw_probs, _ = self.search(board, net)

        if temperature <= 0.0 or len(raw_probs) == 1:
            # Deterministic: pick the most-visited move.
            best = max(raw_probs, key=lambda m: raw_probs[m])
            probs = {m: (1.0 if m == best else 0.0) for m in raw_probs}
            return probs, best

        # Apply temperature: p_i^(1/T) then renormalize.
        inv_temp = 1.0 / temperature
        scaled = {
            m: p ** inv_temp for m, p in raw_probs.items()
        }
        total = sum(scaled.values())
        probs = {m: s / total for m, s in scaled.items()}

        # Sample a move from the distribution.
        selected_move = self._sample_move(probs)
        return probs, selected_move

    # ------------------------------------------------------------------ #
    # SELECT phase
    # ------------------------------------------------------------------ #
    def _select_child(self, node: MCTSNode) -> Tuple[Move, MCTSNode]:
        """Select the child with the highest PUCT score.

        PUCT formula:
            score(a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)

        Parameters
        ----------
        node : MCTSNode
            The parent node to select from.

        Returns
        -------
        (move, child) : Tuple[Move, MCTSNode]
            The selected move and its corresponding child node.
        """
        best_score = -math.inf
        best_move: Optional[Move] = None
        best_child: Optional[MCTSNode] = None

        sqrt_parent_visits = math.sqrt(node.visit_count)

        for move, child in node.children.items():
            # Q value of the child
            q_value = child.value
            # Exploration term (PUCT)
            exploration = (
                self.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1.0 + child.visit_count)
            )
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        assert best_move is not None and best_child is not None
        return best_move, best_child

    # ------------------------------------------------------------------ #
    # EXPAND phase
    # ------------------------------------------------------------------ #
    def _expand_node(
        self,
        node: MCTSNode,
        board: Board,
        net: PolicyValueNet,
    ) -> None:
        """Expand a leaf node by creating children for all legal moves.

        Uses the neural network policy to assign prior probabilities to each
        child. The policy logits are masked to legal moves and softmaxed.

        Parameters
        ----------
        node : MCTSNode
            The leaf node to expand.
        board : Board
            The board state corresponding to this node.
        net : PolicyValueNet
            The neural network for policy priors.
        """
        moves = legal_moves(board)
        if not moves:
            # No legal moves — this is a terminal node (loss for side to move).
            return

        # Get raw policy logits from the network.
        policy_logits, _ = net.predict(board)

        # Mask to legal moves and compute softmax probabilities.
        legal_indices = {move_to_index(m): m for m in moves}
        logits_for_legal = []
        move_list = []
        for idx, move in legal_indices.items():
            logit = policy_logits.get(idx, -1e9)
            logits_for_legal.append(logit)
            move_list.append(move)

        # Softmax over legal move logits.
        priors = self._softmax(logits_for_legal)

        # Create child nodes with the computed priors.
        for move, prior in zip(move_list, priors):
            child = MCTSNode(parent=node, move=move, prior=prior)
            node.children[move] = child

    # ------------------------------------------------------------------ #
    # BACKPROPAGATE phase
    # ------------------------------------------------------------------ #
    def _backpropagate(
        self,
        search_path: List[MCTSNode],
        value: float,
    ) -> None:
        """Backpropagate the leaf value up the search path.

        The value is from the perspective of the side to move at the leaf.
        As we propagate upward, we negate the value at each level because
        consecutive nodes represent alternating sides.

        Parameters
        ----------
        search_path : List[MCTSNode]
            The path from root to leaf (inclusive).
        value : float
            The evaluation of the leaf position, from the perspective of the
            side to move at that leaf.
        """
        # Traverse from leaf back to root, flipping the value sign at each
        # level (since parent and child are on opposite sides).
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Flip perspective for the parent (opponent's turn).
            value = -value

    # ------------------------------------------------------------------ #
    # Policy extraction
    # ------------------------------------------------------------------ #
    def _get_policy(self, root: MCTSNode) -> Dict[Move, float]:
        """Extract the visit-count distribution from the root node.

        The probability of each move is proportional to its visit count.
        This distribution is used as the training target in self-play.

        Parameters
        ----------
        root : MCTSNode
            The root node after all simulations are complete.

        Returns
        -------
        Dict[Move, float]
            Normalized visit-count probabilities (sum to 1.0).
        """
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )
        if total_visits == 0:
            # Fallback: uniform distribution (should not happen in practice).
            n = len(root.children)
            return {move: 1.0 / n for move in root.children} if n > 0 else {}

        return {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }

    def _get_best_move(self, root: MCTSNode) -> Move:
        """Return the move with the highest visit count at the root.

        This is the most robust move choice — it reflects the move the search
        spent the most time exploring.

        Parameters
        ----------
        root : MCTSNode
            The root node after all simulations.

        Returns
        -------
        Move
            The most-visited root move.
        """
        best_move = max(
            root.children,
            key=lambda m: root.children[m].visit_count,
        )
        return best_move

    # ------------------------------------------------------------------ #
    # Dirichlet noise for exploration
    # ------------------------------------------------------------------ #
    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to the root node's priors for exploration.

        This encourages the search to explore moves that the network initially
        considers unlikely, which is critical for self-play improvement.

        The noisy prior is:
            P_noisy(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)

        Parameters
        ----------
        root : MCTSNode
            The root node whose children priors will be modified.
        """
        num_children = len(root.children)
        if num_children == 0:
            return

        # Sample Dirichlet noise.
        noise = self._sample_dirichlet(self.dirichlet_alpha, num_children)

        # Mix noise with existing priors.
        eps = self.dirichlet_epsilon
        for child, eta in zip(root.children.values(), noise):
            child.prior = (1.0 - eps) * child.prior + eps * eta

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _terminal_value(result: GameResult, side_to_move: Color) -> float:
        """Convert a terminal GameResult to a value from the mover's perspective.

        Parameters
        ----------
        result : GameResult
            The terminal game result.
        side_to_move : Color
            The side to move at the terminal position.

        Returns
        -------
        float
            +1.0 if the side to move wins, -1.0 if it loses, 0.0 for a draw.
        """
        if result.outcome is GameOutcome.DRAW:
            return 0.0
        if result.winner is None:
            return 0.0
        if result.winner is side_to_move:
            return 1.0
        return -1.0

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        """Compute softmax probabilities from a list of logits.

        Uses the max-subtraction trick for numerical stability.

        Parameters
        ----------
        logits : List[float]
            Raw logit values.

        Returns
        -------
        List[float]
            Softmax probabilities (sum to 1.0).
        """
        if not logits:
            return []
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _sample_dirichlet(alpha: float, size: int) -> List[float]:
        """Sample from a symmetric Dirichlet distribution.

        Uses the gamma-distribution method: sample Gamma(alpha, 1) for each
        component, then normalize.

        Parameters
        ----------
        alpha : float
            Concentration parameter. Smaller values produce sparser samples.
        size : int
            Dimensionality of the sample.

        Returns
        -------
        List[float]
            A probability vector sampled from Dir(alpha, ..., alpha).
        """
        # Use Python's random module for gamma sampling.
        import random as _random

        samples = [_random.gammavariate(alpha, 1.0) for _ in range(size)]
        total = sum(samples)
        if total == 0.0:
            # Degenerate case: return uniform.
            return [1.0 / size] * size
        return [s / total for s in samples]

    @staticmethod
    def _sample_move(probs: Dict[Move, float]) -> Move:
        """Sample a move from a probability distribution.

        Parameters
        ----------
        probs : Dict[Move, float]
            Move probability distribution (should sum to ~1.0).

        Returns
        -------
        Move
            A move sampled according to the given probabilities.
        """
        import random as _random

        r = _random.random()
        cumulative = 0.0
        moves = list(probs.keys())
        for move in moves:
            cumulative += probs[move]
            if r <= cumulative:
                return move
        # Fallback (floating-point edge case): return last move.
        return moves[-1]
