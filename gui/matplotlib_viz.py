"""gui/matplotlib_viz.py — Static / notebook visualization for Mini Xiangqi.

A renderer that draws a :class:`engine.board.Board` with matplotlib and can
either show it interactively or save it to a PNG. Unlike the pygame GUI, this
needs no live display: ``render(board, save_path="x.png")`` works headlessly,
which makes it ideal for:

  * inspecting a position the AI is thinking about,
  * rendering frames of a self-play game into a folder,
  * embedding boards in a Jupyter notebook (``render(board, show=True)``).

Requires matplotlib + numpy (both already used elsewhere in the project).

Highlights supported (any combination, all optional):
  * ``selected``      — green ring on a piece being analyzed.
  * ``legal_targets`` — green dots on each legal destination for that piece.
  * ``last_move``     — amber rings on the from/to squares of the last move.
  * ``check``         — red ring on the king of the side to move if in check.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from engine.board import BOARD_SIZE, Board, Move, in_bounds
from engine.move_generator import in_check, legal_moves
from engine.piece import Color, PieceType

try:
    import matplotlib
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

# Use the Agg backend by default so saving works without a display. Callers that
# want an interactive window can switch backends before calling render().
if _HAS_MPL and not matplotlib.get_backend().lower().startswith("inline"):
    try:
        matplotlib.use("Agg", force=False)
    except Exception:  # pragma: no cover - never let backend switching crash import
        pass

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
from matplotlib import font_manager  # noqa: E402


def _pick_cjk_font() -> str:
    """Return the first CJK-capable font name matplotlib can find.

    Probed in order of preference; falls back to matplotlib's default if none
    of the known CJK families is installed (glyphs would then render as boxes,
    but board geometry and highlights still draw correctly).
    """
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "Noto Sans CJK TC", "Source Han Sans SC", "PingFang SC",
        "Arial Unicode MS",
    ):
        if candidate in installed:
            return candidate
    return font_manager.findfont(None, fallback_to_default=True)


_CJK_FONT = _pick_cjk_font()

Square = Tuple[int, int]

# Colors.
_BG = "#deb887"          # light wood
_LINE = "#3c2814"        # grid lines
_RED = "#c82828"
_BLACK = "#282828"
_LIGHT_PIECE = "#f5ebd2"  # disc fill
_HIGHLIGHT = "#50c850"    # selected / legal target
_LAST_MOVE = "#dcc850"    # last move from/to
_CHECK = "#e02020"        # king-in-check marker

# Chinese glyph per (color, PieceType).
_GLYPH = {
    (Color.RED, PieceType.KING): "帥",
    (Color.BLACK, PieceType.KING): "將",
    (Color.RED, PieceType.ROOK): "車",
    (Color.BLACK, PieceType.ROOK): "車",
    (Color.RED, PieceType.HORSE): "馬",
    (Color.BLACK, PieceType.HORSE): "馬",
    (Color.RED, PieceType.CANNON): "炮",
    (Color.BLACK, PieceType.CANNON): "砲",
    (Color.RED, PieceType.SOLDIER): "兵",
    (Color.BLACK, PieceType.SOLDIER): "卒",
}


def _board_to_xy(row: int, col: int) -> Tuple[float, float]:
    """Board (row, col) -> plot (x, y). Row 0 (Red) at the bottom."""
    return float(col), float(BOARD_SIZE - 1 - row)


def _palace_bounds() -> Iterable[Tuple[float, float, float, float]]:
    """Yield (x_min, x_max, y_min, y_max) for each palace, in plot coords."""
    for r0, r1, c0, c1 in [(0, 2, 2, 4), (4, 6, 2, 4)]:
        x0, y0 = _board_to_xy(r0, c0)
        x1, y1 = _board_to_xy(r1, c1)
        yield (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def _draw_grid(ax: plt.Axes) -> None:
    for i in range(BOARD_SIZE):
        # Vertical line (col i): from row 0 to row 6.
        x0, y0 = _board_to_xy(0, i)
        x1, y1 = _board_to_xy(BOARD_SIZE - 1, i)
        ax.plot([x0, x1], [y0, y1], color=_LINE, linewidth=1.2, zorder=1)
        # Horizontal line (row i): from col 0 to col 6.
        x0, y0 = _board_to_xy(i, 0)
        x1, y1 = _board_to_xy(i, BOARD_SIZE - 1)
        ax.plot([x0, x1], [y0, y1], color=_LINE, linewidth=1.2, zorder=1)

    # Palace diagonals.
    for x_min, x_max, y_min, y_max in _palace_bounds():
        ax.plot([x_min, x_max], [y_min, y_max], color=_LINE, linewidth=0.8, zorder=1)
        ax.plot([x_min, x_max], [y_max, y_min], color=_LINE, linewidth=0.8, zorder=1)


def _draw_piece(ax: plt.Axes, row: int, col: int, piece) -> None:
    x, y = _board_to_xy(row, col)
    color = _RED if piece.color is Color.RED else _BLACK
    ax.scatter([x], [y], s=1400, c=_LIGHT_PIECE,
               edgecolors=color, linewidths=2.0, zorder=3)
    ax.text(x, y, _GLYPH[(piece.color, piece.ptype)], color=color,
            fontsize=18, ha="center", va="center", zorder=4,
            fontfamily=_CJK_FONT)


def render(
    board: Board,
    *,
    selected: Optional[Square] = None,
    legal_targets: Optional[Iterable[Square]] = None,
    last_move: Optional[Move] = None,
    show_check: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> plt.Figure:
    """Render ``board`` and optionally save/show it. Returns the matplotlib Figure.

    Parameters
    ----------
    selected, legal_targets
        Highlight a piece and its legal destinations (typically set together:
        ``legal_targets`` defaults to the legal moves out of ``selected``).
    last_move
        Amber rings on the move's from/to squares.
    show_check
        Red ring on the side-to-move king if it is in check.
    save_path
        If given, write a PNG (works headlessly).
    show
        If True, call ``plt.show()`` for an interactive window.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for render()")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)
    _draw_grid(ax)

    # Default legal_targets to the selected piece's moves.
    if legal_targets is None and selected is not None:
        legal_targets = {
            m.to_sq for m in legal_moves(board) if m.from_sq == selected
        }

    # Last-move highlight.
    if last_move is not None:
        for sq in (last_move.from_sq, last_move.to_sq):
            x, y = _board_to_xy(*sq)
            ax.scatter([x], [y], s=1700, facecolors="none",
                       edgecolors=_LAST_MOVE, linewidths=2.2, zorder=2)

    # Selected piece + its legal destinations.
    if selected is not None:
        x, y = _board_to_xy(*selected)
        ax.scatter([x], [y], s=1900, facecolors="none",
                   edgecolors=_HIGHLIGHT, linewidths=2.4, zorder=2)
    if legal_targets:
        for (r, c) in legal_targets:
            x, y = _board_to_xy(r, c)
            ax.scatter([x], [y], s=160, c=_HIGHLIGHT, zorder=3, alpha=0.85)

    # Pieces.
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board.piece_at(row, col)
            if piece is not None:
                _draw_piece(ax, row, col, piece)

    # Check indicator.
    if show_check and in_check(board, board.side_to_move):
        ksq = board.find_king(board.side_to_move)
        if ksq is not None:
            x, y = _board_to_xy(*ksq)
            ax.scatter([x], [y], s=2000, facecolors="none",
                       edgecolors=_CHECK, linewidths=2.6, zorder=2)

    # File/rank labels.
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_xticklabels([chr(ord("a") + c) for c in range(BOARD_SIZE)], fontsize=10)
    ax.set_yticks(range(BOARD_SIZE))
    ax.set_yticklabels([str(r + 1) for r in range(BOARD_SIZE - 1, -1, -1)], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.7, BOARD_SIZE - 0.3)
    ax.set_ylim(-0.7, BOARD_SIZE - 0.3)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(f"Mini Xiangqi — {board.side_to_move.name.lower()} to move",
                     fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def render_game(
    moves: Iterable[Move],
    *,
    out_dir: str,
    start: Optional[Board] = None,
    prefix: str = "frame",
) -> int:
    """Render every position of a move list to ``out_dir/<prefix>_NNN.png``.

    Useful for producing a flipbook / gif of a self-play game. Returns the
    number of frames written. The starting position is rendered as frame 000,
    and each move produces the next frame.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    board = start.clone() if start is not None else Board()
    n = 0
    render(board, last_move=None, save_path=os.path.join(out_dir, f"{prefix}_{n:03d}.png"))
    for mv in moves:
        board.make_move(mv)
        n += 1
        render(board, last_move=mv,
               save_path=os.path.join(out_dir, f"{prefix}_{n:03d}.png"))
    return n + 1


def frames_to_gif(frame_paths, gif_path: str, duration_ms: int = 600) -> str:
    """Stitch PNG frames (in given order) into a GIF at ``gif_path``.

    Tries Pillow first (widely available), then imageio. Raises RuntimeError
    with an actionable message if neither is installed.
    """
    if not frame_paths:
        raise ValueError("no frames to stitch")

    # --- Pillow path ---
    try:
        from PIL import Image
        images = [Image.open(p).convert("P") for p in frame_paths]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        return gif_path
    except ImportError:
        pass

    # --- imageio path ---
    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, frames, duration=duration_ms / 1000.0)
        return gif_path
    except ImportError:
        raise RuntimeError(
            "GIF export needs Pillow or imageio. Install one with: "
            "pip install pillow   (or)   pip install imageio"
        )


def animate_game(
    moves: Iterable[Move],
    *,
    out_dir: str,
    start: Optional[Board] = None,
    prefix: str = "frame",
    outcome: Optional[str] = None,
    gif_path: Optional[str] = None,
    duration_ms: int = 600,
) -> int:
    """Render a self-play game frame-by-frame; optionally export a GIF.

    Produces ``<prefix>_000.png`` (start position) ... ``<prefix>_NNN.png``
    (final position), one PNG per ply, each captioned with move number + mover
    + UCI move, and (if given) a final-result banner. If ``gif_path`` is set,
    all frames are stitched into an animated GIF.

    Returns the number of frames written.
    """
    import os

    from engine.piece import Color

    os.makedirs(out_dir, exist_ok=True)
    board = start.clone() if start is not None else Board()

    frames: list[str] = []

    def _write(caption: str, last_move: Optional[Move]) -> None:
        path = os.path.join(out_dir, f"{prefix}_{len(frames):03d}.png")
        render(board, last_move=last_move, title=caption, save_path=path)
        frames.append(path)

    _write("start — red to move", None)

    moves = list(moves)
    for i, mv in enumerate(moves):
        mover = "red" if board.side_to_move is Color.RED else "black"
        move_no = (i // 2) + 1
        caption = f"{move_no}. {mover} {mv.uci()}"
        board.make_move(mv)

        # If this was the last move and we have an outcome, annotate it.
        if i == len(moves) - 1 and outcome:
            result_word = {
                "red": "RED WINS", "black": "BLACK WINS",
                "draw": "DRAW",
            }.get(outcome, outcome.upper())
            caption = f"{caption}   —   {result_word}"
        _write(caption, mv)

    if gif_path is not None:
        frames_to_gif(frames, gif_path, duration_ms=duration_ms)
    return len(frames)
