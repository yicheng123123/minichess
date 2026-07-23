"""gui/pygame_gui.py — Pygame renderer and play loop for Mini Xiangqi.

Features:
  * Human vs Human mode
  * Human vs AI mode (alpha-beta or MCTS, configurable difficulty)
  * Undo (take-back) support
  * New Game / Reset
  * Status bar showing side to move, check, game result
  * Last-move and selected-piece highlights

Pygame is optional: the module imports fine without it, and :func:`run` raises
a clear error if it's missing.

Font note
---------
We deliberately load the CJK font by *file path* via ``pygame.font.Font``,
instead of ``pygame.font.SysFont``. On some pygame-ce / Python combinations
(e.g. pygame-ce 2.5.7 on Windows + Python 3.12) ``SysFont``'s system-font
enumeration crashes, which silently falls back to the default bitmap font that
has no Chinese glyphs — producing the "pieces with no text" symptom. Loading
the TTF/TTC directly sidesteps that bug entirely.
"""

from __future__ import annotations

import os
import sys
import threading
from enum import Enum
from typing import List, Optional, Tuple

from engine.board import BOARD_SIZE, Board
from engine.move import Move, in_bounds
from engine.move_generator import legal_moves, in_check
from engine.rules import game_result, is_game_over
from engine.piece import Color
from engine.game import Game

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:  # pragma: no cover
    _HAS_PYGAME = False


# --------------------------------------------------------------------------- #
# Font discovery
# --------------------------------------------------------------------------- #
def _candidate_cjk_font_paths() -> List[str]:
    win = os.environ.get("WINDIR", r"C:\Windows")
    return [
        os.path.join(win, "Fonts", "simhei.ttf"),
        os.path.join(win, "Fonts", "msyh.ttc"),
        os.path.join(win, "Fonts", "simsun.ttc"),
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        # Linux (Noto / Source Han)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]


# --------------------------------------------------------------------------- #
# Visual constants
# --------------------------------------------------------------------------- #
CELL = 70
MARGIN = 40
BOARD_PX = CELL * (BOARD_SIZE - 1)
STATUS_H = 50  # status bar height at the bottom
WIN_W = MARGIN * 2 + BOARD_PX
WIN_H = MARGIN * 2 + BOARD_PX + STATUS_H

# Colors.
_BG = (222, 184, 135)       # light wood
_LINE = (60, 40, 20)        # grid lines
_RED = (200, 40, 40)
_BLACK_P = (40, 40, 40)
_HIGHLIGHT = (80, 200, 80)
_LAST_MOVE = (220, 200, 60)
_STATUS_BG = (50, 50, 50)
_STATUS_FG = (240, 240, 240)
_CHECK_COLOR = (255, 60, 60)
_BTN_BG = (70, 70, 70)
_BTN_FG = (230, 230, 230)
_BTN_HOVER = (100, 100, 100)

# Chinese characters used for each piece, by (color, type).
_GLYPH = {
    (Color.RED, "K"): "帥", (Color.BLACK, "K"): "將",
    (Color.RED, "R"): "車", (Color.BLACK, "R"): "車",
    (Color.RED, "N"): "馬", (Color.BLACK, "N"): "馬",
    (Color.RED, "C"): "炮", (Color.BLACK, "C"): "砲",
    (Color.RED, "P"): "兵", (Color.BLACK, "P"): "卒",
}


# --------------------------------------------------------------------------- #
# Game mode / difficulty
# --------------------------------------------------------------------------- #
class GameMode(Enum):
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_AI = "human_vs_ai"


class Difficulty(Enum):
    EASY = 1      # alpha-beta depth 2
    MEDIUM = 2    # alpha-beta depth 3
    HARD = 3      # alpha-beta depth 4
    EXPERT = 4    # MCTS 200 simulations


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _board_to_screen(row: int, col: int, flip: bool = False) -> Tuple[int, int]:
    """Board (row, col) -> screen pixel (x, y).

    By default Row 0 (Red) is at the bottom. When *flip* is True the board is
    rendered upside-down so that Black (row 6) is at the bottom — used when the
    human plays Black.
    """
    if flip:
        x = MARGIN + (BOARD_SIZE - 1 - col) * CELL
        y = MARGIN + row * CELL
    else:
        x = MARGIN + col * CELL
        y = MARGIN + (BOARD_SIZE - 1 - row) * CELL
    return x, y


def _screen_to_board(x: int, y: int, flip: bool = False) -> Optional[Tuple[int, int]]:
    """Screen pixel -> nearest board (row, col), or None if off the grid."""
    if flip:
        col = BOARD_SIZE - 1 - round((x - MARGIN) / CELL)
        row = round((y - MARGIN) / CELL)
    else:
        col = round((x - MARGIN) / CELL)
        row = BOARD_SIZE - 1 - round((y - MARGIN) / CELL)
    if in_bounds(row, col):
        return row, col
    return None


# --------------------------------------------------------------------------- #
# AI move computation (runs in a thread to avoid blocking the GUI)
# --------------------------------------------------------------------------- #
def _compute_ai_move(board: Board, difficulty: Difficulty) -> Optional[Move]:
    """Compute the AI's move for the given difficulty level."""
    if difficulty in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        from search.alphabeta import alphabeta
        depth = difficulty.value + 1  # EASY=2, MEDIUM=3, HARD=4
        _score, move = alphabeta(board, depth=depth)
        return move
    else:
        # EXPERT: use MCTS with the default network
        from search.mcts import MCTS
        from nn.network import default_net
        net = default_net()
        mcts = MCTS(num_simulations=200)
        _probs, move = mcts.search(board, net)
        return move


# --------------------------------------------------------------------------- #
# Color selection screen (shown at startup in HvAI mode)
# --------------------------------------------------------------------------- #
def _select_color_screen() -> Color:
    """Show a visual selection screen letting the player choose Red or Black.

    Returns the chosen Color. The caller is responsible for pygame.init().
    """
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Mini Xiangqi — 选择阵营")
    font_title = _load_standalone_font(44)
    font_sub = _load_standalone_font(22)
    font_piece = _load_standalone_font(60)
    clock = pygame.time.Clock()

    # Button geometry
    btn_w, btn_h = 200, 100
    gap = 60
    total_w = btn_w * 2 + gap
    start_x = (WIN_W - total_w) // 2
    btn_y = WIN_H // 2 - btn_h // 2 + 20

    red_rect = pygame.Rect(start_x, btn_y, btn_w, btn_h)
    black_rect = pygame.Rect(start_x + btn_w + gap, btn_y, btn_w, btn_h)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if red_rect.collidepoint(event.pos):
                    return Color.RED
                if black_rect.collidepoint(event.pos):
                    return Color.BLACK
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_r:
                    return Color.RED
                if event.key == pygame.K_b:
                    return Color.BLACK

        mouse_pos = pygame.mouse.get_pos()
        screen.fill((40, 40, 50))

        # Title
        title = font_title.render("选择阵营", True, (240, 240, 240))
        screen.blit(title, title.get_rect(center=(WIN_W // 2, WIN_H // 2 - 120)))
        hint = font_sub.render("Choose your side  (R / B 快捷键)", True, (160, 160, 160))
        screen.blit(hint, hint.get_rect(center=(WIN_W // 2, WIN_H // 2 - 75)))

        # Red button
        r_hover = red_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (120, 30, 30) if r_hover else (80, 25, 25),
                         red_rect, border_radius=10)
        pygame.draw.rect(screen, _RED, red_rect, 2, border_radius=10)
        glyph_r = font_piece.render("帥", True, _RED)
        screen.blit(glyph_r, glyph_r.get_rect(center=(red_rect.centerx, red_rect.centery - 10)))
        label_r = font_sub.render("执红 (先手)", True, (230, 200, 200))
        screen.blit(label_r, label_r.get_rect(center=(red_rect.centerx, red_rect.bottom - 18)))

        # Black button
        b_hover = black_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (50, 50, 60) if b_hover else (35, 35, 45),
                         black_rect, border_radius=10)
        pygame.draw.rect(screen, (180, 180, 180), black_rect, 2, border_radius=10)
        glyph_b = font_piece.render("將", True, (220, 220, 220))
        screen.blit(glyph_b, glyph_b.get_rect(center=(black_rect.centerx, black_rect.centery - 10)))
        label_b = font_sub.render("执黑 (后手)", True, (200, 200, 210))
        screen.blit(label_b, label_b.get_rect(center=(black_rect.centerx, black_rect.bottom - 18)))

        pygame.display.flip()
        clock.tick(30)


def _load_standalone_font(size: int):
    """Load a CJK font without requiring the GUI class (used by selection screen)."""
    probe = "帥"
    for path in _candidate_cjk_font_paths():
        if not os.path.exists(path):
            continue
        try:
            font = pygame.font.Font(path, size)
        except Exception:
            continue
        try:
            surf = font.render(probe, True, (0, 0, 0))
            arr = pygame.surfarray.array2d(surf)
            if (arr != 0).any():
                return font
        except Exception:
            continue
    return pygame.font.Font(None, size)


# --------------------------------------------------------------------------- #
# Main GUI class
# --------------------------------------------------------------------------- #
class MiniXiangqiGUI:
    """Encapsulates the pygame window, drawing, interaction, and AI opponent."""

    def __init__(
        self,
        board: Optional[Board] = None,
        mode: GameMode = GameMode.HUMAN_VS_AI,
        difficulty: Difficulty = Difficulty.MEDIUM,
        human_color: Color = Color.RED,
    ) -> None:
        if not _HAS_PYGAME:
            raise RuntimeError(
                "pygame is required for the GUI. Install with: pip install pygame"
            )
        self.game = Game(board=board)
        self.mode = mode
        self.difficulty = difficulty
        self.human_color = human_color

        self.selected: Optional[Tuple[int, int]] = None
        self.legal_dests: set = set()
        self.last_move: Optional[Move] = None

        # AI state
        self._ai_thinking = False
        self._ai_move: Optional[Move] = None
        self._ai_thread: Optional[threading.Thread] = None

        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Mini Xiangqi (迷你象棋)")
        self.font = self._load_font(38)
        self.font_small = self._load_font(20)
        self.clock = pygame.time.Clock()

    @property
    def _flipped(self) -> bool:
        """True when the board should be drawn flipped (human plays Black)."""
        return self.mode == GameMode.HUMAN_VS_AI and self.human_color is Color.BLACK

    def _load_font(self, size: int):
        """Load a CJK-capable font by file path (see module docstring)."""
        probe = "帥"
        for path in _candidate_cjk_font_paths():
            if not os.path.exists(path):
                continue
            try:
                font = pygame.font.Font(path, size)
            except Exception:
                continue
            try:
                surf = font.render(probe, True, (0, 0, 0))
                arr = pygame.surfarray.array2d(surf)
                if (arr != 0).any():
                    return font
            except Exception:
                continue
        return pygame.font.Font(None, size)

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw(self) -> None:
        self.screen.fill(_BG)
        self._draw_grid()
        self._draw_palaces()
        self._draw_highlights()
        self._draw_pieces()
        self._draw_status_bar()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        f = self._flipped
        for i in range(BOARD_SIZE):
            x0, y0 = _board_to_screen(0, i, f)
            x1, y1 = _board_to_screen(BOARD_SIZE - 1, i, f)
            pygame.draw.line(self.screen, _LINE, (x0, y0), (x1, y1), 2)
            xr0, yr0 = _board_to_screen(i, 0, f)
            xr1, yr1 = _board_to_screen(i, BOARD_SIZE - 1, f)
            pygame.draw.line(self.screen, _LINE, (xr0, yr0), (xr1, yr1), 2)

    def _draw_palaces(self) -> None:
        f = self._flipped
        for r0, r1, c0, c1 in [(0, 2, 2, 4), (4, 6, 2, 4)]:
            tl = _board_to_screen(r1, c0, f)
            br = _board_to_screen(r0, c1, f)
            tr = _board_to_screen(r1, c1, f)
            bl = _board_to_screen(r0, c0, f)
            pygame.draw.line(self.screen, _LINE, tl, br, 1)
            pygame.draw.line(self.screen, _LINE, tr, bl, 1)

    def _draw_highlights(self) -> None:
        f = self._flipped
        if self.last_move is not None:
            for sq in (self.last_move.from_sq, self.last_move.to_sq):
                x, y = _board_to_screen(*sq, f)
                pygame.draw.circle(self.screen, _LAST_MOVE, (x, y), CELL // 2 - 4, 3)
        if self.selected is not None:
            x, y = _board_to_screen(*self.selected, f)
            pygame.draw.circle(self.screen, _HIGHLIGHT, (x, y), CELL // 2 - 4, 3)
        for (r, c) in self.legal_dests:
            x, y = _board_to_screen(r, c, f)
            pygame.draw.circle(self.screen, _HIGHLIGHT, (x, y), 8)
        # Check indicator
        board = self.game.board
        if in_check(board, board.side_to_move):
            ksq = board.find_king(board.side_to_move)
            if ksq:
                x, y = _board_to_screen(*ksq, f)
                pygame.draw.circle(self.screen, _CHECK_COLOR, (x, y), CELL // 2 - 2, 3)

    def _draw_pieces(self) -> None:
        board = self.game.board
        f = self._flipped
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board.piece_at(row, col)
                if piece is None:
                    continue
                x, y = _board_to_screen(row, col, f)
                color = _RED if piece.color is Color.RED else _BLACK_P
                pygame.draw.circle(self.screen, (245, 235, 210), (x, y), CELL // 2 - 6)
                pygame.draw.circle(self.screen, color, (x, y), CELL // 2 - 6, 3)
                glyph = _GLYPH[(piece.color, piece.ptype.value)]
                text = self.font.render(glyph, True, color)
                self.screen.blit(text, text.get_rect(center=(x, y)))

    def _draw_status_bar(self) -> None:
        """Draw the bottom status bar with game info and buttons."""
        bar_y = WIN_H - STATUS_H
        pygame.draw.rect(self.screen, _STATUS_BG, (0, bar_y, WIN_W, STATUS_H))

        # Status text
        board = self.game.board
        if self.game.is_over:
            result = self.game.result
            status = f"Game Over: {result.outcome.value} ({result.reason})"
        elif self._ai_thinking:
            status = "AI thinking..."
        else:
            side = "Red" if board.side_to_move is Color.RED else "Black"
            check_str = " [CHECK!]" if in_check(board, board.side_to_move) else ""
            mode_str = "HvH" if self.mode == GameMode.HUMAN_VS_HUMAN else "HvAI"
            you_str = ""
            if self.mode == GameMode.HUMAN_VS_AI:
                you_str = f" | You: {'Red' if self.human_color is Color.RED else 'Black'}"
            status = f"{side} to move{check_str}  |  {mode_str} | {self.difficulty.name}{you_str}"

        text = self.font_small.render(status, True, _STATUS_FG)
        self.screen.blit(text, (10, bar_y + 15))

        # Buttons: Switch, Undo, New Game
        self._draw_button("Side (S)", WIN_W - 300, bar_y + 10, 85, 30)
        self._draw_button("Undo (Z)", WIN_W - 200, bar_y + 10, 85, 30)
        self._draw_button("New (N)", WIN_W - 105, bar_y + 10, 85, 30)

    def _draw_button(self, label: str, x: int, y: int, w: int, h: int) -> None:
        rect = pygame.Rect(x, y, w, h)
        mouse_pos = pygame.mouse.get_pos()
        color = _BTN_HOVER if rect.collidepoint(mouse_pos) else _BTN_BG
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, _BTN_FG, rect, 1, border_radius=4)
        text = self.font_small.render(label, True, _BTN_FG)
        self.screen.blit(text, text.get_rect(center=rect.center))

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def _is_human_turn(self) -> bool:
        """Whether the current side to move is controlled by the human."""
        if self.mode == GameMode.HUMAN_VS_HUMAN:
            return True
        return self.game.side_to_move == self.human_color

    def on_click(self, pos) -> None:
        # Check button clicks first
        bar_y = WIN_H - STATUS_H
        if pos[1] >= bar_y:
            if WIN_W - 300 <= pos[0] <= WIN_W - 215 and bar_y + 10 <= pos[1] <= bar_y + 40:
                self._switch_side()
                return
            if WIN_W - 200 <= pos[0] <= WIN_W - 115 and bar_y + 10 <= pos[1] <= bar_y + 40:
                self._undo()
                return
            if WIN_W - 105 <= pos[0] <= WIN_W - 20 and bar_y + 10 <= pos[1] <= bar_y + 40:
                self._new_game()
                return
            return

        # Only allow clicks during human's turn and when AI isn't thinking
        if not self._is_human_turn() or self._ai_thinking:
            return
        if self.game.is_over:
            return

        sq = _screen_to_board(*pos, self._flipped)
        if sq is None:
            return
        row, col = sq

        # If a piece is selected and this is a legal destination, move.
        if self.selected is not None and (row, col) in self.legal_dests:
            move = Move(self.selected, (row, col))
            self._apply_move(move)
            self.selected = None
            self.legal_dests = set()
            return

        # Otherwise, try to select a piece of the side to move.
        board = self.game.board
        piece = board.piece_at(row, col)
        if piece is not None and piece.color is board.side_to_move:
            self.selected = (row, col)
            self.legal_dests = {
                m.to_sq for m in self.game.legal_moves() if m.from_sq == (row, col)
            }
        else:
            self.selected = None
            self.legal_dests = set()

    def _apply_move(self, move: Move) -> None:
        """Apply a move through the Game controller."""
        self.game.play(move)
        self.last_move = move

    def _undo(self) -> None:
        """Undo the last move (or last two in HvAI mode to undo both AI + human)."""
        if self._ai_thinking:
            return
        if self.mode == GameMode.HUMAN_VS_AI and self.game.ply >= 2:
            self.game.undo()
            self.game.undo()
        elif self.game.ply >= 1:
            self.game.undo()
        self.selected = None
        self.legal_dests = set()
        # Update last_move display
        if self.game.move_history:
            self.last_move = Move.from_uci(self.game.move_history[-1])
        else:
            self.last_move = None

    def _new_game(self) -> None:
        """Reset to a fresh game."""
        self._ai_thinking = False
        self._ai_move = None
        self.game = Game()
        self.selected = None
        self.legal_dests = set()
        self.last_move = None
        pygame.display.set_caption("Mini Xiangqi (迷你象棋)")

    def _switch_side(self) -> None:
        """Switch which color the human plays (HvAI mode only). Starts a new game."""
        if self.mode != GameMode.HUMAN_VS_AI:
            return
        self.human_color = self.human_color.opponent
        self._new_game()

    # ------------------------------------------------------------------ #
    # AI turn handling
    # ------------------------------------------------------------------ #
    def _maybe_trigger_ai(self) -> None:
        """If it's the AI's turn and the game isn't over, start thinking."""
        if self.mode != GameMode.HUMAN_VS_AI:
            return
        if self.game.is_over:
            return
        if self.game.side_to_move == self.human_color:
            return
        if self._ai_thinking:
            return

        self._ai_thinking = True
        self._ai_move = None
        board_snapshot = self.game.board.clone()

        def _think():
            move = _compute_ai_move(board_snapshot, self.difficulty)
            self._ai_move = move
            self._ai_thinking = False

        self._ai_thread = threading.Thread(target=_think, daemon=True)
        self._ai_thread.start()

    def _check_ai_result(self) -> None:
        """If the AI has finished thinking, apply its move."""
        if not self._ai_thinking and self._ai_move is not None:
            self._apply_move(self._ai_move)
            self._ai_move = None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z:
                        self._undo()
                    elif event.key == pygame.K_n:
                        self._new_game()
                    elif event.key == pygame.K_s:
                        self._switch_side()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # AI turn management
            self._check_ai_result()
            self._maybe_trigger_ai()

            self.draw()

            # Update caption on game over
            if self.game.is_over:
                result = self.game.result
                pygame.display.set_caption(
                    f"Mini Xiangqi — {result.outcome.value} ({result.reason})"
                )

            self.clock.tick(30)
        pygame.quit()


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run(
    board: Optional[Board] = None,
    mode: GameMode = GameMode.HUMAN_VS_AI,
    difficulty: Difficulty = Difficulty.MEDIUM,
    human_color: Color = Color.RED,
    choose_color: bool = True,
) -> None:
    """Entry point: open the window and run the play loop.

    If *choose_color* is True and mode is HUMAN_VS_AI, a visual selection
    screen is shown first so the player can pick Red or Black.
    """
    if not _HAS_PYGAME:
        raise RuntimeError(
            "pygame is required for the GUI. Install with: pip install pygame"
        )
    pygame.init()

    # Show color selection screen in HvAI mode
    if choose_color and mode == GameMode.HUMAN_VS_AI:
        human_color = _select_color_screen()

    MiniXiangqiGUI(
        board=board, mode=mode, difficulty=difficulty, human_color=human_color
    ).run()


if __name__ == "__main__":
    run()
