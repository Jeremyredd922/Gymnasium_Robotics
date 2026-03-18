"""
ascii_renderer.py — Convert ViZDoom GRAY8 frame buffers to ASCII art.
"""
import os
import sys

import numpy as np

_RAMP       = " `.-_':,;^=+!|/\\1tfilI?rjJFfxX2YV3nuvczUCL0OZmwqpdbkhao*#M&8%B@"
_RAMP_LEN   = len(_RAMP)
_RAMP_ARRAY = np.array(list(_RAMP), dtype='U1')

_CURSOR_HOME  = "\033[H"
_CURSOR_HIDE  = "\033[?25l"
_CURSOR_SHOW  = "\033[?25h"
_CLEAR_SCREEN = "\033[2J"
_RESET        = "\033[0m"

_HP_FULL  = "#"
_HP_EMPTY = "-"
_HP_BAR   = 20


def _terminal_size():
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 80, 24


def _compute_dims(term_cols, term_rows):
    available_rows = max(5, term_rows - 4)
    cols_from_rows = int(available_rows * 8 / 3)
    cols = min(term_cols, cols_from_rows)
    rows = int(cols * 3 / 8)
    rows = min(rows, available_rows)
    return max(20, cols), max(5, rows)


class AsciiRenderer:

    def __init__(self, width=None, height=None):
        self._fixed_w  = width
        self._fixed_h  = height
        self._last_tw  = self._last_th = 0
        self._cols     = self._rows = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, screen_buf, health=100, ammo=50, kills=0,
               episode=1, strategy="", player_mode=False):
        tw, th = self._terminal_dims()
        if tw != self._last_tw or th != self._last_th:
            self._cols, self._rows = _compute_dims(tw, th)
            self._last_tw, self._last_th = tw, th

        frame_str = self._frame_to_ascii(screen_buf)
        hud_str   = self._build_hud(health, ammo, kills, episode,
                                    strategy, self._cols, player_mode)
        sys.stdout.write(_CURSOR_HOME + frame_str + "\n" + hud_str)
        sys.stdout.flush()

    def render_episode_end(self, episode, kills, total_kills, duration,
                           map_name=None, completed=False, next_map=None):
        tw, _ = self._terminal_dims()
        mins, secs = divmod(int(duration), 60)
        if completed and next_map:
            status = f"{map_name} cleared! → {next_map}"
        elif completed:
            status = f"{map_name} cleared!"
        elif map_name:
            status = f"Died on {map_name}"
        else:
            status = f"Episode {episode} ended"
        line = (f"  [ {status} | "
                f"Kills: {kills}  Total: {total_kills}  "
                f"Time: {mins:02d}:{secs:02d} ]")
        sys.stdout.write("\n" + line[:tw])
        sys.stdout.flush()

    def show_loading(self):
        sys.stdout.write(_CLEAR_SCREEN + _CURSOR_HOME +
                         "  Loading ViZDoom engine...\n")
        sys.stdout.flush()

    def show_banner(self, name: str, player_mode: bool = False):
        tw, _ = self._terminal_dims()
        if player_mode:
            from agent import HumanAgent
            banner = (f"  PLAYER MODE  |  {HumanAgent.CONTROLS}"
                      f"  |  Ctrl+C to quit")
        else:
            banner = f"  AI agent: {name.upper()}  |  Press Ctrl+C to quit"
        sys.stdout.write(_CLEAR_SCREEN + _CURSOR_HOME + banner[:tw] + "\n")
        sys.stdout.flush()

    def hide_cursor(self):
        sys.stdout.write(_CURSOR_HIDE)
        sys.stdout.flush()

    def show_cursor(self):
        sys.stdout.write(_CURSOR_SHOW + _RESET + "\n")
        sys.stdout.flush()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _terminal_dims(self):
        if self._fixed_w and self._fixed_h:
            return self._fixed_w, self._fixed_h
        tw, th = _terminal_size()
        if self._fixed_w: tw = self._fixed_w
        if self._fixed_h: th = self._fixed_h
        return tw, th

    def _frame_to_ascii(self, buf):
        orig_h, orig_w = buf.shape
        rows, cols = self._rows, self._cols
        row_idx = np.linspace(0, orig_h - 1, rows, dtype=int)
        col_idx = np.linspace(0, orig_w - 1, cols, dtype=int)
        small   = buf[np.ix_(row_idx, col_idx)]
        indices = (small.astype(np.float32) / 255.0 * (_RAMP_LEN - 1)).astype(int)
        return "\n".join("".join(_RAMP_ARRAY[indices[r]]) for r in range(rows))

    @staticmethod
    def _build_hud(health, ammo, kills, episode, strategy, width, player_mode):
        hp_filled = int(max(0, min(health, 100)) / 100 * _HP_BAR)
        hp_bar    = (_HP_FULL * hp_filled).ljust(_HP_BAR, _HP_EMPTY)

        if player_mode:
            label = "YOU"
            info  = "W/S:fwd  A/D:turn  SPC:fire  F:use"[:18].ljust(18)
        else:
            label = "AI "
            info  = (strategy[:18] if strategy else "").ljust(18)

        hud = (f" EP:{episode:3d}  "
               f"HP:[{hp_bar}]{health:3d}  "
               f"AMMO:{ammo:3d}  "
               f"KILLS:{kills:3d}  "
               f"{label}:[{info}]  "
               f"Ctrl+C=quit")
        return hud[:width]
