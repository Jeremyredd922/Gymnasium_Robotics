"""
graphics_renderer.py — Pygame-based renderer for doom-ascii.

Displays the ViZDoom RGB24 screen buffer in a scaled window with a
custom HUD bar showing health, ammo, kills, episode number, AI strategy,
and live FPS.

Interface matches AsciiRenderer so main.py can use either interchangeably.
"""
import sys
import os
import numpy as np

try:
    import pygame
except ImportError:
    print("pygame not found. Install it with:  pip install pygame")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BLACK     = (  0,   0,   0)
_DARK      = ( 18,  18,  18)
_DOOM_RED  = (170,   0,   0)
_GREEN     = (  0, 210,   0)
_YELLOW    = (255, 210,   0)
_RED       = (220,   0,   0)
_WHITE     = (255, 255, 255)
_DIM       = (110, 110, 110)

_HUD_H     = 52   # pixel height of HUD bar below game view
_GAME_W    = 320
_GAME_H    = 240


class GraphicsRenderer:
    """
    Pygame window renderer.

    Parameters
    ----------
    scale : int
        Integer scale applied to the 320×240 ViZDoom buffer.
        scale=3 → 960×720 game area.
    """

    def __init__(self, scale: int = 3):
        self._scale           = scale
        self._win_w           = _GAME_W * scale
        self._win_h           = _GAME_H * scale + _HUD_H
        self._active          = False
        self._clock           = None
        self._screen          = None
        self._font            = None
        self._font_b          = None
        self._fps             = 0.0
        self.restart_requested = False
        self._init_pygame()

    # ------------------------------------------------------------------
    # Shared renderer interface
    # ------------------------------------------------------------------

    def show_loading(self):
        self._screen.fill(_BLACK)
        self._blit_centre("Loading ViZDoom engine...", self._font_b, _GREEN)
        pygame.display.flip()
        pygame.event.pump()

    def show_banner(self, name: str, player_mode: bool = False):
        self._screen.fill(_BLACK)
        if player_mode:
            from agent import HumanAgent
            lines = [
                "PLAYER MODE",
                HumanAgent.CONTROLS,
                "ESC / close window to quit",
            ]
        else:
            lines = [
                f"AI Agent: {name.upper()}",
                "ESC / close window to quit",
            ]
        y = self._win_h // 2 - len(lines) * 22
        for line in lines:
            surf = self._font_b.render(line, True, _GREEN)
            self._screen.blit(surf, ((self._win_w - surf.get_width()) // 2, y))
            y += 44
        pygame.display.flip()
        pygame.event.pump()

    def hide_cursor(self):
        pygame.mouse.set_visible(False)

    def show_cursor(self):
        pygame.mouse.set_visible(True)

    def render(self, screen_buf, health=100, ammo=50, kills=0,
               episode=1, strategy="", player_mode=False) -> bool:
        """
        Draw one frame.  Returns False if the user closed the window / pressed ESC.

        screen_buf : numpy array (H, W, 3) RGB uint8
        """
        if not self._active:
            return False

        # Poll OS events — mandatory to keep the window alive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_r:
                    self.restart_requested = True

        # --- Game frame -------------------------------------------------------
        buf = self._normalise_buffer(screen_buf)
        H, W = buf.shape[:2]
        try:
            surf = pygame.image.frombuffer(
                np.ascontiguousarray(buf).tobytes(), (W, H), "RGB"
            )
        except Exception:
            # Fallback: blank frame on format mismatch.
            surf = pygame.Surface((W, H))

        scaled = pygame.transform.scale(surf, (self._win_w, _GAME_H * self._scale))
        self._screen.blit(scaled, (0, 0))

        # --- HUD bar ----------------------------------------------------------
        hud_y = _GAME_H * self._scale
        pygame.draw.rect(self._screen, _DARK, (0, hud_y, self._win_w, _HUD_H))
        pygame.draw.line(self._screen, _DOOM_RED, (0, hud_y), (self._win_w, hud_y), 2)
        self._draw_hud(hud_y, health, ammo, kills, episode, strategy, player_mode)

        pygame.display.flip()
        self._clock.tick(15)
        self._fps = self._clock.get_fps()
        return True

    def render_episode_end(self, episode, kills, total_kills, duration,
                           map_name=None, completed=False, next_map=None):
        if not self._active:
            return
        # Semi-transparent dark overlay.
        overlay = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self._screen.blit(overlay, (0, 0))

        mins, secs = divmod(int(duration), 60)
        if completed and next_map:
            status = f"{map_name} cleared!  Advancing to {next_map}..."
        elif completed:
            status = f"{map_name} cleared!"
        elif map_name:
            status = f"Died on {map_name}"
        else:
            status = f"Episode {episode} ended"
        lines = [
            status,
            f"Kills: {kills}   Total: {total_kills}",
            f"Time: {mins:02d}:{secs:02d}",
        ]
        y = self._win_h // 2 - len(lines) * 22
        for line in lines:
            surf = self._font_b.render(line, True, _WHITE)
            self._screen.blit(surf, ((self._win_w - surf.get_width()) // 2, y))
            y += 40

        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if self._active:
            self._active = False
            pygame.quit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_pygame(self):
        # Tell SDL to position the window at the centre of the primary display
        # before the window is created — avoids it appearing behind other windows.
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

        pygame.init()
        pygame.display.set_caption("doom-ascii  |  AI Spectator")
        self._screen  = pygame.display.set_mode(
            (self._win_w, self._win_h), pygame.SHOWN
        )
        self._clock   = pygame.time.Clock()
        fsize         = max(13, self._scale * 5)
        self._font    = pygame.font.SysFont("monospace", fsize)
        self._font_b  = pygame.font.SysFont("monospace", fsize + 4, bold=True)
        self._active  = True
        self._raise_window()

    def _raise_window(self):
        """Bring the pygame window to the foreground on all platforms."""
        pygame.event.pump()   # let SDL process pending events first

        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    user32 = ctypes.windll.user32
                    # Briefly set HWND_TOPMOST then restore to HWND_NOTOPMOST
                    # so the window pops to the front without staying on top.
                    SWP_NOMOVE  = 0x0002
                    SWP_NOSIZE  = 0x0001
                    HWND_TOPMOST    = -1
                    HWND_NOTOPMOST  = -2
                    flags = SWP_NOMOVE | SWP_NOSIZE
                    user32.SetWindowPos(hwnd, HWND_TOPMOST,   0, 0, 0, 0, flags)
                    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
                    user32.SetForegroundWindow(hwnd)
            except Exception:
                pass   # non-fatal — window still opens, just may not be on top

    def _draw_hud(self, hud_y, health, ammo, kills, episode, strategy,
                  player_mode=False):
        pad  = 10
        row1 = hud_y + 6
        row2 = hud_y + 6 + self._font.get_height() + 2

        hp_col = _GREEN if health > 60 else (_YELLOW if health > 30 else _RED)

        # Left — episode & health.
        self._screen.blit(
            self._font.render(f"EP:{episode:3d}", True, _DIM), (pad, row1))
        self._screen.blit(
            self._font_b.render(f"HP:{health:3d}%", True, hp_col), (pad, row2))

        # Centre — ammo & kills.
        cx = self._win_w // 2
        for text, col, row in (
            (f"AMMO:{ammo:3d}", _YELLOW, row1),
            (f"KILLS:{kills:3d}", _WHITE, row2),
        ):
            surf = self._font.render(text, True, col)
            self._screen.blit(surf, (cx - surf.get_width() // 2, row))

        # Right — player label or AI strategy, plus FPS.
        if player_mode:
            right_top = ("YOU", _GREEN)
            right_bot = ("W/S/A/D  SPC:fire", _DIM)
        else:
            right_top = (f"AI:{strategy[:20]}", _GREEN)
            right_bot = (f"FPS:{self._fps:4.1f}", _DIM)

        for (text, col), row in zip((right_top, right_bot), (row1, row2)):
            surf = self._font.render(text, True, col)
            self._screen.blit(surf, (self._win_w - surf.get_width() - pad, row))

    def _blit_centre(self, text, font, colour):
        surf = font.render(text, True, colour)
        self._screen.blit(
            surf,
            ((self._win_w - surf.get_width()) // 2,
             (self._win_h - surf.get_height()) // 2),
        )

    @staticmethod
    def _normalise_buffer(buf):
        """
        Ensure buf is (H, W, 3) uint8.
        ViZDoom RGB24 is usually already (H, W, 3) but guard against (3, H, W).
        """
        if buf.ndim == 3 and buf.shape[0] == 3 and buf.shape[2] != 3:
            # Channels-first → channels-last.
            buf = buf.transpose(1, 2, 0)
        return np.ascontiguousarray(buf, dtype=np.uint8)
