"""
tests.py — unit tests for doom-ascii (no ViZDoom install required).

Run with:  python tests.py
"""
import sys
import types
import unittest
import numpy as np

# ---------------------------------------------------------------------------
# Stub out vizdoom so agent.py / game.py can be imported without it installed
# ---------------------------------------------------------------------------

_vzd = types.ModuleType("vizdoom")

class _Button:
    MOVE_LEFT = MOVE_RIGHT = MOVE_FORWARD = MOVE_BACKWARD = None
    TURN_LEFT = TURN_RIGHT = ATTACK = USE = None

class _GameVariable:
    HEALTH = AMMO2 = KILLCOUNT = POSITION_X = POSITION_Y = ANGLE = None

class _ScreenResolution:
    RES_320X240 = None

class _ScreenFormat:
    RGB24 = GRAY8 = None

class _Mode:
    PLAYER = None

class _DoomGame:
    def set_doom_scenario_path(self, *a): pass
    def set_doom_game_path(self, *a): pass
    def set_doom_map(self, *a): pass
    def set_screen_resolution(self, *a): pass
    def set_screen_format(self, *a): pass
    def set_render_hud(self, *a): pass
    def set_render_crosshair(self, *a): pass
    def set_render_weapon(self, *a): pass
    def set_render_decals(self, *a): pass
    def set_render_particles(self, *a): pass
    def set_depth_buffer_enabled(self, *a): pass
    def set_labels_buffer_enabled(self, *a): pass
    def set_window_visible(self, *a): pass
    def set_mode(self, *a): pass
    def set_episode_timeout(self, *a): pass
    def set_episode_start_time(self, *a): pass
    def add_available_button(self, *a): pass
    def add_available_game_variable(self, *a): pass
    def set_living_reward(self, *a): pass
    def init(self): pass

_vzd.Button          = _Button
_vzd.GameVariable    = _GameVariable
_vzd.ScreenResolution= _ScreenResolution
_vzd.ScreenFormat    = _ScreenFormat
_vzd.Mode            = _Mode
_vzd.DoomGame        = _DoomGame
_vzd.scenarios_path  = "."
sys.modules["vizdoom"] = _vzd

# Now safe to import project modules
from game import read_stats, make_no_op, N_BUTTONS          # noqa: E402
from ascii_renderer import _compute_dims, AsciiRenderer      # noqa: E402
import agent as _agent                                        # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for ViZDoom state / label objects
# ---------------------------------------------------------------------------

class FakeLabel:
    def __init__(self, name, x=0, y=0, width=10, height=10, object_id=1):
        self.object_name = name
        self.x           = x
        self.y           = y
        self.width       = width
        self.height      = height
        self.object_id   = object_id


class FakeState:
    def __init__(self, labels=None, depth=None, screen=None, game_vars=None):
        self.labels        = labels or []
        self.depth_buffer  = depth
        self.screen_buffer = screen
        self.game_variables = game_vars or [100, 50, 0, 0.0, 0.0, 0.0]


# ===========================================================================
# game.py
# ===========================================================================

class TestReadStats(unittest.TestCase):

    def test_none_state(self):
        self.assertEqual(read_stats(None), (0, 0, 0))

    def test_normal_state(self):
        state = FakeState(game_vars=[75, 30, 5, 0, 0, 0])
        self.assertEqual(read_stats(state), (75, 30, 5))

    def test_short_game_vars(self):
        state = FakeState(game_vars=[50])
        h, a, k = read_stats(state)
        self.assertEqual(h, 50)
        self.assertEqual(a, 0)
        self.assertEqual(k, 0)


class TestMakeNoOp(unittest.TestCase):

    def test_length(self):
        action = make_no_op()
        self.assertEqual(len(action), N_BUTTONS)

    def test_all_false(self):
        self.assertTrue(all(v is False for v in make_no_op()))


# ===========================================================================
# ascii_renderer.py
# ===========================================================================

class TestComputeDims(unittest.TestCase):

    def test_wide_terminal(self):
        cols, rows = _compute_dims(200, 50)
        self.assertGreaterEqual(cols, 20)
        self.assertGreaterEqual(rows, 5)

    def test_narrow_terminal(self):
        cols, rows = _compute_dims(30, 10)
        self.assertGreaterEqual(cols, 20)
        self.assertGreaterEqual(rows, 5)

    def test_rows_match_cols(self):
        cols, rows = _compute_dims(160, 40)
        # rows ≈ cols * 3/8
        self.assertAlmostEqual(rows / cols, 3 / 8, delta=0.15)

    def test_minimum_enforced(self):
        cols, rows = _compute_dims(1, 1)
        self.assertGreaterEqual(cols, 20)
        self.assertGreaterEqual(rows, 5)


class TestBuildHud(unittest.TestCase):

    def _hud(self, **kw):
        defaults = dict(health=100, ammo=50, kills=0,
                        episode=1, strategy="idle", width=200, player_mode=False)
        defaults.update(kw)
        return AsciiRenderer._build_hud(**defaults)

    def test_contains_health(self):
        self.assertIn("HP:", self._hud(health=75))

    def test_contains_ammo(self):
        self.assertIn("AMMO:", self._hud(ammo=25))

    def test_contains_kills(self):
        self.assertIn("KILLS:", self._hud(kills=3))

    def test_truncated_to_width(self):
        hud = self._hud(width=20)
        self.assertLessEqual(len(hud), 20)

    def test_zero_health_bar(self):
        hud = self._hud(health=0)
        self.assertIn("HP:", hud)

    def test_full_health_bar(self):
        hud = self._hud(health=100)
        self.assertIn("HP:", hud)

    def test_player_mode_label(self):
        self.assertIn("YOU", self._hud(player_mode=True))

    def test_ai_mode_label(self):
        self.assertIn("AI", self._hud(player_mode=False))

    def test_strategy_truncated(self):
        long_strategy = "x" * 100
        hud = self._hud(strategy=long_strategy, width=200)
        # strategy is capped at 18 chars inside the hud
        self.assertLessEqual(hud.index("]") - hud.index("["), 22)


class TestFrameToAscii(unittest.TestCase):

    def _renderer(self, cols=40, rows=10):
        r = AsciiRenderer(width=cols * 2, height=rows + 4)
        r._cols, r._rows = cols, rows
        return r

    def test_output_line_count(self):
        buf = np.zeros((240, 320), dtype=np.uint8)
        r   = self._renderer(cols=40, rows=10)
        out = r._frame_to_ascii(buf)
        self.assertEqual(out.count("\n"), 9)   # 10 rows → 9 newlines

    def test_output_col_count(self):
        buf = np.zeros((240, 320), dtype=np.uint8)
        r   = self._renderer(cols=40, rows=10)
        lines = r._frame_to_ascii(buf).split("\n")
        for line in lines:
            self.assertEqual(len(line), 40)

    def test_black_frame_uses_dark_chars(self):
        buf = np.zeros((240, 320), dtype=np.uint8)
        r   = self._renderer(cols=20, rows=5)
        out = r._frame_to_ascii(buf)
        # Black (0) maps to index 0 → first character in ramp (' ')
        self.assertTrue(all(c == " " for c in out if c != "\n"))

    def test_white_frame_uses_bright_chars(self):
        buf = np.full((240, 320), 255, dtype=np.uint8)
        r   = self._renderer(cols=20, rows=5)
        out = r._frame_to_ascii(buf)
        bright = out.replace("\n", "")
        # All chars should be the last ramp character
        self.assertTrue(len(set(bright)) == 1)


# ===========================================================================
# agent.py — helper functions
# ===========================================================================

class TestScreenWidth(unittest.TestCase):

    def test_none_state(self):
        self.assertEqual(_agent._screen_width(None), 320)

    def test_gray_buffer(self):
        state = FakeState(screen=np.zeros((240, 320), dtype=np.uint8))
        self.assertEqual(_agent._screen_width(state), 320)

    def test_rgb_channels_last(self):
        state = FakeState(screen=np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(_agent._screen_width(state), 320)

    def test_rgb_channels_first(self):
        state = FakeState(screen=np.zeros((3, 240, 320), dtype=np.uint8))
        self.assertEqual(_agent._screen_width(state), 320)

    def test_none_buffer(self):
        state = FakeState(screen=None)
        self.assertEqual(_agent._screen_width(state), 320)


class TestAimAction(unittest.TestCase):

    def _action(self):
        return make_no_op()

    def test_attack_always_set(self):
        lbl    = FakeLabel("DoomImp", x=150, width=20)  # centre ≈ 160
        action = self._action()
        _agent._aim_action(action, lbl, screen_width=320)
        self.assertTrue(action[_agent.BTN_ATTACK])

    def test_turn_left_when_enemy_right(self):
        # Enemy centre at 280 → offset = +120 → turn right
        lbl    = FakeLabel("DoomImp", x=270, width=20)
        action = self._action()
        _agent._aim_action(action, lbl, screen_width=320)
        self.assertTrue(action[_agent.BTN_TURN_RIGHT])
        self.assertFalse(action[_agent.BTN_TURN_LEFT])

    def test_turn_right_when_enemy_left(self):
        # Enemy centre at 30 → offset = -130 → turn left
        lbl    = FakeLabel("DoomImp", x=20, width=20)
        action = self._action()
        _agent._aim_action(action, lbl, screen_width=320)
        self.assertTrue(action[_agent.BTN_TURN_LEFT])
        self.assertFalse(action[_agent.BTN_TURN_RIGHT])

    def test_aligned_no_turn(self):
        # Enemy centre exactly at screen centre (160) → no turn
        lbl    = FakeLabel("DoomImp", x=150, width=20)
        action = self._action()
        aligned = _agent._aim_action(action, lbl, screen_width=320,
                                     turn_threshold_px=20)
        self.assertTrue(aligned)
        self.assertFalse(action[_agent.BTN_TURN_LEFT])
        self.assertFalse(action[_agent.BTN_TURN_RIGHT])


class TestVisibleEnemies(unittest.TestCase):

    def test_none_state(self):
        self.assertEqual(_agent._visible_enemies(None), [])

    def test_no_labels(self):
        self.assertEqual(_agent._visible_enemies(FakeState(labels=[])), [])

    def test_filters_non_enemies(self):
        labels = [FakeLabel("Medikit"), FakeLabel("Clip")]
        self.assertEqual(_agent._visible_enemies(FakeState(labels=labels)), [])

    def test_returns_enemy(self):
        labels = [FakeLabel("DoomImp")]
        result = _agent._visible_enemies(FakeState(labels=labels))
        self.assertEqual(len(result), 1)

    def test_filters_dead_enemies(self):
        labels = [FakeLabel("DeadDoomImp"), FakeLabel("DoomImp")]
        result = _agent._visible_enemies(FakeState(labels=labels))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].object_name, "DoomImp")

    def test_sorted_nearest_first_by_depth(self):
        # Two enemies: far one has larger bounding box (would sort first by area)
        # but smaller depth value on the closer one should win.
        depth = np.full((240, 320), 500.0, dtype=np.float32)
        # Enemy A: small bbox but close (depth=50 at its centre)
        depth[120, 80] = 50.0
        # Enemy B: large bbox but far (depth=500 at its centre)
        lbl_near = FakeLabel("ZombieMan",  x=75,  y=115, width=10, height=10, object_id=1)
        lbl_far  = FakeLabel("ShotgunGuy", x=120, y=100, width=80, height=40, object_id=2)
        state  = FakeState(labels=[lbl_far, lbl_near], depth=depth)
        result = _agent._visible_enemies(state)
        self.assertEqual(result[0].object_id, 1)   # near enemy first

    def test_sorted_by_area_when_no_depth(self):
        lbl_small = FakeLabel("ZombieMan",  x=0, width=5,  height=5,  object_id=1)
        lbl_large = FakeLabel("ShotgunGuy", x=0, width=50, height=50, object_id=2)
        state  = FakeState(labels=[lbl_small, lbl_large], depth=None)
        result = _agent._visible_enemies(state)
        self.assertEqual(result[0].object_id, 2)   # largest area first


class TestWeaponSeekerDecide(unittest.TestCase):

    def _label(self, name):
        return FakeLabel(name)

    def test_ammo_guaranteed_when_low(self):
        lbl = self._label("Clip")
        for _ in range(20):
            self.assertTrue(_agent._WeaponSeeker._decide(ammo=5, label=lbl))

    def test_ammo_sometimes_skipped_when_medium(self):
        lbl     = self._label("Clip")
        results = [_agent._WeaponSeeker._decide(ammo=50, label=lbl)
                   for _ in range(200)]
        # Should not be all True or all False with p=0.65
        self.assertTrue(any(results))
        self.assertTrue(not all(results))

    def test_ammo_rarely_taken_when_full(self):
        lbl     = self._label("Clip")
        results = [_agent._WeaponSeeker._decide(ammo=100, label=lbl)
                   for _ in range(200)]
        # p=0.25 — most should be False
        self.assertLess(sum(results), 100)

    def test_weapon_almost_always_taken(self):
        lbl     = self._label("Shotgun")
        results = [_agent._WeaponSeeker._decide(ammo=50, label=lbl)
                   for _ in range(200)]
        # p=0.85 — most should be True
        self.assertGreater(sum(results), 100)


# ===========================================================================
# agent.py — _GridAgent internal helpers (tested via HPAStarAgent instance)
# ===========================================================================

class TestGridAgentHelpers(unittest.TestCase):

    def setUp(self):
        self.ag = _agent.HPAStarAgent()

    # _world_to_grid / _grid_to_world
    def test_origin_maps_to_grid_centre(self):
        gx, gy = self.ag._world_to_grid(0, 0)
        self.assertEqual(gx, self.ag._ORIGIN)
        self.assertEqual(gy, self.ag._ORIGIN)

    def test_world_to_grid_clamped(self):
        gx, gy = self.ag._world_to_grid(1e9, 1e9)
        self.assertEqual(gx, self.ag.GRID_SIZE - 1)
        self.assertEqual(gy, self.ag.GRID_SIZE - 1)

    def test_world_to_grid_negative_clamped(self):
        gx, gy = self.ag._world_to_grid(-1e9, -1e9)
        self.assertEqual(gx, 0)
        self.assertEqual(gy, 0)

    def test_grid_to_world_roundtrip(self):
        for wx, wy in [(0, 0), (128, 64), (-64, -128)]:
            gx, gy = self.ag._world_to_grid(wx, wy)
            cx, cy = self.ag._grid_to_world(gx, gy)
            # round-trip is approximate (cell-level)
            self.assertAlmostEqual(cx / self.ag.CELL_SIZE,
                                   wx / self.ag.CELL_SIZE, delta=1.5)

    # _nearest_unknown_bfs
    def test_bfs_finds_unknown(self):
        # Fresh grid — every cell is _UNKNOWN
        result = self.ag._nearest_unknown_bfs(self.ag._ORIGIN, self.ag._ORIGIN)
        self.assertIsNotNone(result)

    def test_bfs_returns_none_when_fully_explored(self):
        # Mark entire grid as FREE (no unknowns)
        self.ag.grid[:] = _agent._FREE
        result = self.ag._nearest_unknown_bfs(self.ag._ORIGIN, self.ag._ORIGIN)
        self.assertIsNone(result)

    def test_bfs_blocked_by_walls(self):
        # Surround origin with walls
        ox, oy = self.ag._ORIGIN, self.ag._ORIGIN
        self.ag.grid[oy-1:oy+2, ox-1:ox+2] = _agent._WALL
        # BFS from origin cannot leave the walled region
        result = self.ag._nearest_unknown_bfs(ox, oy)
        self.assertIsNone(result)

    # _stamp_visited
    def test_stamp_visited_marks_cells(self):
        ox, oy = self.ag._ORIGIN, self.ag._ORIGIN
        self.ag._stamp_visited(ox, oy)
        self.assertEqual(self.ag.grid[oy, ox], _agent._VISITED)

    def test_stamp_visited_does_not_overwrite_walls(self):
        ox, oy = self.ag._ORIGIN, self.ag._ORIGIN
        self.ag.grid[oy, ox] = _agent._WALL
        self.ag._stamp_visited(ox, oy)
        self.assertEqual(self.ag.grid[oy, ox], _agent._WALL)


# ===========================================================================
# agent.py — _RoomClearer
# ===========================================================================

class TestRoomClearer(unittest.TestCase):

    def setUp(self):
        self.rc = _agent._RoomClearer()

    def test_inactive_by_default(self):
        self.assertFalse(self.rc.active)

    def test_returns_none_when_inactive(self):
        state = FakeState()
        self.assertIsNone(self.rc.update(state, kills=0, angle_deg=0.0))

    def test_activates_on_kill(self):
        state = FakeState()
        # First call establishes baseline (0 kills)
        self.rc.update(state, kills=0, angle_deg=0.0)
        # Kill happens
        result = self.rc.update(state, kills=1, angle_deg=0.0)
        self.assertTrue(self.rc.active)
        self.assertIsNotNone(result)

    def test_sweep_completes_after_360(self):
        state = FakeState()
        self.rc.update(state, kills=0, angle_deg=0.0)
        self.rc.update(state, kills=1, angle_deg=0.0)  # activate

        # Simulate turning 360° in small increments
        angle = 0.0
        for _ in range(400):
            angle = (angle + 1.0) % 360
            result = self.rc.update(state, kills=1, angle_deg=angle)
            if result is None:
                break  # sweep done

        self.assertFalse(self.rc.active)

    def test_reset_clears_state(self):
        state = FakeState()
        self.rc.update(state, kills=0, angle_deg=0.0)
        self.rc.update(state, kills=1, angle_deg=0.0)
        self.rc.reset()
        self.assertFalse(self.rc.active)
        self.assertEqual(self.rc._prev_kills, 0)


# ===========================================================================
# agent.py — _line_of_sight
# ===========================================================================

class TestLineOfSight(unittest.TestCase):

    def _grid(self, size=20):
        return np.zeros((size, size), dtype=np.uint8)

    def test_clear_path(self):
        grid = self._grid()
        self.assertTrue(_agent._line_of_sight(grid, (0, 0), (5, 5)))

    def test_blocked_by_wall(self):
        grid = self._grid()
        grid[3, 3] = _agent._WALL
        # Wall sits on the direct path from (0,0) to (6,6)
        self.assertFalse(_agent._line_of_sight(grid, (0, 0), (6, 6)))

    def test_same_point(self):
        grid = self._grid()
        self.assertTrue(_agent._line_of_sight(grid, (5, 5), (5, 5)))

    def test_out_of_bounds_returns_false(self):
        grid = self._grid(10)
        self.assertFalse(_agent._line_of_sight(grid, (0, 0), (20, 20)))


# ===========================================================================
# agent.py — build_agent / AGENTS registry
# ===========================================================================

class TestBuildAgent(unittest.TestCase):

    def test_all_registered_agents_build(self):
        for name in _agent.AGENTS:
            ag = _agent.build_agent(name)
            self.assertIsInstance(ag, _agent.BaseAgent)

    def test_unknown_agent_raises(self):
        with self.assertRaises(ValueError):
            _agent.build_agent("nonexistent_agent")

    def test_agents_have_act_and_reset(self):
        for name, cls in _agent.AGENTS.items():
            ag = cls()
            self.assertTrue(callable(getattr(ag, "act",  None)), name)
            self.assertTrue(callable(getattr(ag, "reset", None)), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
