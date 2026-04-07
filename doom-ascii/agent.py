"""
agent.py — AI agents and human-player input for doom-ascii.

AI agents
---------
  FlowFieldAgent      RTS-style vector-field navigation (Supreme Commander, They Are Billions).
  WaypointGraphAgent  Sparse waypoint graph with line-of-sight connections (Quake, Half-Life).
  HPAStarAgent        Hierarchical Pathfinding A* — cluster-based two-level search
                      (Dragon Age: Origins, StarCraft series).

Human player
------------
  HumanAgent          Keyboard input.  Graphics: pygame.key.get_pressed().
                      ASCII: background thread (msvcrt on Windows, tty on Unix).
"""
import heapq
import math
import random
import sys
import threading
import time
from collections import deque

import numpy as np

from game import (
    make_no_op,
    BTN_MOVE_LEFT, BTN_MOVE_RIGHT, BTN_MOVE_FORWARD, BTN_MOVE_BACKWARD,
    BTN_TURN_LEFT, BTN_TURN_RIGHT, BTN_ATTACK, BTN_USE,
)


# ---------------------------------------------------------------------------
# Enemy label detection
# ---------------------------------------------------------------------------

# ViZDoom object names that identify hostile actors.
_ENEMY_KEYWORDS = (
    "ZombieMan", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS",
    "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "Archvile",
    "Revenant", "Fatso", "PainElemental", "SpiderMastermind",
    "Cyberdemon", "Monster",  # catch-all for modded enemies
)


def _visible_enemies(state, max_dist=float("inf")):
    """
    Return a list of ViZDoom Label objects that belong to living enemies
    within *max_dist* depth-buffer units, sorted most-dangerous first then
    nearest first.

    Requires labels_buffer to be enabled in game.py.
    Returns [] when no enemies are visible or state is None.
    """
    if state is None or not hasattr(state, "labels") or state.labels is None:
        return []
    enemies = [
        lbl for lbl in state.labels
        if any(kw in lbl.object_name for kw in _ENEMY_KEYWORDS)
        and not lbl.object_name.startswith("Dead")
    ]
    if not enemies:
        return []

    depth = getattr(state, "depth_buffer", None)
    if depth is not None and depth.ndim != 2:
        depth = depth[:, :, 0] if depth.ndim == 3 else None

    def _danger(lbl):
        for kw, lvl in _ENEMY_DANGER.items():
            if kw in lbl.object_name:
                return lvl
        return 0

    if depth is not None:
        H, W = depth.shape
        def _dist(lbl):
            cx = max(0, min(W - 1, int(lbl.x + lbl.width  / 2)))
            cy = max(0, min(H - 1, int(lbl.y + lbl.height / 2)))
            d  = float(depth[cy, cx])
            return d if d > 0 else float("inf")
        enemies = [e for e in enemies if _dist(e) <= max_dist]
        enemies.sort(key=lambda l: (-_danger(l), _dist(l)))
    else:
        enemies.sort(key=lambda l: (-_danger(l), -(l.width * l.height)))
    return enemies


# ---------------------------------------------------------------------------
# Weapon / ammo pickup
# ---------------------------------------------------------------------------

_WEAPON_ITEM_KEYWORDS = (
    "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher",
    "PlasmaRifle", "BFG9000", "Chainsaw",
)
_AMMO_KEYWORDS = (
    "Clip", "ClipBox", "Shell", "ShellBox",
    "RocketAmmo", "RocketBox", "Cell", "CellPack",
)
_HEALTH_KEYWORDS = (
    "HealthBonus", "Stimpack", "Medikit", "MegaSphere",
    "ArmorBonus", "GreenArmor", "BlueArmor",
)
_PICKUP_KEYWORDS = _WEAPON_ITEM_KEYWORDS + _AMMO_KEYWORDS + _HEALTH_KEYWORDS

# Danger level per enemy type — higher = more threatening.
# Used to prioritise targets when multiple enemies are visible.
_ENEMY_DANGER = {
    "Archvile": 10, "Cyberdemon": 9, "SpiderMastermind": 9,
    "Revenant": 8, "Fatso": 8, "PainElemental": 7,
    "HellKnight": 6, "BaronOfHell": 6, "Cacodemon": 5,
    "Arachnotron": 5, "Demon": 4, "Spectre": 4,
    "DoomImp": 3, "ChaingunGuy": 3, "ShotgunGuy": 2, "ZombieMan": 1,
}


def _visible_pickups(state):
    """
    Return ViZDoom Label objects for weapons and ammo visible on screen,
    sorted by descending bbox area (largest / closest first).
    """
    if state is None or not hasattr(state, "labels") or state.labels is None:
        return []
    picks = [
        lbl for lbl in state.labels
        if any(kw in lbl.object_name for kw in _PICKUP_KEYWORDS)
    ]
    picks.sort(key=lambda l: l.width * l.height, reverse=True)
    return picks


class _WeaponSeeker:
    """
    Decides once per object whether the agent should walk toward a weapon,
    ammo, or health/armor pickup, then steers the agent toward it.

    Decision factors
    ----------------
    - Weapons  : 85 % chance regardless of ammo (upgrades are almost always worth it)
    - Ammo     : guaranteed if ammo < 25; 65 % if 25–75; 25 % if > 75
    - Health   : guaranteed if health < 50; 70 % if 50–75; skip if > 75
    - Armor    : always take (free mitigation)

    Once the agent declines an item its object_id is remembered for the
    episode so the decision is not re-rolled every frame.
    """

    _ALIGN_THRESH_PX = 15   # pixels from screen centre treated as "facing"

    def __init__(self):
        self._declined = set()   # object_ids the agent chose to skip

    def reset(self):
        self._declined = set()

    def update(self, state, ammo, health=100):
        """
        Returns an action list steering toward the chosen pickup, or None
        when no pickup is being pursued.
        """
        pickups = _visible_pickups(state)
        if not pickups:
            return None

        sw = _screen_width(state)
        for item in pickups:
            if item.object_id in self._declined:
                continue
            if not self._decide(ammo, health, item):
                self._declined.add(item.object_id)
                continue
            # Steer toward this item.
            action = make_no_op()
            cx     = item.x + item.width / 2
            offset = cx - sw / 2
            if offset < -self._ALIGN_THRESH_PX:
                action[BTN_TURN_LEFT]  = True
            elif offset > self._ALIGN_THRESH_PX:
                action[BTN_TURN_RIGHT] = True
            action[BTN_MOVE_FORWARD] = True
            return action

        return None   # all visible pickups declined

    @staticmethod
    def _decide(ammo, health, label):
        name = label.object_name
        if any(kw in name for kw in _WEAPON_ITEM_KEYWORDS):
            return random.random() < 0.85
        if any(kw in name for kw in ("BlueArmor", "GreenArmor", "ArmorBonus", "MegaSphere")):
            return True   # armor and megasphere — always worth it
        if any(kw in name for kw in ("HealthBonus", "Stimpack", "Medikit")):
            return True   # always grab health — often an enemy drop
        # Ammo pickup — scale probability with need.
        if ammo < 25:
            return True
        if ammo > 75:
            return random.random() < 0.25
        return random.random() < 0.65



def _screen_width(state):
    """Return the screen width in pixels from a game state, defaulting to 320."""
    buf = state.screen_buffer if state is not None else None
    if buf is None:
        return 320
    # GRAY8: (H, W)   RGB24 channels-first: (3, H, W)   channels-last: (H, W, 3)
    if buf.ndim == 2:
        return buf.shape[1]
    if buf.ndim == 3:
        return buf.shape[2] if buf.shape[0] == 3 else buf.shape[1]
    return 320


def _aim_action(action, label, screen_width, turn_threshold_px=20):
    """
    Modify *action* in-place to turn toward the horizontal centre of *label*
    and fire.  Returns True if the crosshair is already aligned (fire only).

    Parameters
    ----------
    action           : mutable action list
    label            : ViZDoom Label with .x and .width attributes
    screen_width     : total screen width in pixels
    turn_threshold_px: pixel band around centre treated as "aligned"
    """
    enemy_cx  = label.x + label.width / 2
    screen_cx = screen_width / 2
    offset    = enemy_cx - screen_cx

    action[BTN_ATTACK] = True
    if offset < -turn_threshold_px:
        action[BTN_TURN_LEFT]  = True
        return False
    if offset > turn_threshold_px:
        action[BTN_TURN_RIGHT] = True
        return False
    return True   # aligned — caller may choose to advance


# ---------------------------------------------------------------------------
# Room-clearing state machine
# ---------------------------------------------------------------------------

class _RoomClearer:
    """
    After each kill the agent stops, rotates a full 360°, and engages any
    enemy that comes into view before resuming exploration.

    Distance gating — only enemies within ~5 metres (~160 Doom map units)
    are considered.  The depth buffer is sampled at each label's screen
    centre; if no depth buffer is available every visible enemy qualifies.

    Usage
    -----
    Instantiate once in __init__, call .update() at the top of act(), and
    return its result immediately when it is not None.
    """

    _CLEAR_RADIUS = 4.0     # Doom units — stop and shoot within this range

    def __init__(self):
        self.active      = False
        self._prev_kills = 0
        self._accum      = 0.0   # degrees turned so far this sweep
        self._prev_ang   = None  # angle (deg) at the previous frame
        self._engaging   = False # True while locked onto a visible enemy

    def reset(self):
        self.__init__()

    def update(self, state, kills, angle_deg):
        """
        Call once per frame.

        Returns
        -------
        action : list or None
            If not None the caller must return this action immediately.
            None means the sweep is inactive; caller proceeds normally.
        """
        if kills > self._prev_kills:
            self.active    = True
            self._accum    = 0.0
            self._prev_ang = angle_deg
            self._engaging = False
        self._prev_kills = kills

        if not self.active:
            return None

        action  = make_no_op()
        enemies = _visible_enemies(state)

        if enemies:
            # Lock onto the closest enemy; pause rotation accumulation.
            self._engaging = True
            self._prev_ang = angle_deg
            _aim_action(action, enemies[0], _screen_width(state))
            # Stand still while shooting — do NOT advance into the enemy.
            return action

        # No enemy in view — spin right and accumulate rotation.
        self._engaging = False
        if self._prev_ang is not None:
            delta        = (angle_deg - self._prev_ang + 180) % 360 - 180
            self._accum += abs(delta)
        self._prev_ang = angle_deg

        if self._accum >= 360.0:
            self.active = False
            return None             # full sweep done, area clear

        action[BTN_TURN_RIGHT] = True
        return action

    def _nearby_enemies(self, state):
        """Visible enemies within _CLEAR_RADIUS Doom units."""
        enemies = _visible_enemies(state)
        if not enemies:
            return []
        depth = (state.depth_buffer
                 if state is not None and state.depth_buffer is not None
                 else None)
        if depth is None:
            return enemies          # no depth info — accept all visible

        if depth.ndim != 2:
            depth = depth[:, :, 0] if depth.ndim == 3 else None
        if depth is None:
            return enemies

        H, W   = depth.shape
        result = []
        for lbl in enemies:
            px = max(0, min(W - 1, int(lbl.x + lbl.width  / 2)))
            py = max(0, min(H - 1, int(lbl.y + lbl.height / 2)))
            dist = float(depth[py, px])
            if dist > 0 and dist <= self._CLEAR_RADIUS:
                result.append(lbl)
        return result


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseAgent:
    """All agents expose act() and reset()."""

    strategy: str = "idle"

    def act(self, state, health: int, ammo: int, kills: int) -> list:
        """Return a ViZDoom boolean action list for the current game state."""
        raise NotImplementedError

    def reset(self):
        """Called at the start of every episode."""


# ---------------------------------------------------------------------------
# FlowFieldAgent — RTS-style vector-field navigation
# ---------------------------------------------------------------------------

def astar(grid, start, goal, max_nodes=4000):
    """
    A* on a 2D numpy occupancy grid.

    Cell costs:
      UNKNOWN  → passable, cost ×2.0  (explore cautiously)
      FREE     → passable, cost ×1.0
      VISITED  → passable, cost ×1.5  (prefer unvisited routes)
      WALL     → impassable

    Parameters
    ----------
    grid      : 2D numpy uint8 array  (rows=Y, cols=X)
    start     : (gx, gy) tuple
    goal      : (gx, gy) tuple
    max_nodes : hard limit on nodes expanded (keeps it real-time safe)

    Returns
    -------
    List of (gx, gy) from start to goal inclusive, or None if unreachable.
    """
    H, W = grid.shape

    def heuristic(a, b):
        # Octile distance — admissible for 8-directional movement.
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def move_cost(cell_val, diagonal):
        if cell_val == _WALL:
            return float("inf")
        base = math.sqrt(2) if diagonal else 1.0
        if cell_val == _UNKNOWN:
            return base * 2.0
        if cell_val == _VISITED:
            return base * 1.5
        return base

    open_heap = [(0.0, start)]
    came_from = {start: None}
    g_score   = {start: 0.0}
    expanded  = 0

    while open_heap and expanded < max_nodes:
        _, current = heapq.heappop(open_heap)
        expanded += 1

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        cx, cy = current
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            cell_val = int(grid[ny, nx])
            cost = move_cost(cell_val, dx != 0 and dy != 0)
            if cost == float("inf"):
                continue
            tg = g_score[current] + cost
            neighbor = (nx, ny)
            if neighbor not in g_score or tg < g_score[neighbor]:
                g_score[neighbor] = tg
                heapq.heappush(open_heap, (tg + heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    return None   # unreachable or budget exhausted
# ---------------------------------------------------------------------------
# Shared base for grid-building agents
# ---------------------------------------------------------------------------



def _line_of_sight(grid, a, b):
    """
    Bresenham line check on the occupancy grid.
    Returns True if no WALL cell lies strictly between a and b.
    Used by WaypointGraphAgent to prune edges without line-of-sight.
    """
    H, W = grid.shape
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while (x0, y0) != (x1, y1):
        if not (0 <= x0 < W and 0 <= y0 < H):
            return False
        if grid[y0, x0] == _WALL:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy
    return True

class _GridAgent(BaseAgent):
    """
    Mixin that provides the occupancy-grid infrastructure shared by
    FlowFieldAgent, WaypointGraphAgent, and HPAStarAgent.

    Sub-classes must implement _plan(gx, gy) → list[(gx,gy)] and set
    self.strategy themselves.
    """

    CELL_SIZE           = 64
    GRID_SIZE           = 256
    _ORIGIN             = GRID_SIZE // 2
    _REPLAN_EVERY       = 30
    _FOV                = 90.0
    _MAX_DEPTH          = 768
    _TURN_THRESH        = 12.0
    _COMBAT_FRAMES      = 20
    _ENEMY_DETECT_RANGE = 650   # depth-buffer units; limits proactive targeting to nearby enemies

    def __init__(self):
        self.grid         = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self._path        = []
        self._replan_t    = 0
        self._combat_t    = 0
        self._prev_health = 100
        self._clearer     = _RoomClearer()
        self._seeker      = _WeaponSeeker()
        self.strategy     = "initialising"
        self._scan_phase  = True   # True = spinning to scan, False = moving forward
        self._scan_t      = 70     # frames remaining in current scan/move phase
        self._target_id   = None   # object_id of the currently locked enemy
        self._strafe_dir  = 1      # 1 = strafe right, -1 = strafe left (flips on damage)
        self._stuck_t     = 0      # frames without meaningful movement
        self._prev_px     = None   # world position last frame (for stuck detection)
        self._prev_py     = None

    def reset(self):
        self._clearer.reset()
        self._seeker.reset()
        self.__init__()

    def _pick_target(self, enemies):
        """
        Return the locked target enemy if it is still visible, otherwise lock
        onto the nearest new enemy.  Returns None when no enemies are visible.

        Locking by object_id prevents the agent from thrashing between two
        nearly-equidistant enemies and ensures it finishes one off before
        moving on to the next.
        """
        if not enemies:
            self._target_id = None
            return None
        if self._target_id is not None:
            for e in enemies:
                if e.object_id == self._target_id:
                    return e
        # Lock on to the nearest visible enemy (enemies is already distance-sorted).
        self._target_id = enemies[0].object_id
        return enemies[0]

    def act(self, state, health, ammo, kills):
        action = make_no_op()
        if state is None:
            return action

        gv = state.game_variables
        if len(gv) < 6:
            action[BTN_MOVE_FORWARD] = True
            self.strategy = "no position data"
            return action

        px, py    = float(gv[3]), float(gv[4])
        angle_deg = float(gv[5])
        gx, gy    = self._world_to_grid(px, py)

        self._stamp_visited(gx, gy)
        if state.depth_buffer is not None:
            self._update_grid(state.depth_buffer, px, py, angle_deg)

        # Stuck detection — press USE after ~30 frames without movement (opens doors).
        if self._prev_px is not None:
            if abs(px - self._prev_px) + abs(py - self._prev_py) < 5:
                self._stuck_t += 1
            else:
                self._stuck_t = 0
        self._prev_px, self._prev_py = px, py

        if health < self._prev_health:
            self._combat_t = self._COMBAT_FRAMES
            self._strafe_dir = random.choice([-1, 1])  # randomise dodge direction
        self._prev_health = health

        if self._combat_t > 0:
            self._combat_t -= 1
            self.strategy = "combat"
            enemies = _visible_enemies(state)
            target  = self._pick_target(enemies)
            if target is not None:
                aligned = _aim_action(action, target, _screen_width(state))
                if not aligned:
                    action[BTN_ATTACK] = False
            else:
                action[BTN_ATTACK] = True
            # Strafe to dodge instead of walking backward into walls.
            if self._strafe_dir > 0:
                action[BTN_MOVE_RIGHT] = True
            else:
                action[BTN_MOVE_LEFT] = True
            return action

        # Engage any nearby enemy (range-limited so far-off enemies are ignored).
        enemies = _visible_enemies(state, self._ENEMY_DETECT_RANGE)
        target  = self._pick_target(enemies)
        if target is not None:
            self.strategy = "targeting"
            aligned = _aim_action(action, target, _screen_width(state))
            if not aligned:
                action[BTN_ATTACK] = False   # hold fire until crosshair is on target
            else:
                action[BTN_MOVE_FORWARD] = True   # close the distance when aimed
            return action

        # No enemy in range — grab any visible items (including enemy drops).
        seek_act = self._seeker.update(state, ammo, health)
        if seek_act is not None:
            self.strategy = "picking up item"
            return seek_act

        # Room clearing — 360° sweep after each kill to find hidden enemies.
        clear_act = self._clearer.update(state, kills, angle_deg)
        if clear_act is not None:
            self.strategy = "clearing"
            return clear_act

        self._replan_t -= 1
        if self._replan_t <= 0 or not self._path:
            self._path     = self._plan(gx, gy)
            self._replan_t = self._REPLAN_EVERY

        while self._path:
            wgx, wgy = self._path[0]
            if abs(gx - wgx) <= 1 and abs(gy - wgy) <= 1:
                self._path.pop(0)
            else:
                break

        if self._path:
            wgx, wgy = self._path[0]
            twx, twy = self._grid_to_world(wgx, wgy)
            action   = self._steer(action, px, py, angle_deg, twx, twy)
        else:
            # No graph path — use A* toward the nearest unexplored cell so the
            # agent routes around walls instead of steering in a straight line.
            bfs_target = self._nearest_unknown_bfs(gx, gy)
            if bfs_target is not None:
                self.strategy = "wandering"
                path = astar(self.grid, (gx, gy), bfs_target, max_nodes=2000)
                if path and len(path) > 1:
                    twx, twy = self._grid_to_world(*path[1])
                else:
                    twx, twy = self._grid_to_world(*bfs_target)
                action = self._steer(action, px, py, angle_deg, twx, twy)
            else:
                # Map fully explored — oscillate to find any remaining enemies.
                self.strategy = "searching"
                self._scan_t -= 1
                if self._scan_t <= 0:
                    self._scan_phase = not self._scan_phase
                    self._scan_t = 70 if self._scan_phase else 30
                if self._scan_phase:
                    action[BTN_TURN_RIGHT] = True
                else:
                    action[BTN_MOVE_FORWARD] = True

        # Press USE when stuck (opens doors); also back up + turn to escape corners.
        if self._stuck_t >= 30:
            action[BTN_USE] = True
            self._replan_t = 0   # force replan to pick a fresh target
            action[BTN_MOVE_BACKWARD] = True
            action[BTN_TURN_LEFT if random.random() < 0.5 else BTN_TURN_RIGHT] = True
            self._stuck_t = 0

        return action

    # ---- shared helpers --------------------------------------------------

    def _plan(self, gx, gy):
        raise NotImplementedError

    def _steer(self, action, px, py, angle_deg, tx, ty):
        dx, dy       = tx - px, ty - py
        target_angle = math.degrees(math.atan2(dy, dx))
        diff         = (target_angle - angle_deg + 180) % 360 - 180
        if abs(diff) > self._TURN_THRESH:
            self.strategy = f"turning {'left' if diff > 0 else 'right'}"
            action[BTN_TURN_LEFT  if diff > 0 else BTN_TURN_RIGHT] = True
        else:
            self.strategy = f"pathing ({type(self).__name__})"
            action[BTN_MOVE_FORWARD] = True
        return action

    def _nearest_unknown_bfs(self, gx, gy):
        q, seen = deque([(gx, gy)]), {(gx, gy)}
        while q:
            cx, cy = q.popleft()
            if self.grid[cy, cx] == _UNKNOWN:
                return (cx, cy)
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                nx, ny = cx+dx, cy+dy
                if (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE
                        and (nx, ny) not in seen
                        and self.grid[ny, nx] != _WALL):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None

    def _update_grid(self, depth_buf, px, py, angle_deg):
        if depth_buf.ndim != 2:
            depth_buf = depth_buf[:, :, 0] if depth_buf.ndim == 3 else None
        if depth_buf is None:
            return
        H, W       = depth_buf.shape
        half_step  = self.CELL_SIZE / 2
        for col in range(0, W, 6):
            offset  = (col / (W - 1) - 0.5) * self._FOV
            ray_rad = math.radians(angle_deg + offset)
            cos_r, sin_r = math.cos(ray_rad), math.sin(ray_rad)
            raw  = float(depth_buf[H // 2, col])
            if raw <= 0:
                continue
            depth = min(raw, self._MAX_DEPTH)
            d = half_step
            while d < depth - half_step:
                cgx, cgy = self._world_to_grid(px + d*cos_r, py + d*sin_r)
                if 0 <= cgx < self.GRID_SIZE and 0 <= cgy < self.GRID_SIZE:
                    if self.grid[cgy, cgx] == _UNKNOWN:
                        self.grid[cgy, cgx] = _FREE
                d += half_step
            if raw < self._MAX_DEPTH:
                wgx, wgy = self._world_to_grid(px + depth*cos_r, py + depth*sin_r)
                if 0 <= wgx < self.GRID_SIZE and 0 <= wgy < self.GRID_SIZE:
                    self.grid[wgy, wgx] = _WALL

    def _stamp_visited(self, gx, gy):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = gx+dx, gy+dy
                if (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE
                        and self.grid[ny, nx] != _WALL):
                    self.grid[ny, nx] = _VISITED

    def _world_to_grid(self, x, y):
        gx = max(0, min(self.GRID_SIZE-1, int(x/self.CELL_SIZE) + self._ORIGIN))
        gy = max(0, min(self.GRID_SIZE-1, int(y/self.CELL_SIZE) + self._ORIGIN))
        return gx, gy

    def _grid_to_world(self, gx, gy):
        return ((gx-self._ORIGIN)*self.CELL_SIZE + self.CELL_SIZE/2,
                (gy-self._ORIGIN)*self.CELL_SIZE + self.CELL_SIZE/2)


# ---------------------------------------------------------------------------
# HPAStarAgent — Hierarchical Pathfinding A*
# ---------------------------------------------------------------------------

class FlowFieldAgent(_GridAgent):
    """
    Flow Field pathfinding — used in Supreme Commander, Planetary Annihilation,
    They Are Billions, and other RTS games.

    Instead of computing a per-agent path, a single cost-to-goal map is built
    by running multi-source Dijkstra *outward from the target cell*.  Every
    cell is then assigned a unit direction vector pointing toward its
    lowest-cost passable neighbour.  The agent simply reads the vector at its
    current grid cell and steers in that direction — O(1) per frame, no
    re-planning needed until the target changes.

    Advantage over A*: the field is computed once and shared; all agents in
    the same level could reuse it.  This is why it dominates in games with
    hundreds of simultaneous units.
    """

    _FIELD_TTL = 60    # recompute field after this many frames

    def __init__(self):
        super().__init__()
        self._field   = None   # (GRID_SIZE, GRID_SIZE, 2) float32
        self._field_t = 0

    def reset(self):
        super().reset()
        self._field   = None
        self._field_t = 0

    def _plan(self, gx, gy):
        return []   # navigation handled directly in act()

    def _build_field(self, goal):
        """
        Dijkstra from *goal* outward to fill a distance map, then derive a
        direction vector for every reachable cell using numpy shift operations.
        """
        GS   = self.GRID_SIZE
        dist = np.full((GS, GS), np.inf, dtype=np.float32)
        dist[goal[1], goal[0]] = 0.0
        heap = [(0.0, goal)]

        while heap:
            d, (cx, cy) = heapq.heappop(heap)
            if d > dist[cy, cx] + 1e-6:
                continue
            for dx, dy in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < GS and 0 <= ny < GS):
                    continue
                cell = int(self.grid[ny, nx])
                if cell == _WALL:
                    continue
                base = math.sqrt(2) if (dx and dy) else 1.0
                if cell == _UNKNOWN: base *= 2.0
                elif cell == _VISITED: base *= 1.5
                nd = d + base
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, (nx, ny)))

        # Build direction vectors: for each direction, shift the distance
        # map so that shifted[y, x] = dist of neighbour in that direction.
        # The direction with the lowest shifted value wins.
        field = np.zeros((GS, GS, 2), dtype=np.float32)
        best  = dist.copy()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            shifted = np.roll(np.roll(dist, -dy, axis=0), -dx, axis=1)
            # Zero out wrapped border values.
            if dx > 0:  shifted[:, -dx:] = np.inf
            elif dx < 0: shifted[:, :-dx] = np.inf
            if dy > 0:  shifted[-dy:, :] = np.inf
            elif dy < 0: shifted[:-dy, :] = np.inf
            mask = shifted < best
            mag  = math.hypot(dx, dy)
            field[:, :, 0] = np.where(mask, dx / mag, field[:, :, 0])
            field[:, :, 1] = np.where(mask, dy / mag, field[:, :, 1])
            best = np.where(mask, shifted, best)

        return field

    def act(self, state, health, ammo, kills):
        action = make_no_op()
        if state is None:
            return action

        gv = state.game_variables
        if len(gv) < 6:
            action[BTN_MOVE_FORWARD] = True
            self.strategy = "no position data"
            return action

        px, py    = float(gv[3]), float(gv[4])
        angle_deg = float(gv[5])
        gx, gy    = self._world_to_grid(px, py)

        self._stamp_visited(gx, gy)
        if state.depth_buffer is not None:
            self._update_grid(state.depth_buffer, px, py, angle_deg)

        # Stuck detection — press USE after ~30 frames without movement (opens doors).
        if self._prev_px is not None:
            if abs(px - self._prev_px) + abs(py - self._prev_py) < 5:
                self._stuck_t += 1
            else:
                self._stuck_t = 0
        self._prev_px, self._prev_py = px, py

        if health < self._prev_health:
            self._combat_t = self._COMBAT_FRAMES
            self._strafe_dir = random.choice([-1, 1])
        self._prev_health = health

        if self._combat_t > 0:
            self._combat_t -= 1
            self.strategy = "combat"
            enemies = _visible_enemies(state)
            target  = self._pick_target(enemies)
            if target is not None:
                aligned = _aim_action(action, target, _screen_width(state))
                if not aligned:
                    action[BTN_ATTACK] = False
            else:
                action[BTN_ATTACK] = True
            if self._strafe_dir > 0:
                action[BTN_MOVE_RIGHT] = True
            else:
                action[BTN_MOVE_LEFT] = True
            return action

        # Engage any nearby enemy (range-limited).
        enemies = _visible_enemies(state, self._ENEMY_DETECT_RANGE)
        target  = self._pick_target(enemies)
        if target is not None:
            self.strategy = "targeting"
            aligned = _aim_action(action, target, _screen_width(state))
            if not aligned:
                action[BTN_ATTACK] = False   # hold fire until crosshair is on target
            else:
                action[BTN_MOVE_FORWARD] = True   # close the distance when aimed
            return action

        # No enemy in range — grab any visible items (including enemy drops).
        seek_act = self._seeker.update(state, ammo, health)
        if seek_act is not None:
            self.strategy = "picking up item"
            return seek_act

        # Room clearing — 360° sweep after each kill to find hidden enemies.
        clear_act = self._clearer.update(state, kills, angle_deg)
        if clear_act is not None:
            self.strategy = "clearing"
            return clear_act

        # Rebuild field periodically or when stale.
        self._field_t -= 1
        if self._field_t <= 0 or self._field is None:
            target = self._nearest_unknown_bfs(gx, gy)
            if target is not None:
                self._field = self._build_field(target)
            self._field_t = self._FIELD_TTL

        if self._field is not None:
            fdx, fdy = float(self._field[gy, gx, 0]), float(self._field[gy, gx, 1])
            if abs(fdx) > 0.01 or abs(fdy) > 0.01:
                tx = px + fdx * self.CELL_SIZE * 2
                ty = py + fdy * self.CELL_SIZE * 2
                action = self._steer(action, px, py, angle_deg, tx, ty)
                if "pathing" in self.strategy or "turning" in self.strategy:
                    self.strategy = "flow field"
            else:
                # At flow-field goal — force a rebuild next frame.
                self._field_t = 0
                bfs_target = self._nearest_unknown_bfs(gx, gy)
                if bfs_target is not None:
                    self.strategy = "wandering"
                    path = astar(self.grid, (gx, gy), bfs_target, max_nodes=2000)
                    if path and len(path) > 1:
                        twx, twy = self._grid_to_world(*path[1])
                    else:
                        twx, twy = self._grid_to_world(*bfs_target)
                    action = self._steer(action, px, py, angle_deg, twx, twy)
                else:
                    self.strategy = "searching"
                    self._scan_t -= 1
                    if self._scan_t <= 0:
                        self._scan_phase = not self._scan_phase
                        self._scan_t = 70 if self._scan_phase else 30
                    if self._scan_phase:
                        action[BTN_TURN_RIGHT] = True
                    else:
                        action[BTN_MOVE_FORWARD] = True
        else:
            # No flow field yet — use A* to route around walls to nearest unknown.
            bfs_target = self._nearest_unknown_bfs(gx, gy)
            if bfs_target is not None:
                self.strategy = "wandering"
                path = astar(self.grid, (gx, gy), bfs_target, max_nodes=2000)
                if path and len(path) > 1:
                    twx, twy = self._grid_to_world(*path[1])
                else:
                    twx, twy = self._grid_to_world(*bfs_target)
                action = self._steer(action, px, py, angle_deg, twx, twy)
            else:
                self.strategy = "searching"
                self._scan_t -= 1
                if self._scan_t <= 0:
                    self._scan_phase = not self._scan_phase
                    self._scan_t = 70 if self._scan_phase else 30
                if self._scan_phase:
                    action[BTN_TURN_RIGHT] = True
                else:
                    action[BTN_MOVE_FORWARD] = True

        # Press USE when stuck (opens doors); also back up + turn to escape corners.
        if self._stuck_t >= 30:
            action[BTN_USE] = True
            self._field_t = 0    # force flow field rebuild to pick a fresh target
            action[BTN_MOVE_BACKWARD] = True
            action[BTN_TURN_LEFT if random.random() < 0.5 else BTN_TURN_RIGHT] = True
            self._stuck_t = 0

        return action


# ---------------------------------------------------------------------------
# WaypointGraphAgent — sparse LOS-connected waypoint graph (NavMesh ancestor)
# ---------------------------------------------------------------------------

class WaypointGraphAgent(_GridAgent):
    """
    Waypoint Graph pathfinding — used in Quake, Half-Life, Counter-Strike,
    and Unreal Tournament (classic era).

    A sparse set of navigation waypoints is sampled from free grid cells at
    regular intervals.  Pairs of waypoints with an unobstructed line of sight
    (Bresenham check on the occupancy grid) are connected with weighted edges.
    The agent plans a path as a sequence of waypoints using A* on this graph,
    then steers through them one by one.

    This is the direct predecessor of modern NavMesh systems: Unity's NavMesh
    and Unreal's Recast/Detour work on the same principle but with walkable
    polygons instead of discrete points.
    """

    _WP_SPACING = 12   # grid cells between sampled waypoints
    _GRAPH_TTL  = 90   # rebuild graph after this many frames

    def __init__(self):
        super().__init__()
        self._waypoints = []   # list of (gx, gy)
        self._edges     = {}   # {i: [(j, dist), ...]}
        self._graph_t   = 0

    def reset(self):
        super().reset()
        self._waypoints = []
        self._edges     = {}
        self._graph_t   = 0

    def _rebuild_graph(self):
        """Sample waypoints from explored cells; connect pairs with LOS."""
        GS   = self.GRID_SIZE
        step = self._WP_SPACING
        wps  = [
            (x, y)
            for y in range(step // 2, GS, step)
            for x in range(step // 2, GS, step)
            if self.grid[y, x] in (_FREE, _VISITED)
        ]
        edges = {i: [] for i in range(len(wps))}
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                if _line_of_sight(self.grid, wps[i], wps[j]):
                    d = math.hypot(wps[i][0] - wps[j][0], wps[i][1] - wps[j][1])
                    edges[i].append((j, d))
                    edges[j].append((i, d))
        self._waypoints = wps
        self._edges     = edges

    def _wp_astar(self, start_i, goal_i):
        """A* search on the waypoint graph."""
        wps = self._waypoints

        def h(i):
            return math.hypot(wps[i][0] - wps[goal_i][0],
                              wps[i][1] - wps[goal_i][1])

        heap      = [(h(start_i), start_i)]
        came_from = {start_i: None}
        g         = {start_i: 0.0}

        while heap:
            _, curr = heapq.heappop(heap)
            if curr == goal_i:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]
            for nb, cost in self._edges.get(curr, []):
                ng = g[curr] + cost
                if ng < g.get(nb, float("inf")):
                    g[nb]         = ng
                    came_from[nb] = curr
                    heapq.heappush(heap, (ng + h(nb), nb))
        return None

    def _plan(self, gx, gy):
        self._graph_t -= 1
        if self._graph_t <= 0 or not self._waypoints:
            self._rebuild_graph()
            self._graph_t = self._GRAPH_TTL

        target = self._nearest_unknown_bfs(gx, gy)
        if target is None:
            return []

        if not self._waypoints:
            # Graph not built yet — fall back to plain A*.
            path = astar(self.grid, (gx, gy), target)
            return path[1:] if path else []

        start_wp = min(range(len(self._waypoints)),
                       key=lambda i: math.hypot(self._waypoints[i][0] - gx,
                                                self._waypoints[i][1] - gy))
        goal_wp  = min(range(len(self._waypoints)),
                       key=lambda i: math.hypot(self._waypoints[i][0] - target[0],
                                                self._waypoints[i][1] - target[1]))

        wp_path = self._wp_astar(start_wp, goal_wp)
        if wp_path is None:
            return []

        cells = [self._waypoints[i] for i in wp_path]
        cells.append(target)
        return cells


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------

# Occupancy grid cell values.
_UNKNOWN = 0
_FREE    = 1
_WALL    = 2
_VISITED = 3


class HPAStarAgent(_GridAgent):
    """
    Hierarchical Pathfinding A* (HPA*) — used in Dragon Age: Origins,
    the StarCraft series, and other large-map strategy games.

    The occupancy grid is divided into fixed-size clusters.  Cells on the
    border between two passable adjacent clusters become "entrance" nodes.
    An abstract graph connects those entrances — within a cluster via local
    A*, across cluster boundaries via direct adjacency.

    Path planning happens in two stages:
      1. Abstract search  — A* on the small entrance graph (fast, coarse).
      2. Local refinement — A* within each cluster segment to produce a
                            smooth, accurate ground-level path.

    Compared to flat A* on the full 256×256 grid the abstract graph is
    orders of magnitude smaller, so planning is much faster on large maps.
    """

    _CLUSTER  = 16   # grid cells per cluster edge (16×16 clusters)
    _GRAPH_TTL = 90   # frames between abstract-graph rebuilds

    def __init__(self):
        super().__init__()
        self._entrances = []   # [(gx, gy), ...]
        self._abs_graph = {}   # {i: [(j, cost), ...]}
        self._graph_t   = 0

    def reset(self):
        super().reset()
        self._entrances = []
        self._abs_graph = {}
        self._graph_t   = 0

    def _cluster_of(self, gx, gy):
        return gx // self._CLUSTER, gy // self._CLUSTER

    def _rebuild_abstract(self):
        """Find border entrances and connect them in an abstract graph.

        Only the midpoint of each contiguous passable run on each cluster
        border is added as an entrance node.  This keeps the entrance count
        small (O(doorways) rather than O(border cells)) and prevents the
        O(N²) pairwise loop from becoming catastrophically slow on explored
        maps.
        """
        GS  = self.GRID_SIZE
        CS  = self._CLUSTER
        NCX = GS // CS
        NCY = GS // CS

        ent_set = {}

        def add(cell):
            if cell not in ent_set:
                ent_set[cell] = len(ent_set)

        def add_run_midpoints(cells):
            """Split cells into contiguous runs; add the midpoint of each."""
            if not cells:
                return
            run = [cells[0]]
            for c in cells[1:]:
                if abs(c[0] - run[-1][0]) + abs(c[1] - run[-1][1]) == 1:
                    run.append(c)
                else:
                    add(run[len(run) // 2])
                    run = [c]
            add(run[len(run) // 2])

        # Horizontal borders: between cluster (cx,cy) and (cx+1,cy).
        for cy in range(NCY):
            for cx in range(NCX - 1):
                bx = (cx + 1) * CS
                left  = [(bx - 1, y) for y in range(cy * CS, (cy + 1) * CS)
                         if self.grid[y, bx - 1] != _WALL and self.grid[y, bx] != _WALL]
                right = [(bx,     y) for y in range(cy * CS, (cy + 1) * CS)
                         if self.grid[y, bx - 1] != _WALL and self.grid[y, bx] != _WALL]
                add_run_midpoints(left)
                add_run_midpoints(right)

        # Vertical borders: between cluster (cx,cy) and (cx,cy+1).
        for cy in range(NCY - 1):
            for cx in range(NCX):
                by = (cy + 1) * CS
                top = [(x, by - 1) for x in range(cx * CS, (cx + 1) * CS)
                       if self.grid[by - 1, x] != _WALL and self.grid[by, x] != _WALL]
                bot = [(x, by)     for x in range(cx * CS, (cx + 1) * CS)
                       if self.grid[by - 1, x] != _WALL and self.grid[by, x] != _WALL]
                add_run_midpoints(top)
                add_run_midpoints(bot)

        entrances = list(ent_set.keys())

        # Safety cap: if the entrance count is still large, fall back to an
        # empty graph so _plan() uses flat A* instead of the abstract layer.
        if len(entrances) > 300:
            self._entrances = []
            self._abs_graph = {}
            return

        graph = {i: [] for i in range(len(entrances))}

        for i, (ax, ay) in enumerate(entrances):
            ci = self._cluster_of(ax, ay)
            for j in range(i + 1, len(entrances)):
                bx, by = entrances[j]
                cj = self._cluster_of(bx, by)
                if ci == cj:
                    path = astar(self.grid, (ax, ay), (bx, by), max_nodes=300)
                    if path:
                        cost = float(len(path) - 1)
                        graph[i].append((j, cost))
                        graph[j].append((i, cost))
                elif (abs(ci[0] - cj[0]) + abs(ci[1] - cj[1])) == 1:
                    d = math.hypot(ax - bx, ay - by)
                    if d <= 1.5:
                        graph[i].append((j, 1.0))
                        graph[j].append((i, 1.0))

        self._entrances = entrances
        self._abs_graph = graph

    def _abs_astar(self, start_i, goal_i):
        """A* on the abstract entrance graph."""
        ents = self._entrances

        def h(i):
            return math.hypot(ents[i][0] - ents[goal_i][0],
                              ents[i][1] - ents[goal_i][1])

        heap      = [(h(start_i), start_i)]
        came_from = {start_i: None}
        g         = {start_i: 0.0}

        while heap:
            _, curr = heapq.heappop(heap)
            if curr == goal_i:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = came_from[curr]
                return path[::-1]
            for nb, cost in self._abs_graph.get(curr, []):
                ng = g[curr] + cost
                if ng < g.get(nb, float("inf")):
                    g[nb]         = ng
                    came_from[nb] = curr
                    heapq.heappush(heap, (ng + h(nb), nb))
        return None

    def _plan(self, gx, gy):
        self._graph_t -= 1
        if self._graph_t <= 0 or not self._entrances:
            self._rebuild_abstract()
            self._graph_t = self._GRAPH_TTL

        target = self._nearest_unknown_bfs(gx, gy)
        if target is None:
            return []

        if not self._entrances:
            path = astar(self.grid, (gx, gy), target)
            return path[1:] if path else []

        # Find entrance nodes nearest to current position and goal.
        start_e = min(range(len(self._entrances)),
                      key=lambda i: math.hypot(self._entrances[i][0] - gx,
                                               self._entrances[i][1] - gy))
        goal_e  = min(range(len(self._entrances)),
                      key=lambda i: math.hypot(self._entrances[i][0] - target[0],
                                               self._entrances[i][1] - target[1]))

        abs_path = self._abs_astar(start_e, goal_e)
        if abs_path is None:
            path = astar(self.grid, (gx, gy), target)
            return path[1:] if path else []

        # Refine: stitch local A* segments between entrance waypoints.
        waypoints  = [(gx, gy)] + [self._entrances[i] for i in abs_path] + [target]
        full_path  = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            seg = astar(self.grid, a, b, max_nodes=500)
            if seg:
                full_path.extend(seg[1:])
        return full_path


# ---------------------------------------------------------------------------
# MDPAgent — Q-learning Markov Decision Process agent
# ---------------------------------------------------------------------------

# Abstract action indices (high-level strategies / options).
_MDP_COMBAT    = 0   # dodge-strafe + shoot at visible enemy
_MDP_AIM_SHOOT = 1   # careful aim then advance when aligned
_MDP_FLEE      = 2   # back away from enemy
_MDP_PICKUP    = 3   # move toward visible pickup
_MDP_CLEAR     = 4   # 360° sweep for hidden enemies
_MDP_EXPLORE   = 5   # A* pathfinding toward unexplored cells
_MDP_SEARCH    = 6   # slow spin scan (map fully explored)
_MDP_USE_DOOR  = 7   # press USE + back up (stuck recovery)
_N_MDP_ACTIONS = 8


def _mdp_state(state, health, ammo, kills, kills_prev, stuck_t):
    """
    Extract a compact discrete feature tuple used as the MDP state (Q-table key).

    Features
    --------
    enemy_vis  : 0/1   — at least one enemy visible
    enemy_cls  : 0/1   — nearest enemy within 250 depth units
    n_enemies  : 0/1/2 — 0, 1, or ≥2 visible enemies
    hp_bucket  : 0–3   — 0=<25  1=<50  2=<75  3=≥75 HP
    ammo_buck  : 0–3   — 0=0  1=<25  2=<75  3=≥75 rounds
    has_pick   : 0/1   — a pickup item is visible
    has_hp_pk  : 0/1   — a health/armor pickup is visible
    just_kill  : 0/1   — kills rose since the last frame
    is_stuck   : 0/1   — no meaningful movement for ≥20 frames

    Up to ~3 072 reachable entries — fits comfortably in an in-memory dict.
    """
    if state is None:
        return (0, 0, 0, 2, 2, 0, 0, 0, 0)

    enemies = _visible_enemies(state)
    n_vis   = min(len(enemies), 2)

    enemy_close = 0
    if enemies:
        depth = getattr(state, 'depth_buffer', None)
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth.ndim == 2:
                H, W = depth.shape
                lbl  = enemies[0]
                cx   = max(0, min(W - 1, int(lbl.x + lbl.width  / 2)))
                cy   = max(0, min(H - 1, int(lbl.y + lbl.height / 2)))
                d    = float(depth[cy, cx])
                enemy_close = int(0 < d < 250)
            else:
                enemy_close = 1
        else:
            enemy_close = 1

    if   health < 25: hp = 0
    elif health < 50: hp = 1
    elif health < 75: hp = 2
    else:             hp = 3

    if   ammo == 0: am = 0
    elif ammo < 25: am = 1
    elif ammo < 75: am = 2
    else:           am = 3

    pickups   = _visible_pickups(state)
    has_pick  = int(bool(pickups))
    has_hp_pk = int(any(
        any(kw in p.object_name for kw in _HEALTH_KEYWORDS)
        for p in pickups
    ))

    return (
        int(bool(enemies)),        # enemy_vis
        enemy_close,               # enemy_cls
        n_vis,                     # n_enemies
        hp,                        # hp_bucket
        am,                        # ammo_buck
        has_pick,                  # has_pick
        has_hp_pk,                 # has_hp_pk
        int(kills > kills_prev),   # just_kill
        int(stuck_t >= 20),        # is_stuck
    )


class MDPAgent(_GridAgent):
    """
    Q-learning Markov Decision Process agent for doom-ascii.

    Decision-making is modelled as a semi-MDP over a compact discrete state
    space.  Eight high-level strategies (options) are available; the agent
    selects among them with an ε-greedy policy backed by a Q-table updated
    online using the TD(0) Q-learning rule:

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a′ Q(s′,a′) − Q(s,a)]

    After each frame the reward signal (kills, health delta, exploration
    progress, stuck penalty, alive bonus) drives table updates.  Over many
    episodes the agent learns when to fight, flee, heal, explore, or clear.

    The Q-table persists across episodes within a session so learning is
    cumulative.  ε decays from 0.40 toward 0.05 as episodes accumulate.

    Hyper-parameters
    ----------------
    α = 0.15    learning rate
    γ = 0.95    discount factor
    ε = 0.40 → 0.05  per-episode ε decay of 0.002
    Commit window: 8 frames (shorter on critical override)
    """

    strategy = "MDP"

    _ALPHA     = 0.15
    _GAMMA     = 0.95
    _EPS_START = 0.40
    _EPS_MIN   = 0.05
    _EPS_DECAY = 0.002    # subtracted from ε each episode
    _COMMIT    = 8        # frames to execute the chosen strategy before re-evaluating

    def __init__(self):
        super().__init__()
        # Q-table: state-tuple → list of Q-values, one per abstract action.
        # Optimistic initialisation (1.0) encourages early exploration of all options.
        self._q          = {}
        self._epsilon    = self._EPS_START
        self._n_ep       = 0

        # Current committed abstract action and countdown.
        self._cur_act    = _MDP_EXPLORE
        self._commit_t   = 0

        # Bookkeeping for reward computation and Q-update.
        self._s_prev     = None   # MDP state at previous frame
        self._a_prev     = None   # abstract action chosen at previous frame
        self._kills_prev = 0      # kill count at previous frame
        self._cells_prev = 0      # explored-cell count at previous frame

    def reset(self):
        # Preserve Q-table and learning state across episodes.
        q_saved  = self._q
        eps      = max(self._EPS_MIN,
                       self._EPS_START - (self._n_ep + 1) * self._EPS_DECAY)
        n_ep     = self._n_ep + 1
        # super().reset() calls _GridAgent.reset() → self.__init__() which
        # would wipe _q; we restore it immediately after.
        super().reset()
        self._q          = q_saved
        self._epsilon    = eps
        self._n_ep       = n_ep
        self._cur_act    = _MDP_EXPLORE
        self._commit_t   = 0
        self._s_prev     = None
        self._a_prev     = None
        self._kills_prev = 0
        self._cells_prev = 0

    # ---- Q-table helpers -------------------------------------------------------

    def _qv(self, s):
        """Return (initialising if absent) the Q-value vector for state s."""
        if s not in self._q:
            self._q[s] = [1.0] * _N_MDP_ACTIONS
        return self._q[s]

    def _best_act(self, s, pool=None):
        qv = self._qv(s)
        candidates = list(range(_N_MDP_ACTIONS)) if pool is None else pool
        return max(candidates, key=lambda a: qv[a])

    def _choose(self, s, pool=None):
        """ε-greedy action selection over *pool* (all actions if None)."""
        candidates = list(range(_N_MDP_ACTIONS)) if pool is None else pool
        if random.random() < self._epsilon:
            return random.choice(candidates)
        return self._best_act(s, candidates)

    def _td_update(self, s, a, r, s_next):
        """One-step Q-learning update."""
        qv       = self._qv(s)
        qv_next  = self._qv(s_next)
        target   = r + self._GAMMA * max(qv_next)
        qv[a]   += self._ALPHA * (target - qv[a])

    # ---- Reward shaping --------------------------------------------------------

    def _reward(self, health, kills, new_cells):
        """
        Per-frame reward signal.

        Kills       : +50 each  (primary objective)
        Health gain : +0.5 / HP  (incentivise picking up health)
        Health loss : -1.0 / HP  (penalise taking damage)
        Exploration : +0.5 per new grid cell revealed
        Alive bonus : +0.05 per frame
        Stuck       : -0.2 per frame when stuck
        """
        r   = (kills - self._kills_prev) * 50.0
        dh  = health - self._prev_health
        r  += dh * (0.5 if dh > 0 else 1.0)
        r  += new_cells * 0.5
        r  += 0.05
        if self._stuck_t >= 20:
            r -= 0.2
        return r

    # ---- Override detection ----------------------------------------------------

    def _override(self, s):
        """
        Return a forced abstract action for critical conditions, or None to
        let the commitment timer and Q-table decide normally.

        Critical overrides bypass Q-table to keep the agent alive:
          • Critical HP + health pickup visible  → grab it immediately
          • Critical HP + enemy close            → flee now
          • Stuck                                → open door / unstick
        """
        _, enemy_cls, _, hp, _, _, has_hp_pk, _, is_stuck = s
        if hp == 0 and has_hp_pk:
            return _MDP_PICKUP
        if hp == 0 and enemy_cls:
            return _MDP_FLEE
        if is_stuck:
            return _MDP_USE_DOOR
        return None

    # ---- Low-level strategy executors ------------------------------------------

    def _do_combat(self, state, angle_deg):
        """Strafe-dodge while shooting at the locked target."""
        action = make_no_op()
        target = self._pick_target(_visible_enemies(state))
        if target is not None:
            _aim_action(action, target, _screen_width(state))
        else:
            action[BTN_ATTACK] = True
        action[BTN_MOVE_RIGHT if self._strafe_dir > 0 else BTN_MOVE_LEFT] = True
        self.strategy = "MDP:combat"
        return action

    def _do_aim(self, state, px, py, angle_deg):
        """Careful aim; advance only when crosshair is aligned."""
        action = make_no_op()
        target = self._pick_target(_visible_enemies(state))
        if target is None:
            return None
        aligned = _aim_action(action, target, _screen_width(state))
        if aligned:
            action[BTN_MOVE_FORWARD] = True
        self.strategy = "MDP:aim_shoot"
        return action

    def _do_flee(self, angle_deg):
        """Back away from enemy while strafing perpendicular."""
        action = make_no_op()
        action[BTN_MOVE_BACKWARD] = True
        action[BTN_MOVE_LEFT if self._strafe_dir > 0 else BTN_MOVE_RIGHT] = True
        self.strategy = "MDP:flee"
        return action

    def _do_pickup(self, state, ammo, health):
        """Steer toward the best visible pickup item."""
        act = self._seeker.update(state, ammo, health)
        if act is not None:
            self.strategy = "MDP:pickup"
        return act

    def _do_clear(self, state, kills, angle_deg):
        """360° room-clearing sweep after a kill."""
        act = self._clearer.update(state, kills, angle_deg)
        if act is not None:
            self.strategy = "MDP:clear"
        return act

    def _do_explore(self, state, px, py, angle_deg, gx, gy):
        """A* navigation toward the nearest unexplored cell."""
        target = self._nearest_unknown_bfs(gx, gy)
        if target is None:
            self.strategy = "MDP:explore(done)"
            return None
        path   = astar(self.grid, (gx, gy), target, max_nodes=2000)
        action = make_no_op()
        if path and len(path) > 1:
            twx, twy = self._grid_to_world(*path[1])
            action   = self._steer(action, px, py, angle_deg, twx, twy)
        else:
            action[BTN_MOVE_FORWARD] = True
        self.strategy = "MDP:explore"
        return action

    def _do_search(self):
        """Alternating spin/forward scan when the map is fully explored."""
        action = make_no_op()
        self._scan_t -= 1
        if self._scan_t <= 0:
            self._scan_phase = not self._scan_phase
            self._scan_t = 70 if self._scan_phase else 30
        if self._scan_phase:
            action[BTN_TURN_RIGHT] = True
        else:
            action[BTN_MOVE_FORWARD] = True
        self.strategy = "MDP:search"
        return action

    def _do_use_door(self):
        """Stuck recovery: USE the wall, back up, and turn randomly."""
        action = make_no_op()
        action[BTN_USE]           = True
        action[BTN_MOVE_BACKWARD] = True
        action[BTN_TURN_LEFT if random.random() < 0.5 else BTN_TURN_RIGHT] = True
        self._stuck_t  = 0
        self._replan_t = 0
        self.strategy  = "MDP:use_door"
        return action

    def _execute(self, act_idx, state, health, ammo, kills,
                 px, py, angle_deg, gx, gy):
        """Dispatch abstract action index to the appropriate low-level executor."""
        if act_idx == _MDP_COMBAT:    return self._do_combat(state, angle_deg)
        if act_idx == _MDP_AIM_SHOOT: return self._do_aim(state, px, py, angle_deg)
        if act_idx == _MDP_FLEE:      return self._do_flee(angle_deg)
        if act_idx == _MDP_PICKUP:    return self._do_pickup(state, ammo, health)
        if act_idx == _MDP_CLEAR:     return self._do_clear(state, kills, angle_deg)
        if act_idx == _MDP_EXPLORE:   return self._do_explore(state, px, py, angle_deg, gx, gy)
        if act_idx == _MDP_SEARCH:    return self._do_search()
        if act_idx == _MDP_USE_DOOR:  return self._do_use_door()
        return None

    # ---- Required by _GridAgent ------------------------------------------------

    def _plan(self, gx, gy):
        """Not used — MDPAgent manages navigation directly in act()."""
        return []

    # ---- Main act() ------------------------------------------------------------

    def act(self, state, health, ammo, kills):
        action = make_no_op()
        if state is None:
            return action

        gv = state.game_variables
        if len(gv) < 6:
            action[BTN_MOVE_FORWARD] = True
            self.strategy = "MDP:no_pos"
            return action

        px, py    = float(gv[3]), float(gv[4])
        angle_deg = float(gv[5])
        gx, gy    = self._world_to_grid(px, py)

        # Update occupancy grid from depth buffer.
        self._stamp_visited(gx, gy)
        if state.depth_buffer is not None:
            self._update_grid(state.depth_buffer, px, py, angle_deg)

        # Track explored cells for reward (_UNKNOWN == 0, so count_nonzero works).
        cur_cells    = int(np.count_nonzero(self.grid))
        new_cells    = max(0, cur_cells - self._cells_prev)
        self._cells_prev = cur_cells

        # Stuck detection — updates self._stuck_t.
        if self._prev_px is not None:
            moved = abs(px - self._prev_px) + abs(py - self._prev_py)
            self._stuck_t = 0 if moved >= 5 else self._stuck_t + 1
        self._prev_px, self._prev_py = px, py

        # Randomise strafe direction on damage (dodge variety).
        if health < self._prev_health:
            self._strafe_dir = random.choice([-1, 1])

        # ------------------------------------------------------------------
        # MDP step: extract current state, compute reward, update Q-table.
        # ------------------------------------------------------------------
        s = _mdp_state(state, health, ammo, kills, self._kills_prev, self._stuck_t)
        r = self._reward(health, kills, new_cells)

        # Apply Q-learning update for the previous (s, a) → (s′, r) transition.
        if self._s_prev is not None:
            self._td_update(self._s_prev, self._a_prev, r, s)

        # ------------------------------------------------------------------
        # Action selection: override → commit → new ε-greedy choice.
        # ------------------------------------------------------------------
        forced = self._override(s)
        if forced is not None:
            # Critical condition — bypass Q-table and commit immediately.
            self._cur_act  = forced
            self._commit_t = self._COMMIT
        elif self._commit_t <= 0:
            # Commitment window expired — pick a new strategy.
            self._cur_act  = self._choose(s)
            self._commit_t = self._COMMIT
        self._commit_t -= 1

        # ------------------------------------------------------------------
        # Execute the chosen strategy; fall back greedily if it returns None.
        # ------------------------------------------------------------------
        result = self._execute(self._cur_act, state, health, ammo, kills,
                               px, py, angle_deg, gx, gy)

        if result is None:
            # Primary strategy had nothing to do — try the next-best action.
            fb_pool = [a for a in range(_N_MDP_ACTIONS) if a != self._cur_act]
            fb      = self._best_act(s, fb_pool)
            self._cur_act  = fb
            self._commit_t = self._COMMIT // 2
            result = self._execute(fb, state, health, ammo, kills,
                                   px, py, angle_deg, gx, gy)

        if result is None:
            result = make_no_op()
            result[BTN_MOVE_FORWARD] = True
            self.strategy = "MDP:default"

        # Save state/action/bookkeeping for next frame.
        self._s_prev      = s
        self._a_prev      = self._cur_act
        self._prev_health = health
        self._kills_prev  = kills

        return result


# ---------------------------------------------------------------------------
# Keyboard reader (used by HumanAgent in ASCII / terminal mode)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import msvcrt as _msvcrt

    class _KeyboardReader:
        def __init__(self):
            self._key     = None
            self._ts      = 0.0
            self._lock    = threading.Lock()
            self._running = True
            threading.Thread(target=self._loop, daemon=True).start()

        def _loop(self):
            while self._running:
                if _msvcrt.kbhit():
                    raw = _msvcrt.getch()
                    if raw in (b'\xe0', b'\x00'):
                        raw = b'\xe0' + _msvcrt.getch()
                    with self._lock:
                        self._key = raw
                        self._ts  = time.monotonic()
                else:
                    time.sleep(0.005)

        def get(self):
            with self._lock:
                return self._key, self._ts

        def stop(self):
            self._running = False

else:
    import tty     as _tty
    import termios as _termios
    import select  as _select

    class _KeyboardReader:
        def __init__(self):
            self._key          = None
            self._ts           = 0.0
            self._lock         = threading.Lock()
            self._running      = True
            self._old_settings = _termios.tcgetattr(sys.stdin)
            _tty.setraw(sys.stdin.fileno())
            threading.Thread(target=self._loop, daemon=True).start()

        def _loop(self):
            while self._running:
                rlist, _, _ = _select.select([sys.stdin], [], [], 0.02)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        r2, _, _ = _select.select([sys.stdin], [], [], 0.02)
                        if r2:
                            ch += sys.stdin.read(2)
                    with self._lock:
                        self._key = ch.encode()
                        self._ts  = time.monotonic()

        def get(self):
            with self._lock:
                return self._key, self._ts

        def stop(self):
            self._running = False
            _termios.tcsetattr(sys.stdin, _termios.TCSADRAIN, self._old_settings)


# ---------------------------------------------------------------------------
# HumanAgent — keyboard-driven player
# ---------------------------------------------------------------------------

class HumanAgent(BaseAgent):
    """
    Passes keyboard input directly to ViZDoom as actions.

    Graphics mode (pygame)
        Uses pygame.key.get_pressed() — captures held keys naturally.
        Controls:  W/↑=forward  S/↓=back  A/←=turn left  D/→=turn right
                   Q=strafe left  E=strafe right  Space=fire  F=use

    ASCII mode (terminal)
        Background thread reads raw keystrokes via msvcrt (Windows) or
        tty/termios (Unix).  Key stays active for KEY_TTL seconds to
        simulate held-key feel.  Press Ctrl-C or ESC to quit.
    """

    CONTROLS = "W/S:fwd/back  A/D:turn  Q/E:strafe  SPC:fire  F:use"
    _KEY_TTL  = 0.12   # seconds a terminal keypress stays active

    def __init__(self, renderer_type: str = "graphics"):
        self._mode             = renderer_type
        self.strategy          = "PLAYER"
        self.quit_requested    = False
        self.restart_requested = False
        self._kb               = _KeyboardReader() if renderer_type == "ascii" else None

    def reset(self):
        self.quit_requested    = False
        self.restart_requested = False

    def stop(self):
        if self._kb:
            self._kb.stop()

    def act(self, state, health, ammo, kills):
        if self._mode == "graphics":
            return self._act_pygame()
        return self._act_terminal()

    # ------------------------------------------------------------------

    def _act_pygame(self):
        import pygame
        k      = pygame.key.get_pressed()
        action = make_no_op()
        action[BTN_MOVE_FORWARD]  = bool(k[pygame.K_w] or k[pygame.K_UP])
        action[BTN_MOVE_BACKWARD] = bool(k[pygame.K_s] or k[pygame.K_DOWN])
        action[BTN_TURN_LEFT]     = bool(k[pygame.K_a] or k[pygame.K_LEFT])
        action[BTN_TURN_RIGHT]    = bool(k[pygame.K_d] or k[pygame.K_RIGHT])
        action[BTN_MOVE_LEFT]     = bool(k[pygame.K_q])
        action[BTN_MOVE_RIGHT]    = bool(k[pygame.K_e])
        action[BTN_ATTACK]        = bool(k[pygame.K_SPACE])
        action[BTN_USE]           = bool(k[pygame.K_f])
        for event in pygame.event.get(pygame.KEYDOWN):
            if event.key == pygame.K_r:
                self.restart_requested = True
        return action

    def _act_terminal(self):
        action   = make_no_op()
        raw, ts  = self._kb.get()

        if raw in (b'\x03', b'\x1b', b'c', b'C'):   # Ctrl-C / ESC / c
            self.quit_requested = True
            return action

        if raw is None or time.monotonic() - ts > self._KEY_TTL:
            return action

        k = raw.lower()
        if   k == b'r':              self.restart_requested = True; return action
        elif k == b'w':              action[BTN_MOVE_FORWARD]  = True
        elif k == b's':              action[BTN_MOVE_BACKWARD] = True
        elif k in (b'a', b'\xe0\x4b'): action[BTN_TURN_LEFT]  = True
        elif k in (b'd', b'\xe0\x4d'): action[BTN_TURN_RIGHT] = True
        elif k == b'q':              action[BTN_MOVE_LEFT]     = True
        elif k == b'e':              action[BTN_MOVE_RIGHT]    = True
        elif k == b' ':              action[BTN_ATTACK]        = True
        elif k == b'f':              action[BTN_USE]           = True

        return action


# ---------------------------------------------------------------------------
# MonsterAgent — Doom enemy AI applied to the player
# ---------------------------------------------------------------------------

class MonsterAgent(BaseAgent):
    """
    Mimics the AI that Doom uses for its own monsters — applied to the player.

    Behaviour
    ---------
    Enemy in sight  → charge straight at it and fire (no pathfinding).
    No enemy in sight → wander: move forward and randomly change facing every
                        WANDER_INTERVAL frames.  Gets stuck on walls just like
                        a real Doom imp.

    No map building, no item pickup, no room-clearing sweep.
    """

    strategy = "monster"

    _WANDER_INTERVAL = 28   # frames between random direction changes

    def __init__(self):
        self._wander_turn  = 0   # -1 left, 0 straight, 1 right
        self._wander_timer = 0

    def reset(self):
        self.__init__()

    def act(self, state, health, ammo, kills):
        action  = make_no_op()
        enemies = _visible_enemies(state)

        if enemies:
            # Charge the closest visible enemy and shoot.
            aligned = _aim_action(action, enemies[0], _screen_width(state))
            action[BTN_MOVE_FORWARD] = True   # always rush forward
            return action

        # No enemies visible — wander randomly like a Doom monster.
        self._wander_timer -= 1
        if self._wander_timer <= 0:
            self._wander_timer = self._WANDER_INTERVAL
            self._wander_turn  = random.choice([-1, -1, 0, 0, 0, 1, 1])

        action[BTN_MOVE_FORWARD] = True
        if self._wander_turn == -1:
            action[BTN_TURN_LEFT]  = True
        elif self._wander_turn == 1:
            action[BTN_TURN_RIGHT] = True
        return action


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# AI-only agents (shown in the AI agent menu).
AGENTS = {
    "flow_field": FlowFieldAgent,
    "waypoint":   WaypointGraphAgent,
    "hpa_star":   HPAStarAgent,
    "monster":    MonsterAgent,
    "mdp":        MDPAgent,
}


def build_agent(name: str, **kwargs) -> BaseAgent:
    """
    Instantiate an agent by name.

    Pass renderer_type=... when name=='human':
        build_agent('human', renderer_type='graphics')
    """
    name = name.lower()
    if name == "human":
        return HumanAgent(**kwargs)
    if name == "random":
        name = random.choice(list(AGENTS))
    if name not in AGENTS:
        raise ValueError(
            f"Unknown agent '{name}'. Choose from: {', '.join(AGENTS)}, random, or 'human'"
        )
    return AGENTS[name]()
