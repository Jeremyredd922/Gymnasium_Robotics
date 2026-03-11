"""
agent.py — AI agents for doom-ascii.

Three agents are provided:

  RandomAgent    Biased random button presses.  Good chaos baseline.
  WanderAgent    Finite-state machine: forward → turn → strafe → shoot.
                 Deterministic structure with randomised durations.
  HeuristicAgent Screen-buffer analysis + game-state heuristics.
                 Detects motion (enemy?), wall proximity, health loss,
                 and adapts its behaviour accordingly.
"""
import random
import numpy as np

from game import (
    make_no_op,
    BTN_MOVE_LEFT, BTN_MOVE_RIGHT, BTN_MOVE_FORWARD, BTN_MOVE_BACKWARD,
    BTN_TURN_LEFT, BTN_TURN_RIGHT, BTN_ATTACK, BTN_USE,
)


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
# RandomAgent
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    """
    Always moves forward; randomly adds turns, strafes, and attacks.
    Useful as a sanity-check baseline.
    """

    strategy = "random"

    def reset(self):
        pass

    def act(self, state, health, ammo, kills):
        action = make_no_op()
        action[BTN_MOVE_FORWARD] = True

        r = random.random()
        if r < 0.20:
            action[BTN_TURN_LEFT]  = True
        elif r < 0.40:
            action[BTN_TURN_RIGHT] = True

        if random.random() < 0.15:
            action[BTN_ATTACK] = True

        if random.random() < 0.08:
            if random.random() < 0.5:
                action[BTN_MOVE_LEFT]  = True
            else:
                action[BTN_MOVE_RIGHT] = True

        return action


# ---------------------------------------------------------------------------
# WanderAgent
# ---------------------------------------------------------------------------

class WanderAgent(BaseAgent):
    """
    Finite-state machine agent with four states:
      FORWARD  — charge ahead
      TURN     — spin to a new heading
      STRAFE   — sidestep
      SHOOT    — hold fire while advancing
    Durations are randomised so the motion looks natural.
    """

    _STATES = ("forward", "turn", "strafe", "shoot")

    def __init__(self):
        self._state      = "forward"
        self._timer      = 0
        self._turn_dir   = 1     # +1 = right, -1 = left
        self._strafe_dir = 1
        self.strategy    = "wandering"

    def reset(self):
        self.__init__()

    def act(self, state, health, ammo, kills):
        action = make_no_op()

        # Transition when the current state's timer expires.
        if self._timer <= 0:
            self._pick_next_state()

        self._timer -= 1

        if self._state == "forward":
            action[BTN_MOVE_FORWARD] = True
            self.strategy = "wandering"

        elif self._state == "turn":
            if self._turn_dir > 0:
                action[BTN_TURN_RIGHT] = True
            else:
                action[BTN_TURN_LEFT]  = True
            self.strategy = "turning " + ("right" if self._turn_dir > 0 else "left")

        elif self._state == "strafe":
            action[BTN_MOVE_FORWARD] = True
            if self._strafe_dir > 0:
                action[BTN_MOVE_RIGHT] = True
            else:
                action[BTN_MOVE_LEFT]  = True
            self.strategy = "strafing"

        elif self._state == "shoot":
            action[BTN_MOVE_FORWARD] = True
            action[BTN_ATTACK]       = True
            self.strategy = "shooting"

        return action

    def _pick_next_state(self):
        r = random.random()
        if r < 0.45:
            self._state = "forward"
            self._timer = random.randint(25, 70)
        elif r < 0.70:
            self._state = "turn"
            self._turn_dir = random.choice((-1, 1))
            self._timer = random.randint(12, 35)
        elif r < 0.88:
            self._state = "strafe"
            self._strafe_dir = random.choice((-1, 1))
            self._timer = random.randint(10, 20)
        else:
            self._state = "shoot"
            self._timer = random.randint(8, 18)


# ---------------------------------------------------------------------------
# HeuristicAgent
# ---------------------------------------------------------------------------

class HeuristicAgent(BaseAgent):
    """
    Screen-buffer analysis agent.

    Each frame it reads the GRAY8 frame and game variables to decide:

      retreating  — just took damage: backpedal and spray fire
      engaging    — motion detected in the centre zone (likely an enemy):
                    advance and shoot
      avoiding    — wall close ahead: back up and turn to open side
      searching   — no obvious threat: slow scan left/right while advancing
      exploring   — default: move forward, periodic blind-fire
    """

    # Thresholds (tuned for GRAY8 320×240).
    _MOTION_THRESHOLD  = 8.0   # mean abs pixel-diff in centre zone
    _WALL_THRESHOLD    = 185   # mean brightness of near-centre → wall close
    _RETREAT_FRAMES    = 18
    _SCAN_PERIOD       = 80    # frames per full left→right scan sweep

    def __init__(self):
        self._frame       = 0
        self._prev_center = None   # previous centre-zone slice (for motion)
        self._prev_health = 100
        self._prev_kills  = 0
        self._retreat_t   = 0
        self._avoid_t     = 0
        self._avoid_dir   = 1
        self.strategy     = "exploring"

    def reset(self):
        self.__init__()

    def act(self, state, health, ammo, kills):
        self._frame += 1
        action = make_no_op()

        if state is None or state.screen_buffer is None:
            return action

        raw = state.screen_buffer
        # Normalise to GRAY8 (H, W) regardless of whether game runs in
        # GRAY8 (ASCII mode) or RGB24 (graphics mode).
        if raw.ndim == 3:
            if raw.shape[0] == 3:          # channels-first (3, H, W)
                raw = raw.transpose(1, 2, 0)
            buf = np.mean(raw, axis=2).astype(np.uint8)
        else:
            buf = raw                      # already (H, W)
        H, W = buf.shape
        view_H = int(H * 0.65)            # top 65% ≈ 3D view (rest is HUD)

        # --- Extract screen regions ------------------------------------------
        # Centre zone used for motion / enemy detection.
        cx0, cx1 = W // 3,       2 * W // 3
        cy0, cy1 = view_H // 4,  3 * view_H // 4
        center   = buf[cy0:cy1, cx0:cx1]

        # Near-wall detector: lower-centre brightness.
        near     = buf[view_H // 2 : view_H, W // 4 : 3 * W // 4]

        # Left / right halves for open-space navigation.
        left_half  = buf[:view_H, :W // 2]
        right_half = buf[:view_H, W // 2:]

        near_bright  = float(np.mean(near))
        left_bright  = float(np.mean(left_half))
        right_bright = float(np.mean(right_half))

        # --- Motion detection (frame differencing) ---------------------------
        motion = 0.0
        if self._prev_center is not None and self._prev_center.shape == center.shape:
            diff   = np.abs(center.astype(np.int16) - self._prev_center.astype(np.int16))
            motion = float(np.mean(diff))
        self._prev_center = center.copy()

        # --- Game state events -----------------------------------------------
        took_damage = health < self._prev_health
        self._prev_health = health
        self._prev_kills  = kills

        if took_damage:
            self._retreat_t = self._RETREAT_FRAMES

        # --- Decision tree ---------------------------------------------------

        # 1. Retreating after taking damage.
        if self._retreat_t > 0:
            self.strategy = "retreating"
            action[BTN_MOVE_BACKWARD] = True
            action[BTN_ATTACK]        = True
            # Face the open side while retreating.
            if left_bright > right_bright:
                action[BTN_TURN_LEFT]  = True
            else:
                action[BTN_TURN_RIGHT] = True
            self._retreat_t -= 1
            return action

        # 2. Wall avoidance.
        if self._avoid_t > 0:
            self.strategy = "avoiding"
            action[BTN_MOVE_BACKWARD] = True
            if self._avoid_dir > 0:
                action[BTN_TURN_RIGHT] = True
            else:
                action[BTN_TURN_LEFT]  = True
            self._avoid_t -= 1
            return action

        if near_bright > self._WALL_THRESHOLD:
            self.strategy = "avoiding"
            # Turn toward whichever side looks more open.
            self._avoid_dir = 1 if right_bright > left_bright else -1
            self._avoid_t   = random.randint(14, 28)
            action[BTN_MOVE_BACKWARD] = True
            return action

        # 3. Enemy engagement — motion in centre zone.
        if motion > self._MOTION_THRESHOLD:
            self.strategy = "engaging"
            action[BTN_MOVE_FORWARD] = True
            action[BTN_ATTACK]       = True
            # Slightly turn toward whichever half has more activity.
            if left_bright > right_bright + 5:
                action[BTN_TURN_LEFT]  = True
            elif right_bright > left_bright + 5:
                action[BTN_TURN_RIGHT] = True
            return action

        # 4. Default: explore, scanning left/right while walking.
        self.strategy = "exploring"
        action[BTN_MOVE_FORWARD] = True

        # Slow left→right scan sweep to not miss corridors.
        phase = self._frame % self._SCAN_PERIOD
        if phase < self._SCAN_PERIOD // 4:
            action[BTN_TURN_LEFT]  = True
        elif phase < self._SCAN_PERIOD // 2:
            pass                               # centre — look straight
        elif phase < 3 * self._SCAN_PERIOD // 4:
            action[BTN_TURN_RIGHT] = True

        # Periodic blind-fire (Doom monsters can be heard; also clears sectors).
        if self._frame % 35 == 0 and ammo > 5:
            action[BTN_ATTACK] = True

        return action


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

AGENTS = {
    "random":    RandomAgent,
    "wander":    WanderAgent,
    "heuristic": HeuristicAgent,
}


def build_agent(name: str) -> BaseAgent:
    """Instantiate an agent by name.  Raises ValueError for unknown names."""
    name = name.lower()
    if name not in AGENTS:
        raise ValueError(
            f"Unknown agent '{name}'. Choose from: {', '.join(AGENTS)}"
        )
    return AGENTS[name]()
