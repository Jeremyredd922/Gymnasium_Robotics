"""
Simple agents for Gymnasium-Robotics environments.
"""

import numpy as np
import gymnasium as gym


class RandomAgent:
    """Takes uniformly random actions — useful as a baseline."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def act(self, obs) -> np.ndarray:
        return self.action_space.sample()


class GoalConditionedAgent:
    """
    Naive proportional controller for GoalEnv environments.

    Works on environments whose observation dict contains 'observation'
    and 'desired_goal', and where the action space is continuous.
    Moves the gripper/end-effector toward the goal using a simple
    proportional signal on the first `action_dim` dimensions.
    """

    def __init__(self, action_space: gym.spaces.Box, gain: float = 2.0):
        self.action_space = action_space
        self.gain = gain
        self.action_dim = action_space.shape[0]

    def act(self, obs: dict) -> np.ndarray:
        # achieved_goal is explicitly the current end-effector/object position,
        # making it a reliable source for the proportional error signal across envs.
        achieved = obs["achieved_goal"]
        goal = obs["desired_goal"]

        n = min(len(goal), self.action_dim)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:n] = self.gain * (goal[:n] - achieved[:n])

        return np.clip(action, self.action_space.low, self.action_space.high)
