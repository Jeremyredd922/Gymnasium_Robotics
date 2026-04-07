"""
Utility helpers for creating and inspecting Gymnasium-Robotics environments.
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np

# Suppress duplicate-registration warnings from gymnasium_robotics
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()


ROBOTICS_PREFIXES = ("Fetch", "Hand", "Adroit", "AntMaze", "PointMaze")


def list_envs(prefix: str | None = None) -> list[str]:
    """Return all registered robotics environment IDs, optionally filtered by prefix."""
    ids = sorted(gym.envs.registry.keys())
    if prefix:
        ids = [e for e in ids if e.startswith(prefix)]
    else:
        ids = [e for e in ids if any(e.startswith(p) for p in ROBOTICS_PREFIXES)]
    return ids


def make_env(env_id: str, render_mode: str | None = None, max_episode_steps: int | None = None) -> gym.Env:
    """Create and return a Gymnasium-Robotics environment."""
    return gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)


def print_env_info(env: gym.Env) -> None:
    """Print observation/action space details for an environment."""
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    if hasattr(env.observation_space, "spaces"):
        for key, space in env.observation_space.spaces.items():
            print(f"  obs[{key!r:12s}] : shape={space.shape}  dtype={space.dtype}")
