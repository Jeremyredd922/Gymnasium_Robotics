"""
Direct MuJoCo simulation entry point.

Usage
-----
    python main.py                                   # built-in double pendulum, viewer on
    python main.py --no-render                       # headless, prints state each 100 steps
    python main.py --model path/to/model.xml         # custom XML model
    python main.py --gymnasium FetchReach-v4         # run a Gymnasium-Robotics env via main.py
    python main.py --gymnasium FetchReach-v4 --episodes 5 --max-steps 150
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_HERE, "models")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "double_pendulum.xml")

DEFAULT_STEPS = 2000
DEFAULT_EPISODES = 3
DEFAULT_MAX_STEPS = 150


# ---------------------------------------------------------------------------
# MuJoCo simulation helpers
# ---------------------------------------------------------------------------

def _load_mujoco(model_path: str):
    """Load a MuJoCo model and return (model, data)."""
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        sys.exit(f"mujoco not installed – run: pip install mujoco\n({exc})")

    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return mujoco, model, data


def run_mujoco_headless(model_path: str, total_steps: int) -> None:
    """Simulate headlessly and print state every 100 steps."""
    mujoco, model, data = _load_mujoco(model_path)

    print(f"Model : {model_path}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print(f"Running {total_steps} steps (no render)…\n")

    for i in range(total_steps):
        # Zero control (passive simulation)
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

        if i % 100 == 0 or i == total_steps - 1:
            qpos_str = np.array2string(data.qpos, precision=4, suppress_small=True)
            qvel_str = np.array2string(data.qvel, precision=4, suppress_small=True)
            print(f"  step={i:5d}  t={data.time:.3f}  qpos={qpos_str}  qvel={qvel_str}")

    print("\nDone.")


def run_mujoco_viewer(model_path: str) -> None:
    """Launch an interactive MuJoCo passive viewer."""
    mujoco, model, data = _load_mujoco(model_path)

    # Perturb initial position so the pendulum actually swings
    if model.nq > 0:
        data.qpos[0] = 0.3

    print(f"Model : {model_path}")
    print(f"nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print("Launching viewer – close the window to exit.\n")

    try:
        import mujoco.viewer as mjviewer  # type: ignore
    except ImportError:
        sys.exit("mujoco.viewer not available – try: pip install mujoco[extras]")

    with mjviewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()


# ---------------------------------------------------------------------------
# Gymnasium-Robotics helpers
# ---------------------------------------------------------------------------

def run_gymnasium(env_id: str, episodes: int, max_steps: int) -> None:
    """Run a Gymnasium-Robotics environment with a random agent."""
    try:
        import gymnasium as gym  # type: ignore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import gymnasium_robotics  # type: ignore
            gymnasium_robotics.register_robotics_envs()
    except ImportError as exc:
        sys.exit(f"gymnasium-robotics not installed – run: pip install gymnasium-robotics\n({exc})")

    env = gym.make(env_id, render_mode="human", max_episode_steps=max_steps)
    print(f"\n=== {env_id} ===")
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}\n")

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        success = False

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info.get("is_success"):
                success = True
            if terminated or truncated:
                break

        returns.append(total_reward)
        status = "SUCCESS" if success else "done"
        print(f"  Episode {ep + 1:2d}: return={total_reward:8.3f}  [{status}]")

    env.close()
    print(f"\nMean return over {episodes} episodes: {np.mean(returns):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct MuJoCo simulation / Gymnasium-Robotics demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to a MuJoCo XML model file (default: models/double_pendulum.xml)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headlessly and print state each 100 steps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of simulation steps for headless mode (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--gymnasium",
        metavar="ENV_ID",
        default=None,
        help="Run a Gymnasium-Robotics environment by ID (e.g. FetchReach-v4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episodes to run when --gymnasium is used (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode when --gymnasium is used (default: {DEFAULT_MAX_STEPS})",
    )

    args = parser.parse_args()

    if args.gymnasium:
        run_gymnasium(args.gymnasium, args.episodes, args.max_steps)
    elif args.no_render:
        run_mujoco_headless(args.model, args.steps)
    else:
        run_mujoco_viewer(args.model)


if __name__ == "__main__":
    main()
