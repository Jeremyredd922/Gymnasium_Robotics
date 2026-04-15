"""
Direct MuJoCo simulation entry point.

Usage
-----
    python main.py                                    # built-in double pendulum, viewer on
    python main.py --no-render                        # headless, prints state each 100 steps
    python main.py --model path/to/model.xml          # custom XML model
    python main.py --gymnasium FetchReach-v4          # run a Gymnasium-Robotics env via main.py
    python main.py --gymnasium FetchReach-v4 --episodes 5 --max-steps 150
"""

import argparse
import sys

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Built-in double-pendulum XML (MuJoCo tutorial model)
# ---------------------------------------------------------------------------
_DOUBLE_PENDULUM_XML = """
<mujoco model="double_pendulum">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint limited="false" damping="0.05"/>
    <geom contype="0" conaffinity="0" rgba="0.8 0.3 0.1 1"/>
  </default>
  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
    <body name="cart" pos="0 0 1">
      <joint name="hinge1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05 0.3" rgba="0.2 0.6 0.9 1"/>
      <body name="pole" pos="0 0 -0.6">
        <joint name="hinge2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04 0.28" rgba="0.9 0.6 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _print_state(step: int, data: mujoco.MjData) -> None:
    """Print a brief summary of the simulation state."""
    pos = data.qpos.copy()
    vel = data.qvel.copy()
    print(f"  step={step:6d}  qpos={np.round(pos, 4)}  qvel={np.round(vel, 4)}")


def run_mujoco(model_xml: str | None, no_render: bool, steps: int) -> None:
    """Load a MuJoCo model and step it forward, optionally rendering."""
    if model_xml is None:
        model = mujoco.MjModel.from_xml_string(_DOUBLE_PENDULUM_XML)
        print("Running built-in double-pendulum model.")
    else:
        model = mujoco.MjModel.from_xml_path(model_xml)
        print(f"Running model from: {model_xml}")

    data = mujoco.MjData(model)
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}  timestep={model.opt.timestep}")

    # Give joints a small initial perturbation so the simulation is visually interesting
    if model.nq > 0:
        rng = np.random.default_rng(42)
        data.qpos[:] += rng.uniform(-0.1, 0.1, size=model.nq)

    if no_render:
        print(f"Headless simulation for {steps} steps (printing every 100 steps):")
        for step in range(steps):
            mujoco.mj_step(model, data)
            if step % 100 == 0:
                _print_state(step, data)
        print("Done.")
    else:
        try:
            import mujoco.viewer as viewer_module
            with viewer_module.launch_passive(model, data) as viewer:
                print(f"Viewer open — running {steps} steps. Close the window to exit early.")
                for step in range(steps):
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if not viewer.is_running():
                        break
        except Exception as exc:
            print(f"[warning] Could not open viewer ({exc}); falling back to headless mode.")
            for step in range(steps):
                mujoco.mj_step(model, data)
                if step % 100 == 0:
                    _print_state(step, data)
        print("Done.")


# ---------------------------------------------------------------------------
# Gymnasium-Robotics path
# ---------------------------------------------------------------------------

def run_gymnasium(env_id: str, episodes: int, max_steps: int) -> None:
    """Run a Gymnasium-Robotics environment for the requested number of episodes."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import gymnasium_robotics
        gymnasium_robotics.register_robotics_envs()

    import gymnasium as gym

    print(f"Creating Gymnasium-Robotics environment: {env_id}")
    env = gym.make(env_id, max_episode_steps=max_steps)

    obs_space = env.observation_space
    print(f"  Observation space : {obs_space}")
    print(f"  Action space      : {env.action_space}")
    print()

    episode_returns = []

    for ep in range(episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        success = False
        steps_taken = 0
        for steps_taken in range(1, max_steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if info.get("is_success"):
                success = True
            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        status = "SUCCESS" if success else "done"
        print(f"  Episode {ep + 1:2d}: return={total_reward:8.3f}  steps={steps_taken:3d}  [{status}]")

    env.close()
    print(f"\nMean return over {episodes} episodes: {np.mean(episode_returns):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct MuJoCo simulation or Gymnasium-Robotics demo"
    )

    # MuJoCo-direct options
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to a MuJoCo XML model file (default: built-in double pendulum)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless and print state every 100 steps instead of opening a viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of simulation steps (MuJoCo-direct mode, default: 2000)",
    )

    # Gymnasium-Robotics options
    parser.add_argument(
        "--gymnasium",
        default=None,
        metavar="ENV_ID",
        help="Run a Gymnasium-Robotics environment instead of a raw MuJoCo model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes (Gymnasium mode, default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum steps per episode (Gymnasium mode, default: 150)",
    )

    args = parser.parse_args()

    if args.gymnasium:
        run_gymnasium(args.gymnasium, args.episodes, args.max_steps)
    else:
        run_mujoco(args.model, args.no_render, args.steps)


if __name__ == "__main__":
    main()
