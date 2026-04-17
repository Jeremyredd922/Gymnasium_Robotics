"""
Train a SAC + HER agent on a Gymnasium-Robotics GoalEnv.

HER (Hindsight Experience Replay) is required here because FetchReach uses
sparse rewards — the agent almost never hits the goal by chance, so vanilla
RL gets no learning signal.  HER relabels failed trajectories as if the
achieved position *was* the goal, turning every episode into useful data.

Usage
-----
    python train.py                            # train FetchReach-v4, 200k steps
    python train.py --timesteps 500000         # longer run
    python train.py --env FetchReachDense-v4   # dense-reward variant (no HER needed)
    python train.py --no-eval                  # skip evaluation after training
    python train.py --model-path models/my_policy.zip
"""

import argparse
import os
import time

import numpy as np

from env_utils import make_env

DEFAULT_ENV = "FetchReach-v4"
DEFAULT_TIMESTEPS = 200_000
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_NAME = "fetchreach_sac_her"


def train(env_id: str, timesteps: int, model_path: str, eval_after: bool) -> None:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    except ImportError:
        raise SystemExit(
            "stable-baselines3 is not installed.\n"
            "Run: pip install stable-baselines3"
        )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Environment : {env_id}")
    print(f"Algorithm   : SAC + HER (future goal strategy, n_sampled_goal=4)")
    print(f"Timesteps   : {timesteps:,}")
    print(f"Save path   : {model_path}")
    print()

    env = make_env(env_id)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        learning_starts=1_000,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0

    model.save(model_path)
    print(f"\nModel saved to {model_path}  (trained in {elapsed:.0f}s)")

    env.close()

    if eval_after:
        evaluate(env_id, model_path)


def evaluate(env_id: str, model_path: str, episodes: int = 20) -> None:
    try:
        from stable_baselines3 import SAC
    except ImportError:
        raise SystemExit("stable-baselines3 is not installed.")

    print(f"\n=== Evaluation: {episodes} episodes ===")
    model = SAC.load(model_path)
    env = make_env(env_id)

    successes = []
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        success = False
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info.get("is_success"):
                success = True

        successes.append(float(success))
        returns.append(total_reward)
        print(f"  Episode {ep + 1:2d}: return={total_reward:7.3f}  {'SUCCESS' if success else 'fail'}")

    env.close()
    print(f"\nSuccess rate : {np.mean(successes) * 100:.1f}%")
    print(f"Mean return  : {np.mean(returns):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train SAC+HER on a Gymnasium-Robotics GoalEnv")
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--model-path",
        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME),
        help="Save path (no .zip extension needed)",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    args = parser.parse_args()

    train(
        env_id=args.env,
        timesteps=args.timesteps,
        model_path=args.model_path,
        eval_after=not args.no_eval,
    )


if __name__ == "__main__":
    main()
