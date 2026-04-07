"""
Entry point for Gymnasium-Robotics demos.

Usage
-----
    python run.py                          # default env, random agent
    python run.py --env FetchReach-v3      # specific env
    python run.py --env FetchReach-v3 --agent goal  # proportional goal agent
    python run.py --list                   # list available envs
    python run.py --episodes 5 --render    # render to screen
"""

import argparse
import sys

import numpy as np

from env_utils import list_envs, make_env, print_env_info
from agent import RandomAgent, GoalConditionedAgent

DEFAULT_ENV = "FetchReach-v4"
DEFAULT_EPISODES = 10
DEFAULT_MAX_STEPS = 150


def run_episodes(env_id: str, agent_type: str, episodes: int, max_steps: int, render: bool):
    env = make_env(env_id, render_mode="human", max_episode_steps=max_steps)

    print(f"\n=== {env_id} ===")
    print_env_info(env)
    print()

    is_goal_env = hasattr(env.observation_space, "spaces") and "desired_goal" in env.observation_space.spaces

    if agent_type == "goal" and is_goal_env:
        agent = GoalConditionedAgent(env.action_space)
        print("Agent: GoalConditionedAgent (proportional controller)")
    else:
        agent = RandomAgent(env.action_space)
        print("Agent: RandomAgent")

    print()
    episode_returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        success = False

        step = 0
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info.get("is_success"):
                success = True

            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        status = "SUCCESS" if success else "done"
        print(f"  Episode {ep + 1:2d}: return={total_reward:8.3f}  steps={step + 1:3d}  [{status}]")

    env.close()

    print(f"\nMean return over {episodes} episodes: {np.mean(episode_returns):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Gymnasium-Robotics demo runner")
    parser.add_argument("--env", default=DEFAULT_ENV, help="Environment ID")
    parser.add_argument("--agent", choices=["random", "goal"], default="random",
                        help="Agent type: 'random' or 'goal' (proportional controller)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--render", action="store_true", help="Render environment visually")
    parser.add_argument("--list", action="store_true", help="List available environments and exit")
    parser.add_argument("--list-prefix", default=None, help="Filter --list by prefix (e.g. Fetch)")
    args = parser.parse_args()

    if args.list:
        envs = list_envs(args.list_prefix)
        print(f"Available Gymnasium-Robotics environments ({len(envs)}):")
        for e in envs:
            print(f"  {e}")
        return

    run_episodes(
        env_id=args.env,
        agent_type=args.agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
    )


if __name__ == "__main__":
    main()
