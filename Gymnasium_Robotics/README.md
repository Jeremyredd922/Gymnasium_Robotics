# Gymnasium_Robotics

Reinforcement learning experiments using [Gymnasium-Robotics](https://robotics.farama.org/) and MuJoCo.

Built with Claude Code (claude-sonnet-4-6).

## Project Structure

```
Gymnasium_Robotics/
├── agent.py        # RandomAgent and GoalConditionedAgent
├── env_utils.py    # Environment creation and inspection helpers
├── run.py          # Episode runner with configurable agent, env, and steps
└── main.py         # Direct MuJoCo simulation entry point
```

## Agents

- **RandomAgent** — uniform random action baseline
- **GoalConditionedAgent** — naive proportional controller; moves the gripper toward `desired_goal` using the error signal between `achieved_goal` and `desired_goal`

## Installation

```bash
pip install gymnasium-robotics mujoco
```

## Usage

### Run episodes

```bash
python run.py                                    # FetchReach-v4, random agent, 10 episodes
python run.py --env FetchPush-v3                 # different environment
python run.py --agent goal                       # goal-conditioned proportional agent
python run.py --episodes 5 --max-steps 100
python run.py --list                             # list all available environments
python run.py --list --list-prefix Fetch         # filter by prefix
```

### Direct MuJoCo simulation

```bash
python main.py                                   # built-in double pendulum, viewer on
python main.py --no-render                       # headless, prints state each 100 steps
python main.py --model path/to/model.xml         # custom XML model
python main.py --gymnasium FetchReach-v4         # run a Gymnasium-Robotics env via main.py
python main.py --gymnasium FetchReach-v4 --episodes 5 --max-steps 150
```
