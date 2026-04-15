# Gymnasium_Robotics

Reinforcement learning experiments using [Gymnasium-Robotics](https://robotics.farama.org/) and MuJoCo.

Built with Claude Code (claude-sonnet-4-6).

## Project Structure

```
Gymnasium_Robotics/
├── agent.py               # RandomAgent and GoalConditionedAgent
├── env_utils.py           # Environment creation and inspection helpers
├── run.py                 # Episode runner with configurable agent, env, and steps
├── main.py                # Direct MuJoCo simulation entry point
└── models/                # MuJoCo MJCF model files
    ├── double_pendulum.xml # Two-link pendulum (default model)
    ├── reacher.xml         # 2-DOF planar arm with target site
    └── cartpole.xml        # Cart-pole (inverted pendulum on a cart)
```

## Agents

- **RandomAgent** — uniform random action baseline
- **GoalConditionedAgent** — naive proportional controller; moves the gripper toward `desired_goal` using the error signal between `achieved_goal` and `desired_goal`

## Models

The `models/` directory contains MJCF XML files ready to be loaded directly with MuJoCo:

| File | Description |
|------|-------------|
| `double_pendulum.xml` | Two-link pendulum suspended from a fixed pivot; two hinge joints and torque actuators |
| `reacher.xml` | Planar 2-DOF arm (shoulder + elbow) in the horizontal plane with a mock-body target |
| `cartpole.xml` | Inverted pendulum on a sliding cart; classic stabilisation benchmark |

Pass any of these (or your own XML) to `main.py` via `--model`:

```bash
python main.py --model models/cartpole.xml
python main.py --model models/reacher.xml --no-render --steps 5000
```

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
