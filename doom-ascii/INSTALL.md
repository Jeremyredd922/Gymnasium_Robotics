# doom-ascii — Installation Guide

## Requirements

- **Python 3.10 or newer** (3.13 recommended)
- **pip** (included with Python)

---

## Step 1 — Install Python

If you don't have Python installed, download it from:

    https://www.python.org/downloads/

During installation on Windows, check **"Add Python to PATH"**.

Verify it works by opening a terminal and running:

    python --version

---

## Step 2 — Install dependencies

Open a terminal, navigate to the `doom-ascii` folder, and run:

    pip install vizdoom pygame

**What each package does:**

| Package  | Purpose                                                    |
|----------|------------------------------------------------------------|
| vizdoom  | ViZDoom game engine — runs Doom and provides the AI API.   |
|          | Also installs numpy and bundles freedoom2.wad automatically.|
| pygame   | Powers the graphics renderer (960×720 window with HUD).    |
|          | Not required if you only use the ASCII terminal renderer.   |

---

## Step 3 — Run the game

From inside the `doom-ascii` folder:

    python main.py

This opens the startup menu where you can choose AI spectator or human
player mode, pick a renderer, and view controls.

---

## Common options

    python main.py --mode human              # play yourself (graphics)
    python main.py --mode human --renderer ascii   # play in the terminal
    python main.py --mode ai --agent hpa_star      # AI plays, graphics window
    python main.py --mode ai --renderer ascii      # AI plays, terminal

---

## Troubleshooting

**"vizdoom not found"**
Run `pip install vizdoom` and try again.

**"pygame not found"**
Run `pip install pygame` and try again. Only needed for graphics mode.

**Black window or no display**
Make sure your system has a display available (not a headless server).
Use `--renderer ascii` as an alternative.

**Game crashes immediately**
ViZDoom requires a compatible OpenGL driver. Update your GPU drivers,
or use `--renderer ascii` to bypass the graphics window entirely.

**Permission errors on Windows**
Run your terminal as Administrator, or install into a virtual environment:

    python -m venv venv
    venv\Scripts\activate
    pip install vizdoom pygame
    python main.py
