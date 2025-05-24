# Simulation Environments — Guest Lecture Edition

Note: This repo was developed for a guest lecture / class project. It includes working demos from 2023, but is not actively maintained.

Minimal, testable, and production-style RL training loops built for teaching.
Implements:

	•	CartPole (DQN) — discrete control baseline; converges on CPU in ~2 minutes.
 
	•	Minitaur (SAC) — continuous control for quadrupeds via PyBullet; GPU recommended.

Originally developed for a 2025 guest lecture in Applied Machine Learning.

---

## Setup

### For macOS (M1/M2) – Recommended (Conda + Pip Hybrid)

```bash
# 1. Create conda environment with Python 3.10
conda create -n simenv python=3.10 -c conda-forge -y
conda activate simenv

# 2. Install TensorFlow dependencies (optimized for Apple Silicon)
conda install -c apple tensorflow-deps -y

# 3. Install the rest of the environment via conda-forge
conda install -c conda-forge pybullet=3.2.5 matplotlib pyvirtualdisplay imageio=2.4.0 tqdm -y

# 4. Install Apple TensorFlow and TF-Agents via pip
pip install tensorflow-macos==2.13.0 tensorflow-metal "tf-agents[reverb]"
```

### Linux & Intel macOS — Pure Pip Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

pip install \
  tensorflow==2.13.0 \
  "tf-agents[reverb]" \
  pybullet==3.2.5 \
  matplotlib \
  pyvirtualdisplay \
  imageio==2.4.0 \
  tqdm
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/jakedugi/simulation_enviroments.git
cd simulation_enviroments

# If using pip:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run CartPole (quick test)
python examples/train_cartpole.py

# Run Minitaur (slow, requires GPU + PyBullet)
python examples/train_minitaur.py
```

---

## Project Structure
```bash
simulation_enviroments/
├── requirements.txt       # Python 3.10 / TF 2.13 compatible
├── examples/              # One-line training scripts
│   ├── train_cartpole.py
│   └── train_minitaur.py
├── notebooks/             # Colab-friendly demos
│   ├── CartPole_DQN_demo.ipynb
│   └── Minitaur_SAC_demo.ipynb
├── src/
│   ├── config.py          # Hyperparameter dataclasses
│   ├── cli.py             # CLI entrypoint: train --agent dqn|sac
│   ├── agents/            # tf-agents glue code (DQN, SAC)
│   ├── environments/      # OpenAI Gym / PyBullet wrappers
│   ├── experience/        # Reverb replay buffer wrapper
│   ├── training/          # Trainer class + progress callbacks
│   └── utils/             # Logging, seed control, misc I/O
```

---

## Notebooks for Exploration

Want visuals or a step-by-step walkthrough?
Check out:

	•	notebooks/CartPole_DQN_demo.ipynb
	•	notebooks/Minitaur_SAC_demo.ipynb


