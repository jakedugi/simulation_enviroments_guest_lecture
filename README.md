# simulation_enviroments

Simple implementation of SAC and DQN in Minitaur enviroment PyBullet for quadroped. And Cartpole in OpenAI Gym.

# Simulation Environments — Guest-Lecture Edition

A **minimal, idiomatic** reference for teaching reinforcement-learning
fundamentals:

* **CartPole (DQN)** — instant feedback; converges in ±2 min on CPU.
* **Minitaur (SAC)** — continuous-control showcase; runs on GPU or CPU overnight.

Originally delivered as a guest lecture in *Applied ML 2025*  

## Quick start

```bash
git clone https://github.com/your-github/simulation_enviroments.git
cd simulation_enviroments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train CartPole
python examples/train_cartpole.py

# Train Minitaur (GPU recommended)
python examples/train_minitaur.py
