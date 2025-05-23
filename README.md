# Simulation Environments — Guest Lecture Edition

Minimal, testable, and production-style RL training loops built for teaching.
Implements:

	•	CartPole (DQN) — discrete control baseline; converges on CPU in ~2 minutes.
 
	•	Minitaur (SAC) — continuous control for quadrupeds via PyBullet; GPU recommended.

Originally developed for a 2025 guest lecture in Applied Machine Learning.

---

## Quick start
```bash
# Clone and set up environment
git clone https://github.com/your-github/simulation_enviroments.git
cd simulation_enviroments
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train CartPole (fast)
python examples/train_cartpole.py

# Train Minitaur (slow, requires PyBullet and GPU)
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

 ---
