from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(slots=True)
class DQNConfig:
    env_name: str = "CartPole-v1"
    num_iterations: int = 20_000
    initial_collect_steps: int = 100
    collect_steps_per_iteration: int = 1
    replay_buffer_max_length: int = 100_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    fc_layer_params: tuple[int, int] = (100, 50)
    log_interval: int = 200
    eval_interval: int = 1_000
    num_eval_episodes: int = 10
    seed: int = 42
    model_dir: Path = ROOT / "runs" / "dqn"

@dataclass(slots=True)
class SACConfig:
    env_name: str = "MinitaurBulletEnv-v0"
    num_iterations: int = 100_000
    initial_collect_steps: int = 10_000
    collect_steps_per_iteration: int = 1
    replay_buffer_max_length: int = 10_000
    batch_size: int = 256
    actor_fc: tuple[int, int] = (256, 256)
    critic_fc: tuple[int, int] = (256, 256)
    learning_rate: float = 3e-4
    log_interval: int = 5_000
    eval_interval: int = 10_000
    num_eval_episodes: int = 20
    seed: int = 42
    model_dir: Path = ROOT / "runs" / "sac"