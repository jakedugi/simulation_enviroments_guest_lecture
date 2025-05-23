from src.config import DQNConfig
from src.environments.cartpole import make as make_envs
from src.agents.dqn import DQNFactory
from src.training.trainer import RLTrainer

if __name__ == "__main__":
    cfg = DQNConfig()
    trainer = RLTrainer(DQNFactory, make_envs, cfg)
    trainer.train()