from src.config import SACConfig
from src.environments.minitaur import make as make_envs
from src.agents.sac import SACFactory
from src.training.trainer import RLTrainer

if __name__ == "__main__":
    cfg = SACConfig()
    trainer = RLTrainer(SACFactory, make_envs, cfg)
    trainer.train()