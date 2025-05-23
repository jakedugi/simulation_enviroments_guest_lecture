import argparse
from src.config import DQNConfig, SACConfig
from src.environments.cartpole import make as make_cartpole
from src.environments.minitaur import make as make_minitaur
from src.agents.dqn import DQNFactory
from src.agents.sac import SACFactory
from src.training.trainer import RLTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "sac"], required=True)
    args = parser.parse_args()

    if args.agent == "dqn":
        cfg = DQNConfig()
        trainer = RLTrainer(DQNFactory, make_cartpole, cfg)
    else:
        cfg = SACConfig()
        trainer = RLTrainer(SACFactory, make_minitaur, cfg)

    trainer.train()

if __name__ == "__main__":
    main()