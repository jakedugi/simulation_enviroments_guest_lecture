from __future__ import annotations
from pathlib import Path
import tensorflow as tf
from tqdm.auto import trange

from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy, random_tf_policy
from tf_agents.metrics import py_metrics
from tf_agents.utils import common

from ..experience.replay import ReplayBuffer
from ..utils.io import set_seed, colour_text


class RLTrainer:
    """Base class – handles seed, checkpointing & evaluation loop."""
    def __init__(self, agent_factory, env_fn, cfg):
        set_seed(cfg.seed)
        self.train_env, self.eval_env = env_fn()
        self.cfg = cfg
        self.agent  = agent_factory(cfg, self.train_env).agent

        # Replay
        self.replay = ReplayBuffer(self.agent.collect_data_spec,
                                   max_len=cfg.replay_buffer_max_length)
        self.dataset = iter(self.replay.dataset(cfg.batch_size))

        # Policies
        self.collect_policy = self.agent.collect_policy
        self.eval_policy    = self.agent.policy
        self.random_policy  = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(), self.train_env.action_spec())

        # Driver for random seeding
        py_driver.PyDriver(
            env=self.train_env.pyenv.envs[0],
            policy=py_tf_eager_policy.PyTFEagerPolicy(self.random_policy, True),
            observers=[self.replay.observer],
            max_steps=cfg.initial_collect_steps
        ).run(self.train_env.pyenv.envs[0].reset())

        # TT / compile
        self.agent.train = common.function(self.agent.train)
        self.global_step = self.agent.train_step_counter

        # Logging
        self.returns: list[float] = []
        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    def train(self):
        """Main loop no side effects besides logs + checkpoints."""
        avg_return = self._evaluate()
        self.returns.append(avg_return)

        driver = py_driver.PyDriver(
            env=self.train_env.pyenv.envs[0],
            policy=py_tf_eager_policy.PyTFEagerPolicy(self.collect_policy, True),
            observers=[self.replay.observer],
            max_steps=self.cfg.collect_steps_per_iteration
        )

        iterator = trange(self.cfg.num_iterations, desc="Training", unit="iter")
        for _ in iterator:
            driver.run(self.train_env.pyenv.envs[0].current_time_step())
            experience, _ = next(self.dataset)
            loss = self.agent.train(experience).loss

            step = int(self.global_step.numpy())
            if step % self.cfg.log_interval == 0:
                iterator.set_postfix(loss=f"{loss:.3f}")

            if step % self.cfg.eval_interval == 0:
                avg_return = self._evaluate()
                self.returns.append(avg_return)
                iterator.write(colour_text(f"Step {step:,}: "
                                           f"Average Return = {avg_return:.2f}"))

    def _evaluate(self):
        total = 0.0
        for _ in range(self.cfg.num_eval_episodes):
            ts = self.eval_env.reset()
            episode_return = 0.0
            while not ts.is_last():
                action = self.eval_policy.action(ts)
                ts = self.eval_env.step(action.action)
                episode_return += ts.reward
            total += episode_return
        return (total / self.cfg.num_eval_episodes).numpy()[0]