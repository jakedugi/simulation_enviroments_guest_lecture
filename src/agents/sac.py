from __future__ import annotations
import tensorflow as tf
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network, critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.utils import common

class SACFactory:
    """Builds a tf-agents SAC agent from a config + environment specs."""
    def __init__(self, config, train_env):
        self.cfg = config
        self._train_env = train_env
        self.agent = self._build()

    def _build(self):
        obs_spec = self._train_env.observation_spec()
        act_spec = self._train_env.action_spec()
        ts_spec  = self._train_env.time_step_spec()

        critic = critic_network.CriticNetwork(
            (obs_spec, act_spec),
            joint_fc_layer_params=self.cfg.critic_fc
        )
        actor = actor_distribution_network.ActorDistributionNetwork(
            obs_spec, act_spec,
            fc_layer_params=self.cfg.actor_fc,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork
        )

        optimizer = lambda: tf.keras.optimizers.Adam(self.cfg.learning_rate)
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        agent = sac_agent.SacAgent(
            ts_spec, act_spec,
            actor_network=actor,
            critic_network=critic,
            actor_optimizer=optimizer(),
            critic_optimizer=optimizer(),
            alpha_optimizer=optimizer(),
            target_update_tau=0.005,
            target_update_period=1,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=0.99,
            reward_scale_factor=1.0,
            train_step_counter=global_step
        )
        agent.initialize()
        return agent