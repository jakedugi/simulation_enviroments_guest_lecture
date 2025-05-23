from __future__ import annotations
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.utils import common


def _dense(units: int):
    return tf.keras.layers.Dense(
        units, activation="relu",
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

class DQNFactory:
    """Builds a tf-agents DQN agent from a config + environment specs"""
    def __init__(self, config, train_env):
        self.cfg = config
        self._train_env = train_env
        self.agent = self._build()

    def _build(self):
        action_spec     = self._train_env.action_spec()
        n_actions       = action_spec.maximum - action_spec.minimum + 1
        dense_layers    = [_dense(u) for u in self.cfg.fc_layer_params]
        q_values_layer  = tf.keras.layers.Dense(n_actions, activation=None)
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(self.cfg.learning_rate)
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        agent = dqn_agent.DqnAgent(
            self._train_env.time_step_spec(),
            action_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=global_step
        )
        agent.initialize()
        return agent