import gymnasium as gym
from tf_agents.environments import suite_gym, tf_py_environment

def make(env_name: str = "CartPole-v1"):
    """Return paired (train_env, eval_env) as TFPyEnvironments"""
    train_py = suite_gym.load(env_name)
    eval_py  = suite_gym.load(env_name)
    return tf_py_environment.TFPyEnvironment(train_py), tf_py_environment.TFPyEnvironment(eval_py)

__all__ = ["make"]