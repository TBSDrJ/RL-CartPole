"""
Dr. J, March 2024

Minimal Reinforcement Learning example, following
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training
"""

from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# 1. Environment
py_env = suite_gym.load('CartPole-v0')
tf_env = TFPyEnvironment(py_env)
