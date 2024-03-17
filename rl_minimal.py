"""
Dr. J, March 2024

Minimal Reinforcement Learning example, following
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training
"""

from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import DqnAgent
import tensorflow.keras.optimizers.legacy as optimizers

# 1. Environment
py_env = suite_gym.load('CartPole-v0')
tf_env = TFPyEnvironment(py_env)

# 2. Network/Model/Agent
model = QNetwork(
    tf_env.time_step_spec().observation,
    tf_env.action_spec(),
    fc_layer_params = (100, 50), # This is what is used in the larger Deep Q ex
)
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network = model,
    optimizer = optimizers.Adam(learning_rate = 0.001)
)
