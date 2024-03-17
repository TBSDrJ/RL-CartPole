"""
Dr. J, March 2024

Minimal Reinforcement Learning example, following
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training
"""

from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
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

# 3. Replay Buffer
replay_buffer = TFUniformReplayBuffer (
    agent.collect_data_spec,
    batch_size = 32, # Wild guess
    max_length = 1000, # Following example
)

# 4. Observer
# Notice that, in the sample code, they put this in a list.  I am waiting,
#    to be clearer that a list is required as an arg later.
observer = replay_buffer.add_batch