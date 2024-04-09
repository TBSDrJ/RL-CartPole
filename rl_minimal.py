"""
Dr. J, March 2024

Minimal Reinforcement Learning example, following
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training
"""
import random
import tensorflow as tf
import numpy as np
import cv2
import PIL
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies import TFPolicy
# from tf_agents.policies.q_policy import QPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents import trajectories
import tensorflow.keras.optimizers.legacy as optimizers

def compute_avg_return(
        env: TFPyEnvironment,
        pol: TFPolicy,
        num_episodes: int = 20,
) -> np.float32:
    """Runs episodes, calculates return.  Max return in CartPole-v0 is 200."""
    total_return = 0.0
    for _ in range(num_episodes):
        episode_ret = 0.0
        time_step = env.reset()
        while not time_step.is_last():
            action_step = pol.action(time_step)
            time_step = env.step(action_step.action)
            episode_ret += time_step.reward
        total_return += episode_ret
    if num_episodes > 0:
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    else:
        return None


# Supress Warnings
tf.get_logger().setLevel('ERROR')

# 1. Environment
py_env = suite_gym.load('CartPole-v1')
tf_env = TFPyEnvironment(py_env)
py_env_2 = suite_gym.load('CartPole-v1')
eval_env = TFPyEnvironment(py_env_2)

# 2. Network/Model/Agent
model = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params = (100, 50),
    # dropout_layer_params = (0.3, 0.3),
)
# lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 25000, 0.1)
lr = 1e-3
train_counter = tf.Variable(0)
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network = model,
    optimizer = optimizers.Adam(learning_rate = lr),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter = train_counter,
)
agent.initialize()

# 3. Replay Buffer
# The first argument here is a description of the required format for the
#    description of one step of the agent in the environment. The pre-built
#    environments have an attribute that contains that information for us.
replay_buffer = TFUniformReplayBuffer (
    data_spec = agent.collect_data_spec,
    batch_size = 1,
    max_length = 1000, 
)

# 5. Policy
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

# 7. Dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size = 64,
    num_steps = 2, 
    num_parallel_calls = 10,
)
iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

# print(tf_env.batch_size)

time_step = tf_env.current_time_step()
# print(time_step)
# quit()
action_step = random_policy.action(time_step)
next_time_step = tf_env.current_time_step()
traj = trajectories.from_transition(time_step, action_step, next_time_step)
# print(traj)
replay_buffer.add_batch(traj)
# quit()
model.summary()
print(compute_avg_return(eval_env, eval_policy))

losses = []
for i in range(20000):
    time_step = tf_env.current_time_step()
    action_step = collect_policy.action(time_step)
    next_time_step = tf_env.step(action_step.action)
    traj = trajectories.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    experience, _ = next(iterator)
    loss = agent.train(experience)
    losses.append(loss.loss)
    while len(losses) > 500:
        losses.pop(0)
    # print(loss)
    if i % 200 == 199:
        if len(losses) > 0:
            loss = sum(losses) / len(losses)
        print(f"{i+1:5} {loss:.8f}", flush=True)
    if i % 1000 == 999:
        print(compute_avg_return(eval_env, eval_policy))

print()

print(compute_avg_return(eval_env, eval_policy))
