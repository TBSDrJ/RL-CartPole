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
from tf_agents.policies.q_policy import QPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents import trajectories
import tensorflow.keras.optimizers.legacy as optimizers

def compute_avg_return(
        env: TFPyEnvironment,
        pol: QPolicy,
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
py_env = suite_gym.load('CartPole-v0')
tf_env = TFPyEnvironment(py_env)
py_env_2 = suite_gym.load('CartPole-v0')
eval_env = TFPyEnvironment(py_env_2)

# 2. Network/Model/Agent
model = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params = (100, 50),
    # dropout_layer_params = (0.9, 0.7, 0.5),
)
# lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 25000, 0.1)
lr = 0.001
train_counter = tf.Variable(0)
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network = model,
    optimizer = optimizers.Adam(learning_rate = lr),
    td_errors_loss_fn=tf.keras.losses.MeanSquaredError(),
    train_step_counter = train_counter,
)
agent.initialize()

# 3. Replay Buffer
# The first argument here is a description of the required format for the
#    description of one step of the agent in the environment. The pre-built
#    environments have an attribute that contains that information for us.
replay_buffer = TFUniformReplayBuffer (
    data_spec = agent.collect_data_spec,
    batch_size = 64,
    max_length = 100000, 
)

# 4. Observer
# Notice that, in the sample code, they put this in a list.  I am waiting,
#    to be clearer that a list is required as an arg later.
observer = replay_buffer.add_batch

# 5. Policy
eval_policy = agent.policy
collect_policy = agent.collect_policy
# q_policy = QPolicy(
#     tf_env.time_step_spec(),
#     tf_env.action_spec(),
#     q_network = model,
# )

# 6. Driver
# This driver is not the same as the Deep Q tutorial, this one seems simpler
#    than the PyDriver used there.
# driver = DynamicStepDriver(
#     tf_env,
#     q_policy,
#     observers = [observer],
#     num_steps = 1, # Following example, not sure why/how to pick
# )

# 7. Dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size = 64, # Matching value chosen earlier
    num_steps = 2, 
    num_parallel_calls = 10,
).prefetch(3)
iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

model.summary()
print(compute_avg_return(eval_env, eval_policy))

# # print(q_policy.policy_step_spec)
# step_0 = tf_env.reset()
# # print(f"Reset: {step}")
# step_1, _ = driver.run()
# # print(f"Next: {next_step}")
# step_2, _ = driver.run()
# traj = Trajectory(
#     tf.reshape([step_0.step_type, step_1.step_type], (1,2)),
#     tf.reshape([step_0.observation, step_1.observation], [1,2,4]),
#     tf.reshape(tf.constant([
#             q_policy.action(step_0).action.numpy()[0], 
#             q_policy.action(step_1).action.numpy()[0]
#             ], 
#             dtype=tf.int64), (1,2)),
#     (),
#     tf.reshape([step_1.step_type, step_2.step_type], (1,2)),
#     tf.reshape([step_1.reward, step_2.reward], (1,2)),
#     tf.reshape([step_1.discount, step_2.discount], (1,2)),
# )
# # print(f"Initial Trajectory: {traj}")

for _ in range(100):
    time_step = tf_env.current_time_step()
    action_step = collect_policy.action(time_step)
    next_time_step = tf_env.current_time_step()
    traj = trajectories.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

losses = []
for i in range(50000):
    time_step = tf_env.current_time_step()
    action_step = collect_policy.action(time_step)
    next_time_step = tf_env.current_time_step()
    traj = trajectories.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)
    experience, _ = next(iterator)
    loss = agent.train(experience)
    losses.append(loss.loss * 10000)
    while len(losses) > 500:
        losses.pop(0)
    # print(loss)
    if i % 100 == 0:
        if len(losses) > 0:
            loss = sum(losses) / len(losses)
        print(f"{i:5} {loss:.8f}", flush=True)
    if i % 1000 == 0:
        print(compute_avg_return(eval_env, eval_policy))

print()

print(compute_avg_return(eval_env, eval_policy))

# total = 0
# length = 20
# for i in range(length):
#     step = tf_env.reset()
#     img = py_env.render()
#     cv2.imshow(f"{i}", img)
#     cv2.waitKey(0)
#     j = 0
#     while step.step_type < 2:
#         step, _ = driver.run()
#         j += 1
#         img = py_env.render()
#         cv2.imshow(f"{i}", img)
#         cv2.waitKey(1)
#     print(i, j, end = " | ", flush=True)
#     total += j
# print(total/length)

# Taken from training loop
    # step_0 = step_1
    # step_1 = step_2
    # step_2, pol_info = driver.run()
    # # print(f"Policy {i}: {pol_info}")
    # traj = Trajectory(
    #     tf.reshape([step_0.step_type, step_1.step_type], (1,2)),
    #     tf.reshape([step_0.observation, step_1.observation], [1,2,4]),
    #     tf.reshape(tf.constant([
    #             q_policy.action(step_0).action.numpy()[0], 
    #             q_policy.action(step_1).action.numpy()[0]
    #             ], 
    #             dtype=tf.int64), (1,2)),
    #     (),
    #     tf.reshape([step_1.step_type, step_2.step_type], (1,2)),
    #     tf.reshape([step_1.reward, step_2.reward], (1,2)),
    #     tf.reshape([step_1.discount, step_2.discount], (1,2)),
    # )
    # print(traj)
    # print(traj.observation)