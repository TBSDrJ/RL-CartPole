"""
https://towardsdatascience.com/cartpole-problem-using-tf-agents-build-your-first-reinforcement-learning-application-3e6006adeba7
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents import trajectories

# Supress Warnings
tf.get_logger().setLevel('ERROR')

num_iterations = 20000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
log_interval = 200
num_eval_episodes = 10
eval_interval = 1000

# variable to store name of the environment
env_name = 'CartPole-v0'

# loading the environment
# env = suite_gym.load(env_name)

# create an empty dataframe
# df = pd.DataFrame(columns=['step_type', 'reward', 'discount', 'observation'])

# initial state
# time_step = env.reset()
# df = df._append(
#     {
#     'step_type': time_step.step_type, 
#     'reward': time_step.reward, 
#     'discount': time_step.discount, 
#     'observation': time_step.observation
#     }, 
#     ignore_index=True
# )

# iterate while the time_step is not the last
# while not time_step.is_last():
#     time_step = env.step(np.array([1]))
#     df = df._append(
#         {
#         'step_type': time_step.step_type, 
#         'reward': time_step.reward, 
#         'discount': time_step.discount, 
#         'observation': time_step.observation
#         }, 
#         ignore_index=True
#     )

# two environments, one for training and one for evaluation
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# fully connected layer architecture
fc_layer_params = (100, )

# defining q network using train_env and fully connected layer architecture
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# adam optimizer to do the training
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

# create agent using the network, specifications, and other parameters
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

# initialize the agent
agent.initialize()

# eval_policy for evaluation and collect_policy for collection
eval_policy = agent.policy
collect_policy = agent.collect_policy

# random policy -- select action from the list of actions randomly
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# utility function to compute average return
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# replay buffer to collect the trajectories
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length
)

# functions to collect data to the replay buffer
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectories.from_transition(time_step, action_step, next_time_step)
    print(traj)
    buffer.add_batch(traj)
    
def collect_data(env, policy, buffer, steps):
    for i in range(steps):
        collect_step(env, policy, buffer)
        print(i)
        
collect_data(train_env, random_policy, replay_buffer, steps=100)

dataset = replay_buffer.as_dataset(
  num_parallel_calls=10, 
  sample_batch_size=batch_size, 
  num_steps=2
).prefetch(3)
iterator = iter(dataset)

# Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# reset the train step
agent.train_step_counter.assign(0)

# evaluate the agent's policy once before training
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]
avg_return_rand = compute_avg_return(eval_env, random_policy, num_eval_episodes)

q_net.summary()
print(f"Average Return with Random Policy = {avg_return_rand:.6f}")
print(f"Initial Average Return = {avg_return:.6f}")

losses = []
for i in range(num_iterations):
  
    # collect few experiences using collect policy and store in the replay buffer    
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    print(i)
        
    # take a batch of data from the replay buffer and train the network
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss
    
    step = agent.train_step_counter.numpy()
    
    losses.append(train_loss.numpy())
    while len(losses) > 500:
        losses.pop(0)

    if step % log_interval == 0:
        print(f"Step = {step:5}: Loss = {sum(losses) / len(losses):10.6f}")
        
    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(f"Average Return = {avg_return:.6f}")
        returns.append(avg_return)
