"""
Dr. J, March 2024

Minimal Reinforcement Learning example, following
https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training
"""
import random
import tensorflow as tf
import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents import trajectories
import tensorflow.keras.optimizers.legacy as optimizers

# Supress Warnings
tf.get_logger().setLevel('ERROR')

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
# The first argument here is a description of the required format for the
#    description of one step of the agent in the environment. The pre-built
#    environments have an attribute that contains that information for us.
replay_buffer = TFUniformReplayBuffer (
    agent.collect_data_spec,
    batch_size = 32, # Wild guess
    max_length = 1000, # Following example
)

# 4. Observer
# Notice that, in the sample code, they put this in a list.  I am waiting,
#    to be clearer that a list is required as an arg later.
observer = replay_buffer.add_batch

# 5. Policy
collect_policy = agent.collect_policy

# 6. Driver
# This driver is not the same as the Deep Q tutorial, this one seems simpler
#    than the PyDriver used there.
driver = DynamicStepDriver(
    tf_env,
    collect_policy,
    observers = [observer],
    num_steps = 10, # Following example, not sure why/how to pick
)

# Gotta get the iterator started with two initial batches. Why two? All the 
#    examples I see with the cartpole problem use num_steps=2 when building
#    the dataset from the replay buffer.  As far as I can tell, that's because 
#    that's how many steps are needed for the training process, you just need 
#    to see what the state is before and after one action to decide if you've 
#    made a good move or a bad move.
# Because we don't have reverb, there doesn't seem to be a convenience
#    function to produce batches for us. So, we have to build it from the
#    ground up, which is never easy.
# So, a batch always has to be in a specific format. The pre-built environments
#    come with attributes that describe the 'data_spec' of the environment,
#    which includes a listing of the inputs and what format they are in.  There
#    are ___Spec objects in tf to represent these descriptions as Python
#    objects and to facilitate assertion-based type checking.
# Notice that, in 3. above, we set up the Replay Buffer to require the spec
#    agent.collect_data_spec.  So, print this out so we can see the format.
# print(agent.collect_data_spec)
# Strangely, agent.collect_data_spec is not actually a ___Spec object, it is 
#    Trajectory object with several ___Spec objects inside it.
# There are 7 fields in the agent.collect_data_spec. The first 2 come from   
#    the state of the environment *before* the action is taken, the last 3  
#    come from the state of the environment *after* the action is taken.
#    c.f. https://www.tensorflow.org/agents/api_docs/python/tf_agents/trajectories/Trajectory

# So, building up the object:
# tf_env.reset() shows the state of the environment before.
# print(tf_env.reset())
# tf_env.step(1) shows the state of the environment after, using action=1.
# print(tf_env.step(1))
# Arg 1: step_type: this is a class variable that is just an enumerated 
#    type with 3 possible values: 0 = FIRST, 1 = MID, 2 = LAST. But 0 is
#    not an int, it's an np.ndarray, so use the class variable. 
# Arg 2: observation: This comes directly out of the environment as-is,
#    using the environment state from before the action.
# Arg 3: action: In cartpole, this is either 0 (left) or 1 (right). However,
#    we have to submit a tf tensor, using a data type recognizable by tf.
#    The spec calls for a one-dimensional, length 1 tensor of type int64.
# Arg 4: policy_info, I'm leaving this as an empty tuple for now because
#    this will build with that in there, and I'm not sure what the data 
#    type of the object actually is -- the docs refer to it as 'an
#    arbitrary nest.' There is a tf.nest submodule, it seems to be a tf
#    adaptation of a dictionary, but I'm not sure of the details yet.
# Arg 5: next_step_type: same as 1, but this is after the first action is
#    taken, so we go from FIRST before to MID here.
# Arg 6: reward: This comes directly out of the environment as-is,
#    using the environment state from after the action.
# Arg 7: discount: Similar to reward.
before = tf_env.reset()
action = tf.constant([random.randrange(0,2)], dtype=tf.int64)
after_1 = tf_env.step(action)
# Four of these got a tf.reshape() applied to them. Coming out of the 
#    environment, they were represented as 1-dimensional tensors with
#    length of 1, except the observation, which was a 2-d tensor of
#    shape (1, 4), so it also had an extra dimension more than needed.
#    If I submitted a single trajectory as a batch, this was great,
#    and the build went through because it treated this dim as the 
#    batch dimension.  But, when I tried to build a batch, this acted like
#    a second batch dimension, which doesn't make sense, so it caused a
#    crash. So, I needed to remove a dimension, making them into 0-dim tensors
#    instead, then the batch dimension is the only dimension provided.
#    It's possible that I could have rebuilt the TensorSpec arguments in the
#    agent.collect_data_spec to include that dimenstion, but I don't know if
#    that would have caused other problems elsewhere.
# Also note: When I tried to submit a single trajectory as a batch, with no
#    batch dimension, I needed to make the StepTypes into 1-d tensors, because
#    right now they are 0-d np.ndarrays.
# print(action)
# print(tf.reshape(action, ()))
# print(before.observation)
# print(tf.reshape(before.observation, (4)))
traj_1 = trajectories.Trajectory(
    trajectories.StepType.FIRST,
    tf.reshape(before.observation, (4)),
    tf.reshape(action, ()),
    (),
    trajectories.StepType.MID,
    tf.reshape(after_1.reward, ()),
    tf.reshape(after_1.discount, ()),
)
print(traj_1)
# c.f. https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#writing_to_the_buffer
batch_1 = tf.nest.map_structure(lambda t: tf.stack([t] * 32), traj_1)
# print(batch_1)
# print(agent.collect_data_spec)
replay_buffer.add_batch(batch_1)

action = tf.constant([random.randrange(0,2)], dtype=tf.int64)
after_2 = tf_env.step(action)
traj_2 = trajectories.Trajectory(
    trajectories.StepType.MID,
    tf.reshape(after_1.observation, (4)),
    tf.reshape(action, ()),
    (),
    trajectories.StepType.MID,
    tf.reshape(after_2.reward, ()),
    tf.reshape(after_2.discount, ()),
)
print(traj_2)
batch_2 = tf.nest.map_structure(lambda t: tf.stack([t] * 32), traj_2)
replay_buffer.add_batch(batch_1)

# 7. Dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size = 32, # Matching value chosen earlier
    num_steps = 2, 
    single_deterministic_pass = False, # following warning
)

# This iterator produces batches of data as the network trains.
iterator = iter(dataset)

# So, now, after all that jazz, we have two batches that have no 
#    decision-making policy. But we also know a lot more about how to 
#    make batches.  That means that the next() function can finally be 
#    called without throwing an error!
trajectories = next(iterator)
print(trajectories)

# for _ in range(10):

