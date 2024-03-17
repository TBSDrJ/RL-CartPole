# Reinforcement Learning

I am trying to build a minimal reinforcement learning example, solving the cartpole problem.

I started with [Training a Deep Q Network](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial), but that is pretty far from minimal, and I was struggling to minimize it.

Then, in the process of trying to understand how to minimize it, I found this example: [Replay Buffers: Using replay buffers during training](https://www.tensorflow.org/agents/tutorials/5_replay_buffers_tutorial#using_replay_buffers_during_training).

So, I'm going to kick the tires on this more minimal example here first.

Even the most minimal example needs six things.  The docs
present them in a different order, but this is the order they are
needed in the code:
    1. Environment: The context or problem you are training to solve.
    2. Network/Model/Agent: The actual neural network that you are training,
in the form of an agent who actions are being trained.
    3. Replay Buffer: This holds the sequence of actions of the agent.  This 
provides the feed of actions to the model, and can also store the actions so
the programmer can view the actions.
    4. Observer: This collects actions from the replay buffer and feeds them
to the driver.
    5. Policy: This is function that actually makes the decisions for the
agent.
    6. Driver: This connects the replay buffer/observer and the policy to the 
agent.