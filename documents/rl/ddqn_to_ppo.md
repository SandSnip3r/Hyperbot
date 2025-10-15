# DDQN to PPO plan

 - Rather than a large replay buffer, there is still a sort of "replay buffer", but it's size is small (around 2k transitions, called "Horizon" or `T`). This replay buffer is flushed after a training step.
 - Rather than the training loop being tight and constantly sampling from the "replay buffer", it will:
  - Wait until `T` transitions are in the buffer
  - Train `num epochs`
  - Flush buffer
  - Note that during training, the actors will need a place to write their transitions to. It will probably be best to, in the train step, immediately copy the "replay buffer" locally, then clear and unlock the global one.
 - Rather than epsilon-greedy exploration, we will always invoke the model for action selection, even in the case when the agent cannot send a packet