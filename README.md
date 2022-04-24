## Introduction

Atari Skiing is a classical Atari game. Developing an AI player is difficult as the reward is only returned at the end of the game. Given the DQN as the baseline, we will try to implement some optimizations and see if there is any improvement. Current methodologies include:

- Adding heuristic data as initialization of replay buffer
- Implementing Double DQN and Dueling DQN
- Using prioritized replay buffer with trick
- Implementing residual DQN
- Implementing unrolled LSTM (DRQN)

