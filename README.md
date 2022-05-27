## Introduction

Atari Skiing is a classical Atari game. Developing an AI player is difficult as the reward is only returned at the end of the game. Given the DQN as the baseline, we will try to implement some optimizations and see if there is any improvement. Current methodologies include:

- Introducing heuristic data as initialization of replay buffer
- Implementing D3QN + Noise (optional)
- Cyclic epsilon 
- Using prioritized replay buffer prioritizing with absolute TD error
- Implementing residual DQN
- Implementing unrolled LSTM (DRQN, Sequential Sampling)
- Rescaling Q function
- Soft update on target model
- Gradient clipping (optional)

## Prerequisites

numpy <= 1.19.5

## Quick Start

As we are building a playground, 
there are several ways of starting experiments.
One example could be as follows:

`python .\dqn.py --double_dqn --dueling --is_rnn --learning_rate 0.001 --training --include_flag_punishment`

## Checkpoints

[Google Drive](https://drive.google.com/drive/folders/18hl2xdMD7aV94PpG5acj40TfFrUeYGZl?usp=sharing)

