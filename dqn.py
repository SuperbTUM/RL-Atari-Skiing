from __future__ import print_function
import numpy as np
import gym
import time
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras as keras
tf.get_logger().setLevel('ERROR')

# Our Experience Replay memory
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
priorities = []
log_path = "skiing.log"
logging.basicConfig(filename=log_path, level=logging.INFO,
                    filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
logger = logging.getLogger(__name__)


class PrioritizedBuffer:
    def __init__(self, capacity=1000, prob_alpha=0.6, beta=0.4):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.action_history = action_history[:min(capacity, len(action_history))]
        self.state_history = state_history[:min(capacity, len(state_history))]
        self.state_next_history = state_next_history[:min(capacity, len(state_next_history))]
        self.rewards_history = rewards_history[:min(capacity, len(rewards_history))]
        self.done_history = done_history[:min(capacity, len(done_history))]
        assert len(self.action_history) <= capacity
        assert len(self.state_history) <= capacity
        assert len(self.state_next_history) <= capacity
        assert len(self.rewards_history) <= capacity
        assert len(self.done_history) <= capacity
        if len(priorities) >= capacity:
            self.priorities = np.asarray(priorities[:capacity], dtype=np.float32)
        else:
            self.priorities = np.asarray(priorities + [0 for _ in range(capacity - len(priorities))], dtype=np.float32)
        self.cnt = len(action_history)
        self.beta = beta

    def push(self, state, action, reward, next_state, done):
        max_prio = np.max(self.priorities) if self.state_history else 1.0
        pos = np.argmin(self.priorities) if self.state_history else 0

        if self.cnt < self.capacity:
            self.action_history.append(action)
            self.state_history.append(state)
            self.state_next_history.append(next_state)
            self.rewards_history.append(reward)
            self.done_history.append(done)
            self.cnt += 1
        else:
            self.action_history[pos] = action
            self.state_history[pos] = state
            self.state_next_history[pos] = next_state
            self.rewards_history[pos] = reward
            self.done_history[pos] = done

        self.priorities[pos] = max_prio

    def _beta_update(self, frame_idx, beta_frames=1000):
        self.beta = min(1.0, self.beta + frame_idx * (1.0 - self.beta) / beta_frames)
        return

    def sample(self, frame_idx, batch_size=20, beta_frames=3000):
        self._beta_update(frame_idx, beta_frames)
        prios = self.priorities[:min(self.capacity, len(self.action_history))]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(range(len(self.action_history)), batch_size, p=probs, replace=False)
        state_samples = np.asarray([self.state_history[idx] for idx in indices])
        next_state_samples = np.asarray([self.state_next_history[idx] for idx in indices])
        action_samples = np.asarray([self.action_history[idx] for idx in indices])
        reward_samples = np.asarray([self.rewards_history[idx] for idx in indices])
        done_samples = np.asarray([self.done_history[idx] for idx in indices])

        total = len(self.action_history)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return state_samples, \
               action_samples, reward_samples, next_state_samples, done_samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return self.cnt


def DQN(num_actions=3, is_rnn=False):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(32, (8, 8), strides=4, padding="same", kernel_initializer="he_normal", activation="relu",
                            ))
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))
    model.add(
        keras.layers.Conv2D(64, (4, 4), strides=2, padding="same", kernel_initializer="he_normal", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))
    model.add(
        keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2), strides=2))
    model.add(keras.layers.Flatten())
    if is_rnn:
        model.add(keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)))
        model.add(keras.layers.LSTM(128, return_sequences=True, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(num_actions))
    return model


class Duel_DQN(keras.Model):
    def __init__(self, num_actions=3, is_rnn=False):
        super(Duel_DQN, self).__init__()
        self.num_actions = num_actions
        self.is_rnn = is_rnn
        self.conv1 = keras.layers.Conv2D(32, (8, 8), strides=4, padding="same", kernel_initializer="he_normal",
                                         activation="relu",
                                         )
        self.pooling1 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.conv2 = keras.layers.Conv2D(64, (4, 4), strides=2, padding="same", kernel_initializer="he_normal",
                                          activation="relu"
                                          )
        self.pooling2 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.conv3 = keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal",
                                         activation="relu"
                                         )
        self.pooling3 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.flatten = keras.layers.Flatten()
        if self.is_rnn:
            self.Lambda = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))
            self.lstm = keras.layers.LSTM(128, activation="relu")
        self.fc1_action = keras.layers.Dense(512, activation="relu")
        self.fc2_action = keras.layers.Dense(self.num_actions)
        self.fc1_value = keras.layers.Dense(512, activation="relu")
        self.fc2_value = keras.layers.Dense(1)

    def call(self, x, training=True):
        first_conv = self.conv1(x)
        first_pooling = self.pooling1(first_conv)
        second_conv = self.conv2(first_pooling)
        second_pooling = self.pooling2(second_conv)
        third_conv = self.conv3(second_pooling)
        third_pooling = self.pooling3(third_conv)
        output = self.flatten(third_pooling)
        if self.is_rnn:
            reshaped = self.Lambda(output)
            output = self.lstm(reshaped)
        action = self.fc1_action(output)
        action = self.fc2_action(action)
        value = self.fc1_value(output)
        value = tf.broadcast_to(self.fc2_value(value), shape=(x.shape[0], self.num_actions))
        Q = action + value - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(action, 1), axis=1),
                                             shape=(tf.shape(x)[0], self.num_actions))
        return Q


def heuristic_agent():
    def get_pos_player(observe):
        ids = np.where(np.sum(observe == [214, 92, 92], -1) == 3)
        return ids[0].mean(), ids[1].mean()

    def get_pos_flags(observe):
        if np.any(np.sum(observe == [184, 50, 50], -1) == 3):
            ids = np.where(np.sum(observe == [184, 50, 50], -1) == 3)
            return ids[0].mean(), ids[1].mean()
        else:
            ids = np.where(np.sum(observe == [66, 72, 200], -1) == 3)
            return ids[0].mean(), ids[1].mean()

    def get_speed(observe, observe_old):
        # As the vertical location of the player is not changed,
        # I estimate the vertical speed by measuring how much frames are shifted up.
        min_val = np.inf
        min_idx = 0
        for k in range(0, 7):
            val = np.sum(np.abs(observe[54:-52, 8:152] - observe_old[54 + k:-52 + k, 8:152]))
            if min_val > val:
                min_idx = k
                min_val = val
        return min_idx

    observe = env.reset()
    step = 0
    done = False
    # states
    r_a, c_a = get_pos_player(observe)
    r_f, c_f = get_pos_flags(observe)
    r_a_old, c_a_old = r_a, c_a
    observe_old = observe
    while not done:
        step += 1
        v_f = np.arctan2(r_f - r_a, c_f - c_a)  # direction from player to target
        spd = get_speed(observe, observe_old)
        v_a = np.arctan2(spd, c_a - c_a_old)  # speed vector of the player
        r_a_old, c_a_old = r_a, c_a
        observe_old = observe
        if spd == 0 and (c_a - c_a_old) == 0:
            # no movement
            act = np.random.choice(3, 1)[0]
        else:
            if v_f - v_a < -0.1:
                act = 1
            elif v_f - v_a > 0.1:
                act = 2
            else:
                act = 0

        observe, reward, done, _ = env.step(act)
        state_next = process_state(observe)
        state = process_state(observe_old)
        if not done:
            action_history.append(act)
            rewards_history.append(reward)
            state_next_history.append(state_next)
            state_history.append(state)
            done_history.append(done)
            priorities.append(max(priorities) if priorities else 1.0)

        r_a, c_a = get_pos_player(observe)
        r_f, c_f = get_pos_flags(observe)
    return


def process_state(state, ratio=0.6):
    state = Image.fromarray(state[28:-35, 8:152, :]).convert('L')
    preprocessed_state = np.asarray(state)
    preprocessed_state = np.where(preprocessed_state >= 180, 236, preprocessed_state)
    preprocessed_state = Image.fromarray(preprocessed_state)
    preprocessed_state = preprocessed_state.resize(
        (int(preprocessed_state.size[0] * ratio), int(preprocessed_state.size[1] * ratio)), Image.LANCZOS)
    assert preprocessed_state.size[0] >= 80
    preprocessed_state = preprocessed_state.filter(ImageFilter.EDGE_ENHANCE_MORE)
    state = np.expand_dims(np.asarray(preprocessed_state), axis=-1)
    return state.astype("float32")


def trainer(gamma=0.1,
            batch_size=4,
            learning_rate=0.001,
            max_memory=3600,
            target_update_every=100,
            max_steps_per_episode=3600,
            max_episodes=1000,
            update_after_actions=4,
            randomly_update_memory_after_actions=True,
            last_n_reward=100,
            target_avg_reward=-4000,
            double_dqn=False,
            dueling_dqn=False
            ):
    global action_history, state_history, state_next_history, rewards_history, done_history
    # Model used for selecting actions (principal)
    if dueling_dqn:
        model = Duel_DQN(is_rnn=False)
        # Then create the target model. This will periodically be copied from the principal network
        model_target = Duel_DQN(is_rnn=False)
    else:
        model = DQN(is_rnn=True)
        model_target = DQN(is_rnn=True)

    model.build((batch_size, resize_shape[0], resize_shape[1], 1))
    model_target.build((batch_size, resize_shape[0], resize_shape[1], 1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    reduction = tf.keras.losses.Reduction.NONE if version == "2" else tf.losses.Reduction.NONE
    loss_function = keras.losses.Huber(
        reduction=reduction)  # You can use the Huber loss function or the mean squared error

    running_reward = 0
    episode_count = 0
    episode_reward_history = []
    static_update_after_actions = update_after_actions
    no_improvement = 0

    # how often to train your model - this allows you to speed up learning
    # by not performing in every iteration learning. See also reference paper
    # you can set this value to other values like 1 as well to learn every time

    epsilon = .3
    running_rewards = list()
    pb = PrioritizedBuffer(capacity=max_memory)

    for episode in range(max_episodes):
        logger.info("epsilon is " + str(epsilon) + ", episode is " + str(episode))
        state = process_state(np.asarray(env.reset()))
        episode_reward = 0
        timestep_count = 0
        done = False
        start = time.time()
        for timestep in range(1, max_steps_per_episode):
            timestep_count += 1
            # exploration
            if np.random.random() < epsilon:
                # Take random action
                action = np.random.choice(3)
            else:
                # Predict action Q-values
                state_t = tf.convert_to_tensor(state.astype(np.float32))
                state_t = tf.expand_dims(state_t, 0)
                action_vals = model(state_t, training=False)
                # Choose the best action
                action = tf.argmax(action_vals[0]).numpy()
            epsilon = max(0.01, epsilon * 0.995)
            # follow selected action
            state_next, reward, done, _ = env.step(action)
            state_next = process_state(state_next)
            if done:
                # # there should be a huge punishment due to not crossing the flags
                # for i in range(len(pb.rewards_history) - timestep_count, len(pb.rewards_history)):
                #     pb.rewards_history[i] += reward / timestep_count
                pass
            else:
                episode_reward += reward

            # Save action/states and other information in replay buffer
            pb.push(state, action, reward, state_next, done)
            # action_history.append(action)
            # state_history.append(state)
            # state_next_history.append(state_next)
            # rewards_history.append(reward)
            # done_history.append(done)

            state = state_next

            # Update every Xth frame to speed up (optional)
            # and if you have sufficient history
            if randomly_update_memory_after_actions:
                update_after_actions = np.random.choice(
                    range(static_update_after_actions // 2, static_update_after_actions + 1))
            if timestep_count % update_after_actions == 0 and len(action_history) > batch_size:
                #  Sample a set of batch_size memories from the history
                # state_history = np.asarray(state_history)
                # state_next_history = np.asarray(state_next_history)
                # rewards_history = np.asarray(rewards_history)
                # action_history = np.asarray(action_history)
                # done_history = np.asarray(done_history)
                #
                # idx = np.random.choice(range(len(state_history)), batch_size, False)
                #
                # state_sample = state_history[idx]
                # state_next_sample = state_next_history[idx]
                # rewards_sample = rewards_history[idx]
                # action_sample = action_history[idx]
                # done_sample = done_history[idx]
                #
                # state_history = state_history.tolist()
                # state_next_history = state_next_history.tolist()
                # rewards_history = rewards_history.tolist()
                # action_history = action_history.tolist()
                # done_history = done_history.tolist()
                state_sample, action_sample, rewards_sample, state_next_sample, done_sample, indices, weights = \
                    pb.sample(timestep, batch_size)

                state_next_sample = tf.convert_to_tensor(state_next_sample)

                if not double_dqn:
                    # Create for the sample states the targets (r+gamma * max Q(...) )
                    Q_next_state = model_target.predict(state_next_sample, batch_size)
                    Q_targets = rewards_sample + gamma * tf.reduce_max(Q_next_state, axis=-1)

                else:
                    max_Q_index = tf.argmax(model.predict(state_next_sample, batch_size), axis=1)
                    Q_next_target = torch_gather(model_target.predict(state_next_sample, batch_size),
                                                 tf.expand_dims(max_Q_index, axis=1), 1)
                    Q_targets = rewards_sample + gamma * tf.squeeze(Q_next_target)
                # If the episode was ended (done_sample value is 1)
                # you can penalize the Q value of the target by some value `penalty`

                # What actions are relevant and need updating
                relevant_actions = tf.one_hot(action_sample, 3)
                # we will use Gradient tape to do a custom gradient
                # in the `with` environment we will record a set of operations
                # and then we will take gradients with respect to the trainable parameters
                # in the neural network
                with tf.GradientTape() as tape:
                    # Train the model on your action selecting network
                    q_values = model(tf.convert_to_tensor(state_sample, dtype=np.float32))
                    # We consider only the relevant actions
                    Q_of_actions = tf.reduce_sum(tf.multiply(q_values, relevant_actions), axis=1)
                    # Calculate loss between principal network and target network
                    loss = loss_function(tf.expand_dims(Q_targets, 1), tf.expand_dims(Q_of_actions, 1))
                    pb.update_priorities(indices, loss.numpy() + 1e-5)
                    try:
                        loss = loss.mean()
                    except:
                        loss = tf.math.reduce_mean(loss)

                # Nudge the weights of the trainable variables towards
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if timestep_count % target_update_every == 0 or done or timestep_count + 1 == max_steps_per_episode:
                # update the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}, epsilon {}"
                logger.info(template.format(running_reward, episode_count, timestep_count, epsilon))
            # Don't let the memory grow beyond the limit
            # if len(rewards_history) > max_memory:
            #     del rewards_history[:len(rewards_history)-max_memory]
            #     del state_history[:len(state_history)-max_memory]
            #     del state_next_history[:len(state_next_history)-max_memory]
            #     del action_history[:len(action_history)-max_memory]
            #     del done_history[:len(done_history)-max_memory]
            if done: break
        if not done:
            # for i in range(len(pb.rewards_history) - timestep_count, len(pb.rewards_history)):
            #     pb.rewards_history[i] -= 10000 / timestep_count
            pass

        # reward of last n episodes
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > last_n_reward: del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        # early stopping
        if running_rewards and running_rewards[-1] > running_reward:
            no_improvement += 1
            if no_improvement >= 3:
                break
        else:
            no_improvement = 0
        running_rewards.append(running_reward)
        episode_count += 1
        # If you want to stop your training once you achieve the reward you want you can
        # have an if statement here. Alternatively you can stop after a fixed number
        # of episodes.
        if running_reward > target_avg_reward:
            break

        end = time.time()
        logger.info("time per episode {:.4f} seconds".format(end - start))
    return running_rewards


def plot_rewards(running_rewards):
    plt.figure()
    plt.plot(range(len(running_rewards)), running_rewards, linewidth=2)
    plt.ylabel("average running rewards")
    plt.imshow()
    plt.show()


def torch_gather(x, indices, gather_axis):

    # create a tensor containing indices of each element
    all_indices = tf.where(tf.fill(indices.shape, True))
    gather_locations = tf.reshape(indices, [indices.shape.num_elements()])

    # splice in our pytorch style index at the correct axis
    gather_indices = []
    for axis in range(len(indices.shape)):
        if axis == gather_axis:
            gather_indices.append(gather_locations)
        else:
            gather_indices.append(all_indices[:, axis])

    gather_indices = tf.stack(gather_indices, axis=-1)
    gathered = tf.gather_nd(x, gather_indices)
    reshaped = tf.reshape(gathered, indices.shape)
    return reshaped


if __name__ == "__main__":
    version = tf.__version__[0]
    if version == "1":
        tf.compat.v1.enable_eager_execution()
    envname = "Skiing-v0"  # environment name
    env = gym.make(envname)
    resize_shape = (86, 88)
    play_times = 5
    for _ in range(play_times):
        heuristic_agent()

    running_rewards = trainer(double_dqn=True, dueling_dqn=True)
    plot_rewards(running_rewards)
