from __future__ import print_function
import gym
import time
import matplotlib.pyplot as plt
import warnings
import logging
from utils import *
from models import *
warnings.filterwarnings("ignore")
import argparse
from model_utils import *
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

    def sample(self, frame_idx, batch_size=20, beta_frames=72000, is_sequential=False):
        self._beta_update(frame_idx, beta_frames)
        prios = self.priorities[:min(self.capacity, len(self.action_history))]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        if is_sequential:
            start_indice = np.random.choice(range(len(self.action_history)), 1, p=probs)
            indices = np.arange(start=start_indice, stop=start_indice+batch_size)
            indices = np.where(indices >= len(self.action_history), indices - len(self.action_history), indices)
        else:
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


def invertible_value_rescale(Q):
    return tf.math.sign(Q) * (tf.math.sqrt(tf.math.abs(Q) + 1) - 1) + 0.01 * Q


def trainer(gamma=0.995,
            batch_size=4,
            learning_rate=0.001,
            max_memory=10800,
            target_update_every=100,
            max_steps_per_episode=3600,
            max_episodes=20,
            update_after_actions=4,
            randomly_update_memory_after_actions=True,
            last_n_reward=100,
            target_avg_reward=-4000,
            double_dqn=False,
            dueling_dqn=False,
            include_flag_punishment=False,
            tao=1.,
            is_noisy=False,
            mixed_loss=0.9,
            is_grad_clip=False,
            is_unrolled=False,
            training=True
            ):
    global action_history, state_history, state_next_history, rewards_history, done_history
    # Model used for selecting actions (principal)
    if dueling_dqn:
        if is_unrolled:
            model = Duel_DQN_Unrolled()
            model_target = Duel_DQN_Unrolled()
        else:
            model = Duel_DQN(is_rnn=True, is_noisy=is_noisy)
            model_target = Duel_DQN(is_rnn=True, is_noisy=is_noisy)

    else:
        model = DQN(is_rnn=True)
        model_target = DQN(is_rnn=True)

    model.build((batch_size, resize_shape[0], resize_shape[1], 1))
    model_target.build((batch_size, resize_shape[0], resize_shape[1], 1))

    if not training:
        return 0, model_target

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-3)
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
    timestep_count = 0

    for episode in range(max_episodes):
        logger.info("epsilon is " + str(epsilon) + ", episode is " + str(episode))
        state = process_state(np.asarray(env.reset()))
        episode_reward = 0
        done = False
        start = time.time()
        for timestep in range(1, max_steps_per_episode):
            timestep_count += 1
            # exploration
            if np.random.random() < epsilon and not is_noisy:
                # Take random action
                action = np.random.choice(3)
            else:
                # Predict action Q-values
                state_t = tf.convert_to_tensor(state.astype(np.float32))
                state_t = tf.expand_dims(state_t, 0)
                action_vals = model(state_t, training=False)
                # Choose the best action
                action = tf.argmax(action_vals[0]).numpy()
            # epsilon = max(0.01, epsilon * 0.995)
            alpha = ((timestep % 128) - 1) / 127
            epsilon = 0.1 ** (alpha + 3 * (1 - alpha))
            # follow selected action
            state_next, reward, done, _ = env.step(action)
            state_next = process_state(state_next)
            if done:
                if include_flag_punishment:
                    # there should be a huge punishment due to not crossing the flags
                    for i in range(len(pb.rewards_history) - timestep, len(pb.rewards_history)):
                        pb.rewards_history[i] += reward / timestep
                    episode_reward += reward
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
                    pb.sample(timestep_count, batch_size, max_episodes * max_steps_per_episode, is_unrolled)

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
                Q_targets = invertible_value_rescale(Q_targets)
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
                    gt = tf.expand_dims(Q_targets, 1)
                    predict = tf.expand_dims(Q_of_actions, 1)
                    loss = loss_function(gt, predict)
                    absolute_loss = tf.abs(gt - predict)
                    pb.update_priorities(indices, absolute_loss.numpy() + 1e-5)
                    loss *= weights
                    try:
                        loss = (1-mixed_loss) * loss.mean() + mixed_loss * loss.max()
                    except:
                        loss = (1-mixed_loss) * tf.math.reduce_mean(loss) + mixed_loss * tf.math.reduce_max(loss)

                # Nudge the weights of the trainable variables towards
                grads = tape.gradient(loss, model.trainable_variables)
                if is_grad_clip:
                    for i, grad in enumerate(grads):
                        if grad is not None:
                            grads[i] = tf.clip_by_norm(grad, 10.)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if timestep_count % target_update_every == 0 or done or timestep + 1 == max_steps_per_episode:
                # soft update the target network with new weights
                local_weights = [tao * weights_online for weights_online in model.get_weights()]
                target_weights = [(1-tao) * weights_target for weights_target in model_target.get_weights()]
                soft_update = [l + t for l, t in zip(local_weights, target_weights)]
                model_target.set_weights(soft_update)
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
            if include_flag_punishment:
                for i in range(len(pb.rewards_history) - timestep, len(pb.rewards_history)):
                    pb.rewards_history[i] -= 10000 / timestep

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
            save_model(model)
            # model = load_model(model)
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
    return running_rewards, model


def plot_rewards(running_rewards):
    plt.figure()
    plt.plot(range(len(running_rewards)), running_rewards, linewidth=2)
    plt.ylabel("average running rewards")
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


def evaluation(model, env, include_flag_punishment, times=10):
    model = load_model(model)
    total_reward = 0
    for _ in range(times):
        count = 0
        state = process_state(env.reset())
        done = False
        while not done and count <= 3600:
            count += 1
            state_t = tf.convert_to_tensor(state)
            state_t = tf.expand_dims(state_t, 0)
            action_vals = model(state_t, training=False)
            action = tf.argmax(action_vals[0]).numpy()
            state_next, reward, done, _ = env.step(action)
            if done:
                if not include_flag_punishment:
                    break
            state = process_state(state_next)
            total_reward += reward
    return total_reward / times


def start(args):
    if args.training:
        for _ in range(args.heuristic):
            heuristic_agent()

    running_rewards, model = trainer(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_update_every=args.target_update_every,
        max_memory=args.max_memory,
        double_dqn=args.double_dqn,
        dueling_dqn=args.dueling,
        include_flag_punishment=args.include_flag_punishment,
        tao=args.tao,
        is_noisy=args.is_noisy,
        is_unrolled=args.is_unrolled,
        training=args.training
    )
    plot_rewards(running_rewards)
    print(evaluation(model, env, args.include_flag_punishment, 2))


if __name__ == "__main__":
    version = tf.__version__[0]
    if version == "1":
        tf.compat.v1.enable_eager_execution()
    envname = "Skiing-v0"  # environment name
    env = gym.make(envname)
    resize_shape = (86, 88)

    parser = argparse.ArgumentParser()
    parser.add_argument("--heuristic",
                        type=int,
                        help="number of heuristic replay as initialization",
                        default=5)
    parser.add_argument("--batch_size",
                        type=int,
                        default=4)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5)
    parser.add_argument("--double_dqn",
                        action="store_true")
    parser.add_argument("--dueling",
                        action="store_true")
    parser.add_argument("--target_update_every",
                        type=int,
                        help="target update every n steps",
                        default=100)
    parser.add_argument("--max_memory",
                        type=int,
                        help="max memory size as replay buffer",
                        default=10800)
    parser.add_argument("--include_flag_punishment",
                        help="include flag punishment to reward",
                        action="store_true")
    parser.add_argument("--tao",
                        type=int,
                        help="hyperparameter for soft update",
                        default=0.25)
    parser.add_argument("--is_noisy",
                        action="store_true")
    parser.add_argument("--is_unrolled",
                        action="store_true")
    parser.add_argument("--training",
                        action="store_true")

    args = parser.parse_args()
    if args.include_flag_punishment and args.training:
        logger.info("flag punishment activated! All rewards will be rectified!")
    start(args)
