import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras

tf.get_logger().setLevel('ERROR')
import numpy as np


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
        model.add(keras.layers.LSTM(128, return_sequences=False, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(num_actions))
    return model


class ResDQN(keras.Model):
    def __init__(self, num_actions=3, is_rnn=False, is_dueling=False):
        super(ResDQN, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, (8, 8), strides=4, padding="same", kernel_initializer="he_normal",
                                         activation="relu",
                                         )
        self.pooling1 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.resconv1 = keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal",
                                            activation="relu",
                                            )
        self.resconv2 = keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.pooling2 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.resconv3 = keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal",
                                            activation="relu",
                                            )
        self.resconv4 = keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.pooling3 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.flatten = keras.layers.Flatten()
        if is_rnn:
            self.lstm = keras.layers.LSTMCell(128, activation="relu")
        self.fc1 = keras.layers.Dense(512, activation="relu")
        self.fc2 = keras.layers.Dense(num_actions)
        if is_dueling:
            self.value_approx1 = keras.layers.Dense(512, activation="relu")
            self.value_approx2 = keras.layers.Dense(1)
        self.is_rnn = is_rnn
        self.is_dueling = is_dueling
        self.num_actions = num_actions

    def get_initial_state(self, batch_size=4):
        return self.lstm.get_initial_state(batch_size=batch_size, dtype=np.float32)

    def call(self, x, training=True):
        initial_state = self.get_initial_state()
        x = self.conv1(x)
        x = self.pooling1(x)
        original = x
        x = self.resconv1(x)
        x = self.resconv2(x)
        x = tf.concat((x, original), axis=-1)
        x = self.pooling2(x)
        original = x
        x = self.resconv3(x)
        x = self.resconv4(x)
        x = tf.concat((x, original), axis=-1)
        x = self.pooling3(x)
        x = self.flatten(x)
        if self.is_rnn:
            outputs = tf.unstack(tf.expand_dims(x, axis=1))  # list of (1, 256)
            core_outputs = list()
            for output in outputs:
                core_output, core_state = self.lstm(output, initial_state)
                core_outputs.append(core_output)
                initial_state = core_state

            x = tf.compat.v1.layers.flatten(tf.stack(core_outputs))  # (4, 128 * 4)?
        action = self.fc2(self.fc1(x))
        if self.is_dueling:
            value = self.value_approx2(self.value_approx1(x))
            value = tf.broadcast_to(value, shape=(x.shape[0], self.num_actions))
            Q = action + value - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(action, 1), axis=1),
                                                 shape=(tf.shape(x)[0], self.num_actions))
            return Q
        return action


class Duel_DQN(keras.Model):
    def __init__(self, num_actions=3, is_rnn=False, is_noisy=False):
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
        if is_noisy:
            in_features = 128 if self.is_rnn else 64
            self.fc1_action = keras.Sequential([noisy_dense(in_features, 512),
                                                keras.layers.ReLU()])
            self.fc2_action = noisy_dense(512, self.num_actions)
            self.fc1_value = keras.Sequential([noisy_dense(in_features, 512),
                                               keras.layers.ReLU()])
            self.fc2_value = noisy_dense(512, 1)
        else:
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


class Duel_DQN_Unrolled(keras.Model):
    def __init__(self, num_actions=3):
        super(Duel_DQN_Unrolled, self).__init__()
        self.num_actions = num_actions
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
        self.lstm = keras.layers.LSTMCell(256)
        self.fc1_action = keras.layers.Dense(512, activation="relu")
        self.fc2_action = keras.layers.Dense(self.num_actions)
        self.fc1_value = keras.layers.Dense(512, activation="relu")
        self.fc2_value = keras.layers.Dense(1)

    def get_initial_state(self, batch_size=4):
        return self.lstm.get_initial_state(batch_size=batch_size, dtype=np.float32)

    def call(self, x, training=True):
        initial_state = self.get_initial_state()
        first_conv = self.conv1(x)
        first_pooling = self.pooling1(first_conv)
        second_conv = self.conv2(first_pooling)
        second_pooling = self.pooling2(second_conv)
        third_conv = self.conv3(second_pooling)
        outputs = self.pooling3(third_conv)  # (4, 64, 2, 2)
        outputs = self.flatten(outputs)  # (4, 256)
        outputs = tf.unstack(tf.expand_dims(outputs, axis=1))  # list of (1, 256)
        core_outputs = list()
        for output in outputs:
            core_output, core_state = self.lstm(output, initial_state)
            core_outputs.append(core_output)
            initial_state = core_state

        core_output = tf.compat.v1.layers.flatten(tf.stack(core_outputs))  # (4, 128 * 4)?
        action = self.fc1_action(core_output)
        action = self.fc2_action(action)
        value = self.fc1_value(core_output)
        value = tf.broadcast_to(self.fc2_value(value), shape=(x.shape[0], self.num_actions))
        Q = action + value - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(action, 1), axis=1),
                                             shape=(tf.shape(x)[0], self.num_actions))
        return Q


class noisy_dense(keras.Model):
    def __init__(self, in_features, units):
        super(noisy_dense, self).__init__()
        self.units = units
        w_shape = [self.units, in_features]
        mu_w = tf.Variable(initial_value=tf.random.truncated_normal(shape=w_shape))
        sigma_w = tf.Variable(initial_value=tf.constant(0.017, shape=w_shape))
        epsilon_w = tf.random.uniform(shape=w_shape)

        b_shape = [self.units]
        mu_b = tf.Variable(initial_value=tf.random.truncated_normal(shape=b_shape))
        sigma_b = tf.Variable(initial_value=tf.constant(0.017, shape=b_shape))
        epsilon_b = tf.random.uniform(shape=b_shape)

        self.w = tf.add(mu_w, tf.multiply(sigma_w, epsilon_w))
        self.b = tf.add(mu_b, tf.multiply(sigma_b, epsilon_b))

    def call(self, x, training=True):
        return tf.matmul(x, tf.transpose(self.w)) + self.b


class GoExplore(keras.Model):
    def __init__(self, num_actions=3, memsize=800):
        super(GoExplore, self).__init__()
        self.num_actions = num_actions
        self.memsize = memsize
        self.conv1 = keras.layers.Conv2D(64, kernel_size=(8, 8), strides=4,
                                         kernel_initializer=self.ortho_init(),
                                         bias_initializer=tf.constant_initializer(),
                                         activation="relu")
        self.conv2 = keras.layers.Conv2D(128, kernel_size=(4, 4), strides=2,
                                         kernel_initializer=self.ortho_init(),
                                         bias_initializer=tf.constant_initializer(),
                                         activation="relu")
        self.conv3 = keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1,
                                         kernel_initializer=self.ortho_init(),
                                         bias_initializer=tf.constant_initializer(),
                                         activation="relu")
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(memsize,
                                      kernel_initializer=self.normc_init(),
                                      bias_initializer=tf.constant_initializer())
        self.layernorm = keras.layers.LayerNormalization(center=False, scale=False)
        self.gru = keras.layers.GRUCell(memsize,
                                        kernel_initializer=self.normc_init(),
                                        bias_initializer=tf.constant_initializer())
        self.rnn_block = keras.layers.RNN(self.gru, return_sequences=True,
                                          return_state=True,
                                          time_major=False)
        self.action = keras.layers.Dense(num_actions,
                                         kernel_initializer=self.normc_init(0.01),
                                         bias_initializer=tf.constant_initializer())
        self.value = keras.layers.Dense(1,
                                        kernel_initializer=self.normc_init(0.01),
                                        bias_initializer=tf.constant_initializer())

    def normc_init(self, std=1.0, axis=0):
        """
        Initialize with normalized columns
        """

        # noinspection PyUnusedLocal
        def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
            return tf.constant(out)

        return _initializer

    def ortho_init(self, scale=1.0):
        # noinspection PyUnusedLocal
        def _ortho_init(shape, dtype, partition_info=None):  # pylint: disable=W0613
            shape = tuple(shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4:  # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v  # pick the one with the correct shape
            q = q.reshape(shape)
            return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

        return _ortho_init

    def get_initial_state(self, batch_size=4):
        return self.gru.get_initial_state(batch_size=batch_size, dtype=np.float32)

    def call(self, x, training=True):
        batch_size = x.shape[0]
        init_state = self.get_initial_state(batch_size)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv_result = self.flatten(x)
        fc1_result = keras.activations.relu(self.layernorm(self.fc1(conv_result)))
        reshape = tf.reshape(fc1_result, [1, -1, self.memsize])

        gru_output, gru_state = self.rnn_block(reshape, initial_state=init_state, training=training)
        fc2 = tf.concat([tf.reshape(gru_output, [batch_size, self.memsize]), fc1_result], axis=1)  # (4, 1600)
        pi = self.action(fc2)
        vf = tf.broadcast_to(self.value(fc2), shape=(batch_size, self.num_actions))
        Q = pi + vf - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(pi, 1), axis=1),
                                      shape=(tf.shape(x)[0], self.num_actions))
        return Q


if __name__ == "__main__":
    # inputs = tf.random.normal([4, 1, 256])
    # inputs = tf.unstack(inputs)
    # rnn = tf.keras.layers.LSTMCell(128)
    # states = rnn.get_initial_state(batch_size=1, dtype=np.float32)
    # for input_ in inputs:
    #     print(input_.shape)
    #     output, core_states = rnn(input_, states)
    #     print(output.shape)
    model = GoExplore()
    model.build((4, 88, 86, 1))
    inputs = tf.random.normal((4, 88, 86, 1))
    print(model(inputs))
