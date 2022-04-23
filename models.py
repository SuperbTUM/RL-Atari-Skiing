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
        self.resconv1 = keras.layers.Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer="he_normal",
                                            activation="relu",
                                            )
        self.resconv2 = keras.layers.Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.pooling2 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.resconv3 = keras.layers.Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer="he_normal",
                                            activation="relu",
                                            )
        self.resconv4 = keras.layers.Conv2D(32, (3, 3), strides=1, padding="same", kernel_initializer="he_normal")
        self.pooling3 = keras.layers.MaxPool2D((2, 2), strides=2)
        self.flatten = keras.layers.Flatten()
        if is_rnn:
            self.Lambda = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))
            self.lstm = keras.layers.LSTM(128, activation="relu")
        self.fc1 = keras.layers.Dense(512, activation="relu")
        self.fc2 = keras.layers.Dense(num_actions)
        if is_dueling:
            self.value_approx1 = keras.layers.Dense(512, activation="relu")
            self.value_approx2 = keras.layers.Dense(1)
        self.is_rnn = is_rnn
        self.is_dueling = is_dueling
        self.num_actions = num_actions

    def call(self, x, training=True):
        x = self.conv1(x)
        x = self.pooling1(x)
        original = x
        x = self.resconv1(x)
        x = self.resconv2(x)
        x += original
        x = self.pooling2(x)
        original = x
        x = self.resconv3(x)
        x = self.resconv4(x)
        x += original
        x = self.pooling3(x)
        x = self.flatten(x)
        if self.is_rnn:
            x = self.lstm(self.Lambda(x))
        action = self.fc2(self.fc1(x))
        if self.is_dueling:
            value = self.value_approx2(self.value_approx1(x))
            value = tf.broadcast_to(value, shape=(x.shape[0], self.num_actions))
            Q = action + value - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(action, 1), axis=1),
                                                 shape=(tf.shape(x)[0], self.num_actions))
            return Q
        return action


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
        self.lstm = keras.layers.LSTMCell(128, activation="relu")
        self.fc1_action = keras.layers.Dense(512, activation="relu")
        self.fc2_action = keras.layers.Dense(self.num_actions)
        self.fc1_value = keras.layers.Dense(512, activation="relu")
        self.fc2_value = keras.layers.Dense(1)

    def get_initial_state(self, batch_size=1):
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
        outputs = tf.unstack(tf.expand_dims(outputs, axis=1))   # list of (1, 256)
        core_outputs = list()
        for output in outputs:
            core_output, core_state = self.lstm(output, initial_state)
            core_outputs.append(core_output)
            initial_state = core_state

        core_output = tf.squeeze(tf.stack(core_outputs), axis=1)  # (4, 1, 128)
        action = self.fc1_action(core_output)
        action = self.fc2_action(action)
        value = self.fc1_value(core_output)
        value = tf.broadcast_to(self.fc2_value(value), shape=(x.shape[0], self.num_actions))
        Q = action + value - tf.broadcast_to(tf.expand_dims(tf.math.reduce_mean(action, 1), axis=1),
                                             shape=(tf.shape(x)[0], self.num_actions))
        return Q


if __name__ == "__main__":
    inputs = tf.random.normal([4, 1, 256])
    inputs = tf.unstack(inputs)
    rnn = tf.keras.layers.LSTMCell(128)
    states = rnn.get_initial_state(batch_size=1, dtype=np.float32)
    for input_ in inputs:
        print(input_.shape)
        output, core_states = rnn(input_, states)
        print(output.shape)
