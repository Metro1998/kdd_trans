# _*_ coding：utf-8_*_
# 开发人员：zm
# 开发时间：15:02
# 文件名：123.py
# 开发工具：PyCharm
import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os

import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0]
print(path)
import sys

sys.path.append(path)
import random

from pathlib import Path
import pickle
# import gym

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Multiply, Add
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda
from keras.models import Model
import keras.backend as K
from agent.Selector_phase  import Selector


# contains all of the intersections


class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        self.memory = deque(maxlen=500000)
        self.learning_start = 180
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.tau = 1e-2

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.8  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01

        self.batch_size = 512
        self.ob_length = 33
        self.with_priortiy = 0
        self.selector = 0

        self.dense_d = 32
        self.action_space = 8

        self.model = self._build_model()
        self.model.compile(Adam(self.learning_rate), 'mse')

        self.selector = 0
        self.ob_length = 33
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        length = (self.ob_length,)
        inp = Input((length))
        cur_phase = inp[-1]
        x = Dense(64, activation='relu')(inp)
        if self.selector == 0:
            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)
            x = Dense(self.action_space + 1, activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                       output_shape=(self.action_space,))(x)
            return Model(inp, x)
        else:
            list_selected_q_values = []
            for phase in range(1, 1 + self.action_space):
                locals()["q_value_{}".format(phase)] = self._separate_network_structure(x, self.dense_d,
                                                                                        self.action_space,
                                                                                        str(phase))
                locals()["selector_{0}".format(phase)] = Selector(
                    phase, name="selector_{0}".format(phase))(cur_phase)
                locals()["q_values_{0}_selected".format(phase)] = Multiply(name="multiply_{0}".format(phase))(
                    [locals()["q_values_{0}".format(phase)],
                     locals()["selector_{0}".format(phase)]]
                )
                list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase)])
            q_values = list_selected_q_values
            return Model(inp, q_values)

    @staticmethod
    def _separate_network_structure(input, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(input)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(
            hidden_1)
        return q_values

    @staticmethod
    def _reshape_ob(ob):
        return np.reshape(ob, (1, -1))

    def _get_next_estimated_q(self, next_obs):
        print(self.model.predict([next_obs]))
        action = np.argmax(self.model.predict([next_obs]), axis=1)[0]
        return action

agent = TestAgent()
action = agent._get_next_estimated_q([[1]*33,[2]*33])
print(action)

"""
class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

b = Input(shape=(1,))
a = Selector(select= 1)(b[-1])
print(a)

"""