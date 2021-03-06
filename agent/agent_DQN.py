""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0]
import sys

sys.path.append(path)
import random

import gym

from pathlib import Path
import pickle
import gym

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model


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

        self.memory = deque(maxlen=3000)
        self.learning_start = 1000
        self.update_model_freq = 5
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97
        self.learning_rate = 0.005
        self.batch_size = 32
        self.ob_length = 9

        self.action_space = 8

        self.model = self._build_model()

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = self._build_model()
        self.update_target_network()

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.last_change_step = dict.fromkeys(self.agent_list, 0)

    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    ################################
    def extract_state(self, agent_id_list: list, agents: dict, roads: dict, infos: dict):
        # Define our state
        vehicle_in_road = {}
        for i in range(1, 6045):
            vehicle_in_road[i] = [0, 0, 0]
        for key, val in infos.items():
            road_id = infos[key]["road"][0]
            lane_id = infos[key]["drivable"][0]
            if roads[road_id]["length"] - infos[key]["distance"][0] < roads[road_id][
                "speed_limit"] * 10:
                if lane_id == road_id * 100 + 0:
                    vehicle_in_road[int(road_id)][0] += 1
                elif lane_id == road_id * 100 + 1:
                    vehicle_in_road[int(road_id)][1] += 1
                elif lane_id == road_id * 100 + 2:
                    vehicle_in_road[int(road_id)][2] += 1
            else:
                pass
        observations_for_agent = {}
        for observations_agent_id in agent_id_list:
            # the last zero is now_phase
            observations_for_agent[observations_agent_id] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            inroads_of_agent = agents[observations_agent_id][0:4]
            for i in range(0, 8, 2):
                if inroads_of_agent[int(i / 2)] != -1:
                    observations_for_agent[observations_agent_id][i] = vehicle_in_road[inroads_of_agent[int(i / 2)]][0]
                    observations_for_agent[observations_agent_id][i + 1] = vehicle_in_road[inroads_of_agent[int(i / 2)]][1]
                else:
                    pass
            observations_for_agent[observations_agent_id][-1] = self.now_phase[observations_agent_id]
        return observations_for_agent

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.
        actions = {}
        # Get actions
        for agent_id in self.agent_list:
            actions[agent_id] = self.get_action(observations_for_agent[agent_id]) + 1  # a list of 1 * 8
            self.now_phase[agent_id] = actions[agent_id]
        return actions

    def act(self, obs):
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = self.extract_state(self.agent_list, self.agents, self.roads, info)

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]) + 1
            self.now_phase[agent] = actions[agent]
        return actions

    def get_action(self, ob):

        # The epsilon-greedy action selector.

        if np.random.rand() <= self.epsilon:
            return self.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict([ob])
        return np.argmax(act_values[0])

    def sample(self):

        # Random samples

        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model

        model = Sequential()
        model.add(Dense(20, input_dim=self.ob_length, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop()
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        target = rewards + self.gamma * np.amax(self.target_model.predict([next_obs]), axis=1)
        target_f = self.model.predict([obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit([obs], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
    # agent_specs[k].load_model(dir="model/dqn_warm_up",step=49)