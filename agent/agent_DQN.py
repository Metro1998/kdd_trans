""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

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
from Selector_phase import Selector


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

        self.memory = deque(maxlen=10000)
        self.learning_start = 90
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
        self.with_priortiy = 1
        self.selector = 0

        self.dense_d = 32
        self.action_space = 8

        self.model = self._build_model()
        self.model.compile(Adam(self.learning_rate), 'mse')
        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.load_model(dir="model/dqn_warm_up",step=4)
        self.target_model = self._build_model()
        self.target_model.compile(Adam(self.learning_rate), 'mse')
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
    @staticmethod
    def extract_queue_and_delay_in_road(agent_id_list, agents, roads: dict, infos, observation):
        # get queue in roads of agent
        queue_and_delay_in_road = {}
        delay_of_agent = {}
        now_simulation_time = observation["42266617929_lane_speed"][0]
        for i in range(1, 6045):
            # the former [0,0] storage queue,the latter storage delay
            queue_and_delay_in_road[i] = [[0, 0], [0, 0]]
        for key, val in infos.items():
            road_id = infos[key]["road"][0]
            lane_id = infos[key]["drivable"][0]
            if roads[road_id]["length"] - infos[key]["distance"][0] < roads[road_id][
                "speed_limit"] * 20:
                delay = (now_simulation_time - infos[key]["start_time"][0]) / infos[key]["t_ff"][0]
                if lane_id == road_id * 100 + 0:
                    queue_and_delay_in_road[int(road_id)][0][0] += 1
                    queue_and_delay_in_road[int(road_id)][1][0] += delay
                elif lane_id == road_id * 100 + 1:
                    queue_and_delay_in_road[int(road_id)][0][1] += 1
                    queue_and_delay_in_road[int(road_id)][1][1] += delay
                else:
                    pass
            else:
                pass
        for i in range(1, 6045):
            if queue_and_delay_in_road[i][0][0] != 0:
                queue_and_delay_in_road[i][1][0] /= queue_and_delay_in_road[i][0][0]
            if queue_and_delay_in_road[i][0][1] != 0:
                queue_and_delay_in_road[i][1][1] /= queue_and_delay_in_road[i][0][1]
            # print("delay_in_road{}:{} ".format(i, queue_and_delay_in_road[i][1]))
        queue_of_agent = {}
        for agent_id in agent_id_list:
            queue_of_agent[agent_id] = []
            delay_of_agent[agent_id] = []
            inroads_of_agent = agents[agent_id][0:4]
            for inroad_id in inroads_of_agent:
                if inroad_id == -1:
                    queue_of_agent[agent_id].extend([0, 0])
                    delay_of_agent[agent_id].extend([0, 0])
                else:
                    queue_of_agent[agent_id].extend(queue_and_delay_in_road[int(inroad_id)][0])
                    delay_of_agent[agent_id].extend(queue_and_delay_in_road[int(inroad_id)][1])
        # print(delay_of_agent[44051539069])
        return queue_of_agent, delay_of_agent

    @staticmethod
    def extract_delay(agent_id_list: list, agents: dict, roads: dict, observation: dict):
        # delay of per lane = 1-(lane_speed/speed_limit)
        delay_in_road = {}  # dict: {agnet_id: list of 8}
        for agent_id in agent_id_list:
            delay_in_road[agent_id] = [0] * 8
            inroads_of_agent = agents[agent_id][0:4]
            speed_limit_per_lane = []
            for inroad_id in inroads_of_agent:
                if inroad_id == -1:
                    speed_limit_per_lane.extend([0] * 2)
                else:
                    speed_limit_per_lane.extend([roads[inroad_id]["speed_limit"]] * 2)
            lane_speed = []
            for i in range(0, 12, 3):
                lane_speed.extend(observation["{}_lane_speed".format(agent_id)][i + 1:i + 3])
            for i in range(0, 8, 1):
                if lane_speed[i] not in [-1, -2]:
                    delay_in_road[agent_id][i] = 1 - (lane_speed[i] / speed_limit_per_lane[i])
        # print(delay_in_road[44051539069])
        return delay_in_road

    @staticmethod
    def extract_traffic_density(agent_id_list: list, agents, roads, observation: dict):
        traffic_density = {}
        for agent_id in agent_id_list:
            traffic_density[agent_id] = []
            inroads_of_agent = agents[agent_id][0:4]
            for i in range(0, 12, 3):
                traffic_density[agent_id].extend(observation["{}_lane_vehicle_num".format(agent_id)][i + 1:i + 3])
            i = 0
            for inroad_id in inroads_of_agent:
                if inroad_id == -1:
                    traffic_density[agent_id][i] = 0
                    traffic_density[agent_id][i + 1] = 0
                else:
                    traffic_density[agent_id][i] /= (roads[inroad_id]["length"] / 1000)
                    traffic_density[agent_id][i + 1] /= (roads[inroad_id]["length"] / 1000)
                i += 2
        return traffic_density

    def extract_state(self, agent_id_list: list, agents: dict, roads: dict, infos: dict, observation: dict):
        # Define our state
        # delay : a list of 8 in one agent
        delay_in_road = self.extract_delay(agent_id_list, agents, roads, observation)
        # get queue of per lane in a specific length
        queue_in_road, delay2_in_road = self.extract_queue_and_delay_in_road(agent_id_list, agents, roads, infos,
                                                                             observation)
        traffic_density = self.extract_traffic_density(agent_id_list, agents, roads, observation)
        observations_for_agent = {}
        for observations_agent_id in agent_id_list:
            # observations_for_agent:The first eight are the number of vehicles in the lane,
            # The middle eight are the lane density,The last is now_phase.
            observations_for_agent[observations_agent_id] = []
            observations_for_agent[observations_agent_id].extend(
                self.normalization(traffic_density[observations_agent_id]))
            observations_for_agent[observations_agent_id].extend(
                self.normalization(queue_in_road[observations_agent_id]))
            observations_for_agent[observations_agent_id].extend(
                self.normalization(delay_in_road[observations_agent_id]))
            observations_for_agent[observations_agent_id].extend(
                self.normalization(delay2_in_road[observations_agent_id]))
            observations_for_agent[observations_agent_id].append(self.now_phase[observations_agent_id])
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
        observations = obs["observations"]
        actions = {}

        # Get state
        observations_for_agent = self.extract_state(self.agent_list, self.agents, self.roads, info, observations)

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
                                                                                        self.action_space, str(phase))
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
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    @staticmethod
    def _reshape_ob(ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def transfer_weights(self):
        """ Transfer Weights from Model to Target at rate Tau
        """
        W = self.model.get_weights()
        tgt_W = self.target_model.get_weights()
        for i in range(len(W)):
            tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
        self.target_model.set_weights(tgt_W)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append([ob, action, reward, next_ob])

    def replay(self):
        # Update the Q network from the memory buffer.
        minibatch = self._sample_memory()
        obs, actions, rewards, next_obs = [np.stack(x) for x in np.array(minibatch).T]
        target = rewards + self.gamma * np.amax(self.target_model.predict([next_obs]), axis=1)
        target_f = self.model.predict([obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit([obs], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _sample_memory(self):
        if self.with_priortiy == 0:
            if self.batch_size > len(self.memory):
                sampled_memory = self.memory
            else:
                sampled_memory = random.sample(self.memory, self.batch_size)
        else:
            sample_weight = []
            for i in range(len(self.memory)):
                obs, action, reward, next_obs = self.memory[i]
                next_estimated_q = self._get_next_estimated_q(next_obs)
                total_reward = reward + self.gamma * next_estimated_q
                target = self.model.predict([[obs]])
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)
            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)), weights=priority, k=self.batch_size)
            sampled_memory = np.array(self.memory)[p]
        return sampled_memory

    def _get_next_estimated_q(self, next_obs):
        action = np.argmax(self.model.predict([[next_obs]]), axis=1)[0]
        next_estimated_q = self.target_model.predict([[next_obs]])[0][action]
        return next_estimated_q

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

    @staticmethod
    def normalization(data):
        data = np.array(data)
        _range = np.max(data) - np.min(data)
        if np.max(data) == 0:
            return data
        else:
            return (data - np.min(data)) / _range

    @staticmethod
    def standardization(data):
        data = np.array(data)
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
