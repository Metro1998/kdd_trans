import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import random
import numpy as np

"""
utils
"""


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


"""
models
"""

epsilon = 1e-6
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(state))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, 1)
        self.log_std_linear = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.action_scale = 4
        self.action_bias = 4

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # Constrict every element in log_std in to [LOG_SIG_MIN, LOG_SIG_MAX]
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        # first sample from Normal(0,1), then output (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        # (-1, 1)
        # continuous action space
        action = y_t * self.action_scale + self.action_bias
        # Why rescaling? Cause the bound of tanh is (-1, 1), but the bound of action is not.
        log_prob = normal.log_prob(x_t)
        # log_prob(value)是计算value在定义的正态分布中对应的概率的对数
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2) + epsilon))
        log_prob = log_prob.sum(0, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



"""
memory
"""


class ReplayMemory:
    def __init__(self):
        random.seed(123456)
        self.capacity = 100000
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


"""
agent_SAC
"""


class SAC():
    def __init__(self):

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.lr = 0.003

        self.target_update_interval = 1
        self.device = torch.device("cpu")

        # 8 phases
        self.num_inputs = 9
        self.num_actions = 1
        self.hidden_size = 256

        self.critic = QNetwork(self.num_inputs, self.hidden_size).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(self.num_inputs, self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        # Copy the parameters of critic to critic_target

        self.target_entropy = -torch.Tensor([1.0]).to(self.device).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)

        self.policy = GaussianPolicy(self.num_inputs, self.hidden_size).to(
            self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)

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
                    observations_for_agent[observations_agent_id][i + 1] = \
                        vehicle_in_road[inroads_of_agent[int(i / 2)]][1]
                else:
                    pass
            observations_for_agent[observations_agent_id][-1] = self.now_phase[observations_agent_id]
        return observations_for_agent

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.item()

    def _act(self, observations_for_agent):
        actions = {}
        for agent_id in self.agent_list:
            actions[agent_id] = int(self.select_action(observations_for_agent[agent_id]) // 1 + 1)
            self.now_phase[agent_id] = actions[agent_id]
        return actions

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size=batch_size)
        action_batch = np.expand_dims(action_batch, axis=1)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch +  self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optimizer.zero_grad()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # env relative
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.last_change_step = dict.fromkeys(self.agent_list, 0)

    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    def load_model(self, dir="model/sac", step=0):
        critic = "sac_critic_{}.h5".format(step)
        critic_path = os.path.join(dir, critic)
        policy = "sac_policy_{}.h5".format(step)
        policy_path = os.path.join(dir, policy)
        print("loading")
        self.critic.load_state_dict(torch.load(critic_path))
        self.policy.load_state_dict(torch.load(policy_path))

    def save_critic_model(self, dir="model/sac", step=0):
        critic = "sac_critic_{}.h5".format(step)
        critic_path = os.path.join(dir, critic)
        print("saving")
        torch.save(self.critic.state_dict(), critic_path)

    def save_policy_model(self, dir="model/sac", step=0):
        policy = "sac_policy_{}.h5".format(step)
        policy_path = os.path.join(dir, policy)
        print("saving")
        torch.save(self.policy.state_dict(), policy_path)



scenario_dirs = [
    "agent_SAC", "memory"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
agent_specs[scenario_dirs[0]] = SAC()
agent_specs[scenario_dirs[1]] = ReplayMemory()

