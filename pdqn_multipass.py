import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from memory.memory import Memory
from utils import soft_update_target_network, hard_update_target_network
from utils.noise import OrnsteinUhlenbeckActionNoise

import requests
import time
import threading
import subprocess
import json
import statistics
import os
import datetime
import math
from copy import deepcopy


print(datetime.datetime.now())

# request rate r
data_rate = 30      # if not use_tm
use_tm = 0  # if use_tm
result_dir = "./MPDQN_result/"

# initial setting (threshold setting) # no use now
# T_max = 0.065  # t_max violation
# T_min = 0.055
set_tmin = 1  # 1 if setting tmin
# cpus = 0.5  # initial cpus
# replicas = 1  # initial replica

## initial
request_num = []
# timestamp    : 0, 1, 2, 31, ..., 61, ..., 3601
# learning step:          0,  ..., 1,     , 120
#
simulation_time = 3602  # 300 s  # 0 ~ 3601:  3600
request_n = simulation_time

## global variable
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request
RFID = 0  # random number for data
event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_Ccontrol = threading.Event()
# Need modify ip if ip change
ip = "192.168.99.121"  # app_mn1
ip1 = "192.168.99.122"  # app_mn2
error_rate = 0.2  # 0.2/0.5
Rmax_mn1 = 30
Rmax_mn2 = 20


## Learning parameter
# S ={k, u , c, r}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 4       # Total episodes
learning_rate = 0.01          # Learning rate
# max_steps = 50               # Max steps per episode
# Exploration parameters
gamma = 0.9                 # Discounting rate
max_epsilon = 1
min_epsilon = 0.1
epsilon_decay = 1/120
memory_size = 100
batch_size = 8
target_update = 100

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)


## 7/8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    f = open('/home/user/flask_test/client/request/request6.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(simulation_time)]


print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name="app_mn1"):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['-r', '-1', '0', '1', 'r']
        self.n_actions = len(self.action_space)

        # Need modify ip if container name change
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        cmd = "sudo docker-machine ssh default docker stack rm app"
        subprocess.check_output(cmd, shell=True)
        cmd1 = "sudo docker-machine ssh default docker stack deploy --compose-file docker-compose.yml app"
        subprocess.check_output(cmd1, shell=True)
        time.sleep(60)

    def get_response_time(self):

        path1 = result_dir + self.service_name + "_response.txt"

        f1 = open(path1, 'a')
        RFID = random.randint(0, 1000000)
        headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
        data = {
            "m2m:cin": {
                "con": "true",
                "cnf": "application/json",
                "lbl": "req",
                "rn": str(RFID + 1000),
            }
        }
        # URL
        service_name_list = ["app_mn1", "app_mn2"]
        url = self.url_list[service_name_list.index(self.service_name)]
        try:
            start = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=0.05)
            response = response.status_code
            end = time.time()
            response_time = end - start
        except requests.exceptions.Timeout:
            response = "timeout"
            response_time = 0.05

        data1 = str(timestamp) + ' ' + str(response) + ' ' + str(response_time) + ' ' + str(self.cpus) + ' ' + str(self.replica) + '\n'
        f1.write(data1)
        f1.close()
        if str(response) != '201':
            response_time = 0.05

        return response_time

    def get_cpu_utilization(self):
        path = result_dir + self.service_name + '_cpu.txt'
        try:
            f = open(path, "r")
            cpu = []
            time = []
            for line in f:
                s = line.split(' ')
                time.append(float(s[0]))
                cpu.append(float(s[2]))

            last_avg_cpu = statistics.mean(cpu[-5:])
            f.close()

            return last_avg_cpu
        except:

            print('cant open')

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action_index, event, done):
        global timestamp, send_finish, change, simulation_time

        action = self.action_space[action_index]
        if action == '-r':
            if self.replica > 1:
                self.replica -= 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == '-1':
            if self.cpus >= 0.5:
                self.cpus -= 0.1
                self.cpus = round(self.cpus, 1)  # ex error:  0.7999999999999999
                change = 1
                cmd = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == '1':
            if self.cpus < 1:
                self.cpus += 0.1
                self.cpus = round(self.cpus, 1)
                change = 1
                cmd = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == 'r':
            if self.replica < 3:
                self.replica += 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if self.service_name == 'app_mn1':
            time.sleep(10)  # wait app_mn1 service start
        time.sleep(30)  # wait service start

        if not done:
            # print(self.service_name, "_done: ", done)
            # print(self.service_name, "_step complete")
            event.set()

        response_time_list = []
        time.sleep(20)
        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())

        if done:
            # print(self.service_name, "_done: ", done)
            time.sleep(10)
            event.set()  # if done and after get_response_time
        # mean_response_time = sum(response_time_list)/len(response_time_list)
        # print(response_time_list)
        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms
        t_max = 0
        if mean_response_time >= 50:
            Rt = 50
        else:
            Rt = mean_response_time
        if self.service_name == "app_mn1":
            t_max = Rmax_mn1
        elif self.service_name == "app_mn2":
            t_max = Rmax_mn2


        tmp_d = math.exp(50 / t_max)
        tmp_n = math.exp(Rt / t_max)
        c_perf = tmp_n / tmp_d

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        path = result_dir + self.service_name + "_agent_get_cpu.txt"
        f1 = open(path, 'a')
        data = str(timestamp) + ' ' + str(self.cpu_utilization) + '\n'
        f1.write(data)
        f1.close()
        u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(u/10/self.cpus)
        next_state.append(self.cpus)
        next_state.append(request_num[timestamp])
        # state.append(req)

        # cost function
        w_pref = 0.5
        w_res = 0.5
        c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)
        c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res

class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # initialise layer weights
        for i in range(0, len(self.layers) - 1):
            nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers - 1):
            x = F.relu(self.layers[i](x))

        Q = self.layers[-1](x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=True,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.squashing_function = squashing_function
        self.activation = activation
        if init_type == "normal":
            assert init_std is not None and init_std > 0
        assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # initialise layer weights
        for i in range(0, len(self.layers)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
            elif init_type == "normal":
                nn.init.normal_(self.layers[i].weight, std=init_std)
            else:
                raise ValueError("Unknown init_type "+str(init_type))
            nn.init.zeros_(self.layers[i].bias)
        if output_layer_init_std is not None:
            nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        else:
            nn.init.zeros_(self.action_parameters_output_layer.weight)
        nn.init.zeros_(self.action_parameters_output_layer.bias)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        # self.action_parameters_passthrough_layer.requires_grad = False
        # self.action_parameters_passthrough_layer.weight.requires_grad = False
        # self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function "+str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)

        if self.squashing_function:
            # assert False  # scaling not implemented yet
            action_params = action_params.tanh()
            action_params = action_params * self.action_param_lim
        return action_params


class PDQNAgent:
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """
    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        # super(PDQNAgent, self).__init__(observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.num_actions = self.action_space.spaces[0].n
        self.action_parameter_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1,self.num_actions+1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        print([self.action_space.spaces[i].high for i in range(1, self.num_actions+1)])
        self.action_parameter_max_numpy = np.concatenate([self.action_space.spaces[i].high for i in range(1, self.num_actions+1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate([self.action_space.spaces[i].low for i in range(1, self.num_actions+1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        print(self.num_actions+self.action_parameter_size)
        self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1+self.action_parameter_size,), next_actions=False)
        self.actor = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_kwargs).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space.shape[0], self.num_actions, self.action_parameter_size, **actor_param_kwargs).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        print("initial_weights.shape: ", initial_weights.shape)
        print("passthrough_layer.weight.data.size() : ", passthrough_layer.weight.data.size())
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            print("initial_bias.shape:", initial_bias.shape)
            print("passthrough_layer.bias.data.size(): ", passthrough_layer.bias.data.size())
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)

        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = self.np_random.choice(self.num_actions)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                              self.action_parameter_max_numpy))
            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters[offset:offset+self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters


    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        act, all_action_parameters = action
        self._step += 1

        # self._add_sample(state, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward, next_state, terminal)
        self._add_sample(state, np.concatenate(([act], all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), terminal=terminal)
        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()  # Select the corresponding action q value
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------

        action_params = self.actor_param(states)
        Q = self.actor(states, action_params)
        Q_val = Q
        Q_loss = -torch.mean(torch.sum(Q_val, 1))

        self.actor_param_optimiser.zero_grad()
        Q_loss.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):

        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):

        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')


def store_cpu(start_time, woker_name):
    global timestamp, cpus, change, reset_complete

    cmd = "sudo docker-machine ssh " + woker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
    while True:

        if send_finish == 1:
            break
        if change == 0 and reset_complete == 1:
            returned_text = subprocess.check_output(cmd, shell=True)
            my_data = returned_text.decode('utf8')
            # print(my_data.find("CPUPerc"))
            my_data = my_data.split("}")
            # state_u = []
            for i in range(len(my_data) - 1):
                # print(my_data[i]+"}")
                my_json = json.loads(my_data[i] + "}")
                name = my_json['Name'].split(".")[0]
                cpu = my_json['CPUPerc'].split("%")[0]
                if float(cpu) > 0:
                    final_time = time.time()
                    t = final_time - start_time
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' ' + str(t) + ' '
                    # for d in state_u:
                    data = data + str(cpu) + ' ' + '\n'

                    f.write(data)
                    f.close()


# reset Environment
def reset():
    cmd1 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn1 "
    cmd2 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn2 "
    cmd3 = "sudo docker-machine ssh default docker service update --limit-cpu 0.5 app_mn1"
    cmd4 = "sudo docker-machine ssh default docker service update --limit-cpu 0.5 app_mn2"
    subprocess.check_output(cmd1, shell=True)
    subprocess.check_output(cmd2, shell=True)
    subprocess.check_output(cmd3, shell=True)
    subprocess.check_output(cmd4, shell=True)


def store_reward(service_name, reward):
    # Write the string to a text file
    path = result_dir + service_name + "_reward.txt"
    f = open(path, 'a')
    data = str(reward) + '\n'
    f.write(data)


def store_loss(service_name, loss):
    # Write the string to a text file
    path = result_dir + service_name + "_loss.txt"
    f = open(path, 'a')
    data = str(loss) + '\n'
    f.write(data)


def store_trajectory(service_name, step, s, a, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
    f.write(data)


def store_error_count(error):
    # Write the string to a text file
    path = result_dir + "error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)



def post_url(url, RFID, content):

    headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
    data = {
        "m2m:cin": {
            "con": content,
            "cnf": "application/json",
            "lbl": "req",
            "rn": str(RFID),
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=0.05)
        response = str(response.status_code)
    except requests.exceptions.Timeout:
        response = "timeout"

    return response

def send_request(stage, request_num, start_time, total_episodes):
    global change, send_finish, reset_complete
    global timestamp, use_tm, RFID
    error = 0
    for episode in range(total_episodes):
        timestamp = 0
        print("episode: ", episode)
        print("reset envronment")
        reset_complete = 0
        reset()  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
        for i in request_num:
            # print("timestamp: ", timestamp)
            event_mn1.clear()
            event_mn2.clear()
            if ((timestamp - 1) % 30) == 0:
                print("wait mn1 mn2 step ...")
                event_mn1.wait()
                event_mn2.wait()
                change = 0
            event_timestamp_Ccontrol.clear()
            exp = np.random.exponential(scale=1 / i, size=i)
            tmp_count = 0
            for j in range(i):
                try:
                    url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                    # change stage
                    url1 = url + stage[(tmp_count * 10 + j) % 8]
                    if error_rate > random.random():
                        content = "false"
                    else:
                        content = "true"
                    response = post_url(url1, RFID, content)
                    RFID += 1

                except:
                    print("eror")
                    error += 1

                if use_tm == 1:
                    time.sleep(exp[tmp_count])
                    tmp_count += 1

                else:
                    time.sleep(1 / i)  # send requests every 1s

            timestamp += 1
            event_timestamp_Ccontrol.set()

    send_finish = 1
    final_time = time.time()
    alltime = final_time - start_time
    store_error_count(error)
    print('time:: ', alltime)



def dqn(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event, service_name):
    global timestamp, simulation_time, change, RFID, send_finish

    env = Env(service_name)
    agent = MPDQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
    agent.train(total_episodes, event)


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=dqn, args=(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event_mn1, 'app_mn1', ))
t5 = threading.Thread(target=dqn, args=(total_episodes, memory_size, batch_size, target_update, epsilon_decay, event_mn2, 'app_mn2', ))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()


t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
