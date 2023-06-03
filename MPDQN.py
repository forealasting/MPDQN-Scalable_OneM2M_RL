import numpy as np
import random
import requests
import time
import threading
import subprocess
import json
import statistics
import os
import datetime
import math
from pdqn_v1 import PDQNAgent
from pdqn_multipass import MultiPassPDQNAgent

print(datetime.datetime.now())

# request rate r
data_rate = 50      # if not use_tm
use_tm = 1  # if use_tm
result_dir = "./mpdqn_result/result5/"  # need to modify pdqn_v1.py result_dir also

## initial
request_num = []
# timestamp    :  0, 1, 2, , ..., 61, ..., 3601
# learning step:   0,  ..., 1,     , 120

simulation_time = 3600 #
request_n = simulation_time + 60

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
ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
error_rate = 0.2  # 0.2/0.5
Tmax_mn1 = 20
Tmax_mn2 = 20


## Learning parameter
# S ={k, u , c, r} {k, u , c}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 16   # Training_episodes


if_test = False
if if_test:
    total_episodes = 1  # Testing_episodes

multipass = False  # False : PDQN  / Ture: MPDQN

# Exploration parameters
epsilon_steps = 840  # episode per step
epsilon_initial = 1
epsilon_final = 0.01

# Learning rate
tau_actor = 0.1
tau_actor_param = 0.01
learning_rate_actor = 0.001
learning_rate_actor_param = 0.001
gamma = 0.9                 # Discounting rate
replay_memory_size = 960  # Replay memory
batch_size = 16
initial_memory_threshold = 16  # Number of transitions required to start learning
use_ornstein_noise = False

layers = [64,]
seed = 9

clip_grad = 10 # no use now
action_input_layer = 0  # no use now
cres_norml = False
# check result directory
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# build dir
os.mkdir(result_dir)
# store setting
path = result_dir + "setting.txt"

# Define settings dictionary
settings = {
    'date': datetime.datetime.now(),
    'data_rate': data_rate,
    'use_tm': use_tm,
    'Tmax_mn1': Tmax_mn1,
    'Tmax_mn2': Tmax_mn2,
    'simulation_time': simulation_time,
    'tau_actor': tau_actor,
    'tau_actor_param': tau_actor_param,
    'learning_rate_actor': learning_rate_actor,
    'learning_rate_actor_param': learning_rate_actor_param,
    'gamma': gamma,
    'epsilon_steps': epsilon_steps,
    'epsilon_final': epsilon_final,
    'replay_memory_size': replay_memory_size,
    'batch_size': batch_size,
    'loss_function': 'MSE loss',
    'layers': layers,
    'cres_norml': cres_norml,
    'if_test': if_test,
}

# Write settings to file
with open(result_dir + 'setting.txt', 'a') as f:
    for key, value in settings.items():
        f.write(f'{key}: {value}\n')


## 8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    f = open('request/request14.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(request_n)]

print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['1', '1', '1']
        self.state_space = [1, 0.0, 0.5, 40]  # [1, 0.0, 0.5, 10]
        self.n_state = len(self.state_space)
        self.n_actions = len(self.action_space)

        # Need modify ip if container name change
        self.url_list = ["http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4",
                                    "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container",
                                    "http://" + ip + ":1111/test", "http://" + ip1 + ":2222/test"]

    def reset(self):
        self.cpus = 0.5
        self.replica = 1

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
        if self.service_name =='app_mn1':
            worker_name = 'worker'
        else:
            worker_name = 'worker1'
        cmd = "sudo docker-machine ssh " + worker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
        returned_text = subprocess.check_output(cmd, shell=True)
        my_data = returned_text.decode('utf8')
        my_data = my_data.split("}")
        cpu_list = []
        for i in range(len(my_data) - 1):
            # print(my_data[i]+"}")
            my_json = json.loads(my_data[i] + "}")
            name = my_json['Name'].split(".")[0]
            cpu = my_json['CPUPerc'].split("%")[0]
            if float(cpu) > 0 and name == self.service_name:
                cpu_list.append(float(cpu))
        avg_replica_cpu_utilization = sum(cpu_list)/len(cpu_list)
        return avg_replica_cpu_utilization

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action, event, done):
        global timestamp, send_finish, change, simulation_time


        action_replica = action[0]
        action_cpus = action[1][action_replica][0]
        if (action_replica + 1) == self.replica and action_cpus == self.cpus:
            cmd = "sudo docker-machine ssh default docker service update --replicas 0 " + self.service_name
            cmd1 = "sudo docker-machine ssh default docker service update --replicas " + str(action_replica + 1) + " " + self.service_name
            returned_text = subprocess.check_output(cmd, shell=True)
            returned_text = subprocess.check_output(cmd1, shell=True)
        else:
            self.replica = action_replica + 1  # 0 1 2 (index)-> 1 2 3 (replica)
            self.cpus = round(action_cpus, 2)
            # print(self.replica, self.cpus)
            change = 1
            cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(self.replica)
            cmd1 = "sudo docker-machine ssh default docker service update --limit-cpu " + str(self.cpus) + " " + self.service_name
            returned_text = subprocess.check_output(cmd, shell=True)
            returned_text = subprocess.check_output(cmd1, shell=True)

        time.sleep(30)  # wait service start

        event.set()

        response_time_list = []
        time.sleep(50)  # wait for monitor ture value

        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())


        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms
        t_max = 0

        if self.service_name == "app_mn1":
            t_max = Tmax_mn1
        elif self.service_name == "app_mn2":
            t_max = Tmax_mn2

        Rt = mean_response_time
        # Cost 1
        # if Rt > t_max:
        #     c_perf = 1
        # else:
        #     tmp_d = 10 * (Rt - t_max) / t_max
        #     c_perf = math.exp(tmp_d)

        # Cost 2
        B = 10
        target = 20 + 2 * math.log(0.9)
        c_perf = np.where(Rt <= target, np.exp(B * (Rt - t_max) / t_max), 0.9 + ((Rt - target) / (50 - target)) * 0.1)

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        path = result_dir + self.service_name + "_agent_get_cpu.txt"
        f1 = open(path, 'a')
        data = str(timestamp) + ' ' + str(self.cpu_utilization) + '\n'
        f1.write(data)
        f1.close()
        # u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(self.cpu_utilization/100/self.cpus)
        next_state.append(self.cpus)
        next_state.append(Rt)
        # next_state.append(request_num[timestamp])

        # cost function
        w_pref = 0.8
        w_res = 0.2
        # c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)  # min max normalize
        # c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)  # min max normalize
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res



def store_cpu(start_time, worker_name):
    global timestamp, cpus, change, reset_complete

    cmd = "sudo docker-machine ssh " + worker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
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
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' '
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



def store_trajectory(service_name, step, s, a_r, a_c, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    a_c_ = list(a_c)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a_r) + ' ' + str(a_c_) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
    f.write(data)


def store_error_count(error):
    # Write the string to a text file
    path = result_dir + "error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)



def post_url(url, RFID):
    if error_rate > random.random():
        content = "false"
    else:
        content = "true"
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
        print("episode: ", episode+1)
        print("reset envronment")
        reset_complete = 0
        reset()  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
        for i in request_num:
            # print('timestamp: ', timestamp)
            event_mn1.clear()  # set flag to false
            event_mn2.clear()
            if ((timestamp) % 60) == 0 and timestamp!=0 :  # and timestamp<(simulation_time)
                print("wait mn1 mn2 step and service scaling ...")
                event_mn1.wait()  # if flag == false : wait, else if flag == True: continue
                event_mn2.wait()
                change = 0
            event_timestamp_Ccontrol.clear()
            # exp = np.random.exponential(scale=1 / i, size=i)
            tmp_count = 0
            for j in range(i):
                try:
                    url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                    # change stage

                    url1 = url + stage[(tmp_count * 10 + j) % 8]
                    s_time = time.time()
                    response = post_url(url1, RFID)
                    t_time = time.time()
                    rt = t_time - s_time
                    RFID += 1

                except:
                    rt = 0.05
                    print("error")
                    error += 1

                # if use_tm == 1:
                #     time.sleep(exp[tmp_count])
                #     tmp_count += 1
                if rt < (1 / i) and (i > 50):
                    time.sleep((1 / i) - rt)
                elif i <= 50:
                    time.sleep(1 / i)
                tmp_count += 1
            timestamp += 1
            event_timestamp_Ccontrol.set()

    send_finish = 1
    final_time = time.time()
    alltime = final_time - start_time
    store_error_count(error)
    print('time:: ', alltime)


def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def mpdqn(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event, service_name, seed):
    global timestamp, simulation_time

    env = Env(service_name)


    agent_class = PDQNAgent
    if multipass:
        agent_class = MultiPassPDQNAgent
    agent = agent_class(
                       env.n_state, env.n_actions,
                       batch_size=batch_size,
                       learning_rate_actor=learning_rate_actor,
                       learning_rate_actor_param=learning_rate_actor_param,
                       epsilon_initial=epsilon_initial,
                       epsilon_steps=epsilon_steps,
                       gamma=gamma,
                       tau_actor=tau_actor,
                       tau_actor_param=tau_actor_param,
                       clip_grad=clip_grad,
                       initial_memory_threshold=initial_memory_threshold,
                       use_ornstein_noise=use_ornstein_noise,
                       replay_memory_size=replay_memory_size,
                       epsilon_final=epsilon_final,
                       actor_kwargs={'hidden_layers': layers,
                                     'action_input_layer': action_input_layer},
                       actor_param_kwargs={'hidden_layers': layers,
                                           'squashing_function': True,
                                           'output_layer_init_std': 0.0001},
                       seed=seed,
                       service_name=service_name)
    # print(agent)

    start_time = time.time()
    init_state = [1, 1.0, 0.5, 20]
    step = 1
    for episode in range(1, total_episodes+1):
        if if_test:  # Test
            agent.epsilon_final = 0.
            agent.epsilon = 0.
            agent.noise = None
        env.reset()
        state = init_state
        done = False
        while True:
            if timestamp == 50:
                response_time_list = []
                for i in range(5):
                    time.sleep(1)
                    response_time_list.append(env.get_response_time())
                mean_response_time = statistics.mean(response_time_list)
                mean_response_time = mean_response_time * 1000
                Rt = mean_response_time
                state[3] = Rt
                state[1] = (env.get_cpu_utilization() / 100 / env.cpus)
                break
        state = np.array(state, dtype=np.float32)

        print("service name:", env.service_name, " episode:", episode)
        act, act_param, all_action_parameters = agent.act(state)

        action = pad_action(act, act_param)

        while True:
            if timestamp == 0:
                done = False
            event_timestamp_Ccontrol.wait()
            if (((timestamp) % 60) == 0) and (not done)and timestamp!=0:
                if timestamp == (simulation_time):
                    done = True
                else:
                    done = False

                next_state, reward, reward_perf, reward_res = env.step(action, event, done)
                # print("service name:", env.service_name, "action: ", action[0] + 1, round(action[1][action[0]][0], 2))

                # Covert np.float32
                next_state = np.array(next_state, dtype=np.float32)
                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)  # next_act: 2 # next_act_param: 0.85845 # next_all_action_parameters: -0.79984,-0.97112,0.85845
                print("service name:", env.service_name, "action: ", act + 1, act_param, all_action_parameters, " step: ", step,
                      " next_state: ",
                      next_state, " reward: ", reward, " done: ", done, "epsilon", agent.epsilon)
                store_trajectory(env.service_name, step, state, act + 1, all_action_parameters, reward, reward_perf,
                                 reward_res,
                                 next_state, done)
                next_action = pad_action(next_act, next_act_param)
                if not if_test:
                    agent.step(state, (act, all_action_parameters), reward, next_state,
                               (next_act, next_all_action_parameters), done)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters

                action = next_action
                state = next_state
                agent.epsilon_decay()

                step += 1
                event_timestamp_Ccontrol.clear()
                if done:
                    break
    if not if_test:
        agent.save_models(result_dir)
    end_time = time.time()
    print(end_time-start_time)


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn1, 'app_mn1', seed, ))

t5 = threading.Thread(target=mpdqn, args=(total_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final,
        clip_grad, layers, multipass, action_input_layer, event_mn2, 'app_mn2', seed, ))

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

