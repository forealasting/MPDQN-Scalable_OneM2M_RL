import requests
import time
import threading
import subprocess
import json
import numpy as np
import random
import statistics

# request rate r
r = 50      # if not use_tm
use_tm = 1  # if use_tm
error_rate = 0.2  # 0.2/0.5

# initial setting (threshold setting) # no use now
T_max = 0.065  # t_max violation
T_min = 0.055
set_tmin = 1  # 1 if setting tmin
cpus = 0.5  # initial cpus
replicas = 1  # initial replica

## initial
request_num = []
simulation_time = 3  # 300 s  # or 3600s
request_n = simulation_time
change = 0   # 1 if take action / 0 if init or after taking action
reset_complete = 0
send_finish = 0
timestamp = 0  # plus 1 in funcntion : send_request
RFID = 0  # choose random number for data
event = threading.Event()

## Learning parameter
# S ={k, u , c}
# k (replica): 1 ~ 3                          actual value : same
# u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
# c (used cpus) : 0.1 0.2 ... 1               actual value : same
# action_space = ['-r', -1, 0, 1, 'r']
total_episodes = 5       # Total episodes
learning_rate = 0.01          # Learning rate
# max_steps = 50               # Max steps per episode
# Exploration parameters
gamma = 0.9                 # Discounting rate
max_epsilon = 1
min_epsilon = 0.1
epsilon_decay = 1/300

## 7/8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]

if use_tm:
    #   Modify the workload path if it is different
    f = open('/home/user/flask_test/client/request/request10.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [r for i in range(simulation_time)]


print("request_num:: ", len(request_num), "simulation_time:: ", simulation_time)


class Env:

    def __init__(self, service_name="app_mn1"):

        self.service_name = service_name
        self.cpus = 0.5
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['-r', '-1', '0', '1', 'r']
        self.n_actions = len(self.action_space)

        # Need modify if ip change
        self.url_list = ["http://192.168.99.115:666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4", "http://192.168.99.116:777/~/mn-cse/mn-name/AE2/Control_Command_Container", "http://192.168.99.115:1111/test", "http://192.168.99.116:2222/test"]


    def reset(self):
        cmd = "sudo docker-machine ssh default docker stack rm app"
        subprocess.check_output(cmd, shell=True)
        cmd1 = "sudo docker-machine ssh default docker stack deploy --compose-file docker-compose.yml app"
        subprocess.check_output(cmd1, shell=True)
        time.sleep(60)

    def get_response_time(self):
        global RFID
        path1 = "result/" + self.service_name + "_response.txt"

        f1 = open(path1, 'a')

        headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
        data = {
            "m2m:cin": {
                "con": "true",
                "cnf": "application/json",
                "lbl": "req",
                "rn": str(RFID + 10000),
            }
        }
        # URL
        service_name_list = ["app_mn1", "app_mn2"]
        url = self.url_list[service_name_list.index(self.service_name)]
        start = time.time()
        response = requests.post(url, headers=headers, json=data)
        end = time.time()
        response_time = end - start
        data1 = str(timestamp) + ' ' + str(response_time) + ' ' + str(self.cpus) + ' ' + str(self.replica) + '\n'
        f1.write(data1)
        f1.close()
        return response_time

    def get_cpu_utilization(self):
        path = "result/" + self.service_name + '_cpu.txt'
        try:
            f = open(path, "r")
            cpu = []
            time = []
            for line in f:
                s = line.split(' ')
                time.append(float(s[0]))
                cpu.append(float(s[2]))

            last_cpu = cpu[-1]
            f.close()

            return last_cpu
        except:
            print("self.service_name:: ",self.service_name)
            print('cant open')

    def discretize_cpu_value(self, value):
        return int(round(value / 10))

    def step(self, action_index):
        global timestamp, send_finish, RFID, change
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

        time.sleep(30)
        event.set()
        response_time_list = []
        for i in range(5):
            time.sleep(3)
            response_time_list.append(self.get_response_time())

        # avg_response_time = sum(response_time_list)/len(response_time_list)
        median_response_time = statistics.median(response_time_list)
        median_response_time = median_response_time*1000  # 0.05s -> 50ms
        if median_response_time >= 50:
            Rt = 50
        else:
            Rt = median_response_time
        if self.service_name == "app_mn1":
            t_max = 25
        elif self.service_name == "app_mn2":
            t_max = 20
        else:
            t_max = 5

        if median_response_time < t_max:
            c_perf = 0
        else:
            tmp_d = 1.4 ** (50 / t_max)
            tmp_n = 1.4 ** (Rt / t_max)
            c_perf = tmp_n / tmp_d

        c_res = (self.replica*self.cpus)/3   # replica*self.cpus / Kmax
        next_state = []
        # k, u, c # r
        self.cpu_utilization = self.get_cpu_utilization()
        u = self.discretize_cpu_value(self.cpu_utilization)
        next_state.append(self.replica)
        next_state.append(u/10)
        next_state.append(self.cpus)
        # state.append(req)
        done = False
        w_pref = 0.5
        w_res = 0.5
        reward = -(w_pref * c_perf + w_res * c_res)
        # print("step_over_next_state: ", next_state)
        return next_state, reward, done


class QLearningTable:

    def __init__(self, actions, learning_rate=0.01, gamma=0.9, max_epsilon=1, min_epsilon=0.1, epsilon_decay=1 / 300):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.full((10, 11, 10, 5), -np.iinfo(np.int32).max)  # -2147483647

    def choose_action(self, state):
        available_actions = self.get_available_actions(state)
        s = list(state)
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])
        s[0] = int(s[0])-1
        # action selection
        if self.epsilon > np.random.uniform():
            # choose random action
            action = np.random.choice(available_actions)
        else:
            # print(state[0], state[1], state[2])
            # choose greedy action
            q_values = self.q_table[s[0], s[1], s[2], :]
            q_values[np.isin(range(5), available_actions, invert=True)] = -np.iinfo(np.int32).max
            action = np.argmax(q_values)

        return action

    def learn(self, state, a, r, next_state, done):
        s = list(state)
        s_ = list(next_state)
        # state  = [1, 0.0, 0.5]
        # transform state to index
        s[2] = int(s[2] * 10 - 1)
        s[1] = int(s[1])
        s[0] = int(s[0]) - 1

        s_[2] = int(s_[2] * 10 - 1)
        s_[1] = int(s_[1])
        s_[0] = int(s_[0])-1

        q_predict = self.q_table[s[0], s[1], s[2], a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * np.max(self.q_table[s_[0], s_[1], s_[2], :])
        self.q_table[s[0], s[1], s[2], a] = q_predict + self.lr * (q_target - q_predict)
        # print(self.q_table[s[0], s[1], s[2], a])
        # linearly decrease epsilon
        self.epsilon = max(
            self.min_epsilon, self.epsilon - (
                    self.max_epsilon - self.min_epsilon
            ) * self.epsilon_decay
        )

    def get_available_actions(self, state):
        # S ={k, u , c}
        # k (replica): 1 ~ 3                          actual value : same
        # u (cpu utilization) : 0.0, 0.1 0.2 ...1     actual value : 0 ~ 100
        # c (used cpus) : 0.1 0.2 ... 1               actual value : same
        # action_space = ['-r', -1, 0, 1, 'r']

        actions = [0, 1, 2, 3, 4]  # action index
        if state[0] == 1:
            actions.remove(1)
        if state[0] == 3:
            actions.remove(3)
        if state[2] == 1:
            actions.remove(4)
        if state[2] == 0.5:
            actions.remove(0)

        return actions


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
    response = requests.post(url, headers=headers, json=data)

    return response


def store_cpu(start_time, woker_name):
    global timestamp, cpus, change, reset_complete

    cmd = "sudo docker-machine ssh " + woker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
    while True:
        if reset_complete:  # time.sleep(70)  # wait environment start
            if send_finish == 1:
                break
            if change == 0:
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
                    # state_u.append(cpu)
                    final_time = time.time()
                    t = final_time - start_time
                    path = "result/" + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' ' + str(t) + ' '
                    # for d in state_u:
                    data = data + str(cpu) + ' ' + '\n'

                    f.write(data)
                    f.close()


def send_request(stage,request_num, start_time, total_episodes):
    global change, send_finish, reset_complete
    global timestamp, use_tm, RFID
    error = 0
    for episode in range(total_episodes):
        print("episode: ", episode)
        print("reset envronment")
        reset_complete = 0
        reset()  # reset Environment
        time.sleep(70)
        print("reset envronment complete")
        reset_complete = 1
        send_finish = 0
        timestamp = 0
        for i in request_num:
            print("timestamp: ", timestamp)
            exp = np.random.exponential(scale=1 / i, size=i)
            tmp_count = 0
            # if change == 1:
            if ((timestamp - 1) % 30) == 0:
                print("timestamp: ", timestamp)
                # print('change!')
                event.wait()
                # time.sleep(30)
                change = 0
            for j in range(i):
                try:
                    s_time = time.time()
                    # Need modify if ip change
                    url = "http://192.168.99.115:666/~/mn-cse/mn-name/AE1/"
                    # change stage
                    url1 = url + stage[(i * 10 + j) % 8]
                    if error_rate > random.random():
                        content = "false"
                    else:
                        content = "true"
                    response = post_url(url1, RFID, content)
                    # print(response)
                    t_time = time.time()
                    rt = t_time - s_time
                    # store_rt(timestamp, rt)
                    RFID += 1
                    break

                except:
                    error += 1
                    # time.sleep(2)

                if use_tm == 1:
                    time.sleep(exp[tmp_count])
                    tmp_count += 1

                else:
                    time.sleep(1/i)  # send requests every 1s
            timestamp += 1
    send_finish = 1
    final_time = time.time()
    alltime = final_time - start_time
    store_error_count(error)
    print('time:: ', alltime)


def store_error_count(error):
    # Write the string to a text file
    path = "result/error.txt"
    f = open(path, 'a')
    data = str(error) + '\n'
    f.write(data)

# reset Environment
def reset():
    cmd = "sudo docker-machine ssh default docker stack rm app"
    subprocess.check_output(cmd, shell=True)
    cmd1 = "sudo docker-machine ssh default docker stack deploy --compose-file docker-compose.yml app"
    subprocess.check_output(cmd1, shell=True)


def store_reward(service_name, reward):

    # Write the string to a text file
    path = "result/" + service_name + "_reward.txt"
    f = open(path, 'a')
    data = str(reward) + '\n'
    f.write(data)


def store_trajectory(service_name, step, s, a, r, s_):
    path = "result/" + service_name + "_trajectory.txt"
    f = open(path, 'a')
    data = str(step) + ' ' + str(s) + ' ' + str(a) + ' ' + str(r) + ' ' + str(s_) + '\n'
    f.write(data)


def q_learning(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, service_name):
    global timestamp, simulation_time, change, RFID, send_finish

    env = Env(service_name)
    actions = list(range(env.n_actions))
    RL = QLearningTable(actions, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay)
    all_rewards = []
    step = 0
    init_state = [1, 0.0, 0.5]

    for episode in range(total_episodes):
        # initial observation
        state = init_state
        rewards = []  # record reward every episode
        while True:
            if ((timestamp - 1) % 30) == 0:
                # print("timestamp: ", timestamp)
                print(service_name, "step: ", step)
                # RL choose action based on state
                action = RL.choose_action(state)
                print("action: ", action)
                # change = 1
                # RL take action and get next state and reward

                s_t = time.time()
                next_state, reward, done = env.step(action)
                e_t = time.time() - s_t
                print("Env step execution time: ", e_t)
                if timestamp == (simulation_time-1):
                    done = True

                print("next_state: ", next_state, "reward: ", reward)
                print("done: ", done)
                store_trajectory(service_name, step, state, action, reward, next_state)
                rewards.append(reward)
                # RL learn from this transition
                RL.learn(state, action, reward, next_state, done)

                # swap state
                state = next_state
                step += 1
                if done:
                    avg_rewards = sum(rewards)/len(rewards)
                    break

        store_reward(service_name, avg_rewards)
        all_rewards.append(avg_rewards)
    # episode end
    print("service:", service_name, all_rewards)



start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, total_episodes, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=q_learning, args=(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, 'app_mn1', ))
t5 = threading.Thread(target=q_learning, args=(total_episodes, learning_rate, gamma, max_epsilon, min_epsilon, epsilon_decay, 'app_mn2', ))


t1.start()
t2.start()
t3.start()
t4.start()
#t5.start()


t1.join()
t2.join()
t3.join()
t4.join()
#t5.join()

