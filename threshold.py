import requests
import time
import threading
import subprocess
import json
import numpy as np
import random
import os
import math
import statistics
# define result path
result_dir = "./threshold_result/result1/"

# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
data_rate = 50  # use static request rate
use_tm = 0  # use dynamic traffic
error_rate = 0.2   # 0.2/0.5

## initial
request_num = []
simulation_time = 3600  # 300 s  # 3600s
cpus1 = 1
cpus2 = 1
replica1 = 1
replica2 = 1
request_n = simulation_time
change = 0   # 1 if take action / 0 if init or after taking action
send_finish = 0
reset_complete = 0
timestamp = 0
RFID = 0

Tmax_mn1 = 20
Tmax_mn2 = 20

# threshold
scale_out_threshold = 80  # if cpu utilization >= 80%, scale out
scale_in_threshold = 20  # if cpu utilization <= 20%, scale in

event_mn1 = threading.Event()
event_mn2 = threading.Event()
event_timestamp_Ccontrol = threading.Event()

ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
# url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"


## 8 stage
stage = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
         "Color_Container", "RFID_Container_for_stage3", "Contrast_Data_Container", "RFID_Container_for_stage4"]



# check result directory
if os.path.exists(result_dir):
    print("Deleting existing result directory...")
    raise SystemExit  # end process

# build dir
os.mkdir(result_dir)

# store setting
path = result_dir + "setting.txt"
f = open(path, 'a')
data = 'data_rate: ' + str(data_rate) + '\n'
data += 'use_tm: ' + str(use_tm) + '\n'
data += 'simulation_time ' + str(simulation_time) + '\n'
data += 'cpus: ' + str(cpus1) + '\n'
data += 'replica ' + str(replica1) + '\n'
f.write(data)
f.close()

if use_tm:
    #   Modify the workload path if it is different
    f = open('request/request6.txt')

    for line in f:
        if len(request_num) < request_n:

            request_num.append(int(float(line)))
else:
    request_num = [data_rate for i in range(simulation_time)]

print('request_num:: ', len(request_num))

class Env:

    def __init__(self, service_name):

        self.service_name = service_name
        self.cpus = 1
        self.replica = 1
        self.cpu_utilization = 0.0
        self.action_space = ['-1', 0, '1']
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

        if action == '-r':
            if self.replica > 1:
                self.replica -= 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(
                    self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if action == 'r':
            if self.replica < 3:
                self.replica += 1
                change = 1
                cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(
                    self.replica)
                returned_text = subprocess.check_output(cmd, shell=True)

        if  action == '0':
            cmd = "sudo docker-machine ssh default docker service scale " + self.service_name + "=" + str(
                self.replica)
            returned_text = subprocess.check_output(cmd, shell=True)

        time.sleep(30)  # wait service start

        if not done:
            # print(self.service_name, "_done: ", done)
            # print(self.service_name, "_step complete")
            event.set()

        response_time_list = []
        time.sleep(50)
        for i in range(5):
            time.sleep(1)
            response_time_list.append(self.get_response_time())

        if done:
            # print(self.service_name, "_done: ", done)
            time.sleep(5)
            event.set()  # if done and after get_response_time
        # mean_response_time = sum(response_time_list)/len(response_time_list)
        # print(response_time_list)
        mean_response_time = statistics.mean(response_time_list)
        mean_response_time = mean_response_time*1000  # 0.05s -> 50ms
        t_max = 0

        if self.service_name == "app_mn1":
            t_max = Tmax_mn1
        elif self.service_name == "app_mn2":
            t_max = Tmax_mn2

        Rt = mean_response_time
        if Rt > t_max:
            c_perf = 1
        else:
            tmp_d = 10*(Rt - t_max)/t_max
            c_perf = math.exp(tmp_d)


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
        # c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)
        c_res = 0 + ((c_res - (1 / 6)) / (1 - (1 / 6))) * (1 - 0)  # normalize to [0, 1]
        reward_perf = w_pref * c_perf
        reward_res = w_res * c_res
        reward = -(reward_perf + reward_res)
        return next_state, reward, reward_perf, reward_res

def get_cpu_utilization(service_name):
    path = result_dir + service_name + '_cpu.txt'
    try:
        f = open(path, "r")
        cpu = []
        time = []
        for line in f:
            s = line.split(' ')
            time.append(float(s[0]))
            cpu.append(float(s[1]))

        last_avg_cpu = statistics.mean(cpu[-5:])
        f.close()

        return last_avg_cpu
    except:

        print('cant open')


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
        response = 'timeout'

    return response


def store_cpu(start_time, woker_name):
    global timestamp, change, reset_complete

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
                path = result_dir + name + "_cpu.txt"
                f = open(path, 'a')
                data = str(timestamp) + ' '
                # for d in state_u:
                data = data + str(cpu) + ' ' + '\n'
                f.write(data)
                f.close()


def store_trajectory(service_name, step, s, a, r, r_perf, r_res, s_, done):
    path = result_dir + service_name + "_trajectory.txt"
    tmp_s = list(s)
    tmp_s_ = list(s_)
    f = open(path, 'a')
    data = str(step) + ' ' + str(tmp_s) + ' ' + str(a) + ' ' + str(r) + ' ' + str(r_perf) + ' ' + str(r_res) + ' ' + str(tmp_s_) + ' ' + str(done) + '\n'
    f.write(data)

def reset():
    cmd1 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn1 "
    cmd2 = "sudo docker-machine ssh default docker service update --replicas 1 app_mn2 "
    cmd3 = "sudo docker-machine ssh default docker service update --limit-cpu 1 app_mn1"
    cmd4 = "sudo docker-machine ssh default docker service update --limit-cpu 1 app_mn2"
    subprocess.check_output(cmd1, shell=True)
    subprocess.check_output(cmd2, shell=True)
    subprocess.check_output(cmd3, shell=True)
    subprocess.check_output(cmd4, shell=True)

def send_request(stage, request_num, start_time):
    global change, send_finish
    global timestamp, use_tm, RFID
    timestamp = 0
    error = 0

    print("reset envronment")
    reset_complete = 0
    reset()  # reset Environment
    time.sleep(70)
    print("reset envronment complete")
    reset_complete = 1
    send_finish = 0

    for j in range(data_rate):
        tmp_count = 0
        try:
            # change stage
            url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
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
        time.sleep(1 / data_rate)

    for i in request_num:
        # print("timestamp: ", timestamp)
        # exp = np.random.exponential(scale=1 / i, size=i)
        tmp_count = 0
        event_mn1.clear()  # set flag to false
        event_mn2.clear()  # set flag to false
        if (timestamp % 60 == 0 and timestamp != 0):
            print("wait mn1 mn2 step ...")
            event_mn1.wait()  # if flag == false : wait, else if flag == True: continue
            event_mn2.wait()
            change = 0
        for j in range(i):
            try:
                # change stage
                url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                url1 = url + stage[(tmp_count * 10 + j) % 8]
                s_time = time.time()
                response = post_url(url1, RFID)
                t_time = time.time()
                rt = t_time - s_time
                RFID += 1

            except:
                rt = 0.1
                print("error")
                error += 1

            if rt < (1 / i) and (i > 50):
                time.sleep((1 / i) - rt)
            elif i <= 50:
                time.sleep(1 / i)
            tmp_count += 1

        timestamp += 1

    final_time = time.time()
    alltime = final_time - start_time
    print('time:: ', alltime)
    send_finish = 1


def agent_threshold_mn1(event):
    global T_max, change, send_finish, replica1, cpus1
    global timestamp
    done = False
    service_name = "app_mn1"
    env = Env(service_name)
    step = 0
    init_state = [1, 0.0, 0.5, 35]
    state = init_state
    # action: +1 scale out -1 scale in
    while True:
        if timestamp == 0:
            done = False
        event_timestamp_Ccontrol.wait()
        if (((timestamp - 1) % 60) == 0) and (not done):
            if timestamp == (simulation_time - 1):
                done = True
            else:
                done = False
            if done:
                break

            cpu_utilization = get_cpu_utilization(service_name)
            if cpu_utilization >= 80.0:
                action = "+1"
            elif cpu_utilization <= 20.0:
                action = "-1"
            else:
                action = "0"

            next_state, reward, reward_perf, reward_res = env.step(action, event, done)
            store_trajectory(env.service_name, step, state, action, reward, reward_perf, reward_res, next_state, done)

            state = next_state
            step += 1
            event_timestamp_Ccontrol.clear()

def agent_threshold_mn2(event):
    global T_max, change, send_finish, replica2, cpus2
    global timestamp

    done = False
    service_name = "app_mn2"
    env = Env(service_name)
    step = 0
    init_state = [1, 0.0, 0.5, 35]
    state = init_state
    # action: +1 scale out -1 scale in
    while True:
        if timestamp == 0:
            done = False
        event_timestamp_Ccontrol.wait()
        if (((timestamp - 1) % 60) == 0) and (not done):
            if timestamp == (simulation_time - 1):
                done = True
            else:
                done = False
            if done:
                break

            cpu_utilization = get_cpu_utilization(service_name)
            if cpu_utilization >= 80.0:
                action = "+1"
            elif cpu_utilization <= 20.0:
                action = "-1"
            else:
                action = "0"

            next_state, reward, reward_perf, reward_res = env.step(action, event, done)
            print("service name:", env.service_name, "action: ", action, " step: ", step, " next_state: ",
                  next_state, " reward: ", reward, " done: ", done)
            store_trajectory(env.service_name, step, state, action, reward, reward_perf, reward_res, next_state, done)

            state = next_state
            step += 1
            event_timestamp_Ccontrol.clear()


start_time = time.time()

t1 = threading.Thread(target=send_request, args=(stage, request_num, start_time, ))
t2 = threading.Thread(target=store_cpu, args=(start_time, 'worker',))
t3 = threading.Thread(target=store_cpu, args=(start_time, 'worker1',))
t4 = threading.Thread(target=agent_threshold_mn1, args=(event_mn1,))
t5 = threading.Thread(target=agent_threshold_mn2, args=(event_mn2,))

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
