import requests
import concurrent.futures
import time
import threading
import subprocess
import json
import numpy as np
import random
import os

# define result path
result_dir = "./static_result/result68/"

# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
data_rate = 50  # use static request rate
use_tm = 0  # use dynamic traffic
error_rate = 0.2   # 0.2/0.5

## initial
request_num = []
simulation_time = 100  # 300 s  # 3600s
cpus1 = 0.5
replica1 = 1

request_n = simulation_time
change = 0   # 1 if take action / 0 if init or after taking action
send_finish = 0
timestamp = 0
RFID = 0
event_mn1 = threading.Event()
event_mn2 = threading.Event()

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


def store_cpu(worker_name):
    global timestamp, change

    cmd = "sudo docker-machine ssh " + worker_name + " docker stats --all --no-stream --format \\\"{{ json . }}\\\" "
    while True:
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
                if float(cpu) > 0:
                    path = result_dir + name + "_cpu.txt"
                    f = open(path, 'a')
                    data = str(timestamp) + ' '
                    data = data + str(cpu) + ' ' + '\n'
                    f.write(data)
                    f.close()

def store_rt(timestamp, response, rt):
    path = result_dir + "app_mn1_response.txt"
    f = open(path, 'a')
    for i in range(len(timestamp)):
        data = str(timestamp[i]) + ' ' + str(response[i]) + ' ' + str(rt[i]) + '\n'
        f.write(data)
    f.close()

# sned request to app_mn2 app_mnae1 app_mnae2
def store_rt2():
    global timestamp, send_finish, change

    path1 = result_dir + "/app_mn2_response.txt"
    path2 = result_dir + "/app_mnae1_response.txt"
    path3 = result_dir + "/app_mnae2_response.txt"

    while True:
        if change == 0:
            f1 = open(path1, 'a')
            f2 = open(path2, 'a')
            f3 = open(path3, 'a')
            RFID = random.randint(0, 100000000)
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
                data = {
                    "m2m:cin": {
                        "con": "true",
                        "cnf": "application/json",
                        "lbl": "req",
                        "rn": str(RFID),
                    }
                }

                # URL 1
                url = "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container"
                try:
                    s_time = time.time()
                    future = executor.submit(requests.post, url, headers=headers, json=data, timeout=0.05)
                    response = future.result()
                    response_time1 = time.time() - s_time
                    response1 = str(response.status_code)
                except requests.exceptions.Timeout:
                    response1 = 'timeout'
                    response_time1 = 0.05

                # # URL 2
                try:
                    s_time = time.time()
                    future = executor.submit(requests.post, "http://" + ip + ":1111/test", headers=headers, json=data, timeout=0.05)
                    response = future.result()
                    response_time2 = time.time() - s_time
                    response2 = str(response.status_code)
                except requests.exceptions.Timeout:
                    response2 = 'timeout'
                    response_time1 = 0.05

                # # URL 3
                try:
                    s_time = time.time()
                    future = executor.submit(requests.post, "http://" + ip1 + ":2222/test", headers=headers, json=data, timeout=0.05)
                    response = future.result()
                    response_time3 = time.time() - s_time
                    response3 = str(response.status_code)
                except requests.exceptions.Timeout:
                    response3 = 'timeout'
                    response_time3 = 0.05

                data1 = str(timestamp) + ' ' + str(response1) + ' ' + str(response_time1) + '\n'
                data2 = str(timestamp) + ' ' + str(response2) + ' ' + str(response_time2) + '\n'
                data3 = str(timestamp) + ' ' + str(response3) + ' ' + str(response_time3) + '\n'
                f1.write(data1)
                f2.write(data2)
                f3.write(data3)
            time.sleep(1)

            if send_finish == 1:
                f1.close()
                f2.close()
                f3.close()
                break


def send_request(stage, request_num):
    global change, send_finish
    global timestamp, use_tm, RFID

    error = 0
    all_rt = []
    all_timestamp = []
    all_response = []
    for i in request_num:
        print("timestamp: ", timestamp)
        exp = np.random.exponential(scale=1 / i, size=i)
        tmp_count = 0

        for j in range(i):
            try:
                # change stage
                url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                url1 = url + stage[(tmp_count * 10 + j) % 8]

                s_time = time.time()
                response = post_url(url1, RFID)
                t_time = time.time()
                rt = t_time - s_time
                all_timestamp.append(timestamp)
                all_response.append(response)
                all_rt.append(rt)
                RFID += 1

            except:
                print("error")
                error += 1

            if use_tm == 1:
                time.sleep(exp[tmp_count])
                tmp_count += 1

            else:
                time.sleep(1 / i)  # send requests every 1s

        timestamp += 1
    store_rt(all_timestamp, all_response, all_rt)
    send_finish = 1


t1 = threading.Thread(target=send_request, args=(stage, request_num, ))
t2 = threading.Thread(target=store_cpu, args=('worker',))
t3 = threading.Thread(target=store_cpu, args=('worker1',))
t4 = threading.Thread(target=store_rt2)


t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()

