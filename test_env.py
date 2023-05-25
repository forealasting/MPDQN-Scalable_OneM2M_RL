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
result_dir = "./static_result/0521/request_70/result14/"

# request rate r
data_rate = 70  # use static request rate
use_tm = 0  # use dynamic traffic
error_rate = 0.2   # 0.2

## initial
request_num = []
simulation_time = 100  # 300 s  # 3600s
cpus1 = 0.9
replica1 = 2

request_n = simulation_time
change = 0   # 1 if take action / 0 if init or after taking action
send_finish = 0  # 1 : finish
timestamp = 0    # time record
RFID = 0   # For different RFID data

ip = "192.168.99.124"  # app_mn1
ip1 = "192.168.99.125"  # app_mn2
# url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"


## Sensor i for every sensors
sensors = ["RFID_Container_for_stage0", "RFID_Container_for_stage1", "Liquid_Level_Container", "RFID_Container_for_stage2",
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
data += 'replica: ' + str(replica1) + '\n'
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
        # response = requests.post(url, headers=headers, json=data, timeout=0.05)
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
            cpu_list = []
            for i in range(len(my_data) - 1):
                # print(my_data[i]+"}")
                my_json = json.loads(my_data[i] + "}")
                name = my_json['Name'].split(".")[0]
                cpu = my_json['CPUPerc'].split("%")[0]
                if float(cpu) > 0 :
                    cpu_list.append(float(cpu))
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



# sned request to app_mn2 # app_mnae1 app_mnae2
def store_rt2():
    global timestamp, send_finish, change

    path1 = result_dir + "/app_mn2_response.txt"

    while True:
        if change == 0:
            f1 = open(path1, 'a')

            RFID1 = random.randint(10000000, 100000000)

            headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
            data = {
                "m2m:cin": {
                    "con": "true",
                    "cnf": "application/json",
                    "lbl": "req",
                    "rn": str(RFID1),
                }
            }

            # URL 1
            url = "http://" + ip1 + ":777/~/mn-cse/mn-name/AE2/Control_Command_Container"

            try:
                s_time = time.time()
                response = requests.post(url, headers=headers, json=data, timeout=0.05)
                response1 = str(response.status_code)
                response_time1 = time.time() - s_time

            except requests.exceptions.Timeout:
                response1 = 'timeout'
                response_time1 = 0.05
            data1 = str(timestamp) + ' ' + str(response1) + ' ' + str(response_time1) + '\n'
            f1.write(data1)
            time.sleep(1)

            if send_finish == 1:
                f1.close()
                break

def store_rt1():
    global timestamp, send_finish, change

    path1 = result_dir + "/app_mn1_response.txt"

    while True:
        if change == 0:
            f1 = open(path1, 'a')

            RFID1 = random.randint(10000000, 100000000)

            headers = {"X-M2M-Origin": "admin:admin", "Content-Type": "application/json;ty=4"}
            data = {
                "m2m:cin": {
                    "con": "true",
                    "cnf": "application/json",
                    "lbl": "req",
                    "rn": str(RFID1),
                }
            }

            # URL 1
            url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/RFID_Container_for_stage4"

            try:
                s_time = time.time()
                response = requests.post(url, headers=headers, json=data, timeout=0.05)
                response1 = str(response.status_code)
                response_time1 = time.time() - s_time

            except requests.exceptions.Timeout:
                response1 = 'timeout'
                response_time1 = 0.05
            data1 = str(timestamp) + ' ' + str(response1) + ' ' + str(response_time1) + '\n'
            f1.write(data1)
            time.sleep(1)

            if send_finish == 1:
                f1.close()
                break



def send_request(sensors, request_num):
    global change, send_finish
    global timestamp, use_tm, RFID

    error = 0
    all_rt = []
    all_timestamp = []
    all_response = []
    tmp_count = 0
    for i in request_num:  #request_num = [data_rate0, data_rate1 ...]
        # print("timestamp: ", timestamp)
        #exp = np.random.exponential(scale=1 / i, size=i)

        for j in range(i):
            try:
                # change sensors
                url = "http://" + ip + ":666/~/mn-cse/mn-name/AE1/"
                url1 = url + sensors[tmp_count % 8]  # just Post to different sensor i

                s_time = time.time()
                response = post_url(url1, RFID)
                t_time = time.time()
                rt = t_time - s_time

                # all_timestamp.append(timestamp)
                # all_response.append(response)
                # all_rt.append(rt)

                RFID += 1  # For different RFID data

            except:
                rt = 0.05
                error += 1

            # if use_tm == 1: no use now
            #     time.sleep(exp[tmp_count])

            if rt < (1 / i) and (i > 50) :
                time.sleep((1 / i) - rt)
            elif i <= 50 :
                time.sleep(1 / i)
            tmp_count += 1

        timestamp += 1
    send_finish = 1
    store_rt(all_timestamp, all_response, all_rt)
    print("error: ", error)



t1 = threading.Thread(target=send_request, args=(sensors, request_num, ))
t2 = threading.Thread(target=store_cpu, args=('worker',))
t3 = threading.Thread(target=store_cpu, args=('worker1',))
t4 = threading.Thread(target=store_rt2)
t5 = threading.Thread(target=store_rt1)

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

