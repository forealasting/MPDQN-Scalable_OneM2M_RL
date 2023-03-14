import matplotlib.pyplot as plt
import statistics
import re
import json
# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
simulation_time = 300  # 3602 s
total_episodes = 6

#  moving for plot
# moving_avg = 1
# move = 10

# limit_cpus = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
tmp_dir = "result"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"

service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]

# path_list = [path2]
# path_list = [path1, path2]
path_list = [path1]
def parse(p):
    with open(p, "r") as f:
        data = f.read().splitlines()
        parsed_data = []
        parsed_line = []

        for line in data:
            # parse data
            match = re.match(r"(\d+) \[(.+)\] (\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)",
                             line)

            if match != None:
                # Convert the parsing result to the corresponding Python object
                line_data = [int(match.group(1)), json.loads("[" + match.group(2) + "]"), int(match.group(3)),
                             float(match.group(4)), float(match.group(5)), float(match.group(6)),
                             json.loads("[" + match.group(7) + "]"), match.group(8) == "True"]
                # line_data = [int(match.group(1)), tmp2_list, int(match.group(3)),
                #              float(match.group(4)), tmp5_list, match.group(6) == "True"]


                parsed_line.append(line_data)

                if match.group(8) == "True":
                    parsed_data.append(parsed_line)
                    parsed_line = []
            # print(parsed_data)


    return parsed_data
    # return step, replica, cpu_utilization, cpus, reward, resource_use


# Plot --------------------------------------

def fig_add_Cpus(x, y, service_name):
    plt.figure()
    plt.plot(x, y, color="blue")  # color=color
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Cpus")
    # plt.ylabel("Resource use ")
    # plt.ylabel("Replica")
    # plt.ylabel("Cpus")
    plt.grid(True)

    plt.xlim(0, total_episodes*120)
    plt.ylim(0, 1.1)
    plt.savefig("Cpus.png")
    plt.tight_layout()
    plt.show()


def fig_add_Replicas(x, y, service_name):
    plt.figure()
    plt.plot(x, y, color="green")  # color=color
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Replicas")
    # plt.ylabel("Replica")
    # plt.ylabel("Cpus")
    plt.grid(True)

    plt.xlim(0, total_episodes*120)
    plt.ylim(0, 4)
    plt.savefig("Replicas.png")
    plt.tight_layout()
    plt.show()


def fig_add_Cpu_utilization(x, y, service_name):
    plt.plot(x, y)  # color=color
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Cpu_utilization")
    plt.grid(True)
    plt.xlim(0, total_episodes*120)
    plt.ylim(0, 100)
    plt.savefig("Cpu_utilization.png")
    plt.tight_layout()
    plt.show()


def fig_add_Resource_use(x, y, service_name):
    plt.figure()
    plt.plot(x, y, color="black")  # color=color
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Resource_use")
    plt.grid(True)
    plt.xlim(0, total_episodes*120)
    plt.ylim(0, 3)
    plt.savefig("Resource_use.png")
    plt.tight_layout()
    plt.show()

def fig_add_reward(x, y, service_name):
    plt.figure()
    #plt.subplot(pos)
    plt.plot(x, y, color="red")  # color=color # label=label
    plt.title(service_name)
    plt.xlabel("step")
    plt.ylabel("Cost")

    plt.grid(True)

    plt.xlim(0, total_episodes*120)
    plt.ylim(-0.6, -0.1)
    plt.savefig("cost.png")
    plt.tight_layout()
    plt.show()

def moving_average(lst, move=10):
    ret = []
    for i in range(len(lst)):
        if i < move:
            ret.append(sum(lst[:i+1]) / (i+1))
        else:
            ret.append(sum(lst[i-move+1:i+1]) / move)
    return ret


def parse_episods_data(episods_data, service_name):
    plot_name = ["replica", "cpu_utilization", "cpus", "reward", "resource_use"]
    step = []
    replicas = []
    cpu_utilization = []
    cpus = []
    reward = []
    # episode_reward = []
    # episode_idx = [x for x in range(total_episodes)]
    for episode in range(1, total_episodes+1):
        # step = []
        # replica = []
        # cpu_utilization = []
        # cpus = []
        # reward = []
        # print(episods_data)
        for parsed_line in episods_data[episode-1]:
            # parsed_line = episods_data[episode-1]
            step.append(parsed_line[0])
            # step.append()
            replicas.append(parsed_line[1][0])
            cpu_utilization.append(parsed_line[1][1]*100)
            cpus.append(parsed_line[1][2])
            reward.append(parsed_line[3])  # cost = -reward
        # episode_reward.append(sum(reward)/len(reward))
        resource_use = [x * y for x, y in zip(replicas, cpus)]
    replicas_ = moving_average(replicas)
    cpu_utilization_ = moving_average(cpu_utilization)
    cpus_ = moving_average(cpus)
    reward_ = moving_average(reward)
    resource_use_ = moving_average(resource_use)
    # plot_lsit = [replica, cpu_utilization, cpus, reward, resource_use]
    fig_add_Cpus(step, cpus, service_name)
    fig_add_Replicas(step, replicas, service_name)
    fig_add_Cpu_utilization(step, cpu_utilization_, service_name)
    fig_add_Resource_use(step, resource_use_, service_name)
    fig_add_reward(step, reward_, service_name)



tmp_count = 0
for p in path_list:
    # print(p)
    # f = open(p, "r")
    episods_data = parse(p)
    # step, replica, cpu_utilization, cpus, reward, resource_use = parse_episods_data(episods_data)
    parse_episods_data(episods_data, service[tmp_count])

    #step = [x * 30 for x in step]
    # print(y)

    ### plot delay
    # fig_add(step, reward, service[tmp_count])
    # fig_add(x, y2, service[tmp_count])
    # fig_add(x, y3, service[tmp_count])
    tmp_count += 1



tmp_dir = "result"
path_1 = tmp_dir + "/app_mn1_response.txt"
path_2 = tmp_dir + "/app_mn1_response.txt"
path_list = [path_1]

service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]

for path in path_list:
    tmp = 0
    with open(path, "r") as f:
        data = f.read().splitlines()
        response_time = []
        for line in data:
            rt = float(line.split()[2])
            if rt > 0.05:
                rt = 0.05
            response_time.append(rt)

        x = []
        y = []
        tmp1 = 0
        for i in range(0, len(response_time), 5):
            rt_sum = 0
            for j in range(i, i + 5):
                rt_sum += float(response_time[j])
            y.append(rt_sum / 5 * 1000)
            x.append(tmp1)
            tmp1 += 1
        y = moving_average(y)
        plt.figure()
        plt.plot(x, y, color="purple")  # color=color # label=label
        plt.title(service[tmp])
        plt.xlabel("step")
        plt.ylabel("Response")
        plt.grid(True)
        plt.xlim(0, total_episodes*120)
        plt.ylim(0, 60)
        plt.savefig("Response.png")
        plt.tight_layout()
        plt.show()