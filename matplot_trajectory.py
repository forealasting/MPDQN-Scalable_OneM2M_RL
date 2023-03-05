import matplotlib.pyplot as plt
import statistics

# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
simulation_time = 300  # 3602 s

# moving for plot
moving_avg = 1
move = 10

# limit_cpus = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
tmp_dir = "all_result/result_0221"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"

service = ["First_level_mn1", "Second_level_mn2", "app_mnae1", "app_mnae2"]

# path_list = [path1]
path_list = [path1, path2]

def parse(p):
    with open(p, "r") as f:
        data = f.read().splitlines()  # 讀取所有行資料，去除換行符號
        parsed_data = []
        parsed_line = []

        for line in data:
            # parse data
            # line = re.sub(r"(?<=\[)\s+|\s+(?=\])", ",", line)
            # line = re.sub(r"(?<=\[)\s*|\s*(?=\])", ",", line)
            # print(line)
            # match = re.match(r"(\d+) \[(.+)\] (\d+) ([-+]?\d*\.\d+) \[(.+)\] (\w+)", line)
            match = re.match(r"(\d+) \[(.*?)\] (\d+) ([-+]?\d*\.\d+) \[(.*?)\] (\w+)", line)
            if match != None:
                print(match)
                # match = re.sub(r"\s+", ",", match.group(2))
                # print(match.group(2))
                tmp2 = match.group(2)
                tmp5 = match.group(5)
                # print(type(tmp2), tmp2)
                # print(tmp2.split("."))
                # print(tmp5.split("."))
                try:
                    tmp22 = float(int(tmp2.split(".")[3])/10)
                except:
                    tmp22 = 1
                try:
                    tmp55 = float(int(tmp5.split(".")[3]) / 10)
                except:
                    tmp55 = 1

                tmp2_list = [int(tmp2.split(".")[0]), int(tmp2.split(".")[1]), float(tmp22)]
                print(tmp2_list)
                tmp5_list = [int(tmp5.split(".")[0]), int(tmp5.split(".")[1]), float(tmp55)]
                #match1 = match.group(2).replace(' ', ',')
                #print(match1)
                # Convert the parsing result to the corresponding Python object
                # line_data  = [int(match.group(1)), eval("[" + match.group(2) + "]"), int(match.group(3)),
                #                float(match.group(4)), eval("[" + match.group(5) + "]"), match.group(6) == "True"]
                line_data = [int(match.group(1)), tmp2_list, int(match.group(3)),
                             float(match.group(4)), tmp5_list, match.group(6) == "True"]


                parsed_line.append(line_data)

                if match.group(6) == "True":
                    parsed_data.append(parsed_line)
                    parsed_line = []


    return parsed_data
    # return step, replica, cpu_utilization, cpus, reward, resource_use



# Plot --------------------------------------

def fig_add(x, y, label):
    #plt.subplot(pos)
    plt.plot(x, y, label=label)  # color=color

def moving_average(lst, move=5):
    ret = []
    for i in range(len(lst)):
        if i < move:
            ret.append(sum(lst[:i+1]) / (i+1))
        else:
            ret.append(sum(lst[i-move+1:i+1]) / move)
    return ret


def parse_episods_data(episods_data, service_name):
    plot_name = ["replica", "cpu_utilization", "cpus", "reward", "resource_use"]
    # step = []
    # replica = []
    # cpu_utilization = []
    # cpus = []
    # reward = []
    for episode in range(1, total_episodes+1):
        step = []
        replica = []
        cpu_utilization = []
        cpus = []
        reward = []
        for parsed_line in episods_data[episode-1]:
            # parsed_line = episods_data[episode-1]
            step.append(parsed_line[0])
            # step.append()
            replica.append(parsed_line[1][0])
            cpu_utilization.append(parsed_line[1][1])
            cpus.append(parsed_line[1][2])
            reward.append(-parsed_line[3])  # cost = -reward

        replica_ = moving_average(replica)
        cpu_utilization_ = moving_average(cpu_utilization)
        cpus_ = moving_average(cpus)
        reward_ = moving_average(reward)

        resource_use = [x * y for x, y in zip(replica, cpus)]
        resource_use_ = moving_average(resource_use)
        # plot_lsit = [replica, cpu_utilization, cpus, reward, resource_use]
        fig_add(step, cpu_utilization, "episode"+str(episode))

    plt.title(service_name)
    # plt.xlabel("timestamp")
    plt.ylabel("Cpu utilization(%) ")
    #plt.ylabel("Cost ")
    # plt.ylabel("Resource use ")
    # plt.ylabel("Replica")
    # plt.ylabel("Cpus")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, total_episodes*120)
    plt.ylim(0, 2)
    plt.savefig("Cpu_utilization.png")
    plt.tight_layout()
    plt.show()


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




# plt.title("Test")
