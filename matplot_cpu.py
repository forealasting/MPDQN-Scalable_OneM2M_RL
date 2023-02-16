import matplotlib.pyplot as plt


# delay modify = average every x delay (x = 10, 50, 100)
# request rate r
# r = '100'
simulation_time = 3602  # 302 s

# moving for plot
moving_avg = 1
move = 10

# limit_cpus = 1
# tmp_str = "result2/result_cpu" # result_1016/tm1
tmp_dir = "result"
path1 = tmp_dir + "/app_mn1_cpu.txt"
path2 = tmp_dir + "/app_mn2_cpu.txt"

service = ["app_mn1", "app_mn2", "app_mnae1", "app_mnae2"]

# path_list = [path1]
path_list = [path1, path2]

def cal_cpu(f):
    cpu = []
    time = []

    for line in f:
        s = line.split(' ')
        if float(s[2]) > 0 and float(s[2]) < 100:
            # print(float(s[0]), float(s[2]))

            time.append(float(s[0]))
            cpu.append(float(s[2]))


    f.close()

    # calculate  cpu (ms) ---------------

    avg = sum(cpu) / len(cpu)
    max_d = max(cpu)
    min_d = min(cpu)
    print(avg, max_d, min_d)

    # calculate  cpu (ms) ---------------
    # print(len(time), len(cpu))
    x = []
    y = cpu

    count = 0
    for i in range(simulation_time):
        r = time.count(i)
        if r > 0:
            d = 1 / r
            for j in range(r):
                x.append(count)
                count += d
        else:
            count += 1

    # print(len(time), len(cpu))

    if moving_avg:
        cpu_m = []
        for i in range(len(cpu)):
            if i < move:

                avg = sum(cpu[:i + 1]) / (i + 1)
            else:
                avg = sum(cpu[i - move:i + 1]) / move

            cpu_m.append(avg)
        y = cpu_m


    return time, y


# Plot --------------------------------------


def fig_add(x, y, label):
    #plt.subplot(pos)
    plt.plot(x, y, label=label)  # color=color

# pos = 141
tmp_count = 0
for p in path_list:

    f = open(p, "r")
    x, y = cal_cpu(f)
    # print(x,y)
    # print(y)

    ### plot delay
    fig_add(x, y, service[tmp_count])
    tmp_count += 1
    # fig_add(x, y1, 'Machine2', 'blue')
    # fig_add(x, y2, 'Machine2', 'blue')
    # fig_add(x, y3, 'Machine2', 'blue')
    # fig_add(x, y4, 'Machine2', 'blue')



# plt.title("Test")

plt.xlabel("timestamp")
plt.ylabel("Cpu utilization(%) ")
plt.grid(True)
plt.legend()
plt.xlim(0, simulation_time)
plt.ylim(0, 110)
plt.savefig("Cpu_utilization.png")
plt.tight_layout()
plt.show()
