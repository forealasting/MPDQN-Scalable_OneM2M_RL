import matplotlib.pyplot as plt

# r = 50
use_tm = 1
# data_name = '_tm1'
# data_name = str(r)
simulation_time = 300  # 300 s

# moving for plot
moving_avg = 0  # choose avg delay
move = 10

tmp_str = "result_om2m/senario19"

path1 = tmp_str + "/app_mn1_response.txt"
path2 = tmp_str + "/app_mn2_response.txt"
path3 = tmp_str + "/app_mnae1_response.txt"
path4 = tmp_str + "/app_mnae2_response.txt"
# path2 = tmp_str + "/output30.txt"
# path_list = [path1, path2, path3, path4]
path_list = [path1, path2, path3, path4]
service = ["app_mn1", "app_mn2", "app_mnae1", "app_mnae2"]

def cal_delay(f, use_tm, simulation_time):
    time = []
    delay = []
    for line in f:
        s = line.split(' ')

        try:
            tmp = float(s[1].rstrip('\n'))
            if tmp > 0 and tmp < 1:
                time.append(float(s[0]))
                delay.append(tmp * 1000)
        except:
            print(s[0])
    f.close()
    # print(len(delay))
    # delay = list(dict.fromkeys(delay))
    # print(len(delay))
    avg = sum(delay) / len(delay)
    max_d = max(delay)
    min_d = min(delay)
    print(avg, max_d, min_d)
    x = []
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

    y = delay
    if moving_avg:
        delay_m = []
        for i in range(len(delay)):
            if i < move:

                avg = sum(delay[:i + 1]) / (i + 1)
            else:
                avg = sum(delay[i - move:i + 1]) / move

            delay_m.append(avg)

        y = delay_m

    return x, y


def fig_add(x, y, label):
    plt.plot(x, y, label=label)


tmp_count = 0
for p in path_list:

    f = open(p, "r")
    x, y = cal_delay(f, use_tm, simulation_time)
    # print(len(x), len(y))
    # print(x, y)
    Rmax = 10
    result = filter(lambda x: x > Rmax, y)

    R1 = len(list(result)) / len(y)
    # print(R1)
    # print(y)

    ### plot # service[tmp_count] for show service name
    fig_add(x, y, service[tmp_count])
    tmp_count += 1

# plt.title("Test")
plt.ylim(0, 100)
plt.fill()
plt.xlabel("timestamp")
plt.ylabel("Responese time(ms)")
plt.grid(True)
plt.legend()
plt.savefig("Responese_time.png")
plt.show()
