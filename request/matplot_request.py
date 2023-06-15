import matplotlib.pyplot as plt


# delay modify = average every x delay (x = 10, 50, 100)
# request rate r

simulation_time = 3660  # 300 s




path1 = "request14.txt"

path_list = [path1]

def cal_req(f):
    # time = [x for x in range(simulation_time)]
    req = []

    for line in f:
        req.append(float(line))

    f.close()

    return req


# Plot --------------------------------------

def fig_add(x, y, label):
    plt.plot(x, y, label=label)


for p in path_list:

    f = open(p, "r")
    x = [k for k in range(simulation_time)]

    y = cal_req(f)


    print(len(x), len(y))
    print(y)

    # print(y)

    ### plot delay
    fig_add(x, y, 'Machine1')


plt.title("Workload")
plt.xlabel("timestamp")
plt.ylabel("Data rate(requests/s) ")
plt.grid(True)
plt.ylim(0, 100)
# plt.legend()
plt.savefig("Data_rate.png", dpi=300)
plt.show()


