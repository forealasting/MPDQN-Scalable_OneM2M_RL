import matplotlib.pyplot as plt


simulation_time = 3660  # 300 s

path1 = "request14.txt"

path_list = [path1]

def get_req(f):
    # time = [x for x in range(simulation_time)]
    req = []
    for line in f:
        req.append(float(line))

    f.close()

    return req

def step_req(y):

    y_ = []
    for i in range(0, len(y), 60):
        y_.append(y[i])

    return y_


# Plot --------------------------------------

def fig_add(x, y):
    plt.plot(x, y)
    plt.title("Workload")
    plt.xlabel("timestamp")
    plt.ylabel("Data rate(requests/s) ")
    plt.grid(True)
    plt.ylim(0, 100)
    # plt.legend()
    plt.savefig("Data_rate.png", dpi=300)
    plt.show()

def fig_add1(x, y):
    plt.plot(x, y)
    plt.title("Workload")
    plt.xlabel("step")
    plt.ylabel("Data rate(requests/s) ")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig("Data_rate1.png", dpi=300)
    plt.show()


for p in path_list:

    f = open(p, "r")

    y = get_req(f)
    x = [k for k in range(len(y))]


    y_ = step_req(y)
    x_ = [x for x in range(len(y_))]

    ### plot delay
    fig_add(x, y)
    fig_add1(x_, y_)
    print(y_)




