import numpy as np
import matplotlib.pyplot as plt
import math


# 𝛤 = 10
# 𝑇_𝑚𝑎𝑥 = 20
# 𝑟_𝑡 = np.linspace(0, 50, 10000)  # Generate 100 equally spaced points between 0 and 50
#
# 𝑦 = np.where(𝑟_𝑡 <= 𝑇_𝑚𝑎𝑥, np.exp(𝛤 * (𝑟_𝑡 - 𝑇_𝑚𝑎𝑥) / 𝑇_𝑚𝑎𝑥), 1)


# cost 2
# B = 10
# t_max = 20
# target = t_max + 2*math.log(0.9)
#
# r = np.linspace(0, 50, 10000)  # Generate 100 equally spaced points between 0 and 50
#
# y = np.where(r <= target, np.exp(B * (r - t_max) / t_max), 0.9 + ((r - target) / (50 - target)) * 0.1)


# cost 3
t_max = 20
B = 0.27
T_upper = 50

r = np.linspace(0, 50, 10000)  # create response time data , 10000 point in (50, 10000)
y = np.where(r <= t_max, 0, np.exp(B*(r - t_max) / t_max)-0.5)



plt.plot(r, y)
plt.xlabel('r')
plt.ylabel('c_perf')
plt.yticks(np.arange(0., 1.1, 0.1))
plt.title('')
plt.grid(True)
plt.savefig("cperf_function.png", dpi=300)
plt.show()


# r = 50
# y = np.where(r <= target, np.exp(B * (r - t_max) / t_max), 0.9 + ((r - target) / (50 - target)) * 0.1)
# print(r, y)

# B = np.log(1+0.5)/((T_upper-t_max)/t_max)
#
# print(B)