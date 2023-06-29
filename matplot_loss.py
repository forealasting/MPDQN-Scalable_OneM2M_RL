import re
import matplotlib.pyplot as plt
tmp_dir = "mpdqn_result/result8/"
path1 = tmp_dir + "/app_mn1_trajectory.txt"
path2 = tmp_dir + "/app_mn2_trajectory.txt"


path = "loss/app_mn1_actor_loss.txt"
path1 = "loss/app_mn1_actor_loss.txt"
path2 = "loss/app_mn1_actor_loss.txt"
path3 = "loss/app_mn1_actor_loss.txt"


with open('loss/app_mn1_actor_loss.txt', 'r') as file:

    data = [float(re.search(r'-?\d+\.\d+', line).group()) for line in file]

# generate index
x = list(range(len(data)))

# plot
plt.plot(x, data)
plt.xlabel('time')
plt.ylabel('loss')
plt.title('actor_loss')
plt.savefig
#plt.title('critic_loss')
plt.show()