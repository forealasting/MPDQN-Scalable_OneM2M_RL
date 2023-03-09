import math
import statistics
t_max = 25
Rt = 0
a = [0.012551069259643555, 0.017151594161987305, 0.013333320617675781, 0.010972738265991211, 0.024749279022216797]
print(statistics.mean(a))
b = statistics.mean(a) * 1000
Rt = b
tmp_d = math.exp(50 / t_max)
tmp_n = math.exp(Rt / t_max)
c_perf = tmp_n / tmp_d


w_pref = 0.5
c_perf = 0 + ((c_perf - math.exp(-50/t_max)) / (1 - math.exp(-50/t_max))) * (1 - 0)
reward_perf = (w_pref * c_perf)

print(reward_perf)