import math
import statistics
t_max = 25
Rt = 0
a = [0.06882095336914062,
0.09272193908691406,
0.10181951522827148,
0.02005767822265625,
0.1 ]
print(statistics.mean(a))
tmp_d = math.exp(50 / t_max)
tmp_n = math.exp(Rt / t_max)
c_perf = tmp_n / tmp_d


w_pref = 0.5
c_perf = 0 + ((c_perf - math.exp(-2)) / (1 - math.exp(-2))) * (1 - 0)
reward_perf = (w_pref * c_perf)

print(reward_perf)