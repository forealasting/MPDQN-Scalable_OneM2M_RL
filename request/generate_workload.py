import numpy as np
import math
path1 = "request17.txt"

f1 = open(path1, 'a')


# def generate_smooth_data_rate_pattern(total_time):
#     data_rate_pattern = []
#     timestamp = 0
#     base_rate = 20
#     increasing = True
#
#     while timestamp < total_time:
#         for t in range(240):
#             if timestamp >= total_time:
#                 break
#             # 使用正弦函數的一半來進行變化，當increasing為True時增加，為False時減少
#             data_rate = base_rate + 10 * (1 - math.cos(math.pi * t / 240)) if increasing else base_rate + 10 * (1 + math.cos(math.pi * t / 240))
#             data_rate_pattern.append(int(data_rate))
#             timestamp += 1
#
#         # 當base_rate達到80時開始減少，達到20時開始增加
#         if base_rate == 80:
#             increasing = False
#         elif base_rate == 20:
#             increasing = True
#
#         # 根據increasing的狀態調整base_rate
#         base_rate += 20 if increasing else -20
#
#     return data_rate_pattern[:total_time]



# request17.txt
def generate_data_rate_pattern(total_time):
    data_rate_pattern = []
    timestamp = 0
    data_rate = 20
    increasing = True

    while timestamp < total_time:
        for _ in range(12):
            data_rate_pattern.append(data_rate)
        if increasing:
            if data_rate < 80:
                data_rate += 20
            else:
                increasing = False
        else:
            if data_rate > 20:
                data_rate -= 20
            else:
                increasing = True
        timestamp += 240

    return data_rate_pattern[:total_time]

pattern = generate_data_rate_pattern(3660)
print(len(pattern))

with open(path1, 'a') as f1:
    for i in pattern:
        data = str(i) + '\n'
        f1.write(data)





