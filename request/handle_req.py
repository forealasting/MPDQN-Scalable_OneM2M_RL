
r = 7
path = "request" + str(r) + ".txt"
path1 = "request9.txt"

f = open(path, "r")
f1 = open(path1, 'a')

# request = []
# tmp_data = 0
# for line in f:
#     data = float(line)
#
#     data = data/5
#     data = int(data)
#
#     # if data != int(tmp_data):
#     #     # print(data, tmp_data)
#     request.append(data)
#
#     tmp_data = data


# f.close()
# print(request)
# print(len(request))
req_m = []

request = []
idx = 10
done = 1
for i in range(6):
    for j in range(50):
        request.append(idx)
    idx += 10
print(request)


for i in request:

    req_m.append(i)
    data = str(i) + '\n'
    f1.write(data)
f1.close()

# for i in request:
#     # for j in range(6):
#     req_m.append(i)
#     data = str(i) + '\n'
#     f1.write(data)
# f1.close()




