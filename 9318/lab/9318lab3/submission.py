## import modules here
import numpy as np


################# Question 1 #################

def hc(data, k):  # do not change the heading of the function

    # two special cases
    if k >= len(data):
        return [i for i in range(len(data))]
    if k == 1:
        return [0] * len(data)

    # make simlarity table
    simtable = np.array([[-1 for _ in range(len(data))] for _ in range(len(data))], dtype=float)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            temp = dot_product(data[i], data[j])
            simtable[i][j] = temp
            simtable[j][i] = temp

    # print(simtable)
    point = []

    for step in range(len(data) - k):

        # find maxvalue
        maxnum = -1
        maxi, maxj = -1, -1
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if simtable[i][j] > maxnum:
                    maxnum = simtable[i][j]
                    maxi = i
                    maxj = j

        point.append([maxi, maxj])
        simtable[maxi][maxj] = -1
        simtable[maxj][maxi] = -1

        # change the simtable
        for col in range(len(data)):
            newnum = min(simtable[col][maxi], simtable[col][maxj])
            simtable[maxi][col] = newnum
            simtable[col][maxi] = newnum
            simtable[maxj][col] = -1
            simtable[col][maxj] = -1
        # remove the value in all bigger index
        for col in range(len(data)):
            simtable[col][maxj] = -1
            simtable[maxj][col] = -1

    res = [0] * len(data)
    visited = [0] * len(data)
    start = 0
    # print(simtable)
    print(point)
    for i in range(len(point) - 1, -1, -1):
        if visited[point[i][0]] == -1 or visited[point[i][1]] == -1:
            continue
        tempset = set()
        tempset.add(point[i][0])
        tempset.add(point[i][1])
        for j in range(i - 1, -1, -1):
            if point[j][0] in tempset or point[j][1] in tempset:
                tempset.add(point[j][0])
                tempset.add(point[j][1])

        for i in tempset:
            res[i] = start
            visited[i] = -1

        start += 1

    for i in range(len(res)):
        if visited[i] == 0:
            res[i] = start
            start += 1

    return res


def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


# def compute_error(data, labels, k):
#     n, d = data.shape
#     centers = []
#     for i in range(k):
#         centers.append([0 for j in range(d)])
#
#     for i in range(n):
#         centers[labels[i]] = [x + y for x, y in zip(centers[labels[i]], data[i])]
#
#     error = 0
#     for i in range(n):
#         error += dot_product(centers[labels[i]], data[i])
#     return error


k = 4
data = np.loadtxt('asset/data.txt', dtype=float)
print(hc(data, k))
# print(compute_error(data, hc(data, k), k))
