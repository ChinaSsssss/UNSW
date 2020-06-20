import numpy as np
from scipy.spatial import distance


def pq(data, P, init_centroids, max_iter):
    M = data.shape[1]
    N = data.shape[0]
    K = init_centroids.shape[1]
    newdata = np.array(np.split(data, P, axis=1))
    codebooks = np.zeros(shape=(P, K, M // P), dtype=np.float32)
    # for each split data part
    for part in range(P):
        # update max_iter times for each part
        for times in range(max_iter):
            # for each data to calculate the distance to every center,768 data
            # 768 * 256 (every center and its distance to 768 data)
            # for each data, and its distance to 256 k
            newdis = manhattanDistance(newdata[part], init_centroids[part])
            # nearest is 1 * 768，means the nearest center to each data
            nearest = np.argmin(newdis, axis=1)

            for k in range(init_centroids[part].shape[0]):
                dimlist = []
                # find all the points belong to certain k and update the median
                for item in np.argwhere(nearest == k):
                    # data index is item and its dim
                    dimlist.append(newdata[part][item][0])
                if not dimlist:
                    continue
                init_centroids[part, k] = np.median(dimlist, axis=0)
            codebooks[part] = init_centroids[part]

    # using the newest codebooks and do the k-means, then get the one part of codes
    codes = np.zeros(shape=(N, P), dtype=np.uint8)
    for part in range(P):
        newdis = manhattanDistance(newdata[part], init_centroids[part])
        nearest = np.array(np.argmin(newdis, axis=1), dtype=np.uint8)
        if part == 0:
            # 1 * 768 and transfer it to 768 * 1 then become N*P
            first = nearest.reshape(N, -1)
        else:
            codes = np.concatenate((first, nearest.reshape(N, -1)), axis=1)
            first = codes[:]

    return codebooks, codes


def query(queries, codebooks, codes, T):
    qsize = queries.shape[0]
    P = codebooks.shape[0]
    M = queries.shape[1]
    N = codes.shape[0]
    newquery = queries.reshape(qsize, P, M // P)  # Q, P,M//P
    candidates = []
    if T >= N:
        for times in range(qsize):
            temp = set()
            for i in range(N):
                temp.add(i)
            candidates.append(temp)
        return candidates

    for times in range(qsize):
        # find the distance for each center to the query  (P,k) matrix
        # codebooks  p,k,m/p
        # newquery[times] , p,m/p
        disarray = np.array([])
        for p in range(codebooks.shape[0]):
            distance = manhattanDistance(newquery[times][p].reshape(1, -1), codebooks[p])
            if p == 0:
                # 1 * 256
                first = distance
            else:
                disarray = np.concatenate((first, distance), axis=0)
                first = disarray[:]
        # 2 * 256， the distance from one data block to all center
        nearindex = np.argsort(disarray, axis=1)
        curset = set()
        heap = []
        direction = generate(P)
        indexstart = [0] * P  # for p = 2 ,it is (0,0)
        cursum = 0
        dataindex = []
        for p, v in list(enumerate(indexstart)):
            cursum += disarray[p][nearindex[p][v]]
            dataindex.append(nearindex[p][v])
        heap.append([cursum, dataindex, indexstart])
        indexset = set()
        indexset.add(tuple(indexstart))
        while len(curset) < T:
            firstdata = heap.pop(0)
            first = np.array(firstdata[1])
            indexstart = np.array(firstdata[2])
            indexlist = data_index(codes, first)
            if indexlist.size != 0:
                for item in indexlist:
                    curset.add(item)
            # according to the newest element, and add one dimension then push them to heap
            for item in (indexstart + direction):
                cursum = 0
                dataindex = []
                for p, v in enumerate(item):
                    cursum += disarray[p][nearindex[p][v]]
                    dataindex.append(nearindex[p][v])
                tolist = item.tolist()
                totuple = tuple(tolist)
                if totuple not in indexset:
                    # binary insert and do not need to sort again
                    insertindex = binary_insert(heap, cursum)
                    heap.insert(insertindex, [cursum, dataindex, tolist])
                    indexset.add(totuple)
        candidates.append(curset)
    return candidates


def manhattanDistance(data1, data2):
    return distance.cdist(data1, data2, 'cityblock')


# indexdata is the data part center，eg,[2,2]means the index of the 0 part is 2,and 1 part is 2
def data_index(codesdata, indexdata):
    # res = []
    # for i in range(codesdata.shape[0]):
    #     if (codesdata[i, :] == indexdata).all():
    #         res.append(i)
    # return res
    return np.where(np.logical_and.reduce(codesdata == indexdata, axis=1))[0]


def generate(dim):
    start = 1
    # add one for each dimension
    res = []
    for _ in range(dim):
        temp = list(map(int, bin(start)[2:].zfill(dim)))
        start = start << 1
        res.append(temp)
    return np.array(res)


# return the index where to insert
def binary_insert(arr, num):
    l = 0
    r = len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if arr[mid][0] < num:
            l = mid + 1
        else:
            r = mid - 1
    return l



