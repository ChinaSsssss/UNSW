## import modules here 
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################

def buc_rec_optimized(df):  # do not change the heading of the function
    res = pd.DataFrame(columns=df.columns.values)
    pre = []
    my_buc_rec_optimized(df, pre, res)
    return res


def my_buc_rec_optimized(df, pre, res):  # help recursive function
    dims = df.shape[1]
    if df.shape[0] == 1:
        single_tuple(df, pre, res)
    elif dims == 1:
        pre.append(sum(helper.project_data(df, 0)))
        res.loc[len(res)] = pre
    else:
        vals = set(helper.project_data(df, 0).values)
        pre_copy = pre.copy()
        for val in vals:
            pre = pre_copy.copy()
            sub_data = helper.slice_data_dim0(df, val)
            pre.append(val)
            my_buc_rec_optimized(sub_data, pre, res)

        sub_data = helper.remove_first_dim(df)
        pre = pre_copy.copy()
        pre.append("ALL")
        my_buc_rec_optimized(sub_data, pre, res)


def single_tuple(df, pre, res):
    cols = df.shape[1]
    last = []
    if cols > 1:
        last.append([df.iloc[0, 0]])
        last.append(["ALL"])
        for i in range(1, cols - 1):
            temp = []
            for j in last:
                jcopy = j.copy()
                jcopy.append(df.iloc[0, i])
                temp.append(jcopy)
            for j in last:
                jcopy = j.copy()
                jcopy.append("ALL")
                temp.append(jcopy)
            last = temp
        for i in last:
            i.append(df.iloc[0, cols - 1])
    else:
        last.append([df.iloc[0, cols - 1]])
    for i in last:
        pre_copy = pre.copy()
        pre_copy.extend(i)
        res.loc[len(res)] = pre_copy


# def read_data(filename):
#     df = pd.read_csv(filename, sep='\t')
#     return (df)

# input_data = read_data('./asset/f_.txt')
# # print(input_data)
# finalres = buc_rec_optimized(input_data)
# print(finalres)
