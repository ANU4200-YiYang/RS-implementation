import pandas as pd
import config
import numpy as np
import random

train_data = pd.read_csv(config.train_rating, sep='\t',
                             header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
test_data = []
# 99 not rated movies as negative label
with open(config.test_negative, 'r') as fd:
    line = fd.readline()
    while line is not None and line != '':
        arr = line.split('\t')
        u = eval(arr[0])[0]
        test_data.append([u, eval(arr[0])[1]])
        for i in arr[1:]:
            test_data.append([u, int(i)])
        line = fd.readline()
test_data=pd.DataFrame(test_data,columns=['user','item'])
# print(test_data)
# print(type(train_data))
# print(type(test_data))


cold_user=pd.DataFrame(train_data.user.unique()).sample(frac=0.1,random_state=1).sort_values(by=0)
cold_user.columns=['user']
# Int64Index([   3,   21,   24,   27,   28,   31,   41,   45,   51,   53,
#             ...
#             5920, 5923, 5924, 5935, 5970, 5975, 5981, 5993, 6018, 6025],
#            dtype='int64', length=604)
# cold_user.index[603]=6025


# train_data[train_data['user']==0].item
# train_data.loc[train_data['user']==0,'item']
np.random.seed(0)
random.seed(0)
test_data_cold=pd.DataFrame([])
for i in cold_user.index:
    # print(i)
    # print(len(train_data[train_data['user']==i].item))
    item_num=len(train_data[train_data['user']==i].item)
    n_drop=random.randint(item_num-9,item_num-1)
    # print(n_drop)
    drop_idx_train=np.random.choice(train_data[train_data['user']==i].item.index,n_drop,replace=False)
    # print(drop_idx_train)
    train_data=train_data.drop(drop_idx_train)

    test_data_cold = test_data_cold.append(test_data[test_data['user'] == i])
    # print(len(test_data_cold))
    test_data=test_data[test_data['user']!=i]
    # print(len(test_data))



    # print(train_data[train_data['user']==i].item)
train_data.to_csv("Data/train_cold", index=True,index_label='index')
# test_data.to_csv("Data/test_cold", index=True,index_label='index')
test_data_cold.to_csv("Data/test_cold", index=True,index_label='index')

