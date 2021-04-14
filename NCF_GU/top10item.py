import config
import pandas as pd
import numpy as np
import evaluate

train_data = pd.read_csv(config.train_rating, sep='\t',
                             header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
recommendlist=train_data['item'].value_counts().iloc[0:10].index.tolist()
print(recommendlist)



test_data_cold=pd.read_csv("Data/test_cold", index_col='index')
HR, NDCG = [], []
count=0
for i in test_data_cold.user.unique():
    # print(i)
    item=test_data_cold[test_data_cold['user']==i].item
    # print(item)
    # exit()
    gt_item=item.iloc[0].item()#get positive label
    # print(type(gt_item))
    # print(gt_item)
    count+=1
    HR.append(evaluate.hit(gt_item, recommendlist))
    NDCG.append(evaluate.ndcg(gt_item, recommendlist))
print(count)
print(np.mean(HR))
print(np.mean(NDCG))

# [104, 124, 44, 64, 113, 48, 97, 132, 22, 128] same recommend but different orders!!!

# 604
# 0.016556291390728478
# 0.010277024968952574