import os
import config
import data_utils
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import torch.nn as nn
import modelGU
import torch.backends.cudnn as cudnn
import model
import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

train_data = pd.read_csv(config.train_rating, sep='\t',
                             header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
user_num=train_data.user.nunique()

user=torch.LongTensor(train_data.user.unique())
# print(type(user))
# print(user)
# print(user.shape[0]) 6040

# item_num=train_data.item.nunique()
item_num=train_data['item'].max() + 1
# print(type(train_data.item.unique()))
item=torch.LongTensor(np.linspace(0,item_num-1,num=item_num,dtype=int))
# print(type(item))
# print(item)
# print(item.shape[0])

# 3704

# embed_user_GMF = nn.Embedding(user_num, 32)
# tensor_user=torch.LongTensor(user)
# tensor([   0,    1,    2,  ..., 6037, 6038, 6039])
# print(type(tensor_user)) <class 'torch.Tensor'>
# print(tensor_user)
# print(tensor_user.shape[0]) 6040
# print(tensor_user)

# embed_user=embed_user_GMF(tensor_user)
# print(type(embed_user)) <class 'torch.Tensor'>
# print(embed_user)
# print(embed_user.shape)
# torch.Size([6040, 32])
# general_user=torch.mean((embed_user),0).repeat(item_num,1)
# user=general_user
# print(general_user)
# print(type(general_user))
# <class 'torch.Tensor'>
# print(general_user.shape)
# torch.Size([3704, 32])
model2= torch.load(config.NeuMF_model_path)
model=modelGU.NCF(user_num, item_num, 32, 3, 0, config.model, GMF_model=None, MLP_model=None).to("cpu")
model.load_state_dict(model2)
model.eval()
# model.NCF(user_num, item_num, 32, 3, 0, config.model, GMF_model, MLP_model).to("cuda" if torch.cuda.is_available() else "cpu")
# print(model.state_dict())
# print(general_user)

predictions = model(user, item)
print(predictions)
_, indices = torch.topk(predictions,10)
recommends = torch.take(item, indices).cpu().numpy().tolist()
print(recommends)


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
    HR.append(evaluate.hit(gt_item, recommends))
    NDCG.append(evaluate.ndcg(gt_item, recommends))
print(count)
print(np.mean(HR))
print(np.mean(NDCG))
