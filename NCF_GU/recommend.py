
import config
import data_utils
import numpy as np
import torch
import torch.utils.data as data

train_data, train_mat, test_data, user_num, item_num = data_utils.load_all()

train_dataset = data_utils.NCFData(train_data, user_num, item_num, train_mat, 4, True)
train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_dataset = data_utils.NCFData(test_data, user_num, item_num, train_mat, 0, False)
test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
model= torch.load(config.NeuMF_model_path)
model.eval()
recommend_list=[]
train_loader.dataset.ng_sample()
for user, item, label in train_loader:
    user = user.cuda()
    # print(type(user))
    # print(user.size())


    item = item.cuda()
    # print(type(item))
    # print(item.size())
    # exit()
    predictions = model(user, item)
    # print(type(predictions))
    # print(predictions.size())
    # exit()
    _, indices = torch.topk(predictions,10)
    recommends = torch.take(item, indices).cpu().numpy().tolist()
    recommend_list+=recommends
recommend_list=np.array(recommend_list).reshape(-1,10)
input=eval(input('userid?'))
print(recommend_list[input])