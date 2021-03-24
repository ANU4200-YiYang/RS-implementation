
import config
import data_utils
import numpy as np
import torch
import torch.utils.data as data

train_data, train_mat, test_data, user_num, item_num = data_utils.load_all()
test_dataset = data_utils.NCFData(test_data, user_num, item_num, train_mat, 0, False)
test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
model= torch.load(config.NeuMF_model_path)
model.eval()
recommend_list=[]
for user, item, label in test_loader:
    user = user.cuda()
    item = item.cuda()
    predictions = model(user, item)
    _, indices = torch.topk(predictions,10)
    recommends = torch.take(item, indices).cpu().numpy().tolist()
    recommend_list+=recommends
recommend_list=np.array(recommend_list).reshape(-1,10)
input=eval(input('userid?'))
print(recommend_list[input])