import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import model
import config
import evaluate
import data_utils
import random
import winsound

# set random seed
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2021)
# hyperparameter
factor_num = 32
dropout = 0# deleted,do not use it
epochs = 60
batch_size = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# cudnn.benchmark = True
# read data
train_data, train_mat, test_data, user_num, item_num, \
user_featurs,user_neighbors,item_neighbors,gender_num,age_num,occupation_num= data_utils.load_all()

# dataloader
train_dataset = data_utils.NCFData(train_data, user_num, item_num, user_featurs,user_neighbors,item_neighbors, train_mat, 4, True)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataset = data_utils.NCFData(test_data, user_num, item_num, user_featurs,user_neighbors,item_neighbors,train_mat, 0, False)
# do not change batch_size of test set -- leave on out test
test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
# do not use pretraining
if config.model == 'GMF':
	GMF_model = None
else:
	# assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	# GMF_model = torch.load(config.GMF_model_path)
	GMF_model = None

# model
model = model.NCF(user_num, item_num, factor_num, dropout,gender_num,age_num,occupation_num,config.model,GMF_model).to("cuda" if torch.cuda.is_available() else "cpu")
# loss=binary cross entropy+sigmoid
loss_function = nn.BCEWithLogitsLoss()
# optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)


best_epoch = 0
best_hr = 0
best_ndcg = 0
print('start training...')
print(config.model)
for epoch in range(epochs):
	model.train()
	# sample negative label for each positive label in train set in every epoch
	train_loader.dataset.ng_sample()

	# count = 0
	# train
	for user, item, label,user_gender,user_age,user_occupation,user_neighbor,item_neighbor in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		user_gender, user_age, user_occupation,  = user_gender.cuda(), user_age.cuda(), user_occupation.cuda(),

		user_neighbor = (torch.tensor([item.numpy() for item in user_neighbor]).cuda()).T
		item_neighbor = (torch.tensor([item.numpy() for item in item_neighbor]).cuda()).T#(256,10)

		model.zero_grad()
		prediction = model(user, item, user_gender,user_age,user_occupation,user_neighbor,item_neighbor)

		loss = loss_function(prediction, label)

		loss.backward()
		optimizer.step()
		# count += 1
		# print('loss:\t',loss.item())
		# print("step:", count)
	# evaluation(test)
	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, 10)
	print(r"HR: {:.3f} NDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
	# save best HR
	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if not os.path.exists(config.model_path):
			os.mkdir(config.model_path)
		torch.save(model, '{}{}.pth'.format(config.model_path, config.model))
	print("epoch:", epoch+1)
print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
print(config.model)
winsound.Beep(600,1000)
