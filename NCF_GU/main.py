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
from torch.utils.tensorboard import SummaryWriter


# environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
# train,test dataset
train_data, train_mat, test_data, user_num, item_num = data_utils.load_all()
train_dataset = data_utils.NCFData(train_data, user_num, item_num, train_mat, 4, True)
# print(train_dataset)
# exit()
train_loader = data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_dataset = data_utils.NCFData(test_data, user_num, item_num, train_mat, 0, False)
# cant change batch size of test dataset because of test strategy
test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None
# load
model = model.NCF(user_num, item_num, 32, 3, 0, config.model, GMF_model, MLP_model).to("cuda" if torch.cuda.is_available() else "cpu")
# loss
loss_function = nn.BCEWithLogitsLoss()
# opt
if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=0.001)
else:
	optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter()
best_epoch = 0
best_hr = 0
best_ndcg = 0
for epoch in range(20):
	model.train()
	#
	train_loader.dataset.ng_sample()
	count = 0
	# train
	for user, item, label in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()
		# print(user)
		# print(item)
		print(label)
		exit()

		optimizer.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		writer.add_scalar("Loss/train", loss, epoch)
		count += 1
		print("step:", count)
	writer.flush()
	# evaluation
	model.eval()
	writer.close()
	HR, NDCG = evaluate.metrics(model, test_loader, 10)
	print(r"HR: {:.3f} NDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
	# save best HR
	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
		if not os.path.exists(config.model_path):
			os.mkdir(config.model_path)
		torch.save(model.state_dict(), '{}{}.pth'.format(config.model_path, config.model))
	print("epoch:", epoch+1)
print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
# End. Best epoch 009: HR = 0.694, NDCG = 0.423
# cold：End. Best epoch 006: HR = 0.677, NDCG = 0.408
