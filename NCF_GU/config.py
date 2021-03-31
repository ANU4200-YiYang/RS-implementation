dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

model = 'NeuMF'
assert model in ['MLP', 'GMF', 'NeuMF', 'NeuMF-pre']

main_path = r'D:\desktop\ANUwattle\2021s1\ENGN4200\NCF\NCF\Data/'

train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'