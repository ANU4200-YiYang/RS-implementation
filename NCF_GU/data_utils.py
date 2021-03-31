import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import config


def load_all():
    # read train ratings, 994169 rows
    train_data = pd.read_csv(config.train_rating, sep='\t',
                             header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    #  6040 user
    user_num = train_data['user'].max() + 1
    # same as train_data.user.nunique()
    # max item id 3706
    item_num = train_data['item'].max() + 1
    # less than train_data.item.nunique()
    # transfer to list, len(data) = 994169
    train_data = train_data.values.tolist()
    # sparse matrix(6040, 3952), row-UserID， column-MovieID，
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    # if user have rating for item,1
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

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
    # test_data:1x([UserID, MovieID(rated)]+99x[UserID, MovieID(not rated)]。。。) =6040x100


    return train_data, train_mat, test_data, user_num, item_num


class NCFData(data.Dataset):
    def __init__(self, features, user_num, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        self.features_ps = features
        self.user = user_num
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.features_fill = []
        self.labels_fill = []
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        # 994169 * (1+num_ng)
        assert self.is_training, 'no need to sampling when testing'
        features_ng = []
        for x in self.features_ps:
            # UserID
            u = x[0]
            # random ng_num not rated movies
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)

                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])
        #
        # positive_label 994169 x 1
        labels_ps = [1 for _ in range(len(self.features_ps))]
        # negative_label 994169 x ng_num 0

        labels_ng = [0 for _ in range(len(features_ng))]

        #
        self.features_fill = self.features_ps + features_ng


        self.labels_fill = labels_ps + labels_ng



    def __len__(self):

        return len(self.labels) * (self.num_ng + 1)

    def __getitem__(self, idx):

        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label