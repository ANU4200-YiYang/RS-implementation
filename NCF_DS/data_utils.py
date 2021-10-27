import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import tqdm

import config
import random

def processMethod(s:str)->int:
    if s=='F':
        return 0
    else:
        return 1

def listLength(neighbors:list,neighbor_num:int,all_num:int)->list:
    """
    :param neighbors: all neighbour nodes for each user/item：interacted item/user
    :param neighbor_num: sampled neighbors numbers，can be set in config
    :param all_num: total number of user/item ，if user/itemdo not have neighbour，randomly pick
    :return:
    """
    length = len(neighbors)
    if length>=neighbor_num:#sample neighbours
        neighbors = neighbors[:neighbor_num]
    elif length>0 and length<neighbor_num:#repeatly sample if do not have enough neighbours
        idxs = np.random.randint(0,length,size=neighbor_num)
        neighbors = [neighbors[i] for i in idxs]
    else:#randomly pick neighbours if no neighbours at all
        neighbors = np.random.randint(all_num,size=(neighbor_num)).tolist()
    return neighbors

def load_all():
    # read ratings for training, read by \t(4 empty character bit)，user first two columns and named [user, item]， 一994169 record
    train_data = pd.read_csv(config.train_rating, sep='\t',
                             header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    # read user demographic feature
    user_features = pd.read_csv(config.user_features,sep='::',header=None,
                                names=['user','gender','age','occupation','preference'],
                                usecols=[0,1,2,3],engine='python')

    # digitize gender
    user_features['gender'] = user_features.apply(lambda x: processMethod(x['gender']), axis=1)
    # two gender
    gender_num = user_features['gender'].max()+1
    # age:7 buckets
    old_age_ids = user_features['age'].unique()

    new_age_ids = list(range(len(old_age_ids)))

    age_id_transformer = dict(zip(old_age_ids,new_age_ids))

    user_features['age'] = user_features['age'].map(lambda x:age_id_transformer[x])
    age_num = user_features['age'].max()+1
    # 21 occupations
    occupation_num = user_features['occupation'].max()+1
    #
    # print(user_features)
    # exit()
    user_features = user_features.values.tolist()


    # user_num 6040
    user_num = train_data['user'].max() + 1
    # item_num 3952
    item_num = train_data['item'].max() + 1
    #  sparse%
    sparerate = 0.5
    train_data = train_data.sample(frac=1.0).reset_index(drop=True)
    if config.isSpareData:
        spare_train_data = []
        for user_is,records in tqdm.tqdm(train_data.groupby("user")):
            num_records = records.shape[0]
            spare_train_data.append(records.iloc[0:int(sparerate * num_records)])
        spare_train_data = pd.concat(spare_train_data)

        train_data = spare_train_data

    # expand to（[UserID, MovieID]。。。), len(data) = 994169 if not sparse
    # print(train_data)
    # exit()
    train_data = train_data.values.tolist()
    # create empty dok matrix(6040, 3952), row:UserID， col:MovieID，
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)


    # set 1 if entry not empty(means interacted)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # cencus user/item neighbour node information，set same neighbour number for convenience
    user_neighbors,item_neighbors = [],[]
    # transfer train_mat to normal matrix
    train_mat_user = train_mat.todense()
    # transpose
    train_mat_item = np.transpose(train_mat_user)
    # sample user/item neighbours
    for user in range(user_num):
        tmp_list = np.nonzero(train_mat_user[user])[1].tolist()
        random.shuffle(tmp_list)
        tmp_list = listLength(tmp_list,config.neighbor_num,item_num)
        user_neighbors.append(tmp_list)
    for item in range(item_num):
        tmp_list = np.nonzero(train_mat_item[item])[1].tolist()
        random.shuffle(tmp_list)
        tmp_list = listLength(tmp_list,config.neighbor_num,user_num)
        item_neighbors.append(tmp_list)

    test_data = []
    # leave-one-out test, each user's latest interaction as positive label
    # sample 99 negative label
    with open(config.test_negative, 'r') as fd:
        #
        line = fd.readline()
        while line is not None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    # testset ([UserID, MovieID(positive)], [UserID, MovieID(negative)]。。。)

    return train_data, train_mat, test_data, user_num, item_num,user_features,user_neighbors,item_neighbors,gender_num,age_num,occupation_num


class NCFData(data.Dataset):
    def __init__(self, features, user_num, num_item, user_featurs,user_neighbors,item_neighbors,train_mat=None, num_ng=0, is_training=None,):
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
        self.user_features = user_featurs
        self.user_neighbors = user_neighbors
        self.item_neighbors = item_neighbors

    def ng_sample(self):
        # sample negative label for each positive label in train set：994169 * (k+1)
        assert self.is_training, 'no need to sampling when testing'
        features_ng = []


        for x in self.features_ps:
            # UserID
            u = x[0]
            # random pick k un interacted item
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                features_ng.append([u, j])
        #
        # positive_label  994169 '1'
        labels_ps = [1 for _ in range(len(self.features_ps))]
        # negative_label 994169 * k '0'
        labels_ng = [0 for _ in range(len(features_ng))]
        # concatenation
        self.features_fill = self.features_ps + features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        # length
        return len(self.labels) * (self.num_ng + 1)

    def __getitem__(self, idx):
        # get item
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
        user_features = self.user_features
        user_neighbors = self.user_neighbors
        item_neighbors = self.item_neighbors
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        user_gender = user_features[user-1][1]
        user_age = user_features[user-1][2]
        user_occupation = user_features[user-1][3]
        user_neighbor = user_neighbors[user]
        item_neighbor = item_neighbors[item]
        return user, item, label,user_gender,user_age,user_occupation,user_neighbor,item_neighbor