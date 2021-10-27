import torch
import torch.nn as nn




class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, dropout,gender_num,age_num,occupation_num,model,GMF_model=None):
        super(NCF, self).__init__()
        self.model = model
        self.GMF_model = GMF_model
        # define user item embeddin layer
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        # define embedding layer for user neighbour/item neighbour/user_gender/user_age/user_occupation
        self.embed_user_sage = nn.Embedding(user_num,factor_num)
        self.embed_item_sage = nn.Embedding(item_num,factor_num)
        self.embed_user_gender = nn.Embedding(gender_num,factor_num)
        self.embed_user_age = nn.Embedding(age_num,factor_num)
        self.embed_user_occupation = nn.Embedding(occupation_num,factor_num)

        # linear layer for fusing one feature(demographic)
        self.single_features_layer = nn.Linear(2*factor_num,factor_num)
        # linear layer for fusing all three feature(demographic)
        self.all_features_layer = nn.Linear(4*factor_num,factor_num)

        # model 2 predict layer
        self.predict_layer_2 = nn.Sequential(
            nn.Linear(4*factor_num,16),
            nn.Linear(16,1)
        )

        # model 4 linear layer for feature fusion(demographic+neighbour)
        self.com_user_layer = nn.Linear(2*factor_num,factor_num)
        self.com_item_layer = nn.Linear(2*factor_num,factor_num)
        # model 4 predict layer
        self.predict_layer_4 = nn.Sequential(
            nn.Linear(2*factor_num,16),
            nn.Linear(16,1)
        )
        # model 5 predict layer
        self.predict_layer_5 = nn.Sequential(

            nn.Linear(factor_num, 8),
            nn.Linear(8, 1)
        )
        # self.predict_layer_6= nn.Linear(factor_num, 1)
  

        self._init_weight_()

    def _init_weight_(self):
        if self.GMF_model != None:#always false,will not happen
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_sage.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_sage.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
        else:
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_sage.weight, std=0.01)
            nn.init.normal_(self.embed_item_sage.weight, std=0.01)
        # initialization
        nn.init.normal_(self.embed_user_age.weight, std=0.01)
        nn.init.normal_(self.embed_user_gender.weight, std=0.01)
        nn.init.normal_(self.embed_user_occupation.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()



    def forward(self, user, item, user_gender,user_age,user_occupation,user_neighbor,item_neighbor):
        user_id_embed = self.embed_user_GMF(user)
        item_id_embed = self.embed_item_GMF(item)

        ### for model ['GMF',,'GMF-sage'],means no demographic feature used
        user_embed_tmp = user_id_embed
        ### judge if use user demographic
        if 'age'in self.model:
            user_age_embed = self.embed_user_age(user_age)
            user_embed_tmp = self.single_features_layer(torch.cat((user_id_embed,user_age_embed),-1))
        elif 'gender' in self.model :
            user_gender_embed = self.embed_user_gender(user_gender)
            user_embed_tmp = self.single_features_layer(torch.cat((user_id_embed,user_gender_embed),-1))
        elif 'occupation' in self.model:
            user_occupation_embed = self.embed_user_occupation(user_occupation)
            user_embed_tmp = self.single_features_layer(torch.cat((user_id_embed,user_occupation_embed),-1))
        elif 'features' in self.model:
            user_age_embed = self.embed_user_age(user_age)
            user_gender_embed = self.embed_user_gender(user_gender)
            user_occupation_embed = self.embed_user_occupation(user_occupation)
            user_embed_tmp = self.all_features_layer(torch.cat((user_id_embed,user_occupation_embed,user_age_embed,user_gender_embed),-1))



        user_GMF_embed, item_GMF_embed = user_embed_tmp, item_id_embed
        ### judge if use neghbour information(sage)
        if 'sage' in self.model:
            # user/item_neighbors_embedding
            user_neighbors_embed_tmp = self.embed_item_sage(user_neighbor)
            item_neighbors_embed_tmp = self.embed_user_sage(item_neighbor)#torch.Size([256, 10, 32])

            # mean aggregate
            user_sage_embed = torch.mean(user_neighbors_embed_tmp, 1)#torch.Size([256, 32])
            item_sage_embed = torch.mean(item_neighbors_embed_tmp, 1)

            ########## model 1 #############
            # user_embed_final = torch.cat((user_GMF_embed,user_sage_embed),-1)
            # item_embed_final = torch.cat((item_GMF_embed,item_sage_embed),-1)

            #output_embedding = user_embed_final*item_embed_final
            #prediction = self.predict_layer_4(torch.cat((user_embed_final, item_embed_final), -1)).view(-1)

            ########## model 2 #############
            # user_embed_final = torch.cat((user_GMF_embed,user_sage_embed),-1)
            # item_embed_final = torch.cat((item_GMF_embed,item_sage_embed),-1)

            # prediction = self.predict_layer_2(torch.cat((user_embed_final,item_embed_final),-1)).view(-1)

            ########## model 3 #############
            # user_embed_final = torch.cat((user_GMF_embed,user_sage_embed),-1)
            # item_embed_final = torch.cat((item_GMF_embed,item_sage_embed),-1)
            # output_embedding = user_embed_final*item_embed_final
            # prediction = torch.sum(output_embedding, dim=-1).view(-1)

            ########## model 4 ################
            # user_embed_final = self.com_user_layer(torch.cat((user_GMF_embed,user_sage_embed),-1))
            # item_embed_final = self.com_item_layer(torch.cat((item_GMF_embed,item_sage_embed),-1))
            # prediction = self.predict_layer_4(torch.cat((user_embed_final,item_embed_final),-1)).view(-1)
            # # print(torch.cat((user_embed_final,item_embed_final)).shape)
            ########## model 5 ################
            user_embed_final = self.com_user_layer(torch.cat((user_GMF_embed, user_sage_embed), -1))#torch.Size([256, 32])
            item_embed_final = self.com_item_layer(torch.cat((item_GMF_embed, item_sage_embed), -1))
            output_embedding =user_embed_final*item_embed_final#torch.Size([256, 32])
            # print(output_embedding)
            # exit()
            prediction = self.predict_layer_5(output_embedding).view(-1)
            # prediction = torch.sum(output_embedding, dim=-1).view(-1)
            ########## model 6################
            # user_embed_final = self.com_user_layer(torch.cat((user_GMF_embed, user_sage_embed), -1))
            # item_embed_final = self.com_item_layer(torch.cat((item_GMF_embed, item_sage_embed), -1))
            # output_embedding = user_embed_final * item_embed_final
            #
            # prediction = self.predict_layer_6(output_embedding).view(-1)

        else:
            output_embedding = user_GMF_embed*item_GMF_embed
            # print(output_embedding)
            # exit()
            prediction = self.predict_layer_5(output_embedding).view(-1)
            # prediction = torch.sum(output_embedding, dim=-1).view(-1)

        # prediction = torch.sum(output_embedding, dim=-1).view(-1)
        return prediction