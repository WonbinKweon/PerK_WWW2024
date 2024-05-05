import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
import time

class BPR(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(BPR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.user_emb = nn.Embedding(self.num_user, emb_dim)
        self.item_emb = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
        
    # outputs logits
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        output = (pos_score, neg_score)
        
        return output

    def forward_pair(self, batch_user, batch_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        
        return pos_score

    def forward_eval(self, batch_user):
        return torch.mm(self.user_emb(batch_user), self.item_emb.weight.data.T)

    def get_loss(self, output):
        return -(output[0] - output[1]).sigmoid().log().sum()
        #return -(output[0] - output[1]).sigmoid().log().mean()
    

class LightGCN(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu, A, A_T, num_layer):
        super(LightGCN, self).__init__()
        self.user_count = user_count
        self.item_count = item_count

        self.user_list = torch.LongTensor([i for i in range(user_count)]).cuda(gpu)
        self.item_list = torch.LongTensor([i for i in range(item_count)]).cuda(gpu)

        self.user_emb = nn.Embedding(self.user_count, dim)
        self.item_emb = nn.Embedding(self.item_count, dim)

        self.A = A.to(gpu)  # user x item (sparse matrix)
        self.A_T = A_T.to(gpu)

        self.A.requires_grad = False
        self.A_T.requires_grad = False

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
        
        self.u_eval = None
        self.i_eval = None
        
        self.num_layer = num_layer

    def forward(self, user, pos_item, neg_item):
        u_0 = self.user_emb(self.user_list) # num_user x dim
        i_0 = self.item_emb(self.item_list)
        
        i_1 = torch.matmul(self.A_T, u_0) # 유저 평균 -> 아이템
        u_1 = torch.matmul(self.A, i_0)   # 아이템 평균 -> 유저
        
        if self.num_layer == 2:
            i_2 = torch.matmul(self.A_T, u_1)
            u_2 = torch.matmul(self.A, i_1)

            u_fin = (u_0 + u_1 + u_2) / 3
            i_fin = (i_0 + i_1 + i_2) / 3
        else:
            u_fin = (u_0 + u_1) / 2
            i_fin = (i_0 + i_1) / 2            
        
        u = torch.index_select(u_fin, 0, user)
        i = torch.index_select(i_fin, 0, pos_item)
        j = torch.index_select(i_fin, 0, neg_item)
         
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)
        
        return (pos_score, neg_score)

    def get_embedding(self):
        u_0 = self.user_emb(self.user_list) # num. user x dim
        i_0 = self.item_emb(self.item_list)

        i_1 = torch.matmul(self.A_T, u_0) # 유저 평균 -> 아이템
        u_1 = torch.matmul(self.A, i_0)   # 아이템 평균 -> 유저

        i_2 = torch.matmul(self.A_T, u_1)
        u_2 = torch.matmul(self.A, i_1)

        user = (u_0 + u_1 + u_2) / 3
        item = (i_0 + i_1 + i_2) / 3
        
        self.u_eval = user
        self.i_eval = item

        return user, item

    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]
        loss = -(pos_score - neg_score).sigmoid().log().sum()

        return loss

    def forward_pair(self, user, item):
        if self.u_eval == None or self.i_eval==None:
            u_0 = self.user_emb(self.user_list) # num_user x dim
            i_0 = self.item_emb(self.item_list)

            i_1 = torch.matmul(self.A_T, u_0) # 유저 평균 -> 아이템
            u_1 = torch.matmul(self.A, i_0)   # 아이템 평균 -> 유저

            if self.num_layer == 2:
                i_2 = torch.matmul(self.A_T, u_1)
                u_2 = torch.matmul(self.A, i_1)
                   
                self.u_eval = (u_0 + u_1 + u_2) / 3
                self.i_eval = (i_0 + i_1 + i_2) / 3
            else:
                self.u_eval = (u_0 + u_1) / 2
                self.i_eval = (i_0 + i_1) / 2
        
        u = torch.index_select(self.u_eval, 0, user)
        i = torch.index_select(self.i_eval, 0, item)
        
        return (u * i).sum(dim=1, keepdim=True)
        
    def forward_eval(self, batch_user):
        if self.u_eval == None or self.i_eval==None:
            u_0 = self.user_emb(self.user_list) # num_user x dim
            i_0 = self.item_emb(self.item_list)

            i_1 = torch.matmul(self.A_T, u_0) # 유저 평균 -> 아이템
            u_1 = torch.matmul(self.A, i_0)   # 아이템 평균 -> 유저

            if self.num_layer == 2:
                i_2 = torch.matmul(self.A_T, u_1)
                u_2 = torch.matmul(self.A, i_1)
                   
                self.u_eval = (u_0 + u_1 + u_2) / 3
                self.i_eval = (i_0 + i_1 + i_2) / 3
            else:
                self.u_eval = (u_0 + u_1) / 2
                self.i_eval = (i_0 + i_1) / 2

        return torch.matmul(self.u_eval[batch_user], self.i_eval.T)
        

class NeuMF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, num_hidden_layer):
        super(NeuMF, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.user_emb_MF = nn.Embedding(self.num_user, emb_dim)
        self.item_emb_MF = nn.Embedding(self.num_item, emb_dim)

        self.user_emb_MLP = nn.Embedding(self.num_user, emb_dim)
        self.item_emb_MLP = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb_MF.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb_MF.weight, mean=0., std= 0.01)

        nn.init.normal_(self.user_emb_MLP.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb_MLP.weight, mean=0., std= 0.01)

        # Layer configuration
        ##  MLP Layers
        MLP_layers = []
        layers_shape = [emb_dim * 2]
        for i in range(num_hidden_layer):
            layers_shape.append(layers_shape[-1] // 2)
            MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
            MLP_layers.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(* MLP_layers)
        print("MLP Layer Shape ::", layers_shape)
        
        ## Final Layer
        self.final_layer  = nn.Linear(layers_shape[-1]+emb_dim, 1)

        # Loss function
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        pos_score = self.forward_pair(batch_user, batch_pos_item)	 # bs x 1
        neg_score = self.forward_pair(batch_user, batch_neg_item)	 # bs x 1

        output = (pos_score, neg_score)

        return output

    def forward_pair(self, batch_user, batch_item):
        # MF
        u_mf = self.user_emb_MF(batch_user)			# batch_size x dim
        i_mf = self.item_emb_MF(batch_item)			# batch_size x dim
        
        mf_vector = (u_mf * i_mf)					# batch_size x dim

        # MLP
        u_mlp = self.user_emb_MLP(batch_user)		# batch_size x dim
        i_mlp = self.item_emb_MLP(batch_item)		# batch_size x dim

        mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_vector = self.MLP_layers(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        output = self.final_layer(predict_vector) 

        return output

    def forward_eval(self, batch_user):
        # MF
        u_mf = self.user_emb_MF(batch_user)			# batch_size x dim
        i_mf = self.item_emb_MF.weight.data #self.item_emb_MF(batch_item)			# num_item x dim
        
        u_mf_repeat = u_mf.repeat_interleave(self.num_item, dim=0) # batch_size*num_item x dim
        i_mf_repeat = i_mf.repeat(len(u_mf), 1) # num_item*batch_size x dim
        
        mf_vector = (u_mf_repeat * i_mf_repeat)					# batch_size*num_item x dim

        # MLP
        u_mlp = self.user_emb_MLP(batch_user)		# batch_size x dim
        i_mlp = self.item_emb_MLP.weight.data #self.item_emb_MLP(batch_item)		# num_item x dim

        u_mlp_repeat = u_mlp.repeat_interleave(self.num_item, dim=0) # batch_size*num_item x dim
        i_mlp_repeat = i_mlp.repeat(len(u_mlp), 1) # num_item*batch_size x dim
        
        mlp_vector = torch.cat([u_mlp_repeat, i_mlp_repeat], dim=-1)
        mlp_vector = self.MLP_layers(mlp_vector)
           
        # final layer
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        output = self.final_layer(predict_vector)

        return output.view(len(u_mlp), self.num_item)
    
    def get_loss(self, output):
        pos_score, neg_score = output[0], output[1]

        pred = torch.cat([pos_score, neg_score], dim=0)
        gt = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        
        return self.BCE_loss(pred, gt)

    
class CML(nn.Module):
    def __init__(self, user_count, item_count, dim, margin, gpu):
        super(CML, self).__init__()
        self.user_count = user_count
        self.item_count = item_count

        self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
        self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)

        # User / Item Embedding
        self.user_emb = nn.Embedding(self.user_count, dim, max_norm=1.) #
        self.item_emb = nn.Embedding(self.item_count, dim, max_norm=1.) #

        nn.init.normal_(self.user_emb.weight, mean=0., std= 1 / (dim ** 0.5))
        nn.init.normal_(self.item_emb.weight, mean=0., std= 1 / (dim ** 0.5))

        self.margin = margin

    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)

        pos_dist = ((u - i) ** 2).sum(dim=1, keepdim=True)
        neg_dist = ((u - j) ** 2).sum(dim=1, keepdim=True)

        output = (pos_dist, neg_dist)

        return output

    def get_loss(self, output):
        pos_dist, neg_dist = output[0], output[1]
        loss = F.relu(self.margin + pos_dist - neg_dist).sum()

        return loss

    def forward_pair(self, batch_user, batch_items):
        u = self.user_emb(batch_user)  # batch_size x dim
        i = self.item_emb(batch_items) # batch_size x dim

        dist = ((u - i) ** 2).sum(dim=1, keepdim=True)

        return -dist # for ranking

    def get_embedding(self):
        users = self.user_emb(self.user_list)
        items = self.item_emb(self.item_list)

        return users, items

class VAE(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(VAE, self).__init__()
        self.num_users = num_user
        self.num_items = num_item
        self.hid_dim = emb_dim
        
        self.E = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_item, emb_dim[0]),
            nn.Tanh(),
        )
        self.E_mean = nn.Linear(emb_dim[0], emb_dim[1])
        self.E_logvar = nn.Linear(emb_dim[0], emb_dim[1])

        self.D = nn.Sequential(
            nn.Linear(emb_dim[1], emb_dim[0]),
            nn.Tanh(),
            nn.Linear(emb_dim[0], num_item),
        ) 
        
    def forward(self, u):
        h = self.E(u)
        mu = self.E_mean(h)
        logvar = self.E_logvar(h)
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
                
        u_recon = self.D(z)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return u_recon, KL

    def get_loss(self, u, u_recon, KL, beta):
        nll = -torch.sum(u * nn.functional.log_softmax(u_recon))
        
        return nll + beta*KL

    def forward_pair(self, u):
        
        return None


class UBPR(nn.Module):
    def __init__(self, num_user, num_item, emb_dim, i_pop, gpu, eta):
        super(UBPR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.gpu = gpu
        self.eta = eta

        self.user_emb = nn.Embedding(self.num_user, emb_dim)
        self.item_emb = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
        
        self.i_propensity = torch.pow(i_pop / i_pop.max(), self.eta).cuda(gpu)

    def propensity(self, u, i):
        #return torch.pow(self.u_pop[u] / torch.max(self.u_pop), self.eta)
        propensities = self.i_propensity[i]

        return torch.max(propensities, torch.ones_like(propensities) * 0.1)
                
    # outputs logits
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        P_pos = self.propensity(batch_user, batch_pos_item)
        
        return pos_score, neg_score, P_pos

    def forward_pair(self, batch_user, batch_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        
        return pos_score     
    
    def get_loss(self, output):
        pos_score, neg_score, P = output[0], output[1], output[2]
        
        loss = -((pos_score - neg_score).sigmoid().log() / P).sum()
        
        return loss

## from https://github.com/Woody5962/Ranked-List-Truncation
class AttnCut(nn.Module):
    def __init__(self, input_size: int=1, d_model: int=256, n_head: int=4, num_layers: int=1, dropout: float=0.4):
        super(AttnCut, self).__init__()
        
        self.encoding_layer = nn.LSTM(input_size=input_size, hidden_size=int(d_model/2), num_layers=1, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1) #,
            #nn.Softmax(dim=1)
        )
        self.decison_layer_sm = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1), #,
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.encoding_layer(x)[0]
        x = self.attention_layer(x)
        x = self.decison_layer(x)
        return x
    
    def forward_sm(self, x):
        x = self.encoding_layer(x)[0]
        x = self.attention_layer(x)
        x = self.decison_layer_sm(x)
        return x 
    
    def get_loss(self, output, metric):
        output = torch.log(output.squeeze())
        loss_matrix = output.mul(metric)
        
        return -torch.sum(loss_matrix).div(output.shape[0])