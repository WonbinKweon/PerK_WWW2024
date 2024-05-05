import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
import scipy
import math

############### For Model training ###############
def is_visited(base_dict, user_id, item_id):
    if user_id in base_dict and item_id in base_dict[user_id]:
        return True
    else:
        return False

def get_user_item_count_dict(interactions):
    user_count_dict = {}
    item_count_dict = {}

    for user, item in interactions:
        if user not in user_count_dict:
            user_count_dict[user] = 1
        else:
            user_count_dict[user] += 1

        if item not in item_count_dict:
            item_count_dict[item] = 1
        else:
            item_count_dict[item] += 1

    return user_count_dict, item_count_dict

def get_adj_mat(user_count, item_count, train_interactions):
    user_count_dict, item_count_dict = get_user_item_count_dict(train_interactions)

    A_indices, A_values = [[], []], []
    A_T_indices, A_T_values = [[], []], []
    for user, item in train_interactions:
        A_indices[0].append(user)
        A_indices[1].append(item)
        A_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

        A_T_indices[0].append(item)
        A_T_indices[1].append(user)
        A_T_values.append(1 / (user_count_dict[user] * item_count_dict[item]))

    A_indices = torch.LongTensor(A_indices)
    A_values = torch.FloatTensor(A_values)

    A = torch.sparse.FloatTensor(A_indices, A_values, torch.Size([user_count, item_count]))

    A_T_indices = torch.LongTensor(A_T_indices)
    A_T_values = torch.FloatTensor(A_T_values)

    A_T = torch.sparse.FloatTensor(A_T_indices, A_T_values, torch.Size([item_count, user_count]))

    return A.coalesce(), A_T.coalesce()

class traindset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_item, train_dic, train_pair, num_neg):
        super(traindset, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.train_dic = train_dic
        self.train_pair = train_pair
                         
    def negative_sampling(self, safe):      
        sample_list = np.random.choice(list(range(self.num_item)), size = safe * len(self.train_pair) * self.num_neg)

        self.train_arr = []
        bookmark = 0
        for user in range(self.num_user):
            train_list = self.train_dic[user]
            num_train = len(train_list) * self.num_neg

            neg_list = sample_list[bookmark:bookmark+num_train]
            bookmark = bookmark+num_train
            _, mask, _ = np.intersect1d(neg_list, train_list, return_indices=True)

            while True:
                if len(mask) == 0:
                    break
                neg_list[mask] = sample_list[bookmark:bookmark+len(mask)]
                bookmark = bookmark+len(mask)
                _, mask, _ = np.intersect1d(neg_list, train_list, return_indices=True)

            for i,j in zip(train_list, neg_list): ## num_neg가 1이 아닐 때 수정해야함 -> repeat
                self.train_arr.append((user, i, j))
       
        self.train_arr = np.array(self.train_arr)

    def __len__(self):
        return len(self.train_pair) * self.num_neg

    def __getitem__(self, idx):
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]

    
def evaluate(KS, topk_matrix, test_dic, num_item, reduce=True, only_NDCG=False):
    num_user = topk_matrix.shape[0]

    ## IDCG 유저별로 계산해두기
    idcg_ = [sum([1/np.log2(l+2) for l in range(K)]) for K in range(1, max(KS)+1)]
    idcg_matrix = np.tile(idcg_, num_user).reshape(num_user, max(KS))
    for u in range(num_user):
        idcg_matrix[u][len(test_dic[u])-1:] = idcg_matrix[u][min(max(KS)-1, len(test_dic[u])-1)] # len(test_dic[u])이 엄청 크면 넘어가는거 아님??

    ## 계산
    dcg_ = np.array([1/np.log2(K+2) for K in range(max(KS))])  
    dcg_matrix = topk_matrix * dcg_
    dcg_sum_matrix = np.cumsum(dcg_matrix, axis=-1)    
    ndcg_matrix = dcg_sum_matrix / idcg_matrix
    NDCG = ndcg_matrix[:, np.array(KS)-1]
    
    np.set_printoptions(precision=4)
    if only_NDCG:
        if reduce:
            return np.mean(NDCG, axis=0, keepdims=True)
        else:
            return NDCG
    
    hr_sum_matrix = np.cumsum(topk_matrix, axis=-1)
    HR = []
    F1 = []
    for u in range(num_user):
        item_pos = test_dic[u] # pos item idx

        HR.append([hr_sum_matrix[u][K-1] / min(len(item_pos), K) for K in KS])

        Pre_u = np.asarray([hr_sum_matrix[u][K-1] / K for K in KS])
        Rec_u = np.asarray([hr_sum_matrix[u][K-1] / len(item_pos) for K in KS])
        F1.append((2*Pre_u*Rec_u / (Pre_u + Rec_u + 0.000001)).tolist())
            
    if reduce:
        return np.mean(NDCG, axis=0, keepdims=True), np.mean(np.asarray(HR), axis=0, keepdims=True), np.mean(np.asarray(F1), axis=0, keepdims=True)
    else:
        return NDCG, np.asarray(HR), np.asarray(F1) 

def evaluate_p(KS, topk_matrix, test_dic, num_item, reduce=True, only_NDCG=False, penalized=False):
    num_user = topk_matrix.shape[0]

    ## IDCG 유저별로 계산해두기
    idcg_ = [sum([1/np.log2(l+2) for l in range(K)]) for K in range(1, max(KS)+1)]
    idcg_matrix = np.tile(idcg_, num_user).reshape(num_user, max(KS))
    for u in range(num_user):
        idcg_matrix[u][len(test_dic[u])-1:] = idcg_matrix[u][min(max(KS)-1, len(test_dic[u])-1)] # len(test_dic[u])이 엄청 크면 넘어가는거 아님??

    ## DCG 계산
    dcg_ = np.array([1/np.log2(K+2) for K in range(max(KS))])
    if penalized:
        dcg_matrix = (2*topk_matrix-1) * dcg_
    else:
        dcg_matrix = topk_matrix * dcg_
    dcg_sum_matrix = np.cumsum(dcg_matrix, axis=-1)    
    ndcg_matrix = dcg_sum_matrix / idcg_matrix
    NDCG = ndcg_matrix[:, np.array(KS)-1]
    
    np.set_printoptions(precision=4)
    if only_NDCG:
        if reduce:
            return np.mean(NDCG, axis=0, keepdims=True)
        else:
            return NDCG
    
    hr_sum_matrix = np.cumsum(topk_matrix, axis=-1)
    HR = []
    F1 = []
    for u in range(num_user):
        num_pos = len(test_dic[u]) # pos item idx

        HR.append([hr_sum_matrix[u][K-1] / min(num_pos, K) for K in KS])

        Pre_u = np.asarray([hr_sum_matrix[u][K-1] / K for K in KS])
        Rec_u = np.asarray([hr_sum_matrix[u][K-1] / num_pos for K in KS])
        F1.append((2*Pre_u*Rec_u / (Pre_u + Rec_u + 0.000001)).tolist())
            
    if reduce:
        return np.mean(NDCG, axis=0, keepdims=True), np.mean(np.asarray(HR), axis=0, keepdims=True), np.mean(np.asarray(F1), axis=0, keepdims=True), np.mean(dcg_sum_matrix, axis=0, keepdims=True)
    else:
        return NDCG, np.asarray(HR), np.asarray(F1) , dcg_sum_matrix

def evaluate_recall(KS, topk_matrix, test_dic, num_item, reduce=True, only_NDCG=False):
    num_user = topk_matrix.shape[0]
    
    hr_sum_matrix = np.cumsum(topk_matrix, axis=-1)
    Recall = []
    for u in range(num_user):
        item_pos = test_dic[u] # pos item idx

        Recall.append(np.array([hr_sum_matrix[u][K-1] / len(item_pos) for K in KS]))
            
    if reduce:
        return np.mean(np.array(Recall), axis=0, keepdims=True)
    else:
        return np.array(Recall)
    
def PB_CDF_RNA(ps, n):
    mu = ps.sum()
    std = np.sqrt(np.dot(ps,1-ps))
    gamma = np.power(std, -3)*np.dot(np.multiply(ps,1-ps), 1-2*ps)
    
    x = (n + 0.5 - mu) / std
    phi_x = scipy.stats.norm(0, 1).pdf(x)
    big_phi_x = scipy.stats.norm(0, 1).cdf(x)
    
    return big_phi_x + (gamma*(1-x**2)*phi_x)/6