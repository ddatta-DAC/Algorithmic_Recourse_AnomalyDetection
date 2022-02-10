#!/usr/bin/env python
# coding: utf-8


import os
import torch
import sys
sys.path.append('./..')
from joblib import Parallel, delayed
import multiprocessing as MP
import pandas as pd
from typing import *
from scipy.linalg import det as Determinant
from scipy.special import softmax
from torch.nn import functional as F
import pickle
from tqdm import tqdm
from itertools import product
import numpy as np
from AD_model.AD_processor import AD_processor
from torch import FloatTensor as FT
from torch import LongTensor as LT
from timeit import default_timer as timer

class RCEAA:
    def __init__(
        self,
        dataset=None,
        domain_dims:list = None,
        domain_names:list = None,
        data_id_col:str = 'PanjivaRecordID',
        pretrained_ADmodel_emb_dim = 32,
        anom_score_threshold_perc = 10,
        hinge_W = 1,
        LR = 0.05
    ):
        
        self.hinge_W = hinge_W
        self.domain_names = domain_names
        self.domain_dims = domain_dims
        self.vector_dim = np.sum(domain_dims)
        self.ad_proc_obj = AD_processor(dataset, device = torch.device("cpu"))
        self.pretrained_ADmodel_emb_dim = pretrained_ADmodel_emb_dim
        self.anom_score_threshold = self.ad_proc_obj.threshold_dict[self.pretrained_ADmodel_emb_dim][anom_score_threshold_perc]
        self.split_idx = [0 for _ in domain_dims ]
        self.ad_obj  = self.ad_proc_obj.model_dict[pretrained_ADmodel_emb_dim]
        for i,d in enumerate(domain_dims):
            if i > 0:
                self.split_idx[i] = d + self.split_idx[i-1]
            else:
                self.split_idx[i] = d
        self.split_idx = self.split_idx[:-1] 
        self.LR = LR
        return
    
    def convertTo_01(
        self, 
        X: FT # N samples, d dimensions
    ):
        """
        Perform soft thresholding using softmax
        """
        # step 1 :split
        
        x1 = torch.split( X, self.domain_dims, dim = -1)
        res = []
        for _x in x1:
            res.append(torch.argmax(F.softmax(_x,dim=-1), dim=-1))
        
        # convert to 1-0
        x_10 = []
        for d, x in zip(self.domain_dims ,res):
            x = x.reshape([-1,1]).to(int)
            tmp = F.one_hot(x,d)
            x_10.append(tmp)
        res = torch.cat(x_10, dim=-1) 
        return res 
    
    def convert_cat_to_10(
        self,
        X : LT
    ): 
        x1 = torch.split(X,1,  dim = -1)
        x_10 = []
        for d, x in zip(self.domain_dims ,x1):
            x = x.reshape([-1,1]).to(int)
            tmp = F.one_hot(x,d)
            x_10.append(tmp)
        res = torch.cat(x_10, dim=-1) 
        return res 
        
    def convertTo_categorical(
        self, 
        X: FT # N samples, d dimensions
    ):
        """
        Perform soft thresholding using softmax
        """
        # step 1 :split
        x1 = torch.split( X, self.domain_dims , dim = -1)
        
        res = []
        for _x in x1:
            _x = _x.squeeze(1)
            res.append(torch.argmax(F.softmax(_x , dim=-1), dim=-1))
        res = torch.stack(res, dim=-1).to(torch.long) 
        return res 
   
    
#     def convertTo_categorical(
#         self, 
#         X: FT # N samples, d dimensions
#     ):
#         """
#         Perform soft thresholding using softmax
#         """
#         # step 1 :split
#         # x1 = torch.split( X, self.domain_dims , dim = -1)
#         x1 = X.reshape(-1, self.domain_dims, self.vector_dim ) 
#         x2 = torch.argmax(F.softmax(_x , dim=-1), dim=-1)
#         return x2
        
    
    
    def recLoss_surr(
        self, 
        X: FT # N samples, d dimensional
    ):
        """
        Surrogate loss for the Anomaly loss
        hinge function
        We substitute the AE loss by the AD model score 
        
        max(0, w * (threshold - score))
        """
        
        # 1. convert the input to categorical vectors
        x1 = self.convertTo_categorical(X)
       
        scores = self.ad_obj.predict_single_score(x1)
        scores = scores.reshape(-1)
        
        y = scores - self.anom_score_threshold 
        z = torch.abs(y * torch.lt(y,0).to(float))
        
        _loss = z * self.hinge_W
        return _loss
    
    @staticmethod
    def pairwiaseCosineDist(
        x
    ):
        a_norm = x / x.norm(dim=1)[:, None]
        b_norm = x / x.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0,1))
        res = 1 - res
        return res
    
    @staticmethod
    def kernel_DPP(
        x :FT,  # N samples, d dimensional
        
    ):
        """
        K_ij = 1/ (dist(x_i,x_j) +1)
        """
        x = x.squeeze()
        d1 = RCEAA.pairwiaseCosineDist (x)
        d2 = torch.reciprocal(d1 + 1)
        return d2
        
    def div_Loss(
        self, 
        X  # X is k * d dimensional vector
    ):
        """
        Diversity Loss
        """
        X = torch.split(X, int(self.vector_dim), dim=-1)
        X = torch.stack(X, dim=0)
        
        kernel_matrix = RCEAA.kernel_DPP(
            X
        )
        
        # Add perturbation to diagonal for numerical stability
        n = kernel_matrix.shape[0]
        p = torch.eye(n) * torch.randn(size = [n,n]) * 1e-4
        res = torch.linalg.det(kernel_matrix + p )
        return  res
    
    def dist_Loss(
        self,
        X_cf: FT,  # N shamples d dimension
        X_anom: FT # A vector of 1 row,d dimension, 1-0 encoded data
    ):
        """
        d(x,y) = Sum(|x-y|)/ MAD(attribute)
        """
        # columnwise MAD
        def MAD(idx, vector):
            _med = torch.median(vector)
            val = torch.median(torch.abs( vector - _med))
            return (idx, val)
        
        X_cf = X_cf.squeeze(1)
        MAD_x = torch.cat([X_cf, X_anom],dim=0)
            
        MAD_j = torch.ones(X_anom.shape[1])   
        for i in range(MAD_x.shape[1]):
             MAD_j[i] = MAD(i, MAD_x[:,i])[1]
        
        dist = torch.abs(X_cf - X_anom)/MAD_j                             
        return dist
    
    def loss_function(
        self,
        x_cf : FT,
        x_anom_01: FT,
        lambda_1,
        lambda_2
    ):
        
        x_cf = torch.split(x_cf, int(self.vector_dim), dim=-1)
        x_cf = [ _.squeeze(1) for _ in x_cf]
        x_cf = torch.stack(x_cf, dim=0)
        
        rec_loss = torch.mean(self.recLoss_surr(x_cf))
        div_loss = self.div_Loss(x_cf)
        dist_loss = self.dist_Loss(x_cf,x_anom_01)
        
        _loss = torch.mean(rec_loss) + lambda_1* torch.mean(dist_loss) + lambda_2 * torch.mean(div_loss)
        return _loss
    
    def optimize(
        self,
        x_input,
        x_anom_01,
        lambda_1,
        lambda_2,
        max_iter, 
    ):
        key = str(lambda_1) + '_' + str(lambda_2)
        print(key)
        x_input = x_input.to(float)
        x_input.requires_grad = True
        opt = torch.optim.Adam([x_input], lr = self.LR)
        loss_values = []
        for i in range(max_iter):
            opt.zero_grad()
            loss = self.loss_function(x_input, x_anom_01,lambda_1,lambda_2)
            loss.backward()
            loss_values.append(np.mean(loss.cpu().data.numpy()))
            opt.step()   
        # print(key, 'Loss', np.mean(loss_values))
        return (key, np.mean(loss_values[-3:]), x_input)
    
    def find_CF(
        self,
        anomaly_record : pd.Series, # single row,
        num_cf = 10,
        max_iter = 100,
        hyp_step = 0.25,
        lambda_1_range = [0,1],
        lambda_2_range = [0,1]
    ):
        # Initialize input
        x_anom = []
        for dom in self.domain_names:
            x_anom.append(anomaly_record[dom])
        x_anom = np.array(x_anom).reshape([1,-1])
        x_anom = FT(x_anom)
        x_anom_10 = self.convert_cat_to_10(x_anom)
        x_anom_10 = x_anom_10.squeeze(1)
        
        
        # Initialize the result vector
        x_inp = torch.tile( x_anom_10.reshape(-1), [num_cf])
        x_inp = x_inp.reshape(1,x_inp.shape[0])
        x_inp = x_inp + torch.randn(x_inp.shape)
    
        lambda_1_values = np.arange(lambda_1_range[0] , lambda_1_range[1] + hyp_step,hyp_step)
        lambda_2_values = np.arange(lambda_1_range[0] , lambda_1_range[1] + hyp_step,hyp_step)
        n_jobs = len(lambda_1_values) * len(lambda_2_values)
        # n_jobs = 1
            
        result = Parallel(n_jobs = n_jobs, prefer="threads")(delayed(self.optimize)(x_inp,
            x_anom_10,
            l1,
            l2,
            max_iter ) for l1,l2 in product(lambda_1_values,lambda_2_values))
        # result = []
        # for l1,l2 in product(lambda_1_values,lambda_2_values):
        #     res = self.optimize(x_inp,x_anom_10,l1,l2, max_iter )
        #     result.append(res)
        dict_key_loss = {}
        dict_key_vec = {}
        for item in result:
            dict_key_loss[item[0]] = item[1]
            dict_key_vec[item[0]] = item[2]
            
        dict_key_loss = sorted(dict_key_loss.items(), key = lambda item: item[1])
        key = dict_key_loss[0][0]
        vec = dict_key_vec[key].reshape([-1, int(self.vector_dim)])
        res = self.convertTo_categorical(vec)
        return res


# In[3]:


# DIR = 'us_import1'
# df1 = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# df2 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_2.csv', index_col=None)

# with open('./../../GeneratedData/us_import1/domain_dims.pkl', 'rb') as fh:
#      domain_dims = pickle.load(fh)

# obj = RCEAA(
#     dataset=DIR,
#     domain_dims = list(domain_dims.values()),
#     domain_names = list(domain_dims.keys())
# )

# start = timer()

# x = obj.find_CF(
#     df2.iloc[100],
#     max_iter = 100,
#     hyp_step = 0.25
# )

# end = timer()
# print('[Time]', end-start)


# In[ ]:





# In[ ]:




