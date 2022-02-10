#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
import sys
import torch
from torch import nn
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch.nn import Module
from torch.nn import functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import OrderedDict
from pathlib import Path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device( "cpu")

class proxy_clf_network(Module):
    def __init__(
        self,
        domain_dims: list = [], 
        layer_dims:list  = [32,256,256],
        dropout_prob = 0.1
    ):
        """
        A simple 3 layered neural network, as per the paper
        """
        super(proxy_clf_network, self).__init__()
        self.domain_dims = domain_dims
        self.num_domains = len(domain_dims)
        self.layer_dims = layer_dims
        self.dropout_prob = dropout_prob
        self.__build__()
        return
    
    def __build__(self):
        """
        Build the architecture
        """
        emb_layer_dim = self.layer_dims[0]
        # Create an embedding layer for each domain
        embModule_list = []
        for  dim in self.domain_dims:
            embModule_list.append(nn.Embedding(dim, emb_layer_dim))
        self.embModule_list = nn.ModuleList(embModule_list)
        
        # The outputs should be concatenated
        fcn_layers = []
        dropout_prob = self.dropout_prob
        num_layers = len(self.layer_dims)
        inp_dim = emb_layer_dim * len(self.domain_dims)
        for i in range(1, num_layers):
            op_dim =  self.layer_dims[i]
            fcn_layers.append(nn.Linear(inp_dim,op_dim))
            fcn_layers.append(nn.Dropout(dropout_prob))
            fcn_layers.append(nn.ReLU())
            inp_dim = op_dim
        
        # Last layer for binary output
        fcn_layers.append(nn.Linear(inp_dim, 1))
        fcn_layers.append(nn.Sigmoid())                 
        self.fcn = nn.Sequential(*fcn_layers)
        return 
    
    def forward(self,X):
        """ 
        Input X : has shape [batch, num_domains, 1]
        """
       
        emb = []
        for i in range(self.num_domains):
            r = self.embModule_list[i](X[:,i])
            emb.append(r)
        emb = torch.cat(emb, dim =-1)
        
        x1 = self.fcn(emb)
        return x1


class proxy_clf(ClassifierMixin, BaseEstimator):
    """
    Container for the proxy model 
    """
    def __init__(
        self, 
        model: proxy_clf_network,
        dataset :str = None,
        batch_size: int = 512,
        LR: float = 0.001,
        save_dir = None,
        device = torch.device("cpu")
    ):
        self.model = model
        self.device = device
        
        self.signature = 'proxy_clf_{}'.format(dataset) 
        
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, self.signature  + '.pth')
        self.batch_size = batch_size
        self.LR = LR 
        self.dataset = dataset
        return
    
    def fit(
        self,
        X : np.array, 
        Y : np.array,
        num_epochs: int = 50,
        log_interval = 100
    ):
        self.model.train()
        self.model.to(self.device)
        bs = self.batch_size
        opt = torch.optim.Adam(list(self.model.parameters()), lr = self.LR)
        num_batches = X.shape[0] // bs + 1
        idx = np.arange(X.shape[0])
        loss_values = []
        clip_value = 5
        # train model 
        for epoch in range(num_epochs):
            np.random.shuffle(idx)
            epoch_loss = []
            for b in range(num_batches):
                opt.zero_grad() 
                b_idx = idx[b*bs:(b+1)*bs]
                x = LT(X[b_idx]).to(self.device) 
                pred_y = self.model(x)
                target_y = FT(Y[b_idx]).to(self.device)
                # Calculate loss
                loss = F.binary_cross_entropy(pred_y, target_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                opt.step()
                
                if b % log_interval == 0 :
                    print('[Epoch] {}  | batch {} | Loss {:4f}'.format(epoch, b, loss.cpu().data.numpy()))
                epoch_loss.append(loss.cpu().data.numpy())
            epoch_loss = np.mean(epoch_loss)
            loss_values.append(epoch_loss)
        return  loss_values  
    
    def predict(
        self, 
        X
    ):
        self.model.eval()
        result = []
        with torch.no_grad():
            bs = self.batch_size
            num_batches = X.shape[0] // bs + 1
            idx = np.arange(X.shape[0])
            for b in range(num_batches):
                b_idx = idx[b*bs:(b+1)*bs]
                x = LT(X[b_idx]).to(self.device)
                pred_y = self.model(x)
                pred_y = pred_y.cpu().data.numpy()
                result.extend(pred_y)
        return result
    

    def save_model(
        self
    ):
        """
        Save model 
        """
        
        torch.save(self.model, self.save_path)
        return

    def load_model(
        self, 
        path: str = None
    ):
        """
        Load Model
        """
        if self.save_path is None and path is None:
            print('Error . Null path given to load model ')
            return None
        print('Device', self.device)
        if path is None:
            path = self.save_path 
        
        self.model = torch.load(path)
        self.model.eval()
        return
    

# with open('./../../GeneratedData/us_import1/domain_dims.pkl','rb') as fh:
#     domain_dims = OrderedDict(pickle.load(fh))
# df = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# try:
#     del df['PanjivaRecordID']
# except:
#     pass
# X = df.head(1000).values
# Y = np.random.randint(0,2, size=[1000,1])
# network = proxy_clf_network(list(domain_dims.values()))
# clf_obj = proxy_clf(
#     model = network,
#     batch_size=512,
#     device = DEVICE    
# )
# clf_obj.fit(X,Y)