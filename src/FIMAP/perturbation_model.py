#!/usr/bin/env python
# coding: utf-8

# In[38]:

from time import time
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
import proxyClassifier
from proxyClassifier import proxy_clf, proxy_clf_network

class gumbel(nn.Module):
    def __init__(self, dim, tau):
        super(gumbel, self).__init__()
        self.layer = F.gumbel_softmax
        self.size = dim
        self.tau = tau
        
    def forward(self, x):
        x = self.layer(
            x, 
            self.tau
        )
        return x


class perturb_network(nn.Module):
    def __init__(
        self,
        domain_dims: list = [], 
        layer_dims:list  = [32,256,128,128],
        dropout_prob = 0.2,
        gumbel_tau = 0.5
    ):
        super(perturb_network, self).__init__()
        self.gumbel_tau = gumbel_tau
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
        self.fcn = nn.Sequential(*fcn_layers)
        # Projection layer 
        self.projModule_list = []
        for dim in self.domain_dims:
            self.projModule_list.append(nn.Sequential(
                nn.Linear(op_dim, dim),
                gumbel(dim, self.gumbel_tau)
            ))
        self.projModule_list = nn.ModuleList(self.projModule_list)      
        return 
    
    
    def constrain(
        self,
        X 
    ):
        """
        This function allows for perturbation in pre-defined
        indices
        ToDo : implement 
        """
        return 
    
    def forward(
        self, 
        X
    ):
        emb = []
        for i in range(self.num_domains):
            r = self.embModule_list[i](X[:,i])
            emb.append(r)
        emb = torch.cat(emb, dim =-1)
        x1 = self.fcn(emb)
        x2 = []
        for i in range(self.num_domains):
            r = self.projModule_list[i](x1)
            r = torch.argmax(r , dim=1, keepdims=True)
            x2.append(r)
        x2 = torch.cat(x2, dim=-1)
        return x2




class perturb_clf(ClassifierMixin, BaseEstimator):
    """
    Container for the proxy model 
    """
    def __init__(
        self, 
        perturb_model: perturb_network,
        clf_model: proxy_clf_network,
        signature = None,
        dataset :str = None,
        batch_size: int = 512,
        LR: float = 0.001,
        eta:float = 2.0, # From the paper
        device = torch.device("cpu"),
    ):
        self.perturb_model = perturb_model
        self.signature = 'proxy_{}'.format(dataset) 
        if signature is not None:
            self.signature += '_' + str(signature)
        self.device = device
        self.batch_size = batch_size
        self.LR = LR 
        self.clf_model = clf_model
        self.eta = eta
        self.dataset= dataset
        self.num_domains = perturb_model.num_domains
        t = str(int(time()))
        self.chkpt_path = './checkpoints_' + dataset + '/{}_{}'.format(t,signature)
        Path(self.chkpt_path).mkdir(exist_ok=True,parents=True)
        self.chkpt_path = os.path.join(self.chkpt_path, 'chkpt_{}.pt')
        return
    
    def calc_discrete_reg(self, X_data, X_pert):
        diff = []
        for idx in range(self.num_domains):
            x1 = X_data[:,idx]
            x2 = X_pert[:,idx]
            _diff = torch.eq(x1, x2).to(int)
            diff.append(_diff)
        diff = torch.cat(diff,dim=-1)
        diff = diff.to(float)
        reg = self.eta * torch.mean(diff)
        return reg
    
    
    def save_checkpoint(
        self,
        epoch_idx):
        save_path = self.chkpt_path.format(epoch_idx) 
        torch.save({
            'model_state_dict': self.perturb_model.state_dict(),
            'epoch': epoch_idx
        }, save_path)
        

    def load_checkpoint(self, epoch_idx):
        load_path = self.chkpt_path.format(epoch_idx) 
        # Loading from checkpoint
        checkpoint = torch.load(load_path)
        self.perturb_model.load_state_dict(checkpoint['model_state_dict'])
        
        return 
    
    def fit(
        self,
        X : np.array, 
        Y : np.array, # Should be the inverted label ( 1 - f(x))
        num_epochs:int = 50,
        log_interval:int = 100,
        last_K = 15
    ):
        self.perturb_model.train()
        self.clf_model.eval()
        self.perturb_model.to(self.device)
        self.clf_model.to(self.device)
        bs = self.batch_size
        opt = torch.optim.Adam(list(self.perturb_model.parameters()), lr = self.LR)
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
                pred_x = self.perturb_model(x)
                
                # pass this to classification model 
                pred_y = self.clf_model(
                    pred_x
                )
                target_y = FT(Y[b_idx]).to(self.device)
                # Calculate loss
                loss = F.binary_cross_entropy(pred_y, target_y)
                
                # Add regulaization
                # \eta * 1( x_i != x_j )
                reg = self.calc_discrete_reg(x,pred_x)
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.perturb_model.parameters(), clip_value)
                opt.step()
                if b % log_interval == 0 :
                    pass
                    # print('[Epoch] {}  | batch {} | Loss {:4f}'.format(epoch, b, loss.cpu().data.numpy()))
                epoch_loss.append(loss.cpu().data.numpy())
            epoch_loss = np.mean(epoch_loss)
            loss_values.append(epoch_loss)
            self.save_checkpoint(epoch)
            
        idx = np.argmin(loss_values[-last_K:]) + (num_epochs - last_K)
        self.load_checkpoint(epoch)
        # Find the least loss value in the last 10 epochs
        
        return  loss_values  
    
    def predict(self, X):
        """
        Given X from data space ,
        provide a perturbed instance
        """
        self.clf_model.eval()
        self.perturb_model.eval()
        result = []
        with torch.no_grad():
            bs = self.batch_size
            num_batches = X.shape[0] // bs + 1
            idx = np.arange(X.shape[0])
            for b in range(num_batches):
                b_idx = idx[b*bs:(b+1)*bs]
                x = LT(X[b_idx]).to(self.device)
                pred_y = self.perturb_model(x)
                pred_y = pred_y.cpu().data.numpy()
                result.extend(pred_y)
        return result
         
    





