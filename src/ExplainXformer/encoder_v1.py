
import os
import sys
sys.path.append('./..')
import torch
import pandas as pd
import numpy as np
from torch import FloatTensor as FT
from torch import nn
from torch import LongTensor as LT
from torch.nn import Module
from torch.nn import functional as F

from typing import *
import math


# from modules import PE
try:
    import PE
except:
    from . import PE
'''
This is the feature encoder 
Captures the overall context of the record
'''

class Encoder(Module):
    def __init__(
        self,
        emb_dim: int,
        xformer_model_dims: int,
        xformer_heads: int,
        cardinality: List,
        count_numerical_attr:int,
        device,
        num_xformer_layers:int = 3, 
        include_PE:bool = True
        
    ):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.cat_count = len(cardinality)
        self.num_count = count_numerical_attr
        self.include_PE = include_PE
        '''
        1. Assume that the categorical columns precede the numerical ones; needs explicit specification
        2. Create column embedding
        '''
        self.field_embedding = nn.ModuleList(
            [nn.Embedding(cardinality[i]+1, emb_dim) for i in range(self.cat_count)]
        )
        self.PE = PE.PositionalEncoding(emb_dim)
        self.PE.to(self.device)
        self.cat_xformer = nn.TransformerEncoderLayer(
            d_model = emb_dim, 
            nhead = xformer_heads, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True,  
            device=self.device
        )
        self.numAttr_LN = nn.LayerNorm(normalized_shape = self.num_count)
        if self.num_count > 0 :
            self.num_projLayer = nn.Linear(self.num_count , emb_dim)
        else:
            self.num_projLayer = None
            
        _enc_layer = nn.TransformerEncoderLayer(
            d_model = self.emb_dim , 
            nhead = xformer_heads,
            batch_first=True,  
            device=self.device
        )
        
        self.encoder_xformer_joint = nn.TransformerEncoder(
            _enc_layer, 
            num_layers = num_xformer_layers
        )
 
    '''
    Takes the tabular input data
    Returns:
    1. Compressed representation
    2. Embedding of each categorical variable and the numerical variables
    '''
    def forward(self, x):
        x_cat = (x[:,:self.cat_count]).long()
        if self.num_count > 0:
            x_num = x[:,self.cat_count:]
        x_cat_emb_0 = [self.field_embedding[i](x_cat[:,i]) for i in range(self.cat_count)]
        x_cat_emb_0 = torch.stack(
            x_cat_emb_0,
            dim=1
        )
        if self.include_PE :
            x_cat_emb_1 = self.PE(x_cat_emb_0)
        else:
            x_cat_emb_1 = x_cat_emb_0
            
        x_cat_emb = self.cat_xformer(x_cat_emb_1)  
        
        
        if self.num_count > 0:
            x_num = self.numAttr_LN(x_num)
            x_num_emb = self.num_projLayer(x_num)
            x_emb_0 = torch.hstack([ x_cat_emb_0,x_num_emb.reshape(-1,1,self.emb_dim)]) 
            bs = x_num_emb.size()[0]
            x_cat_num = torch.hstack([x_cat_emb, x_num_emb.reshape(-1,1,self.emb_dim)])
        else: 
            x_emb_0 = x_cat_emb_0
            x_cat_num = x_emb_0 
       
        x1 = self.encoder_xformer_joint(x_cat_num)
                
        return x1, x_emb_0