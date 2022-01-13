import pandas as pd
import numpy as np
import os
import torch
import sys
import math
from torch.nn import Module
from torch import nn
from typing import *
sys.path.append('./..')
# from modules import PE
try:
    import PE
except:
    from . import PE
'''
Decoder MLM
'''

class decoder_MLM_layer(Module):
    def __init__(
        self,
        emb_dim: int,
        ffn_layer_dims: List,
        attribute_cardinality: List,
        device,
        dropout:float = 0.2,
        include_PE:bool = True
    ):
        super().__init__()
        self.device = device
        self.include_PE = include_PE
        self.fcn = nn.Sequential(
            nn.Linear(
                in_features = emb_dim, 
                out_features = ffn_layer_dims[0]
            ),
            nn.GELU()
        )
        self.seq_len = len(attribute_cardinality)
        _modules_ = []
        _multiplier = 2 if self.include_PE else 1
        for i in range(self.seq_len):
            tmp = nn.Sequential(
                nn.Linear(ffn_layer_dims[0]* _multiplier, ffn_layer_dims[-1]),
                nn.GELU(),
                nn.Linear(ffn_layer_dims[-1],attribute_cardinality[i]), 
                nn.GELU(),
                nn.Softmax(dim=-1)
            )
            _modules_.append(tmp)
            
        self.op_layer = nn.ModuleList(_modules_)     
        self.PE = PE.PositionalEncoding(d_model = ffn_layer_dims[0])
        
        return 
    
    '''
    encoder_input has shape [Batch, seq_length, emb_dim]
    '''
    def forward(self, encoder_input):
        x = self.fcn(encoder_input)
        if self.include_PE:
            pe_vector = torch.zeros(x.shape).to(self.device)
            pe_vector = self.PE(pe_vector)
            # concatenate 
            x = torch.cat([x, pe_vector ], dim =-1)
            
        token_pred_op = []
        for i in range(self.seq_len):
            x_i = self.op_layer[i](x[:,i])
            token_pred_op.append(x_i)
        'Reurns: List [ ], Each element has shape [Batch, number of classes ]'
        return token_pred_op