import numpy as np
import torch 
import os
import sys
import pandas as pd
import numpy as np
import glob
import re
from pprint import pprint
from typing import *

'''
Set the MASK token to be 0
token weights = 0 - means the loss does not need to be accounted for 
'''
def get_masked_tokens(
    data, 
    column_cardinalities, 
    k1 = 0.8, 
    k2 = 0.2,
    uniform_column_choice = True
):
    column_indices = np.array(data)
    mask_token_id = 0

    # =======
    num_attr = data.shape[-1]
    if uniform_column_choice:
        column_prob = np.array([1/num_attr for _ in range(num_attr)])
    else:
        column_prob = np.array([np.power(column_cardinalities[i]/sum(column_cardinalities),0.5) for i in range(num_attr)]) 
        column_prob = column_prob/np.sum(column_prob)
        
    
    inp_mask = np.random.multinomial(1, column_prob, column_indices.shape[:1])
    inp_mask = np.array(inp_mask, dtype=bool)
    print('column_indices.shape', column_indices.shape)
    # ======
    # Set targets to -1 by default, it means ignore
    # the labels to be predicted 
    # ======

    labels = -1 * np.ones(column_indices.shape, dtype=int)
    labels[inp_mask] = column_indices[inp_mask]

    column_indices_masked = np.copy(column_indices)
    # Replace k1 fraction of tokens with [MASK] following MLM
    
    inp_mask_2mask = inp_mask & (np.random.rand(*column_indices_masked.shape) < k1)
    column_indices_masked[inp_mask_2mask] = mask_token_id  

    # The indices that are not going to be changed
    # Set this fraction to k2
   
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*column_indices_masked.shape) < k2)
    
    # ----------------------
    # Replace masks in each column by randomly sampling from candidate values
    # ----------------------
    for i in range(num_attr):
        _values= np.random.randint(1, column_cardinalities[i]+1, inp_mask_2random[:,i].sum())
        column_indices_masked[:,i][inp_mask_2random[:,i]] = _values
    

    data_masked = column_indices_masked
    token_weights = np.ones(labels.shape)
    token_weights[labels == -1] = 0
    # Replace labels -1 with 0, will be negated by token weights
    labels = np.where(labels == -1, 0, labels) 
    return data_masked, labels, token_weights

'''
Sample run
'''
# column_indices = [
#     [12,90,64,24, 240],
#     [22,91,65,25,241],
#     [32,92,66,26,242],
#     [42,93,67,27,243]
# ]
# column_indices = np.array(column_indices)
# column_cardinalities = [45,100,70,30,250]

# get_maksed_tokens(
#     column_indices, 
#     column_cardinalities,
#     uniform_column_choice = False
# )