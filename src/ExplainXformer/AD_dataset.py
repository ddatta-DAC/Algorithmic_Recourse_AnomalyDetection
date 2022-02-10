import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from torch.nn import functional as F
sys.path.append('./..')
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class anomaly_dataset(Dataset):
    def __init__(
        self,
        data,
        domain_dims,
        alpha_max = 0.25,
        max_perturb = 2
    ):
        self.data = data
        self.alpha_max = alpha_max
        self.domain_dims = domain_dims
        self.max_perturb = max_perturb
        return
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        record = self.data[idx]
        seq_len = self.data.shape[1]
        labels = np.ones([seq_len])
        # select 
        alpha = np.random.uniform()
        if alpha < self.alpha_max:
            return record, np.array(labels)
            
        # Create anomaly
        # perturn between 1 and 2
        count = np.random.randint(1,self.max_perturb+1)
        p_idx = np.random.choice(np.arange(seq_len),count,replace=False)
                
        for c in range(count):
            labels[p_idx[c]] = 0
            record[p_idx[c]] = np.random.choice(self.domain_dims[p_idx[c]],1) + 1
        
        return np.array(record), np.array(labels)