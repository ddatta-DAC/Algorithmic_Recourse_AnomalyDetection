#!/usr/bin/env python
# coding: utf-8




import os
import torch
import pandas as pd
import torch 
import numpy as np
import sys
sys.path.append('./..')
from typing import *

class sparsity:
    def __init__(
        self,
        dataset:str,
        domain_list: list,
        data_id_col: str = 'PanjivaRecordID'
    ):
        self.data_id_col = data_id_col
        self.domain_list = domain_list
    
    def calculate(
        self,
        record_anomaly : pd.DataFrame,
        record_cf: pd.DataFrame
    ):
        values_1 = record_anomaly[self.domain_list].values
        values_2 = record_cf[self.domain_list].values
        
        values_1 = np.tile(values_1, values_2.shape[0]).reshape(values_2.shape)
        
        res = np.where( values_1!=values_2, 1,0)
        res = np.mean(res , axis=1)
        res = np.reciprocal(np.power(res,1) +1)
        return np.mean(res)
    

