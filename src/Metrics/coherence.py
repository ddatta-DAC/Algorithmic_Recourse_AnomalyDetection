#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import os
import numpy as np
import sys
from typing import *
from collections import OrderedDict
from itertools import combinations
import pickle
from joblib import delayed, Parallel
import multiprocessing as MP

# Coherence
class coherence_calculator:
    def __init__(
        self,
        domain_dims: OrderedDict,
        train_data_df: pd.DataFrame = None,
        data_id_col: str = 'PanjivaRecordID',
    ):
        self.domain_dims = domain_dims
        self.__create_coOccMatrix__(train_data_df)
        return 
    
    def __create_coOccMatrix__(self, train_data_df:pd.DataFrame):
        self.coOccMatrix = OrderedDict({})
        
        def _aux_(col_pair):
            nonlocal train_data_df
            col_1, col_2 = list(sorted(col_pair))
            key = '_'.join(sorted([col_1,col_2]))
            df = train_data_df[[col_1, col_2]].copy(deep=True)
            new_df = df.groupby([col_1, col_2]).size().reset_index(name='count')
            count_1 =  self.domain_dims[col_1]
            count_2 =  self.domain_dims[col_2]
            coocc = np.zeros([count_1, count_2])
            for _, row in new_df.iterrows():
                i = row[col_1]
                j = row[col_2]
                coocc[i][j] = row['count']
            return (key,coocc)
        
        res = Parallel(n_jobs = MP.cpu_count())(delayed(_aux_)(pair) for pair in combinations(list(self.domain_dims.keys()),2))
        for _item in res:
            key,coocc = _item[0], _item[1]
            self.coOccMatrix[key] = coocc 
        
        return
    
    def calc_value(
        self,
        row,
        modified_cols
    ):
        """
        p_i|p_j = p_ij| p_j
        """
        res = []
        count = 0
        fixed_cols =  [ _ for _ in self.domain_dims.keys() if _ not in modified_cols]
        
        for mod_col in modified_cols:
            for fc in fixed_cols:
                key = '_'.join(sorted([mod_col, fc]))
                _matrix = np.copy(self.coOccMatrix[key])
                dom1 = mod_col
                dom2 = fc
                if mod_col < fc:
                    pass
                else:
                    _matrix = np.transpose(_matrix)
                
                i = row[dom1]
                j = row[dom2]
                p_ij = _matrix[i][j]/np.sum(_matrix)
                # p_j is the denominator
                p_i = np.sum(_matrix[i,:])/np.sum(_matrix)
                p_j = np.sum(_matrix[:,j])/np.sum(_matrix)
                if mod_col < fc:
                    p_denom = p_j
                else:
                    p_denom = p_i
                # print(p_ij, p_i,p_j)
                if p_denom > 0:
                    p = p_ij/(p_denom)
                else:
                    p = 0
                p = p_ij
                res.append(p)
                count +=1
        coherence = np.mean(res)
        return coherence
    

# with open('./../../GeneratedData/us_import1/domain_dims.pkl','rb') as fh:
#     domain_dims = OrderedDict(pickle.load(fh))

# df = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# obj = coherence_calculator(domain_dims,df.copy(deep=True))
# obj.calc_value(
#     df.iloc[195],
#     modified_cols = ['Carrier', 'HSCode']
# )



