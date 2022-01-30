#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append('./..')

from glob import glob
import pickle
from joblib import delayed, Parallel
import multiprocessing as MP
from collections import OrderedDict
from itertools import combinations

from recourseExpDataFetcher import anomDataFetcher

class heterogeneity_evaluator:
    def __init__(
        self,
        dataset,
        data_id_col = 'PanjivaRecordID'
    ):
        self.dataset = dataset 
        self.data_id_col = data_id_col
        anomDataFetcher_obj  = anomDataFetcher(dataset)
        data = anomDataFetcher_obj.fetch_data()
        self.perturbation_index = {}
        for _ , _dict in data.items():
            self.perturbation_index.update(_dict['perturbations_labels'])
            df = _dict['data']
        self.perturbation_index = { k:v[0] for k,v in self.perturbation_index.items()}  
        self.data_columns = [ _ for _ in list(df.columns) if _ != data_id_col]
        return 
    
    def eval_het(
        self,
        anomaly_record,
        cf_records
    ):
        pert_idx = self.perturbation_index[anomaly_record[self.data_id_col]]
        # do column wise
        het = []
        m = len(pert_idx)
        K = len(cf_records)
        
        def _aux_(i,col, pert_idx):
            # invalid
            if i not in pert_idx: 
                return -1
            values = cf_records[col].values
            W = np.where(values!=anomaly_record[col],1,0)
            W = W * int(i in pert_idx)
            idx = np.arange(len(cf_records)).astype(int)
            dist = [] 
            for idx1,idx2 in combinations(idx,2):
                dist_ij = int(values[idx1] != values[idx2]) * W[idx1] * W[idx2]
                dist.append(dist_ij)
            return np.mean(dist)
                
        het = Parallel(n_jobs = MP.cpu_count())(delayed(_aux_)(i,col, pert_idx,)for i, col in enumerate(self.data_columns) )
        het = [ _ for _ in het if _!= -1]
        if len(het) == 0: 
            het =[0]
        res = np.mean(het)
        return res


# eval_obj = heterogeneity_evaluator('us_import1')
# df1 = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# df2 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_2.csv', index_col=None)
# eval_obj.eval_het(df2.iloc[10], df1.iloc[1000:1040])



