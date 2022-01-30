#!/usr/bin/env python
# coding: utf-8

# In[56]:


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
from recourseExpDataFetcher import anomDataFetcher


class accuracy_evaluator:
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
    
    def calc_acc(
        self,
        anomaly_record : pd.Series,
        cf_record : pd.Series,
    ):
        # Find what has changed
        tp,fp,tn,fn = 0,0,0,0 
        per_idx = self.perturbation_index[anomaly_record[self.data_id_col]]
        
        for i,col in enumerate(self.data_columns,0):
            
            if anomaly_record[col] == cf_record[col]:
                if i in per_idx:
                    fn+= 1
                else:
                    tn+= 1
            else:
                if i in per_idx:
                    tp+=1
                else:
                    fp+=1
        acc = (tp+tn)/(tp+fp+tn+fn)  
        
        return acc
    
    def get_mean_acc(
        self, 
        anomaly_record,
        cf_records
    ):
        res = Parallel(n_jobs=MP.cpu_count())(delayed(self.calc_acc)(anomaly_record,cf_record,) for i, cf_record in cf_records.iterrows())
        return np.mean(res)

# acc_eval_obj = accuracy_evaluator('us_import1')
# df1 = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# df2 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_2.csv', index_col=None)
# acc_eval_obj.get_mean_acc(
#     df2.iloc[100],
#     df1.iloc[100:150]
# )

