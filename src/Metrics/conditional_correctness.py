#!/usr/bin/env python
# coding: utf-8

"""
Rank Improvement Metric
In paper : Conditional Correctness
"""

import os
import sys
import pandas as pd
import pickle
import multiprocessing as MP
sys.path.append('./../')
from joblib import Parallel,delayed
import numpy as np
from tqdm import tqdm
from AD_model.AD_processor import AD_processor
import yaml
from joblib import Parallel,delayed
import multiprocessing as MP
from glob import glob

try:
    file_path = os.path.realpath(__file__)
    base_path = os.path.dirname(file_path)
except:
    base_path = './'

config_file = os.path.join(base_path, 'cc_config.yaml')
print(config_file)

class condCorrectnessEvaluator:
    def __init__(
        self,
        dataset:str,
        samples_frac = 0.2,
        data_id_col = 'PanjivaRecordID',
        refresh=False
    ):
        global base_path
        global config_file
        self.dataset = dataset
        self.data_id_col = data_id_col
        with open( config_file, 'r') as fh:
            self.config = yaml.safe_load(fh)
        self.data_id_col = data_id_col
        self.ad_model_emb = self.config['ad_model_emb'] 
        self.AD_obj = AD_processor(self.dataset)
        
        score_df_file = os.path.join(base_path, 'CC_reference_df.csv')
        if os.path.exists(score_df_file) and refresh is False:
            self.score_df = pd.read_csv(score_df_file, index_col=None)
        else:
            self.score_df = self.get_scores_df(samples_frac)
            self.score_df.to_csv(score_df_file)
        return
    
    def get_scores_df(self, samples_frac):
        anomaly_df_path = self.config['anomaly_data_path']
        normal_df_path = self.config['normal_data_path']
        anomaly_df = None
        files = glob(os.path.join(base_path, anomaly_df_path, self.dataset, 'anomalies**.csv'))
        for _path in files:
            df = pd.read_csv(_path, index_col=None)
            if anomaly_df is None:
                anomaly_df = df
            else:
                anomaly_df = anomaly_df.append(df)

        normal_df = pd.read_csv(os.path.join(base_path, normal_df_path, self.dataset,  'train_data.csv'), index_col=None)
        
        anomaly_df = anomaly_df.sample(frac = samples_frac)
        normal_df = normal_df.sample(frac = samples_frac)

        id_anomalyDf = anomaly_df[self.data_id_col].values
        id_normalDf = normal_df[self.data_id_col].values
        id_list = np.hstack([id_anomalyDf,id_normalDf])


        self.scores_anom = self.AD_obj.score_samples_batch(anomaly_df)[self.ad_model_emb]
        self.scores_norm = self.AD_obj.score_samples_batch(normal_df)[self.ad_model_emb]

        score_list = np.hstack([ self.scores_anom , self.scores_norm])
        data = np.vstack([id_list,score_list]).transpose()
        self.score_df = pd.DataFrame(data = data, columns = [self.data_id_col, 'score'])
        self.score_df = self.score_df.sort_values(by = ['score'])
        self.score_df[ self.data_id_col] = self.score_df[ self.data_id_col ].astype(int)

        self.score_df = self.score_df.reset_index(drop=True)
        return self.score_df 
            
    def calculate(
        self, 
        record_ref : pd.DataFrame,  # anomaly
        record_target: pd.DataFrame  # counterfactual
    ):
        # print(record_ref)
        # print(record_target)
        
        def get_rank(record):
            df = self.score_df.copy(deep=True)
            df = df.reset_index(drop=True)
            
            
            record = pd.DataFrame(record).reset_index(drop=True)
            new_id = int('9999999' + str(record[self.data_id_col].values[0])[:4] )
            record.loc[0,self.data_id_col] = new_id
            _id = new_id

            new_score =  self.AD_obj.score_samples_batch(
                pd.DataFrame(record), 
                model_emb_dim = self.ad_model_emb
            )
            
            record['score'] = new_score[0]
            record = pd.DataFrame(record)[[self.data_id_col, 'score']]
            
            df = df.append(record, ignore_index=True)
            df = df.sort_values(by = ['score'])
            df = df.reset_index(drop=True)
            rank = df.loc[df[self.data_id_col]==_id].index[0]
            del df
            return rank 
        
        
        rank_1 = get_rank(record_ref) 
        rank_2 = get_rank(pd.DataFrame([record_target]))
        
        res = float( rank_1 < rank_2 )
        return res
    
    def get_average_score(
            self,
            record_ref,
            record_targets
        ):
        res = []
        
        # for i, record_target in record_targets.iterrows():
        #     r = self.calculate(record_ref, record_target)
        #     res.append(r)
        res = Parallel(n_jobs=MP.cpu_count(), prefer="threads")(delayed(self.calculate)(record_ref, record_target) for i, record_target in record_targets.iterrows())
        # print (res)
        return np.mean(res)                                    

# 
# DIR = 'us_import1'
# obj = condCorrectnessEvaluator(DIR)
# df1 = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# df2 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_2.csv', index_col=None)
# obj.get_average_score(df2.iloc[1200], df1.iloc[200:250])
# obj.calculate(df2.iloc[10], df1.iloc[100])
