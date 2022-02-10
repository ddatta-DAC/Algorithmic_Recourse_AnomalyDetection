#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

"""
Rank Improvement Metric
In paper : Conditional Correctness V2
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
from AD_APE import AD_interface as APE_interface
from AD_MEAD import AD_interface as MEAD_interface

try:
    file_path = os.path.realpath(__file__)
    base_path = os.path.dirname(file_path)
except:
    base_path = './'

config_file = os.path.join(base_path, 'cc_config.yaml')

class condCorrectnessEvaluator:
    
    def __init__(
        self,
        dataset:str,
        samples_frac = 0.1,
        samples_count = 10000,
        data_id_col = 'PanjivaRecordID',
        refresh=False,
        ad_model = None
    ):
        global base_path
        global config_file
        self.dataset = dataset
        self.data_id_col = data_id_col
        with open( config_file, 'r') as fh:
            self.config = yaml.safe_load(fh)
        self.data_id_col = data_id_col
        self.ad_model_emb = self.config['ad_model_emb'] 
        if ad_model is None:
            self.AD_obj = AD_processor(self.dataset)
            score_df_file = os.path.join(base_path, 'CC_reference_df_{}.csv'.format(dataset))
        else:
            if ad_model == 'APE':
                score_df_file = os.path.join(base_path, 'CC_reference_df_APE_{}.csv'.format(dataset))
                self.AD_obj  = APE_interface.AD_model_interface(dataset)
            if ad_model == 'MEAD':
                score_df_file = os.path.join(base_path, 'CC_reference_df_MEAD_{}.csv'.format(dataset))
                self.AD_obj  = MEAD_interface.AD_model_interface(dataset)
        
        self.ad_model = ad_model
        
        
        if os.path.exists(score_df_file) and refresh is False:
            self.score_df = pd.read_csv(score_df_file, index_col=None)
        else:
            if ad_model is None:
                self.score_df = self.get_scores_df(samples_frac)
                self.score_df.to_csv(score_df_file,index= False)
            else:
                self.score_df = self.get_modelSpec_scores_df(samples_count)
                self.score_df.to_csv(score_df_file,index= False)
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
        # normal_df = normal_df.sample(frac = samples_frac)
        normal_df = normal_df.sample(n = len(anomaly_df)*4)
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
        self.score_df[ self.data_id_col] = np.arange(1,len(self.score_df)+1).astype(int)
        self.score_df = self.score_df.reset_index(drop=True)
        return self.score_df 
    
 
    def calculate(
        self, 
        score_ref, 
        score_targets
        
    ):
        rank_1 = np.searchsorted(self.score_df['score'].values, score_ref)
        rank_2 = np.searchsorted(self.score_df['score'].values, score_targets)
        res = np.where( rank_2 > rank_1, 1, 0 )
        return res
    
    def get_average_score(
            self,
            record_ref,
            cf_record_targets : pd.DataFrame,
            cf_record_scores : pd.DataFrame
        ):
        
        targets_df = cf_record_targets.merge( cf_record_scores, on = self.data_id_col, how= 'inner')
        if self.ad_model is None:
            del record_ref[self.data_id_col]
            # The anomaly record 
            ref_score =  self.AD_obj.score_single_sample(
                        record_ref.iloc[0], 
                        model_emb_dim = self.ad_model_emb
                    )
        else:
            ref_score = self.AD_obj.score_samples_batch(
                        record_ref
                    )
            ref_score = ref_score[0]
            
        ref_score = ref_score
        res = self.calculate(ref_score, targets_df['score'].values)
        
        return np.mean(res)     

    def get_modelSpec_scores_df(self, sample_count):
        normal_df_path = self.config['normal_data_path']
        df_train = pd.read_csv(os.path.join(base_path, normal_df_path, self.dataset,  'train_data.csv'), index_col=None)
        df_test = pd.read_csv(os.path.join(base_path, normal_df_path, self.dataset,  'test_data.csv'), index_col=None)
        df = df_train.append(df_test,ignore_index=True).sample(n = sample_count)
        scores = self.AD_obj.score_samples_batch(df)
        df['score'] = scores
        df = df.sort_values(by = ['score'])
        
        df[self.data_id_col] = np.arange(1,len(df)+1).astype(int)
        score_df = df.reset_index(drop=True)
        score_df = score_df[[self.data_id_col, 'score']]
        return score_df
# In[1]:


# DIR = 'us_import1'
# obj = condCorrectnessEvaluator(DIR)

# cf_dir = './../recourse_Xformer_random/CF_results/us_import1'
# files = glob(os.path.join(cf_dir,'**.csv'))

# 
# DIR = 'us_import1'
# obj = condCorrectnessEvaluator(DIR)
# df1 = pd.read_csv('./../../GeneratedData/us_import1/train_data.csv', index_col=None)
# df2 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_2.csv', index_col=None)
# obj.get_average_score(df2.iloc[1200], df1.iloc[200:250])
# obj.calculate(df2.iloc[10], df1.iloc[100])

# df1 = pd.read_csv('./../recourseExpDataFetcher/FilteredAnomalies/us_import1/anomalies_1_1.csv', index_col=None)

    

# pd.DataFrame([df1.iloc[0]])

# df2 = pd.read_csv(files[0], index_col=None)

# files[0]

# df3 = pd.read_csv('./../recourse_Xformer_random/CF_results/us_import1/precomputed_scores_AD/cf_107624766001001_scores.csv', index_col=None)

# from timeit import default_timer as timer
# s = timer()
# r = obj.get_average_score(
#     pd.DataFrame([df1.iloc[0]]), 
#     df2,
#     df3                     
# )
# e = timer()

# e-s

