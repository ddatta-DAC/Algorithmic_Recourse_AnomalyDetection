#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
import os
import sys
sys.path.append('./..')
import pandas as pd
import numpy as np
import sklearn
from glob import glob
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import *
from AD_model.AD_processor import AD_processor
from joblib import Parallel,delayed
import multiprocessing as MP
from collections import OrderedDict
import pickle

try:
    file_path = os.path.dirname(os.path.realpath(__file__))
    CONFIG_FILE = os.path.join( file_path ,'config.yaml')
except: 
    CONFIG_FILE = 'config.yaml'



class anomDataFetcher:
    # -------------------------
    # Read in anomaly data 
    # -------------------------
    def __init__(self,DIR):
        self.DIR = DIR
        self.id_col = 'PanjivaRecordID'
        self.basepath = os.path.dirname(os.path.realpath(__file__))
        try:
            file_path = os.path.dirname(os.path.realpath(__file__))
            CONFIG_FILE = os.path.join( file_path ,'config.yaml')
        except: 
            CONFIG_FILE = 'config.yaml'
        print(CONFIG_FILE)
        with open(CONFIG_FILE) as fh:
            self.config = yaml.safe_load(fh)
        Path(self.basepath, self.config['save_dir']).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(self.basepath, self.config['save_dir'], self.DIR)).mkdir(exist_ok=True, parents=True)  
        if not self.__checkExists__():
            self.load_data()
        
    def __checkExists__(self):
        for file in self.config['read_files']:
            fpath = os.path.join(self.basepath, self.config['save_dir'], self.DIR, file)
            if not os.path.exists(fpath):
                return False
        return True
    
    
    def load_data(self):  
        ad_proc_obj = AD_processor(self.DIR)
        normal_trainData = pd.read_csv(os.path.join(self.config['data_loc'],  self.DIR , 'train_data.csv'), index_col=None)
        score_dict_n = ad_proc_obj.score_samples_batch(normal_trainData.copy(deep=True))   
        K = self.config['normalData_scorePerc_threshold']
        scoreCutOff_normalData_dict = {}
        # calculate nth-percentile
        for emb, scores in score_dict_n.items():
            scoreCutOff_normalData_dict[emb] = np.percentile(scores, K)
        
        for file in self.config['read_files']:
            # read file
            fpath = os.path.join(self.config['data_loc'],  self.DIR, self.config['anomalySubDir'], file)
            df = pd.read_csv(fpath, index_col=None)
            id_list = df[self.id_col].values.tolist()
            _scores_dict = ad_proc_obj.score_samples_batch(df.copy(deep=True))
            tmp_df = pd.DataFrame(columns = [self.id_col,'score'])
            num_keys = len(list(_scores_dict.keys()))
            for emb, scores in _scores_dict.items():
                tmp_data = [[i,j] for i,j in zip(id_list,scores)]
                tmp_data_df = pd.DataFrame(tmp_data, columns = [self.id_col,'score'])
                tmp_data_df = tmp_data_df.loc[tmp_data_df['score'] < scoreCutOff_normalData_dict [emb]]
                tmp_df = tmp_df.append(tmp_data_df,ignore_index=True)
            # select id s which are present in all cases
            tmp_1 = tmp_df.groupby([self.id_col]).size().reset_index(name='count')
            valid_ids = tmp_1.loc[tmp_1['count']==num_keys][self.id_col].values.tolist()
            tmp_df = tmp_df.loc[tmp_df[self.id_col].isin(valid_ids)]
            tmp_df = tmp_df.groupby(['PanjivaRecordID']).mean().reset_index(drop=False)
            tmp_df = tmp_df.sort_values(by=['score'],ascending=True)
            # Pick the most "anomalous"
            valid_ids = tmp_df.head( self.config['record_count'])[self.id_col].values.tolist()
            df = df.loc[df[self.id_col].isin(valid_ids)]
            # Save file
           
            df.to_csv(
                os.path.join( self.basepath, self.config['save_dir'], self.DIR, file),
                index = False
            ) 
        return                    
     
    # ======================================
    # Auxillary function
    # ======================================
    def func_getPertIdx(self, df_row_true, df_row_target):
        cols = list(df_row_true.to_dict().keys())
        cols.remove(self.id_col)
        res =  []
        for i,c in enumerate(cols):
            if df_row_true[c] != df_row_target[c]:
                res.append(i)
        res = (df_row_true[self.id_col] ,res)
        return res
    
    # ===============================
    # Main function to get data 
    # ===============================
    
    def fetch_data(self):
        result_dict = {}
        
        for file in self.config['read_files']:
            fpath = os.path.join(self.basepath, self.config['save_dir'], self.DIR, file)
            df = pd.read_csv(fpath,index_col=None)
            p_fname = file.replace('.csv','') + '_pertIdx.pkl'
            pert_file_path = os.path.join(self.basepath, self.config['save_dir'], self.DIR, p_fname) 
            
            # Check if the perturbations are stored
            if not os.path.exists( pert_file_path ):
                # find the perturbed idx
                replace_str = str(df.iloc[0][self.id_col])[-6:] # e.g 001001,001002
                data_loc = os.path.join(self.config['data_loc'],self.DIR, 'test_data.csv')
                test_df = pd.read_csv(data_loc, index_col=None)

                df_1 = df.copy(deep=True)
                df_1[self.id_col] = df_1[self.id_col].apply(lambda x: int(str(x).replace(replace_str,'')))

                test_df = test_df.loc[test_df[self.id_col].isin(list(df_1[self.id_col]))]
                _perturbations = Parallel(n_jobs=MP.cpu_count())(
                    delayed(self.func_getPertIdx)(test_df.iloc[i], df_1.iloc[i]) for i in tqdm(range(len(test_df))))

                perturbations = OrderedDict({})
                for _ in _perturbations:
                    _id = int(str(_[0]) + replace_str)
                    perturbations[_id] = _[1:]
                with open(pert_file_path,'wb') as fh:
                    pickle.dump(perturbations, fh, pickle.HIGHEST_PROTOCOL)
                    
            else:
                with open(pert_file_path,'rb') as fh:
                    perturbations = pickle.load(fh)
            key = file.replace('.csv','')     
            result_dict[key] = {
                'data': df, 
                'perturbations_labels': perturbations
            }
        return result_dict
    

# DIR = 'us_import3'
# obj = anomDataFetcher(DIR)
# result_dict = obj.fetch_data()


# In[34]:





# In[ ]:




