#!/usr/bin/env python
# coding: utf-8


from collections import OrderedDict
import os
import sys
sys.path.append('./..')
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import pickle
from glob import glob
import torch
from utils import util
from typing import *
from torch import LongTensor as LT
import pandas as pd
import yaml
import numpy as np
from .model_APE import APE_container

class AD_model_interface(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self, 
        DIR,
        device =  torch.device("cpu"),
        id_col = 'PanjivaRecordID'
    ):
        base_path =  os.path.dirname(os.path.realpath(__file__))
        self.rel_path = base_path 
        CONFIG_FILE = os.path.join( self.rel_path ,'config.yaml')
        self.DIR = DIR
        self.id_col = id_col
        with open(CONFIG_FILE, 'r') as fh:
            CONFIG = yaml.safe_load(fh)
        self.device = device
        
        
        self.model_save_dir = os.path.join(self.rel_path, CONFIG['model_save_dir'], DIR )
        print(self.model_save_dir)
        self.data_loc = CONFIG['DATA_LOC']
        

        self.emb_dim = CONFIG['emb_dim']
        self.domain_dims_df = pd.read_csv(os.path.join(self.data_loc, DIR, 'data_dimensions.csv'),index_col=None)
        with open(os.path.join(self.data_loc, DIR, 'domain_dims.pkl'),'rb') as fh:
            self.domain_dims = OrderedDict(pickle.load(fh))
         
        self.num_entities = np.sum(self.domain_dims_df['dimension'].values)
        
        self.domain_list = self.domain_dims_df['column']
       
        self.read_model()
        
        idMapper_file = os.path.join(self.data_loc, self.DIR, 'idMapping.csv')
        idMapping_df = pd.read_csv(idMapper_file, index_col=None)
        mapping_dict = {}
        
        for domain in set(idMapping_df['domain']):
            tmp =  idMapping_df.loc[(idMapping_df['domain'] == domain)]
            serial_id = tmp['serial_id'].values.tolist()
            entity_id = tmp['entity_id'].values.tolist()
            mapping_dict[domain] = {k:v for k,v in zip(entity_id,serial_id)}
        self.entityID2serialID_mapping_dict = mapping_dict
        return
    
    def read_model(self):
        model_file_path = sorted(
            glob(os.path.join(self.model_save_dir, '**{}_**'.format(self.emb_dim)))
        )[-1]

        ad_obj = APE_container(
            domain_dims = self.domain_dims,
            emb_dim =  self.emb_dim,
            device = self.device
        ) 
        
        ad_obj.load_model(model_file_path)
        self.model_obj = ad_obj
        return 
    
    def fit(self,X=None,Y=None):
        """
        dummy function
        """
        return 
    
    def predict(
        self,
        df_records : pd.DataFrame,
    ):
        records_serialized = util.convert_to_serializedID_format( 
            df_records,
            self.DIR
        )
        try:
            del records_serialized[self.id_col]
        except:
            pass
        x_values = records_serialized.values
        ad_obj = self.model_obj
        scores = ad_obj.predict(x_values)
        
        return scores
     
    def score_samples_batch(
        self,
        df_records : pd.DataFrame,
    ):
        return self.predict(df_records)
    



