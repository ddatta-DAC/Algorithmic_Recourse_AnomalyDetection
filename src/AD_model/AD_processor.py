#!/usr/bin/env python
# coding: utf-8
import yaml
import os
import pandas as pd
import numpy as np
import sys
import pickle
from glob import glob
import torch
from utils import util
from .model_AD_1 import AD_model_container
from typing import *
from torch import LongTensor as LT

# ----------------------------------------------------------------
'''
This class is for utilizing the trained anomaly detyection models and the thresholds stored
'''
# ----------------------------------------------------------------
class AD_processor:
    def __init__(
        self, 
        DIR,
        device =  torch.device("cpu")
    ):
        file_path = os.path.realpath(__file__)
        self.rel_path = os.path.dirname(file_path) 
        CONFIG_FILE = os.path.join( self.rel_path ,'config.yaml')
        self.DIR = DIR
        self.id_col = 'PanjivaRecordID'
        with open(CONFIG_FILE, 'r') as fh:
            CONFIG = yaml.safe_load(fh)
        self.device = device
        self.model_save_dir = os.path.join(CONFIG['model_save_dir'], DIR )
        self.threshold_dict = {}
        self.perc_threshold = CONFIG['perc_threshold']
        self.model_save_dir = os.path.join(self.rel_path, CONFIG['model_save_dir'], DIR )
        self.data_loc = CONFIG['DATA_LOC']
        self.domain_dims = pd.read_csv(os.path.join(self.data_loc, DIR, 'data_dimensions.csv'),index_col=None)
        
        MEAD_emb_list = CONFIG['emb_dims']
        self.MEAD_emb_list = [int(_) for _ in MEAD_emb_list]
        self.num_entities = np.sum(self.domain_dims['dimension'].values)
        self.model_dict = {}
        self.domain_list = self.domain_dims['column']
        self.read_thresold_dict()
        self.read_models()
        
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
    
    def read_thresold_dict(self):
        # print('[Reading in thresholds]')
       
        threshold_save_file =  os.path.join(
            self.rel_path, self.model_save_dir, 'threshold_dict_{}.pkl'.format('-'.join([str(_) for _ in self.perc_threshold]))
        )
        # print(threshold_save_file)
        with open(threshold_save_file, 'rb') as fh:
            self.threshold_dict = pickle.load(fh)
        # print (self.threshold_dict)
        return 

    def read_models(self):
        
        print('[Reading in models]')
        model_dict = {}
        for emb_dim in self.MEAD_emb_list:
            model_file_path = sorted(
                glob(os.path.join(self.model_save_dir, '**{}_**.pth'.format(emb_dim)))
            )[-1]
            
            ad_obj = AD_model_container(
                entity_count = self.num_entities,
                emb_dim =  emb_dim,
                device = self.device
            ) 
            
            ad_obj.load_model(model_file_path)
            self.model_dict[emb_dim] = ad_obj
        return 

    '''
    This is an external facing function
    Input : a single row of pandas dataframe
    This record is not serialized is a single row of a dataframe
    Returns:
    { <emb_dim>: { percentile : ( threshold, record_score ) } 
    '''
    def score_new_sample_wThresh(self, record : pd.DataFrame, perc_thrhld = None ):
        # perform serialization
        serialized_record = self._convert_to_serializedID_format_(
            record
        )
        
        try:
            del serialized_record[self.id_col]
        except:
            pass
        
        x_values = serialized_record.values
        x_values = x_values.reshape([1,-1])
        
        result = {}
        
        for emb_dim in self.MEAD_emb_list:
            ad_obj = self.model_dict[emb_dim] 
            score = ad_obj.predict(x_values)
           
            cutoff_perc_values = self.threshold_dict[emb_dim]
            if perc_thrhld is None:
                _res = {}
                for perc, v in cutoff_perc_values.items():
                    _res[perc] = (v, score[0])
            else:
                _res = [cutoff_perc_values[perc_thrhld], score[0]]
            result[emb_dim] = _res
        return result

    def _convert_to_serializedID_format_(
        self, 
        record: pd.Series
    ):
        record = record.copy(deep=True)
        for d in self.domain_list:
            record[d] = self.entityID2serialID_mapping_dict[d][record[d]]
        return record
    
    def score_samples_batch(
        self, 
        records : pd.DataFrame, 
        model_emb_dim = None):
        records_serialized = util.convert_to_serializedID_format( 
            records,
            self.DIR
        )
        try:
            del records_serialized[self.id_col]
        except:
            pass
        x_values = records_serialized.values
        score_dict = {}
        
        for emb_dim in self.MEAD_emb_list:
            if model_emb_dim is not None and emb_dim!= model_emb_dim:
                continue
            ad_obj = self.model_dict[emb_dim] 
            scores = ad_obj.predict(x_values)
            score_dict[emb_dim] = scores
        if model_emb_dim is not None:
            return score_dict[model_emb_dim]
        return score_dict
    
    def score_single_sample(
        self,
        record: pd.Series,
        model_emb_dim = 32
    ):
        """
        Get the score a single sample with the known model embedding dimension
        """
        record_ser = self._convert_to_serializedID_format_(record)
        ad_obj = self.model_dict[model_emb_dim] 
        x_values = record_ser.values.reshape([1,-1])
        
        score = ad_obj.predict_single_score(x_values)
        return score.cpu().data.numpy()
    
    def score_sample_noBatch(
        self,
        records: pd.DataFrame,
        model_emb_dim = 32
    ):
        """
        Get the score a single sample with the known model embedding dimension
        """
        records_serialized = util.convert_to_serializedID_format( 
            records,
            self.DIR
        )
        try:
            del serialized_record[self.id_col]
        except:
            pass
        
        ad_obj = self.model_dict[model_emb_dim] 
       
       
        x_values = records_serialized.values
        # predict_single_score can take more than 1 row of data
        scores = ad_obj.predict_single_score(x_values)
        
        return scores
    
    def score_tensorBatch(
        self,
        records: LT,
        model_emb_dim = 32
    ):
        """
        not implemented 
        """
        ad_obj = self.model_dict[model_emb_dim]
        return 