#!/usr/bin/env python
# coding: utf-8

# In[26]:

import argparse
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./..')
from pandarallel import pandarallel
from joblib import Parallel,delayed
pandarallel.initialize()
from tqdm import tqdm
import pickle
import torch
import multiprocessing as mp
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from collections import OrderedDict, defaultdict
from itertools import combinations
from glob import glob
import yaml
from utils import util
from typing import *

# ======================================== #
try:
    from .model_AD_1 import AD_model_container, MEAD
except:
    from model_AD_1 import AD_model_container, MEAD
# ======================================== #

DEVICE =  torch.device("cpu")
file_path = os.path.realpath(__file__)
print(file_path)
CONFIG_FILE = os.path.join( os.path.dirname(file_path) ,'config.yaml')

# CONFIG_FILE = './config.yaml'
ad_model_data_dir = None
MEAD_emb_list = []
CONFIG = None
domain_dims = None
train_epochs = 100
error_tol = 0.001
learning_rate = 0.001
model_save_dir = None
num_entities = 0
id_col = 'PanjivaRecordID'
threshold_dict = {}
model_dict = {}
perc_threshold = [2,5,10]

def get_domain_dims(data_loc):
    # with open(os.path.join(data_loc,'domain_dims.pkl'),'rb') as fh:
    #     domain_dims = pickle.load(fh)
    
    domain_dims = pd.read_csv(os.path.join(data_loc, 'data_dimensions.csv'),index_col=None)
    return domain_dims
    
'''
Set up globals
'''

def setup_config(subDIR):
    global CONFIG
    global CONFIG_FILE
    global DIR, ad_model_data_dir, MEAD_emb_list, batch_size, error_tol, train_epochs, learning_rate, model_save_dir
    global domain_dims, num_entities
    global perc_threshold
    DIR = subDIR
    with open(CONFIG_FILE, 'r') as fh:
        CONFIG = yaml.safe_load(fh)
        
    ad_model_data_dir = os.path.join(
        CONFIG['DATA_LOC'], 
        DIR,
        CONFIG['AD_model_data_subdir']
    )
    if 'perc_threshold' in CONFIG.keys():
        perc_threshold = CONFIG['perc_threshold']
    data_loc = CONFIG['DATA_LOC']
    domain_dims = get_domain_dims(os.path.join(data_loc, subDIR))
    MEAD_emb_list = CONFIG['emb_dims']
    MEAD_emb_list = [int(_) for _ in MEAD_emb_list]
    train_epochs = CONFIG['train_epochs']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']
    error_tol = CONFIG['error_tol']
    model_save_dir = os.path.join(CONFIG['model_save_dir'], DIR )
    num_entities = np.sum(domain_dims['dimension'].values)
    
    return 


'''
Procedure to train the models
'''
def train_AD_models():
    global DIR, ad_model_data_dir, MEAD_emb_list, train_epochs, model_save_dir, DEVICE
    
    def aux(emb_dim):
        global ad_model_data_dir, train_epochs, learning_rate, batch_size, error_tol, model_save_dir, DEVICE  , num_entities
        train_x_pos = os.path.join(ad_model_data_dir, 'train_x_pos.npy')
        train_x_neg = os.path.join(ad_model_data_dir, 'train_x_neg.npy')
        train_x_pos = np.load(train_x_pos)
        train_x_neg = np.load(train_x_neg)
        train_x_neg = train_x_neg.reshape([train_x_pos.shape[0], -1, train_x_pos.shape[-1]])
        ad_obj = AD_model_container(
            entity_count = num_entities,
            emb_dim = emb_dim,
            device = DEVICE,
            lr = learning_rate
        ) 
        
        ad_obj.train_model(
            train_x_pos, 
            train_x_neg, 
            batch_size = batch_size, 
            epochs = train_epochs,
            log_interval = 100,
            tol = error_tol
        )
        
        print('model trained')
        
        ad_obj.save_model( 
            model_save_dir
        )
        print('model saved')
        return 
        
    Parallel(n_jobs=4)(delayed(aux)(emb_dim,) for emb_dim in MEAD_emb_list)
    # for emb_dim in MEAD_emb_list:
    #     aux(emb_dim)
    return 
    
    

'''
Stores the pth percentile values for the likelihood scores of training samples.
'''
def calculate_thresholds():
    global model_save_dir
    global MEAD_emb_list
    global num_entities
    global ad_model_data_dir
    global perc_threshold
    global DEVICE
    
    model_dict = {}
    for emb_dim in MEAD_emb_list:
        model_file_path = sorted(glob(os.path.join(model_save_dir, '**{}_**.pth'.format(emb_dim))))[0]
        ad_obj = AD_model_container(
            entity_count = num_entities, 
            emb_dim = emb_dim,
            device = DEVICE
        ) 
        ad_obj.load_model(model_file_path)
        model_dict[emb_dim] = ad_obj
    
    dict_embDim_thresholdValue = defaultdict() 
    # Load the training data set 
    train_x_pos = os.path.join(ad_model_data_dir, 'train_x_pos.npy')
    train_x_pos =  np.load(train_x_pos)
    for emb_dim in MEAD_emb_list:
        ad_obj =  model_dict[emb_dim]
        scores = ad_obj.score_samples(train_x_pos)
        dict_embDim_thresholdValue[emb_dim] = {}
        # Calculate the n-th percentile values
        for p in perc_threshold:
            dict_embDim_thresholdValue[emb_dim][p] = np.percentile(np.array(scores).reshape(-1), p)
    '''
    Save the values
    '''
    threshold_save_file =  os.path.join(model_save_dir, 'threshold_dict_{}.pkl'.format('-'.join([str(_) for _ in perc_threshold])))
    with open(threshold_save_file, 'wb') as fh:
        pickle.dump(dict_embDim_thresholdValue, fh, pickle.HIGHEST_PROTOCOL)
    return 


'''
Call before test mode
'''

def read_thresold_dict():
    global DIR
    global perc_threshold
    global threshold_dict
    threshold_save_file =  os.path.join(model_save_dir, 'threshold_dict_{}.pkl', format('-'.join([str(_) for _ in perc_threshold])))
    with open(threshold_save_file, 'rb') as fh:
        threshold_dict = pickle.load(fh)
    return 

def read_models():
    global model_save_dir, model_dict
    global MEAD_emb_list
    global num_entities
    global ad_model_data_dir, DIR
    model_dict = {}
    for emb_dim in MEAD_emb_list:
        model_file_path = sorted(glob(os.path.join(model_save_dir, '**{}_**.pth'.format(emb_dim))))[0]
        ad_obj = AD_model_container(
            emb_dim,
            num_entities
        ) 
        ad_obj.load_model(model_file_path)
        model_dict[emb_dim] = ad_obj
    return 

'''
This is an external facing function
Input : a single row of pandas dataframe
This record is not serialized is a single row of a dataframe
Returns:
{ <emb_dim>: { percentile : ( threshold, record_score ) } 

'''
def score_new_sample(record : pd.DataFrame):
    global DIR, id_col
    global model_dict
    global threshold_dict
    # perform serialization
    serialized_record = util.convert_to_serializedID_format(
        record, 
        DIR
    )
    try:
        del serialized_record[id_col]
    except:
        pass
    x_values = serialized_record.values[0]
    x_values = x_values.reshape([1,-1])
    result = {}
    for emb_dim in MEAD_emb_list:
        ad_obj = model_dict[emb_dim] 
        score = ad_obj.predict(x_values)
        _res = {}
        cutoff_perc_values = threshold_dict[emb]
        for perc,v in cutoff_perc_values.items():
            _res[perc] = (v, score[0])
        result[emb_dim] = _res
    return result

# ======================================================================================
if __name__ == "main":
    print('Executing.')
    parser = argparse.ArgumentParser(description='Train anomaly detection model (MEAD)')
    parser.add_argument('--dir', type = str, help='Which dataset ? us__import{1,2...}' ) 
    args = parser.parse_args()
    DIR = args.dir
    setup_config(DIR)
    train_AD_models()
    calculate_thresholds()
    
    
# ======================================================================================
'''
Calling externally after initialization
'''
# 
# setup_config(DIR)
# read_thresold_dict()
# read_models()


# In[ ]:




