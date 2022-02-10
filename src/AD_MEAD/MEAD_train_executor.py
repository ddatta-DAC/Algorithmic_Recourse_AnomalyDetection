#!/usr/bin/env python
# coding: utf-8


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
    from .model_MEAD import MEAD_model_container, MEAD
except:
    from model_MEAD import MEAD_model_container, MEAD
# ======================================== #

DEVICE =  torch.device("cpu")
file_path = os.path.realpath(__file__)
print(file_path)
CONFIG_FILE = os.path.join( os.path.dirname(file_path) ,'config.yaml')

# CONFIG_FILE = './config.yaml'
ad_model_data_dir = None
emb_dim = 32
CONFIG = None
domain_dims = None
train_epochs = 100
error_tol = 0.001
learning_rate = 0.001
model_save_dir = None
num_entities = 0
id_col = 'PanjivaRecordID'



def get_domain_dims(data_loc):
    
    domain_dims = pd.read_csv(os.path.join(data_loc, 'data_dimensions.csv'),index_col=None)
    return domain_dims
    
'''
Set up globals
'''

def setup_config(subDIR):
    global CONFIG
    global CONFIG_FILE
    global DIR, ad_model_data_dir, emb_dim, batch_size, error_tol, train_epochs, learning_rate, model_save_dir
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
    emb_dim = CONFIG['emb_dim']
    
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
def train_AD_model():
    global DIR, ad_model_data_dir, emb_dim, train_epochs, model_save_dir, DEVICE
    
   
    global ad_model_data_dir, train_epochs, learning_rate, batch_size, error_tol, model_save_dir, DEVICE  , num_entities
    train_x_pos = os.path.join(ad_model_data_dir, 'train_x_pos.npy')
    train_x_neg = os.path.join(ad_model_data_dir, 'train_x_neg.npy')
    train_x_pos = np.load(train_x_pos)
    train_x_neg = np.load(train_x_neg)
    train_x_neg = train_x_neg.reshape([train_x_pos.shape[0], -1, train_x_pos.shape[-1]])
    ad_obj = MEAD_model_container(
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


