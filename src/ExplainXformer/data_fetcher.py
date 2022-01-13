import pandas as pd
import os
import sys
import numpy as np
import math
from typing import *
import collections
from pathlib import Path
import yaml
import inspect

_rel_path_ = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(_rel_path_,'config.yaml')

id_col = 'PanjivaRecordID'
num_neg_samples_ape = None
use_cols = None
DATA_SOURCE = None

def set_up_config(_DIR = None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global DATA_SOURCE
    global DIR_LOC
    global NUMERIC_COLUMNS
    global id_col
  
    '''
    Return if already initialized!!
    '''
    if DATA_SOURCE is not None:
        return 
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']

    save_dir =  CONFIG['data_loc']
    data_loc = os.path.join(
        CONFIG['data_loc'],
        DIR
    )
    DATA_SOURCE = data_loc
    return 


def get_training_set_data(DIR):
    global DATA_SOURCE
    global id_col
    set_up_config(DIR)
    df = pd.read_csv(os.path.join(DATA_SOURCE, 'train_data.csv'), index_col=None)
    # drop duplicates
    columns = list(df.columns)
    columns.remove(id_col)
    df = df.drop_duplicates(subset=columns)
    return df
    
def get_domain_dims(DIR):
    global DATA_SOURCE
    set_up_config(DIR)
    df = pd.read_csv(os.path.join(DATA_SOURCE, 'data_dimensions.csv'), index_col=None)
    return df

def get_testing_set_data(DIR):
    global DATA_SOURCE
    global id_col
    set_up_config(DIR)
    df = pd.read_csv(os.path.join(DATA_SOURCE, 'test_data.csv'), index_col=None)
    # drop duplicates
    columns = list(df.columns)
    columns.remove(id_col)
    df = df.drop_duplicates(subset=columns)
    return df