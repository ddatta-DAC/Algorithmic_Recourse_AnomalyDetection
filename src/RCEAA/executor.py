#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
sys.path.append('./..')
import torch
from utils import log
import os
from glob import glob
import pickle
from tqdm import tqdm 
import pickle
import torch
import argparse
from pathlib import Path
from typing import *
from torch import nn
import yaml
import multiprocessing as MP
from joblib import Parallel,delayed
from recourseExpDataFetcher import anomDataFetcher 
from RCEAA_model import RCEAA
from collections import OrderedDict

from utils import log
from  timeit import default_timer as timer
try:
    file_path = os.path.realpath(__file__)
    base_path =  os.path.dirname(basepath)
except:
    base_path = './'
CONFIG_FILE = os.path.join(base_path, 'config.yaml')
ID_COL = 'PanjivaRecordID'
LOGGER = None
Data_Dir = None
domain_dims = None
max_opt_iter = 100
hyperparam_step = 0.25
opt_LR = 0.01
cf_save_dir = None


def setup_config(_dir_):
    global DIR
    global domain_dims
    global LOGGER, Data_Dir
    global hyperparam_step
    global max_opt_iter, opt_LR
    global cf_save_dir
    
    DIR = _dir_
    
    with open(CONFIG_FILE,'r') as fh:
        CONFIG = yaml.safe_load(fh)
        
    hyperparam_step = CONFIG['hyperparam_step']
    max_opt_iter = CONFIG['max_opt_iter']
    opt_LR = CONFIG['opt_LR']
    cf_save_dir = os.path.join(CONFIG['cf_save_dir'],DIR)
    Path(cf_save_dir).mkdir(exist_ok=True,parents=True)
    Data_Dir = CONFIG['data_loc']
    domain_dims = get_domain_dims()
    LOG_FILE = os.path.join(base_path,'log_results_{}.txt'.format(_dir_))
    LOGGER = log.get_logger(LOG_FILE)
    return 


def get_domain_dims():
    global DIR, Data_Dir
    with open(os.path.join(Data_Dir, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return domain_dims

def generate_cf(
    record: pd.Series,
    num_cf = 50
): 
    global hyperparam_step
    global max_opt_iter
    global ID_COL
    global domain_dims
    global opt_LR
    global cf_save_dir
    
    fname = 'cf_{}.csv'.format(record[ID_COL])
    f_path = os.path.join( cf_save_dir, fname)
    if os.path.exists(f_path):
        return -1
    
    start = timer()
    obj = RCEAA(
        dataset=DIR,
        domain_dims = list(domain_dims.values()),
        domain_names = list(domain_dims.keys()),
        LR = opt_LR
    )
    
    x = obj.find_CF(
        record,
        num_cf = num_cf,
        max_iter = max_opt_iter,
        hyp_step = hyperparam_step,
        lambda_1_range = [0.25,0.75],
        lambda_2_range = [0.25,0.75]
    )
      
    # convert to dataframe
    data = np.array(x)
    _id_list_ = np.arange(num_cf).reshape([-1,1])
    df_data = np.hstack([_id_list_, data])
    
    fname = 'cf_{}.csv'.format(record[ID_COL])
    res_df = pd.DataFrame (df_data, columns = [ID_COL] + list(domain_dims.keys()))
    res_df.to_csv( os.path.join( cf_save_dir, fname),index=False)
    end = timer()
    time_taken = end -start
    return time_taken  
        
def main(
    num_anomalies= 10,
    num_cf = 50
):
    global domain_dims
    global DIR
    global LOGGER
    anomDataFetcher_obj  = anomDataFetcher(DIR)
    data = anomDataFetcher_obj.fetch_data()
    # keys are anomalies_1_1, anomalies_1_2
    keys = data.keys()
    
    for anom_type in keys:
        df = data[anom_type]['data'].head(num_anomalies).sample(frac=1)
        time_taken = []
        for i,row in  df.iterrows():
            t = generate_cf(row, num_cf)  
            time_taken.append(t)
        # time_taken = Parallel(MP.cpu_count(), prefer="threads")(delayed(generate_cf)(row, num_cf) for i,row in  df.iterrows())
        time_taken = np.mean(time_taken)
        LOGGER.info('Num anomalies: {} |  Num CF: {} | Time taken {:.4f} |  | '.format(num_anomalies, num_cf, time_taken ))
    return 




parser = argparse.ArgumentParser(description='FINMAP')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--num_anomalies', type = int,  help='How many anomalies' ) 
parser.add_argument('--num_cf', type = int,  help='How many anomalies' ) 

args = parser.parse_args()
_DIR = args.dir
num_anomalies = args.num_anomalies
num_cf = args.num_cf

setup_config(_DIR)
main(num_anomalies,num_cf)

