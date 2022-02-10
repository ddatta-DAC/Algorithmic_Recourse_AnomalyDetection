#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./..')
import argparse
import numpy as np
import pandas as pd
import pickle
from glob import glob
import torch
from utils import util
from typing import *
from torch import LongTensor as LT
import pandas as pd
import yaml
import numpy as np
from recourseExpDataFetcher import anomDataFetcher 
from joblib import Parallel, delayed
import multiprocessing as MP
from utils import log

from Metrics import sparsity 
LOGGER = None
DIR = None
cf_results_dir = None
model = None
Data_Dir = None

def setup_config(_dir_, _model_):
    global DIR, cf_results_dir
    global LOGGER, Data_Dir
    DIR = _dir_
    
    with open('./config.yaml', 'rb') as fh:
        config = yaml.safe_load(fh)
    Data_Dir = config['data_loc']
    base_path = './' 
    LOG_FILE = os.path.join(base_path,'log_results_Sparsity_{}.txt'.format(_model_ ))
    LOGGER = log.get_logger(LOG_FILE)
    cf_results_dir = config['cf_results_dir'][_model_].format(DIR)
    read_domain_dims()
    
def read_domain_dims():
    global DIR
    global domain_dims
    with open(os.path.join(Data_Dir,DIR, 'domain_dims.pkl'),'rb') as fh:
        domain_dims =  OrderedDict(pickle.load(fh))

def get_ref_data():
    global DIR
    anomDataFetcher_obj  = anomDataFetcher(DIR)
    data = anomDataFetcher_obj.fetch_data()
    df1 = data['anomalies_1_1']['data']
    df2 = data['anomalies_1_2']['data'] 
    ref_data = df1.append(df2, ignore_index= True)
    return ref_data

def main():
    global DIR
    global LOGGER
    global cf_results_dir
    global domain_dims
    
    domain_list = list(domain_dims.keys())
    eval_obj = sparsity.sparsity(_dir_, domain_list)
    ref_data = get_ref_data()
    
    def aux(eval_obj, file):
        nonlocal ref_data
        record_id = int(os.path.basename(file).split('.')[0].split('_')[1])
        
        anomaly_record = ref_data.loc[ref_data['PanjivaRecordID'] == record_id]
        cf_df = pd.read_csv( file, index_col=None)
        _spar = eval_obj.calculate(anomaly_record, cf_df)
        
        return _spar
    
    files = glob(os.path.join(cf_results_dir,'**.csv'))
    res = []
    # for file in files:
    #     val = aux(eval_obj, file) 
    #     res.append(val)
    
    res = Parallel(
        n_jobs = MP.cpu_count()//10, prefer="threads"
    )(delayed(aux)(eval_obj, file) for file in files)
    print('{}, {} , Sparsity , mean {:.4f}, std{:.4f}'.format(model, DIR, np.mean(res), np.std(res)))
    LOGGER.info('{}, {} , Sparsity , mean {:.4f}, std{:.4f}'.format(model, DIR, np.mean(res), np.std(res)))
    return    

# _dir_ = 'us_import2'
# model = 'FINMAP'
# setup_config(_dir_, model)
# main()    

parser = argparse.ArgumentParser(description='FINMAP')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--model', type = str,  choices=['xformer_random', 'xformer','RCEAA', 'FINMAP','exhaustive'] ) 
args = parser.parse_args()

_dir_ = args.dir
model = args.model

setup_config(_dir_, model)
main()

