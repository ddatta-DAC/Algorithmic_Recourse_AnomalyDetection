#!/usr/bin/env python
# coding: utf-8

# In[72]:


import os
import sys
sys.path.append('./..')
import argparse
import numpy as np
import pandas as pd
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
from recourseExpDataFetcher import anomDataFetcher 
from joblib import Parallel, delayed
import multiprocessing as MP
from utils import log

from Metrics import accuracy
LOGGER = None
DIR = None
cf_results_dir = None
model = None

def setup_config(_dir_, _model_):
    global DIR, cf_results_dir
    global LOGGER
    DIR = _dir_
    
    with open('./config.yaml', 'rb') as fh:
        config = yaml.safe_load(fh)
    base_path = './' 
    LOG_FILE = os.path.join(base_path,'log_results_accuracy_{}.txt'.format(_model_ ))
    LOGGER = log.get_logger(LOG_FILE)
    cf_results_dir = config['cf_results_dir'][_model_].format(DIR)
    


def get_ref_data():
    global DIR
    anomDataFetcher_obj  = anomDataFetcher(DIR)
    data = anomDataFetcher_obj.fetch_data()
    df1 = data['anomalies_1_1']['data']
    df2 = data['anomalies_1_2']['data'] 
    ref_data = df1.append(df2, ignore_index= True)
    return ref_data


def _eval_(file, ref_data, eval_obj):
    
    record_id = int(os.path.basename(file).split('.')[0].split('_')[1])
    anomaly_record = ref_data.loc[ref_data['PanjivaRecordID'] == record_id]
    cf_df = pd.read_csv(file, index_col=None)
    acc = Parallel(n_jobs = MP.cpu_count(),prefer="threads")(delayed(eval_obj.calc_acc)(anomaly_record.iloc[0],cf_df.iloc[i]) for i in range(len(cf_df)) )
    return np.mean(acc)


def main():
    global DIR, LOGGER, cf_results_dir, model
    print('[Accuracy]', DIR, model)
    ref_data = get_ref_data()
    eval_obj = accuracy.accuracy_evaluator(DIR)
    acc = []
    files = glob(os.path.join(cf_results_dir,'**.csv'))
    for file in files:
        acc.append(_eval_(file, ref_data, eval_obj))
    LOGGER.info('{}, {} , Accuracy , mean {:.4f}, std{:.4f}'.format(model, DIR,np.mean(acc), np.std(acc)))

parser = argparse.ArgumentParser(description='FINMAP')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--model', type = str,  choices=['xformer_random', 'xformer','RCEAA', 'FINMAP','exhaustive'] ) 
 

args = parser.parse_args()

_dir_ = args.dir
model = args.model

setup_config(_dir_, model)
main()





