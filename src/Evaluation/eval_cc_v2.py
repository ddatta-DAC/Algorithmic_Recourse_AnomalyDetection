#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('./..')
import argparse
import numpy as np
import pandas as pd
import numpy as np
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

from Metrics import conditional_correctness_v2 as CC
LOGGER = None
DIR = None
cf_results_dir = None
model = None
Data_Dir = None
cf_results_scores_dir = None


def setup_config(_dir_, _model_):
    global DIR, cf_results_dir, cf_results_scores_dir
    global LOGGER, Data_Dir
    DIR = _dir_
    
    with open('./config.yaml', 'rb') as fh:
        config = yaml.safe_load(fh)
    Data_Dir = config['data_loc']
    base_path = './' 
    LOG_FILE = os.path.join(base_path,'log_results_conditionalCorrectness_{}.txt'.format(_model_ ))
    LOGGER = log.get_logger(LOG_FILE)
    cf_results_dir = config['cf_results_dir'][_model_].format(DIR)
    cf_results_scores_dir = os.path.join(cf_results_dir, 'precomputed_scores_AD')
    read_doamin_dims()
    
def read_doamin_dims():
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
    global cf_results_dir, cf_results_scores_dir
    
    eval_obj = CC.condCorrectnessEvaluator(_dir_)
    ref_data = get_ref_data()
    
    
    
    def aux(eval_obj, file):
        nonlocal ref_data
        record_id = int(os.path.basename(file).split('.')[0].split('_')[1])    
        try:
            score_file = glob(os.path.join(cf_results_scores_dir, '**{}**.csv'.format(record_id)))[0]
        except:
            print('[ERROR]', file)
        cf_scores_df = pd.read_csv(score_file, index_col=None)
        anomaly_record = ref_data.loc[ref_data['PanjivaRecordID'] == record_id]
        cf_df = pd.read_csv( file, index_col=None)
        cc = eval_obj.get_average_score(
            anomaly_record, 
            cf_df,
            cf_scores_df
        )
        return cc
    
    files = glob(os.path.join(cf_results_dir,'**.csv'))
    res = []
    for file in files:
        cc = aux(eval_obj, file) 
        res.append(cc)
    LOGGER.info('{}, {} , Conditional Correctness , mean {:.4f}, std{:.4f}'.format(model, DIR, np.mean(res), np.std(res)))

    
# ================================================================================================================================

parser = argparse.ArgumentParser(description='FINMAP')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--model', type = str,  choices=['xformer_random', 'xformer','RCEAA', 'FINMAP','exhaustive'] ) 
args = parser.parse_args()

_dir_ = args.dir
model = args.model

setup_config(_dir_, model)
main()


