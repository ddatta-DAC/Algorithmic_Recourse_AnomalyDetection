
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
from collections import OrderedDict
from main import recourse_generator

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
cf_save_dir = None
metapath_file = None
metapath_list = None
num_NN = 100
AD_model_emdDim = 32

def get_domain_dims():
    global DIR, Data_Dir
    with open(os.path.join(Data_Dir, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return domain_dims

def read_metapaths():
    global metapath_file
    global metapath_list
    with open(metapath_file) as fh:
        metapath_list = fh.readlines()
    metapath_list = [ _.strip('\n') for _ in metapath_list]
    metapath_list = [ _.split(',') for _ in metapath_list]
    return metapath_list

def setup_config(_dir_):
    global DIR
    global domain_dims
    global LOGGER, Data_Dir
    global cf_save_dir
    global metapath_file, metapath_list
    global num_NN, AD_model_emdDim
    DIR = _dir_
    
    with open(CONFIG_FILE,'r') as fh:
        CONFIG = yaml.safe_load(fh)

    cf_save_dir = os.path.join(CONFIG['cf_save_dir'],DIR)
    Path(cf_save_dir).mkdir(exist_ok=True,parents=True)
    Data_Dir = CONFIG['data_loc']
    domain_dims = get_domain_dims()
    LOG_FILE = os.path.join(base_path,'log_results_{}.txt'.format(_dir_))
    LOGGER = log.get_logger(LOG_FILE)
    metapath_file = os.path.join(base_path, CONFIG['metapath_file'][DIR])
    read_metapaths()
           
    num_NN = CONFIG['num_NN']
    AD_model_emdDim = CONFIG['AD_model_emdDim']  
    return 


def generate_cf(record, num_cf):
    global domain_dims
    global DIR 
    global ID_COL
    global cf_save_dir
    global fixed_domains
    global num_NN
    global AD_model_emdDim
    global metapath_list
    data_domains = list(domain_dims.keys()) 
    
    fname = 'cf_{}.csv'.format(record[ID_COL])
    
    start = timer()
    gen_obj = recourse_generator(DIR, domain_dims, metapath_list, num_NN, AD_model_emdDim)
    
    candidate_df = gen_obj.generate_recourse( pd.DataFrame([record]).copy(deep=True), num_cf = num_cf)
    
    
    candidate_df.to_csv(os.path.join( cf_save_dir, fname), index=False)
    end = timer()
    time_taken = end -start
    return time_taken  


def main(num_anomalies,num_cf):
    global DIR 
    global ID_COL
    global LOGGER 
    
    anomDataFetcher_obj  = anomDataFetcher(DIR)
    data = anomDataFetcher_obj.fetch_data()
    # keys are anomalies_1_1, anomalies_1_2
    keys = data.keys()
    
    for anom_type in keys:
        df = data[anom_type]['data'].head(num_anomalies)
        time_taken = []
        for i,row in  df.iterrows():
            t = generate_cf(row, num_cf)
            time_taken.append(t)
        # time_taken = Parallel(MP.cpu_count(), prefer="threads")(delayed(generate_cf)(row, num_cf) for i,row in  df.iterrows())
        time_taken = [ _ for _ in time_taken if _ > 0]
        time_taken = np.mean(time_taken)
        LOGGER.info('Num anomalies: {} |  Num CF: {} | Time taken {:.4f} |  | '.format(num_anomalies, num_cf, time_taken ))
    return 



parser = argparse.ArgumentParser(description='xformer_recourse')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--num_anomalies', type = int,  help='How many anomalies' ) 
parser.add_argument('--num_cf', type = int,  help='How many anomalies' ) 

args = parser.parse_args()
_DIR = args.dir
num_anomalies = args.num_anomalies
num_cf = args.num_cf

setup_config(_DIR)
main(num_anomalies,num_cf)

