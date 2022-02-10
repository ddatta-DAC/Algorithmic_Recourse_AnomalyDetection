#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
sys.path.append('./..')
import pandas as pd
import numpy as np
import pickle
import torch
import argparse
from torch import nn
import yaml
import multiprocessing as MP
from joblib import Parallel,delayed
from recourseExpDataFetcher import anomDataFetcher
from pathlib import Path
from collections import OrderedDict
import proxyClassifier
import perturbation_model
from proxyClassifier import proxy_clf_network, proxy_clf
from perturbation_model import perturb_network, perturb_clf
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import log
from  timeit import default_timer as timer
try:
    file_path = os.path.realpath(__file__)
    base_path =  os.path.dirname(basepath)
except:
    base_path = './'
CONFIG_FILE = os.path.join(base_path, 'config.yaml')
ID_COL = 'PanjivaRecordID'



# Globals
DIR = None
Data_Dir = None
Anom_Dir = None
data_count = None 
clf_batch_size = None
clf_model_save_dir = None 
perturbModel_train_epochs = None
LOGGER = None
cf_save_dir = None
data_columns = None

def setup_config(_dir):
    global DIR, Data_Dir, Anom_Dir
    global CONFIG_FILE
    global data_count
    global clf_batch_size
    global clf_model_save_dir
    global base_path
    global perturbModel_train_epochs
    global cf_save_dir
    global data_columns
    global LOGGER
    
    with open(CONFIG_FILE,'r') as fh:
        CONFIG = yaml.safe_load(fh)
    Data_Dir = CONFIG['data_loc']
    Anom_Dir = CONFIG['anom_data_path']
    DIR = _dir
    data_count = CONFIG['data_count']
    clf_batch_size = CONFIG['clf_batch_size']
    clf_model_save_dir = os.path.join(base_path, './clf_model_save_dir', DIR)
    Path(clf_model_save_dir).mkdir(exist_ok=True, parents=True)
    Path(clf_model_save_dir).mkdir(exist_ok=True, parents=True)
    perturbModel_train_epochs = CONFIG['perturbModel_train_epochs']
      
    LOG_FILE = os.path.join(base_path,'log_results_{}.txt'.format(_dir))
    LOGGER = log.get_logger(LOG_FILE)
    log.log_time(LOGGER)
    cf_save_dir = os.path.join(CONFIG['cf_save_dir'], DIR)
    Path(cf_save_dir).mkdir(exist_ok=True, parents=True)
    domain_dims = get_domain_dims()
    data_columns = list(domain_dims)
    
    return



def get_domain_dims():
    global DIR, Data_Dir
    with open(os.path.join(Data_Dir, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = OrderedDict(pickle.load(fh))
    return domain_dims

def proxy_training_data():
    global DIR, Data_Dir, Anom_Dir, data_points, ID_COL, data_count
    # normal_instances
    path_normal = os.path.join(Data_Dir, DIR, 'train_data.csv')
    df_normal = pd.read_csv(path_normal,index_col=None)
    
    path_anom = os.path.join(Anom_Dir, DIR, 'anomalies_1_1.csv')
    df_anomalies_1 = pd.read_csv(path_normal,index_col=None)
    path_anom = os.path.join(Anom_Dir, DIR, 'anomalies_1_2.csv')
    df_anomalies_2 = pd.read_csv(path_normal,index_col=None)    
    df_anomalies = df_anomalies_1.append(df_anomalies_2,ignore_index=True)
    
    
    if data_count > 0:
        if len(df_anomalies) < data_count:
            df_anomalies = df_anomalies.sample(n = data_count, replace=True)
        else:
            df_anomalies = df_anomalies.sample(n = data_count)

        df_normal = df_normal.sample(n = data_count) 
        
    # create X, Y
    Y1 = np.ones([len(df_normal)])
    Y2 = np.zeros([len(df_anomalies)])
    
    try: del df_normal[ID_COL]
    except: pass
    try: del df_anomalies[ID_COL]
    except: pass
    X1 = df_normal.values
    X2 = df_anomalies.values
    
    X = np.vstack([X1,X2])
    Y = np.hstack([Y1,Y2])            
    return X,Y
    


def get_proxy_clf():
    """
    Train model  and store
    """
    global clf_batch_size, DIR
    global DEVICE
    global clf_model_save_dir
    domain_dims = get_domain_dims()
    
    network = proxy_clf_network(
        list(domain_dims.values())
    )
    clf_obj = proxy_clf(
            model = network,
            dataset = DIR, 
            save_dir = clf_model_save_dir,
            batch_size= clf_batch_size,
            device = DEVICE    
        )
    
    try:
        clf_obj.load_model()
    except:
        
        train_X, train_Y = proxy_training_data()
        clf_obj.fit(train_X, train_Y.reshape([-1,1]), num_epochs=200)
        clf_obj.save_model()
    return clf_obj

def get_pert_model(_id_= None):
    global DEVICE, DIR
    domain_dims = get_domain_dims()
    clf_obj = get_proxy_clf()
    clf_network = clf_obj.model
    
    perturb_network_obj  = perturb_network(
        list(domain_dims.values())
    )
    
    perturb_clf_obj = perturb_clf(
        perturb_model = perturb_network_obj,
        clf_model = clf_network,
        dataset = DIR,
        device = DEVICE,
        signature = _id_
    )
    return perturb_clf_obj

def train_pert_model(
    perturb_clf_obj,
    X, 
    Y
):
    global LOGGER
    global DIR
    global ID_COL
    global perturbModel_train_epochs
    loss_values = perturb_clf_obj.fit(
        X , 
        Y , 
        num_epochs = perturbModel_train_epochs
    )
    return perturb_clf_obj

def generate_cf(row,num_cf):
    global LOGGER
    global ID_COL
    global cf_save_dir
    global data_columns
    
    start = timer()
    _id_ = row[ID_COL]
    perturb_clf_obj = get_pert_model(_id_)
    X = row[data_columns].values.reshape([1,-1])
    Y = np.zeros([1,1])
    perturb_clf_obj = train_pert_model(perturb_clf_obj, X, Y)
    res = []
    for j in range(num_cf):
        x = perturb_clf_obj.predict(X)[0]
        res.append(x)

    res = np.vstack(res)
    _id_ = np.arange(num_cf).reshape([-1,1])
    res_df_data = np.hstack([_id_, res])
    res_df = pd.DataFrame(res_df_data, columns = [ID_COL] + data_columns)
    fname = 'cf_{}.csv'.format(row[ID_COL])
    res_df.to_csv( os.path.join(cf_save_dir,fname),index=False)
    end = timer()
    time = end - start
    return time

def main(
    num_anomalies = 1000,
    num_cf = 50
):
    global DIR 
    global LOGGER
    global ID_COL
    global cf_save_dir
    global data_columns
    anomDataFetcher_obj  = anomDataFetcher(DIR)
    data = anomDataFetcher_obj.fetch_data()
    # keys are anomalies_1_1, anomalies_1_2
    keys = data.keys()
    domain_dims = get_domain_dims()
    get_proxy_clf()
    
    for anom_type in keys:
        df = data[anom_type]['data'].head(num_anomalies)
        res = Parallel(MP.cpu_count(), prefer="threads")(delayed(generate_cf)(row, num_cf) for i,row in  df.iterrows())
        time_taken = np.mean(res)
        LOGGER.info('Num anomalies: {}|  Time taken {:.4f} | Epochs {} | Num CF: {}'.format(num_anomalies, time_taken, perturbModel_train_epochs, num_cf))
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







