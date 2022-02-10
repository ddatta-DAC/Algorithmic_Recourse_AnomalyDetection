import torch
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import os
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import yaml
import sys
import pickle

sys.path.append('./../..')
sys.path.append('./..')

ID_COL = 'PanjivaRecordID'
RESULTS_OP_PATH = 'APE_output'
DATALOC = './../../generated_data_v1'

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
    from .model_APE import APE_container, APE
except:
    from model_APE import APE_container, APE
# ======================================== #

DEVICE =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_path = os.path.dirname( os.path.realpath(__file__))
CONFIG_FILE = os.path.join( base_path ,'config.yaml')

data_loc = None
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
emb_dim = 32

def get_domain_dims(data_loc):
    domain_dims = pd.read_csv(os.path.join(data_loc, 'data_dimensions.csv'),index_col=None)
    return domain_dims
    
'''
Set up globals
'''

def setup_config(subDIR):
    global CONFIG
    global CONFIG_FILE
    global DIR, ad_model_data_dir, MEAD_emb_list, batch_size,  train_epochs, learning_rate, model_save_dir, emb_dim
    global domain_dims, num_entities
    global perc_threshold
    global data_loc
    global error_tol
    
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
    domain_dims = get_domain_dims()
    emb_dim = CONFIG['emb_dim']
    
    train_epochs = CONFIG['train_epochs']
    learning_rate = CONFIG['learning_rate']
    batch_size = CONFIG['batch_size']
    error_tol = CONFIG['error_tol']
    model_save_dir = os.path.join(CONFIG['model_save_dir'], DIR )
    num_entities = np.sum(domain_dims.values())
    
    return 



def get_domain_dims():
    global data_loc, DIR
    with open(os.path.join(data_loc, '{}/domain_dims.pkl'.format(DIR)), 'rb')  as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def get_training_data():
    global ad_model_data_dir
    global DIR
    train_x_pos = os.path.join(ad_model_data_dir, 'train_x_pos.npy')
    train_x_neg = os.path.join(ad_model_data_dir, 'train_x_neg.npy')
    train_x_pos = np.load(train_x_pos)
    train_x_neg = np.load(train_x_neg)
        
    return train_x_pos, train_x_neg


def main():
    global ID_COL
    global DIR
    global data_loc
    global RESULTS_OP_PATH
    global model_save_dir
    global learning_rate
    global train_epochs
    global batch_size
    global error_tol
    global emb_dim
    path_obj = Path(model_save_dir)
    path_obj.mkdir( parents=True, exist_ok=True)


    x_pos, x_neg = get_training_data()
    x_neg = x_neg.reshape([x_pos.shape[0], -1, x_pos.shape[1]])
    domain_dims = get_domain_dims()

    container = APE_container(
        emb_dim, 
        domain_dims, 
        device, 
        batch_size=batch_size, 
        LR=learning_rate,
        model_save_dir = model_save_dir
    )

    loss = container.train_model(
        x_pos,
        x_neg,
        num_epochs=train_epochs,
        tol = error_tol
    )
    container.save_model()
    return


# ===================================== #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dir', choices=['us_import1', 'us_import2', 'us_import3', 'colombia_export', 'ecuador_export'],
    default='us_import1'
)


args = parser.parse_args()
setup_config(args.dir)

main()
