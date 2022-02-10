#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Precompute CF anomaly scores for conditional correctness
"""

import os
import sys
import torch 
import pandas as pd
import numpy as np
from glob import glob 
from pathlib import Path
from joblib import Parallel,delayed
import multiprocessing as MP
import yaml
from tqdm import tqdm

sys.path.append('./..')

from AD_model import AD_processor
import argparse

'''
AD_proc_obj.score_samples_batch( 
    records : pd.DataFrame, 
    model_emb_dim = None
)
'''
cf_results_dir = None
DIR = None
AD_model_emb_dim = 32
save_dir = None
model = None
ID_COL = 'PanjivaRecordID'


def setup_config(_dir_, model):
    global DIR
    global save_dir
    global AD_model_emb_dim
    global cf_results_dir
    global ID_COL
    ID_COL = 'PanjivaRecordID'
    DIR = _dir_
    with open('config.yaml','rb') as fh:
        config = yaml.safe_load(fh)
    cf_results_dir = config['cf_results_dir'][model].format(DIR)
    save_dir = os.path.join(cf_results_dir,  'precomputed_scores_AD')
    Path(save_dir).mkdir(exist_ok=True,parents=True)
    
    return

def main():
    global DIR
    global cf_results_dir
    global AD_model_emb_dim
    global save_dir
    global model
    global ID_COL
    AD_proc_obj = AD_processor.AD_processor(DIR)
    input_file_list = glob(os.path.join(cf_results_dir, '**.csv'))
    print(DIR,  model, 'CF count', len(input_file_list))
    
    def aux(file):
        nonlocal AD_proc_obj
        global save_dir
        
        df = pd.read_csv(file, index_col=None)
        id_list = df[ID_COL].values.reshape([-1,1])
        del df[ID_COL]
        
        scores = AD_proc_obj.score_samples_batch(
            df, 
            model_emb_dim = AD_model_emb_dim
        )
        scores = np.array(scores).reshape([-1,1])
        tmp_df = pd.DataFrame( np.hstack([id_list, scores]), columns = [ID_COL, 'score'])
        tmp_df[ID_COL] = tmp_df[ID_COL] .astype(int)
        fname = os.path.basename(file).split('.')[0] + '_scores.csv'
        fpath = os.path.join(save_dir, fname)
        
        tmp_df.to_csv(fpath,index=False)
        print(fpath)
        return 
    for file in tqdm(input_file_list):
        aux(file)  
    # Parallel(n_jobs = MP.cpu_count()//4, prefer="threads")(delayed(aux)(file) for file in tqdm(input_file_list))

    
parser = argparse.ArgumentParser(description='FINMAP')
parser.add_argument('--dir', type = str,  help='Which dataset ? us__import{1,2...}' ) 
parser.add_argument('--model', type = str,  choices=['xformer_random', 'xformer','RCEAA', 'FINMAP','exhaustive'] ) 
args = parser.parse_args()

_dir_ = args.dir
model = args.model

setup_config(_dir_, model)
main()
        


