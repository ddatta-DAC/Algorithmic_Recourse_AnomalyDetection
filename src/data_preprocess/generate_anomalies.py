import os
import sys
import pandas as pd
import multiprocessing as MP
import numpy as np
from typing import *
from collections import defaultdict,OrderedDict
from joblib import Parallel, delayed
import multiprocessing as MP
import pickle
import argparse
from tqdm import tqdm

from itertools import combinations

from tqdm import tqdm
from pathlib import Path
from glob import glob 
import yaml
from scipy.sparse import csr_matrix, lil_matrix


# =======================
# Globals
# =======================

DIR = None
data_loc = None
anomaly_save_dir = None
data_df_path = None
domain_dims = None
test_data_path = None
CONFIG_FILE = 'anomalyGen_config.yaml'
id_col = 'PanjivaRecordID'
entity_coocc_matrix = defaultdict()
CONFIG = None

def setup_config(DIR):
    global id_col
    global CONFIG_FILE, CONFIG
    global data_loc, anomaly_save_dir, domain_dims, data_df_path, test_data_path
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)
    data_loc = CONFIG['data_dir']
    anomaly_save_dir = os.path.join(data_loc, DIR, CONFIG['save_dir'])
    Path(anomaly_save_dir).mkdir(exist_ok=True, parents=True)
    
    domain_dims_file = CONFIG['domain_dims_file']
    subDIR = DIR
    with open(os.path.join(data_loc, subDIR, domain_dims_file), 'rb') as fh :
        domain_dims = pickle.load(fh)
    data_df_path = os.path.join( data_loc, DIR, CONFIG['train_data_file'])
    test_data_path = os.path.join( data_loc, DIR, CONFIG['test_data_file'])
    
    return
    


def setup_coOcc_matrix(data_df):
    global id_col, entity_coocc_matrix, domain_dims
    columns = list(data_df.columns)
    columns.remove(id_col)
    
    def aux(pair):
        global entity_coocc_matrix, domain_dims
        pair = sorted(pair)
        attr1,attr2 = pair[0], pair[1]
        tmp_df = data_df[[attr1,attr2]]
        key = (attr1,attr2)
        tmp_df = tmp_df.drop_duplicates()
        entity_coocc_matrix[key] = lil_matrix(
            np.zeros([domain_dims[attr1],domain_dims[attr2]],dtype=bool)
        )
        
        for _ , _row in tmp_df.iterrows():
            idx1 = _row[attr1]
            idx2 = _row[attr2]
            entity_coocc_matrix[key][idx1,idx2] = 1
        return 
    Parallel(n_jobs = MP.cpu_count(),backend="threading")(delayed(aux)(pair,) for pair in tqdm(combinations(columns,2)))
    return 
        
 

           

'''
Anomaly type = 1  ::: 
    With repect to one of the fixed types change something
Anomaly type = 2 :::
    Choose any pair of attributes
'''
def aux_genrateAnomaly(
    row,
    perturb_columns: List,
    fixed_columns: List,
    num_perturb: int = 2,
    anom_type:int  = 1
):
    global id_col, domain_dims, CONFIG
    # Sparse matrix to speed up computation
    global entity_coocc_matrix 
    # columns = [ _ for _ in list(df.columns) if _ not in fixed_columns + [id_col])
   
    new_id = int(str(row[id_col]) +  '00' + str(anom_type) + '00' + str(num_perturb))
    new_record = row.copy()
    new_record[id_col] = new_id
  
    
    if anom_type == 1 :
        # choose columns at random
        attr_p = np.random.choice(perturb_columns, size=num_perturb, replace= False)
        attr_f = np.random.choice(fixed_columns, size=num_perturb, replace= False)
        for i in range(num_perturb):
            domain1 = attr_p[i] # domain being perturbed
            domain2 = attr_f[i]
            target_domain = domain1
            fixed_val = row[domain2]
            _order_ = int(domain1 < domain2)
            tmp = sorted([domain1,domain2])
            key_d1, key_d2 = tmp[0], tmp[1]
            key = (key_d1, key_d2)
            
            if _order_ == 1 : 
                choices = entity_coocc_matrix[key][:,fixed_val].toarray().reshape(-1)
            else:
                choices = entity_coocc_matrix[key][fixed_val,:].toarray().reshape(-1)
            choices = np.argwhere(choices == False).reshape(-1)
            new_record[domain1] = int(np.random.choice(choices, 1))      
    else:
        # Read in the matapaths
        with open(CONFIG['anomaly_relations'], 'r') as fh:
            lines = fh.readlines()
            relations_list = [ _.strip().split(',') for _ in lines]
        
        _idx = np.random.choice(np.arange(len(relations_list)), 1)[0]
        candidate_relation = relations_list[_idx]
        candidiate_domains_p = np.random.choice(
            list(set(candidate_relation).intersection(perturb_columns)), 
            num_perturb, 
            replace=False
        )
        candidiate_domains_f =  np.random.choice(
            list(set(candidate_relation).intersection(fixed_columns)), 
            1, 
            replace=False
        )
        for i in range(num_perturb):
            domain1 = candidiate_domains_p[i] # domain being perturbed
            domain2 = candidiate_domains_f[0]
            target_domain = domain1
            fixed_val = row[domain2]
            _order_ = int(domain1 < domain2)
            tmp = sorted([domain1,domain2])
            key_d1, key_d2 = tmp[0], tmp[1]
            key = (key_d1, key_d2)
            
            if _order_ == 1 : 
                choices = entity_coocc_matrix[key][:,fixed_val].toarray().reshape(-1)
            else:
                choices = entity_coocc_matrix[key][fixed_val,:].toarray().reshape(-1)
            choices = np.argwhere(choices == False).reshape(-1)
            new_record[domain1] = int(np.random.choice(choices, 1))    
            
       
    return new_record




def generate_anomalies(
    data_df, 
    num_perturb,
    anomaly_type  
):
    global id_col, anomaly_save_dir
    fixed_columns = ['ConsigneePanjivaID', 'ShipperPanjivaID', 'HSCode']
    perturb_columns = ['PortOfLading', 'PortOfUnlading', 'Carrier', 'ShipmentOrigin', 'ShipmentDestination']
    results = []
    results = Parallel(n_jobs = MP.cpu_count()) (delayed(aux_genrateAnomaly)(
                    row,
                    perturb_columns,
                    fixed_columns,
                    num_perturb,
                    anomaly_type
    ) for _ , row in tqdm(data_df.iterrows()))
    
    
    
    df = pd.DataFrame(columns=list( data_df.columns))
    for  r in results:
        df = df.append(r,ignore_index=True)
    df.to_csv(os.path.join(anomaly_save_dir,'anomalies_{}_{}.csv'.format(anomaly_type, num_perturb)), index=None)
    return

# ----------------------
parser = argparse.ArgumentParser(description='Generate anomalies')
parser.add_argument('--dir', type = str, help='Which dataset ? us__import{1,2...}' )
parser.add_argument('--num_perturb', type = int, default = 2, choices=[1,2,3], help='Number of entities to be perturbed. Default is 2')
parser.add_argument('--anomaly_type', type= int, default = 1, choices=[1,2], help='Anomaly generation logic (see code comments)')
parser.print_help()
args = parser.parse_args()

DIR = args.dir
num_perturb = args.num_perturb
anomaly_type = args.anomaly_type
# ----------------------

# DIR = 'us_import1'
# num_perturb = 2
# anomaly_type = 2

setup_config(DIR)
data_df = pd.read_csv( data_df_path, index_col=None)
entity_coocc_matrix = defaultdict()
setup_coOcc_matrix(data_df.copy(deep=True))
anom_base_df = pd.read_csv( test_data_path, index_col=None)

attribute_cols = list(anom_base_df.columns)
attribute_cols.remove(id_col)
anom_base_df = anom_base_df.drop_duplicates(attribute_cols)

generate_anomalies(
    anom_base_df.copy(deep=True),
    num_perturb,
    anomaly_type
)