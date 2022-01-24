#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

sys.path.append('./../..')
sys.path.append('./..')
import yaml
import multiprocessing as MP
from collections import OrderedDict    
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from pandarallel import pandarallel
import re
pandarallel.initialize()
import yaml
from collections import Counter
import pickle
sys.path.append('./..')
sys.path.append('./../..')


DATA_SOURCE = './../../Data_Raw'
CONFIG = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = './config.yaml'
id_col = 'PanjivaRecordID'
use_cols = None
freq_bound = None
column_value_filters = None


save_dir = None
NUMERIC_COLUMNS = None
DISCRETE_COLUMNS = None

def set_up_config(_DIR = None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global freq_bound
   
    global save_dir
    global column_value_filters
    
    global DATA_SOURCE
    global DIR_LOC
    global NUMERIC_COLUMNS
    global id_col
    global DISCRETE_COLUMNS
    
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']

    DIR_LOC = re.sub('[0-9]', '', DIR)
    DATA_SOURCE = os.path.join(DATA_SOURCE, DIR_LOC)
    save_dir =  CONFIG['save_dir']
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    column_value_filters = CONFIG[DIR]['column_value_filters']
    if column_value_filters is None:
        column_value_filters ={}
        
    NUMERIC_COLUMNS = CONFIG[DIR]['numeric_columns']
    if NUMERIC_COLUMNS is None:
        NUMERIC_COLUMNS = []  
    
    _cols = list(use_cols)
    _cols.remove(id_col)
    for nc in NUMERIC_COLUMNS:
        try:
            _cols.remove(nc)
        except:
            pass
        
    DISCRETE_COLUMNS = list(sorted(_cols))
    return 




def get_regex(_type='train'):
    global DIR
    
    if DIR == 'us_import1':
        if _type == 'train':
            return '.*0[1]_2015.csv'
        if _type == 'test':
            return '.*02_2015.csv'

    elif DIR == 'us_import2':
        if _type == 'train':
            return '.*0[3]_2016.csv'
        if _type == 'test':
            return '.*04_2016.csv'
    
    elif DIR == 'us_import3':
        if _type == 'train':
            return '.*0[5]_2016.csv'
        if _type == 'test':
            return '.*06_2016.csv'
    elif DIR == 'ecuador_export':   
        if _type == 'train':
            return '.*_2015.csv'
        if _type == 'test':
            return '.*0[1-8]_2016.csv'
    elif DIR == 'colombia_export':
        if _type == 'train':
            return '.*0[4-5]_2016.csv'
        if _type == 'test':
            return '.*0[6-7]_2016.csv'
def get_files(DIR, _type='all'):
    global DATA_SOURCE
    print(DATA_SOURCE)
    data_dir = DATA_SOURCE
    regex = get_regex(_type)
    c = glob.glob(os.path.join(data_dir, '*'))
   
    def glob_re(pattern, strings):
        return filter(re.compile(pattern).match, strings)
    
    files = sorted([_ for _ in glob_re(regex, c)])
    print('DIR ::', DIR, ' Type ::', _type, 'Files count::', len(files))
    return files



def remove_low_frequency_values(df):
    global id_col
    global freq_bound
    global NUMERIC_COLUMNS
    global DISCRETE_COLUMNS
    freq_column_value_filters = {}
    
    feature_cols = list(DISCRETE_COLUMNS)
    print ('feature columns ::' , feature_cols)
    # ----
    # figure out which entities are to be removed
    # ----
    
    counter_df = pd.DataFrame(columns=['domain', 'count'])
    
    for c in feature_cols:
        count = len(set(df[c]))
        counter_df = counter_df.append({
            'domain': c, 'count': count
        }, ignore_index=True)
        
        z = np.percentile(
            list(Counter(df[c]).values()), 5)
        print(c, count, z)

    counter_df = counter_df.sort_values(by=['count'], ascending=False)
    
    
    for c in list(counter_df['domain']):
        
        values = list(df[c])
        freq_column_value_filters[c] = []
        obj_counter = Counter(values)

        for _item, _count in obj_counter.items():
            if _count < freq_bound:
                freq_column_value_filters[c].append(_item)

    print('Removing :: ')
    for c, _items in freq_column_value_filters.items():
        print('column : ', c, 'count', len(_items))

    print(' DF length : ', len(df))
    for col, val in freq_column_value_filters.items():
        df = df.loc[~df[col].isin(val)]

    print(' DF length : ', len(df))
    return df

def apply_value_filters(list_df):
    global column_value_filters

    if type(column_value_filters) != bool:
        list_processed_df = []
        for df in list_df:
            for col, val in column_value_filters.items():
                df = df.loc[~df[col].isin(val)]
            list_processed_df.append(df)
        return list_processed_df
    return list_df

'''
4 digit hs code
'''
def HSCode_cleanup_aux(val):
    val = str(val)
    val = val.split(';')
    _list =['9401','9403','9201','9614','9202','9302', '9304', '6602','8201','9207','9504', '9205', '9206', '9209','9202']
    _list =['9401', '9403','9201','9202', '9205','9206', '9207', '9209',  '9302', '9304' ]
    val = str(val[0])
    val = val.replace('.','')
    val = str(val[:6])
    
    if val[:2] == '44': 
        return val[:4]
    
    elif val[:4] in _list: 
        return val 
    return val[:4]

def HSCode_cleanup(list_df):
    new_list = []
    for _df in list_df :
        _df['HSCode'] = _df['HSCode'].parallel_apply(HSCode_cleanup_aux)
        _df = _df.dropna()
        print(' In HSCode clean up , length of dataframe ', len(_df))
        new_list.append(_df)
    return new_list


def clean_train_data():
    global DIR
    global CONFIG
    global DIR_LOC
    global use_cols
    
    files = get_files(DIR, 'train')
    print('Columns read ', use_cols)
    list_df = [pd.read_csv(_file, usecols=use_cols, low_memory=False) for _file in files]
    list_df = [_.dropna() for _  in list_df]
    if 'HSCode' in use_cols:
        list_df = HSCode_cleanup(list_df)
    list_df_1 = apply_value_filters(list_df)
    master_df = None
    
    for df in list_df_1:
        if master_df is None:
            master_df = pd.DataFrame(df, copy=True)
        else:
            master_df = master_df.append(
                df,
                ignore_index=True
            )
    master_df = remove_low_frequency_values(master_df)
    return master_df

def order_cols(df):
    global NUMERIC_COLUMNS
    global DISCRETE_COLUMNS
    global id_col
    print('>>>', NUMERIC_COLUMNS)
    ord_cols = [id_col] + DISCRETE_COLUMNS + NUMERIC_COLUMNS
    return df[ord_cols]

def convert_to_ids(
        df,
        save_dir
):
    global id_col
    global freq_bound
    global DISCRETE_COLUMNS

    feature_columns = list(sorted(DISCRETE_COLUMNS))
    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in feature_columns:
        vals = list(set(df[col]))
        vals = list(sorted(vals))

        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }
        print(' > ',col ,':', len(id2val_dict))

        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict


        # Replace
        df[col] = df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )

        dict_DomainDims[col] = len(id2val_dict)

    print(' Feature columns :: ', feature_columns)
    print(' dict_DomainDims ', dict_DomainDims)
    # -------------
    # Save the domain dimensions
    # -------------

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            dict_DomainDims,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    file = 'col_val2id_dict.pkl'
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            col_val2id_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    return df, col_val2id_dict

def replace_attr_with_id(row, attr, val2id_dict):
    val = row[attr]
    if val not in val2id_dict.keys():
        print(attr, val)
        return None
    else:
        return val2id_dict[val]


def setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
):
    global id_col
    global save_dir
    global DISCRETE_COLUMNS
    test_df = test_df.dropna()

    # Replace with None if ids are not in train_set
    feature_cols = list(DISCRETE_COLUMNS)
   
    for col in feature_cols:
        valid_items = list(col_val2id_dict[col].keys())
        test_df = test_df.loc[test_df[col].isin(valid_items)]

    # First convert to to ids
    for col in feature_cols:
        val2id_dict = col_val2id_dict[col]
        test_df[col] = test_df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )
    test_df = test_df.dropna()
    test_df = test_df.drop_duplicates()
    test_df = order_cols(test_df)

    print(' Length of testing data', len(test_df))
    test_df = order_cols(test_df)
    return test_df

def create_train_test_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    global CONFIG
    global DIR_LOC
    global NUMERIC_COLUMNS
    
    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    
    train_raw_df_file = os.path.join(save_dir, 'train_data_raw.csv')
    test_raw_df_file = os.path.join(save_dir, 'test_data_raw.csv')
    
    column_valuesId_dict_file = 'column_valuesId_dict.pkl'
    column_valuesId_dict_path = os.path.join(save_dir, column_valuesId_dict_file)
    
    # --- Later on - remove using the saved file ---- #
    if os.path.exists(train_df_file) and os.path.exists(test_df_file) and False:
        train_df = pd.read_csv(train_df_file)
        test_df = pd.read_csv(test_df_file)
        with open(column_valuesId_dict_path, 'rb') as fh:
            col_val2id_dict = pickle.load(fh)

        return train_df, test_df, col_val2id_dict

    train_df = clean_train_data()
    train_df = order_cols(train_df)
    train_raw_df = train_df.copy(deep=True)
    
    train_df, col_val2id_dict = convert_to_ids(
        train_df,
        save_dir
    )
    print('Length of train data ', len(train_df))
    train_df = order_cols(train_df)

    '''
         test data preprocessing
    '''
    # combine test data into 1 file :
    test_files = get_files(DIR, 'test')
    list_test_df = [
        pd.read_csv(_file, low_memory=False, usecols=use_cols)
        for _file in test_files
    ]
    list_test_df = [ _.dropna() for _ in list_test_df]
    if 'HSCode' in use_cols:
        list_test_df = HSCode_cleanup(list_test_df)

    test_df = None
    
    for _df in list_test_df:
        if test_df is None:
            test_df = _df
        else:
            test_df = test_df.append(_df)

    print('size of  Test set ', len(test_df))
    test_raw_df = test_df.copy(deep=True)
    test_df = setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
    )
    train_raw_df.to_csv(train_raw_df_file, index=False)
    test_raw_df.to_csv(test_raw_df_file, index=False)
    
    test_df.to_csv(test_df_file, index=False)
    train_df.to_csv(train_df_file, index=False)
    
    # Save data_dimensions.csv ('column', dimension')
    dim_df = pd.DataFrame(columns=['column','dimension'])
    for col in DISCRETE_COLUMNS:
        _count = len(col_val2id_dict[col])
        dim_df = dim_df.append({'column':col, 'dimension': _count},ignore_index=True
        )
        
    dim_df.to_csv(os.path.join(save_dir, 'data_dimensions.csv'), index=False)
        
    # -----------------------
    # Save col_val2id_dict
    # -----------------------
    with open(column_valuesId_dict_path, 'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df, test_df, col_val2id_dict


DIR = 'colombia_export'
set_up_config(DIR)
train_df, test_df, col_val2id_dict = create_train_test_sets()




