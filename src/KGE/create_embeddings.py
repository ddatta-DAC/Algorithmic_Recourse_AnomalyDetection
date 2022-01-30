#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import sys
import yaml
import numpy as np
sys.path.append('./..')
import pandas as pd
import pickle
import stellargraph
from pathlib import Path
from stellargraph import StellarGraph
from stellargraph.mapper import *
from stellargraph.layer import DistMult,regularizers
from stellargraph import datasets, utils
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model
from collections import OrderedDict
# from IPython.display import HTML
import argparse

DATA_DIR = None
DIR = None
id_col = 'PanjivaRecordID'
CONFIG_FILE = 'config.yaml'
KG_emb_dim = 64
kg_train_numNegSamples = 10
domain_dims = None
KG_emb_save_dir = None
KG_train_batchsize = 512
KG_train_patience = 10
KG_train_epochs = 100
metapath_file = None
# ----------------------------------------------

def setup_config(_DIR):
    global DATA_DIR, DIR, KG_emb_dim, kg_train_num_neg_samples, domain_dims
    global KG_emb_save_dir
    global KG_train_epochs
    global KG_train_patience
    global KG_train_batchsize
    global metapath_file
    DIR = _DIR
    with open(CONFIG_FILE,'r') as fh:
        config = yaml.safe_load(fh)
    DATA_DIR = config['DATA_DIR']
    KG_emb_dim = config['embedding_dimension']
    kg_train_num_neg_samples  = config ['kg_train_num_neg_samples']
    domain_dims = pd.read_csv( os.path.join(DATA_DIR, _DIR, 'data_dimensions.csv'), index_col= None)
    KG_emb_save_dir = os.path.join(config['kg_emb_save_dir'], _DIR)
    KG_train_epochs = config['kg_train_epochs']
    KG_train_patience = config['kg_train_epochs']
    KG_train_batchsize = config['kg_train_batch_size']
    Path(os.path.join(KG_emb_save_dir)).mkdir(exist_ok=True, parents=True)
    
    metapath_file = config['metapath_file'][DIR]
    
    return 



def get_graphTrainingData():
    global DATA_DIR, DIR, id_col
    global metapath_file
    
    with open(metapath_file,'r') as fh:
        mp = fh.readlines()
    mp = [ _.strip('\n') for _ in mp]
    mp = [ _.split(',') for _ in mp]
    # Create data
    df_data_file = os.path.join(DATA_DIR, DIR, 'train_data.csv')
    df = pd.read_csv(df_data_file, index_col = None)

    try:
        del df[id_col]
    except:
        pass

    df_named = df.copy(deep=True)
    columns = list(df_named.columns)

    for col in columns:
        df_named[col] = df_named[col].apply(lambda x : col+'_'+ str(x))

    relationships = set()
    for _mp in mp:
        for i in range(len(_mp)-1):
            relationships = relationships.union( [(_mp[i],_mp[i+1])])


    edge_data_wLabels = pd.DataFrame(columns=['source','label', 'target'])
    for relation in relationships:
        _columns = list(relation)
        df_tmp = df_named[_columns]
        df_tmp = df_tmp.rename(columns = {_columns[0]:'source', _columns[1]:'target'})
        df_tmp.loc[:,'label'] = '_'.join( sorted([_columns[0], _columns[1]]))
        df_w = df_tmp.groupby(['source', 'target']).size().reset_index(name='weight')
        df_tmp = df_tmp.merge(df_w, on =['source','target'], how='inner')
        edge_data_wLabels = edge_data_wLabels.append(df_tmp,ignore_index=True)
        edge_data_wLabels = edge_data_wLabels.drop_duplicates(subset =['source', 'target']).reset_index(drop=True)
    node_dict = {}
    for column in list(df_named.columns):
        nodes = list(sorted(set(df_named[column])))
        node_dict[column] = pd.DataFrame( None, index= nodes)
    return node_dict, edge_data_wLabels

'''
Main function to train the model
'''

def train_model():
    global KG_train_epochs
    global KG_train_patience
    global KG_emb_dim
    global KG_emb_save_dir
    global KG_train_batchsize
    
    node_dict, edge_data_wLabels = get_graphTrainingData()
    
    sg = StellarGraph(
        node_dict, 
        edge_data_wLabels,
        edge_type_column="label"
    )

    _gen = KGTripleGenerator(
        sg,
        batch_size=KG_train_batchsize  # ~10 batches per epoch
    )

    distmult = DistMult(
        _gen,
        embedding_dimension=KG_emb_dim,
        embeddings_regularizer=regularizers.l2(1e-7),
    )

    _inp, _out = distmult.in_out_tensors()
    _model = Model(
        inputs=_inp, 
        outputs=_out
    )
    _model.compile(
        optimizer=optimizers.Adam(lr=0.001),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[metrics.BinaryAccuracy(threshold=0.0)],
    )


    _train_gen = _gen.flow(
        edge_data_wLabels, 
        negative_samples=kg_train_num_neg_samples, 
        shuffle=True
    )

    earlyStop_callback = callbacks.EarlyStopping(
        monitor="binary_accuracy", 
        patience=KG_train_patience
    )

    _model.fit(
        _train_gen,
        epochs=KG_train_epochs,
        callbacks = [earlyStop_callback],
        verbose=1,
    )
    '''
    Obtain embeddings of node and relations
    '''

    node_embeddings = distmult.embeddings()[0]
    relation_embeddings = distmult.embeddings()[1]
    domains_list = list(domain_dims['column'].values)
    node_emb_dict = OrderedDict( { _ : OrderedDict({}) for _ in domains_list }  )
    relation_emb_dict = {}

    for entity,emb in zip(list(sg.nodes()),node_embeddings):
        dom, e = entity.split('_')
        e = int(e)
        node_emb_dict[dom][e] = emb


    for idx,e_type in enumerate(sg.edge_types,0):
        relation_emb_dict[e_type] = relation_embeddings[idx]

    '''
    Save KGE embeddings
    '''
    nodeEmb_saveFile_name = 'KG_DM_nodeEmb_{}.pkl'.format(KG_emb_dim)
    edgeEmb_saveFile_name = 'KG_DM_edgeEmb_{}.pkl'.format(KG_emb_dim)

    with open(os.path.join(KG_emb_save_dir, nodeEmb_saveFile_name), 'wb') as fh:
        pickle.dump(node_emb_dict, fh, pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(KG_emb_save_dir, edgeEmb_saveFile_name), 'wb') as fh:
        pickle.dump(relation_emb_dict, fh, pickle.HIGHEST_PROTOCOL)
    print('[INFO]  Saved embeddings at :',  os.path.join(KG_emb_save_dir, edgeEmb_saveFile_name), os.path.join(KG_emb_save_dir, edgeEmb_saveFile_name))
    return

# -----------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate anomalies')
parser.add_argument('--dir', type = str, help='Which dataset ? us__import{1,2...}' )
args = parser.parse_args()
DIR = args.dir

setup_config(DIR)
train_model()


