#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import os
import sys
sys.path.append('./..')
import pandas as pd
import yaml
from pathlib import Path
import pickle
from collections import OrderedDict
import faiss  
from typing import *
from joblib import Parallel, delayed
import multiprocessing as MP
# -------------------------
file_path = os.path.realpath(__file__)
base_path =  os.path.dirname(file_path)
print(base_path)

CONFIG_FILE = os.path.join( base_path ,'config.yaml')
print(CONFIG_FILE)


# --------------------------
class node_fetcher:
    def __init__(self, DIR, embedding_dim = None):
        self.DIR = DIR
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.load_embeddings()
        
        return 
    
    def load_embeddings(self):
        global CONFIG_FILE
        global base_path
        with open(CONFIG_FILE, 'r') as fh:
            config = yaml.safe_load(fh)
        if self.embedding_dim  is None:
            self.embedding_dim  = config['embedding_dimension']
        emb_save_dir = os.path.join(config['kg_emb_save_dir'], self.DIR)
        with open(os.path.join(base_path, emb_save_dir, 'KG_DM_nodeEmb_{}.pkl'.format(self.embedding_dim)), 'rb') as fh:
            self.node_emb_dict = pickle.load(fh)
        
        with open(os.path.join(base_path, emb_save_dir, 'KG_DM_edgeEmb_{}.pkl'.format(self.embedding_dim)), 'rb') as fh:
            self.edge_emb_dict = pickle.load(fh)
            
        
        # -------------------------------------------
        # Create mapping dict 
        # ------------------------------------------
        idx = 0
        self.serialId_2_entityID = OrderedDict({})
        node_emb_array = []
        for domain, emb_dict in self.node_emb_dict.items():
            for _id,_emb in emb_dict.items(): 
                entity_key = domain + '_' + str(_id)
                self.serialId_2_entityID[idx] = entity_key
                idx += 1
                node_emb_array.append(_emb)
        self.node_emb_array = np.array(node_emb_array).astype(np.float32)
        
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim) 
        self.faiss_index.add(self.node_emb_array)  
     
        return
    
                               
    
    '''
    Distmult calculates similarity as inner product <head, rel, tail>
    also note DistMult is symmetric
    head: [ domain, entity_id ]
    rel : [start node domain, end node domain]
    '''
    def find_NN(
        self, 
        head: List, 
        rel: List, 
        num_NN:int = 10,
        max_iter = 10
    ):
        domain, entity_id = head[0], head[1]
        
        head_emb = self.node_emb_dict[domain][entity_id]
        edge_type = '_'.join(sorted(rel))
        edge_emb = self.edge_emb_dict[edge_type]
        
        
        y = np.zeros([self.embedding_dim,self.embedding_dim])
        np.fill_diagonal( y, edge_emb)
        
        x = np.inner(head_emb.reshape([1,-1]),y).astype(np.float32)
        
        rel.remove(domain)
        tail_domain = rel[0]
       
     
        # Use an expanding search approach --- nearest neighbors may not be of desired domain type
        k = num_NN * 100
        res = []
        
        def aux(_idx,_dist):
            d_entity = self.serialId_2_entityID[_idx] # d_entity has form <domain>_<entity_id>
            # Validate 
            cand_dom = d_entity.split('_')[0]
            if cand_dom != tail_domain: 
                return None
            return (d_entity.split('_')[1],_dist)
        
        iter = 0
        while True and iter <= max_iter:
            iter+=1
            # Find the nearest neighbors
            
            dist , fetched_idx = self.faiss_index.search(x, k) 
           
            _fetched_idx = fetched_idx[0].tolist()
            _dist = dist[0].tolist()
            dist = []
            fetched_idx = []
            for _fi,_dist in zip(_fetched_idx, _dist):
                if _fi > 0 :
                    dist.append(_dist)
                    fetched_idx.append(_fi)
            
            if len(fetched_idx)>0:
                res = Parallel(
                    n_jobs = MP.cpu_count(),prefer="threads"
                )(delayed(aux) ( _idx, _dist,) for _idx, _dist in zip(fetched_idx, dist ))
                res = [ _ for _ in res if _ is not None]
            else:
                res =[]
            if len(res)>= num_NN:
                res = res[:num_NN]
                break
            else:
                k = int(k * 2.5) 
            
        try:
            res = sorted(res, key=lambda x: x[1], reverse=True )
            NN = [int(_[0]) for _ in res]  
            return NN
        except:
            return []
                    
        





'''
Example usage
'''
# obj = node_fetcher('us_import1')
# obj.find_NN( 
#         head =[ 'Carrier',10],
#         rel = ['Carrier', 'HSCode']
# )



