#!/usr/bin/env python
# coding: utf-8

import os 
import sys
sys.path.append('./..')
sys.path.append('./../ExplainXformer')
import pandas as pd
import numpy as np
import yaml
import multiprocessing as MP
from AD_model.AD_processor import AD_processor
# from recourseExpDataFetcher import anomDataFetcher
from itertools import combinations 
from collections import defaultdict, OrderedDict
from KGE import KG_emb_util
from ExplainXformer import  AD_Explainer
from itertools import product
from timeit import default_timer as timer

# base_path =  os.path.dirname( os.path.realpath(__file__) )
base_path = '.'
id_col = 'PanjivaRecordID'
CONFIG_FILE = os.path.join(base_path, 'config.yaml')

def get_orderedDomain_list(DIR):
    domain_dims = None
    dims_df = pd.read_csv('./../../GeneratedData/{}/data_dimensions.csv'.format(DIR),index_col=None) 
    domains_list = dims_df['column'].values.tolist()
    return domains_list



def read_metapaths():
    global base_path
    with open(os.path.join(base_path, 'metapaths.txt'), 'r') as fh:
        metapath_list = fh.readlines()
    metapath_list = [ _.strip('\n') for _ in metapath_list]
    metapath_list = [ _.split(',') for _ in metapath_list]
    return metapath_list

class recourse_generator:
    def __init__(
        self, 
        DIR,
        domain_dims,
        metapath_list,
        num_NN,
        AD_model_emdDim
    ):
        global base_path, CONFIG_FILE
        self.DIR = DIR
        self.simEntity_fetcherObj = KG_emb_util.node_fetcher(DIR)
        # self.anomDataFetcher_obj  = anomDataFetcher(DIR)
        self.explainer_obj = AD_Explainer.getTrainedModel(DIR)
        self.metapath_list = metapath_list
        self.ad_proc_obj =  AD_processor(DIR)
        self.domains_list = list(domain_dims.keys())
        self.domain_dims = domain_dims
        self.metapath_list = metapath_list
        self.num_NN =  num_NN
        self.AD_model_emdDim = AD_model_emdDim
        return 
    
    def get_targetDomains(
        self,
        df_record
    ):
        xformer_res = self.explainer_obj.predict_entityProb(
            df_record.copy(deep=True)
        )
        idx = np.where(np.array(xformer_res)[0]<0.5)[0].tolist()
    
        # If idx is empty select one entity whose score is lowest
        if len(idx) == 0:
            idx = np.array([np.argmin(np.array(xformer_res)[0])]).reshape(-1).tolist()
        return idx
    
    def generate_recourse(
        self, 
        df_record,
        num_cf = 100
    ):
        
        xformer_res = self.explainer_obj.predict_entityProb(
            df_record.copy(deep=True)
        )
        target_idx = self.get_targetDomains(df_record)
        record = df_record.iloc[0]  # Convert to pd.Series object
        
        target_domains = [self.domains_list[_] for _ in target_idx]
        # ====================
        # Replace the entities of the target indices with random         
        # ====================
        num_NN = self.num_NN
        candidate_domEnt = OrderedDict({})
        
        for cur_dom in target_domains:
            if self.domain_dims[cur_dom] < num_cf:
                replace_flag = True
            else:
                replace_flag = False
            candidate_entities = np.random.choice(np.arange(self.domain_dims[cur_dom]), size=num_cf, replace=replace_flag) 
            candidate_domEnt[cur_dom] = candidate_entities.tolist()
            
        candidate_records = []
        for i in range(num_cf):
            new_record = record.copy(deep=True)
            for _dom in target_domains:
                new_record[_dom] = candidate_domEnt[_dom][i]
            candidate_records.append(new_record)
        
        candidate_records = pd.DataFrame(candidate_records)
        candidate_records['PanjivaRecordID'] = np.arange(1,len(candidate_records)+1)
        
        candidate_records = candidate_records.reset_index(drop=True)
        return candidate_records





# DIR = 'us_import1'
# gen_obj = recourse_generator('us_import1')
# data = anomDataFetcher_obj.fetch_data()
# df1 = data['anomalies_1_2']['data']
# start = timer()
# candidate_recourse_df = gen_obj.generate_recourse(pd.DataFrame([df1.iloc[1045]]).copy(deep=True))
# end = timer()
# print(end - start)
