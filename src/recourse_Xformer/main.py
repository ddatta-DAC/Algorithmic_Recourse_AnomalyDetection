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
        AD_model_emdDim = 32  
    ):
        global base_path, CONFIG_FILE
        self.DIR = DIR
        self.simEntity_fetcherObj = KG_emb_util.node_fetcher(DIR)
        # self.anomDataFetcher_obj  = anomDataFetcher(DIR)
        self.explainer_obj = AD_Explainer.getTrainedModel(DIR)
        self.metapath_list = metapath_list
        self.ad_proc_obj =  AD_processor(DIR)
        self.domains_list = list(domain_dims)
        self.domain_dims = domain_dims
        self.metapath_list = metapath_list
        self.num_NN =  num_NN
        self.AD_model_emdDim = AD_model_emdDim
        print('Initialized recourse_generator')
        return 
    
    def set_AD_obj(self, ad_obj):
        self.ad_proc_obj  = ad_obj
    
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
        idx = self.get_targetDomains(df_record)
       
        # idx = np.where(np.array(xformer_res)[0] < 0.5)[0].tolist()
        target_idx = idx
        metapath_list = self.metapath_list
        record = df_record.iloc[0]  # Convert to pd.Series object
       
        
        target_domains = [self.domains_list[_] for _ in idx]
        target_mp_list = []
        
        for mp in metapath_list:
            if len(set(mp).intersection(set(target_domains))) > 0:
                target_mp_list.append(mp)
         
     
        num_NN = self.num_NN
        candidate_domEnt = OrderedDict({})
        
        for cur_dom in target_domains: 
            candidate_entities = []
            for mp in target_mp_list:
                try:
                    idx = mp.index(cur_dom)
                except:
                    continue
                    
                nbr_domains = []
                if idx > 0 and idx < len(mp)-1:
                    _dom = mp[idx-1]
                    if _dom not in target_domains:
                        nbr_domains.append(_dom)
                    _dom = mp[idx+1]
                    if _dom not in target_domains:
                        nbr_domains.append(_dom)
                elif idx == 0:
                    _dom = mp[idx+1]
                    if _dom not in target_domains:
                        nbr_domains.append(_dom)
                elif idx == len(mp)-1:
                    _dom = mp[idx-1]
                    if _dom not in target_domains:
                        nbr_domains.append(_dom)

                if len(nbr_domains) == 0: 
                    continue

                if len(nbr_domains) == 1:
                    entity_value = record[nbr_domains[0]]
                    rel = list(sorted([cur_dom, nbr_domains[0]]))
                    res = self.simEntity_fetcherObj.find_NN( 
                        head = [nbr_domains[0],entity_value],
                        rel = rel,
                        num_NN = num_NN
                    ) 
                    candidate_entities.extend(res)

                elif len(nbr_domains) > 1:  
                    common_res = set([])
                    for j in  range(len(nbr_domains)):
                        entity_value = record[nbr_domains[j]]
                        rel = list(sorted([cur_dom, nbr_domains[j]]))
                        res = self.simEntity_fetcherObj.find_NN( 
                            head = [nbr_domains[j],entity_value],
                            rel = rel,
                            num_NN = num_NN 
                        )
                        if len(common_res)== 0 :
                            common_res = common_res.union(set(res))
                        else:
                            common_res = common_res.intersection(set(res))
                        
                    candidate_entities.extend(common_res)
            
            # ======================================================
            # No neighbors can be efound ::: Exclude the domain
            # =====================================================
            if len(candidate_entities) > 0:
                candidate_domEnt[cur_dom] = list(set(candidate_entities))
             
        cand_domain_list = list(candidate_domEnt.keys())
        cand_entity_list = [ candidate_domEnt[dom] for dom in cand_domain_list]
        candidate_records = []
        
        for _entityList in product(*cand_entity_list):
            new_record = record.copy(deep=True)
            for _dom, _entity in zip(cand_domain_list,_entityList):
                new_record[_dom] = _entity
            candidate_records.append(new_record)
        
        candidate_records = pd.DataFrame(candidate_records)
        candidate_records['PanjivaRecordID'] = np.arange(1,len(candidate_records)+1)
        
        try:
            scores = self.ad_proc_obj.score_samples_batch(candidate_records.copy(deep=True), model_emb_dim = self.AD_model_emdDim)
        except:
            scores = self.ad_proc_obj.score_samples_batch(candidate_records.copy(deep=True))
        candidate_records['score'] = scores
        candidate_records = candidate_records.sort_values(by=['score'],ascending=False)
        del candidate_records['score']
        candidate_records = candidate_records.reset_index(drop=True)
        return candidate_records.head(num_cf)



# DIR = 'us_import1'
# gen_obj = recourse_generator('us_import1')
# data = anomDataFetcher_obj.fetch_data()
# df1 = data['anomalies_1_2']['data']
# start = timer()
# candidate_recourse_df = gen_obj.generate_recourse(pd.DataFrame([df1.iloc[1045]]).copy(deep=True))
# end = timer()
# print(end - start)
