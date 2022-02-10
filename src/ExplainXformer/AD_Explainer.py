#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
sys.path.append('./../')
import os
from pathlib import Path

from joblib import Parallel, delayed
import multiprocessing as MP
import yaml
import pandas as pd
import torch 
from tqdm import tqdm
from torch import nn

from torch.nn import Module
from torch.optim.lr_scheduler import ChainedScheduler,CosineAnnealingLR,CyclicLR
from torch import LongTensor as LT
from torch import FloatTensor as FT
from torch.nn import functional as F
try:
    import encoder_v1
    from encoder_v1 import Encoder
    import data_fetcher
    from AD_dataset import anomaly_dataset
    import PE
except:
    from . import encoder_v1
    from .encoder_v1 import Encoder
    from . import data_fetcher
    from .AD_dataset import anomaly_dataset
    from . import PE

id_col = 'PanjivaRecordID'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
_rel_path_ = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(_rel_path_,'config.yaml')
# -------------------------

'''
This is the decoder for the explainer 
To be used in stage 2
'''
class adExp_decoder(Module):
    def __init__(
        self,
        entity_emb_dim,
        seq_len,
        density_ffn,
        device
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        
        # for each attribute calculate 
        # Calculate the probability
        # Assume entity_emb_dim == base embedding dimension
        _modules_ = []
        for _ in range(seq_len):
            _modules_.append(nn.Bilinear(entity_emb_dim, entity_emb_dim*2, entity_emb_dim))
        self.biLinearLayer = nn.ModuleList(_modules_)
        self.PE = PE.PositionalEncoding(entity_emb_dim)
        self.PE.to(self.device)
        # Assume that entity_emb_dim == base embedding dimension
        density_fcn_inp_dim = entity_emb_dim + entity_emb_dim
        
        self.likelihood_fcn = []
        for _ in range(seq_len):
            _layers = []
            inp_dim = density_fcn_inp_dim
            for idx in range(len(density_ffn)):
                op_dim = density_ffn[idx]
                _layers.append(nn.Linear(inp_dim,op_dim))
                inp_dim = op_dim
                if op_dim == density_ffn[-1]: 
                    _layers.append(nn.Sigmoid())
                else:
                    _layers.append(nn.ReLU())            
                               
            self.likelihood_fcn.append(nn.Sequential(*_layers))
        self.likelihood_fcn = nn.ModuleList(self.likelihood_fcn)  
        
       
            
            
    def forward(self, enc_seq_data, entity_emb):
        
        pe_vector = torch.zeros(enc_seq_data.shape).to(self.device)
        pe_vector = self.PE(pe_vector).to(self.device)
        
        token_density = []
        entity_emb.to(self.device)
        enc_seq_data.to(self.device)
        for token_idx in range(self.seq_len):
            _x0 = torch.cat( [entity_emb[:,token_idx,:], pe_vector[:,token_idx,:]], dim=-1)
            _x1 = self.biLinearLayer[token_idx](enc_seq_data[:,token_idx,:], _x0 )
            # Elementwise product
            _x2 = entity_emb[:,token_idx,:] * enc_seq_data[:,token_idx,:]
            _x3 = torch.cat([_x1,_x2],dim=-1)
            token_density.append(self.likelihood_fcn[token_idx](_x3))
        # token_density should have shape [batch, seq_len]
        token_density = torch.cat(token_density, dim =-1)
  
        return token_density
    

class xformer_ADExp_v1(Module):
    def __init__(
        self,
        entity_emb_dim,
        seq_len,
        cardinality,
        encoder_xformer_heads,
        encoder_include_PE,
        encoder_num_xformer_layers,
        pretrained_encoder_model_path,
        density_ffn = [64,16,1],
        device=None
    ):
        super().__init__()
        self.entity_emb_dim = entity_emb_dim
        self.encoder_op_dim = entity_emb_dim
        self.seq_len = seq_len
        self.device = device
        # Encoder 
        self.encoder_obj = Encoder(
            emb_dim = entity_emb_dim,
            xformer_heads = encoder_xformer_heads,
            xformer_model_dims =  encoder_xformer_heads * entity_emb_dim,
            cardinality = cardinality,
            count_numerical_attr=0,
            device = device,
            num_xformer_layers = encoder_num_xformer_layers,
            include_PE = encoder_include_PE
        )
        if str(self.device)=="cpu":
            self.encoder_obj = torch.load(pretrained_encoder_model_path,  map_location=torch.device('cpu'))
        else:
            self.encoder_obj = torch.load(pretrained_encoder_model_path)
        self.encoder_obj.eval()
        
        self.decoder_obj = adExp_decoder(
            entity_emb_dim,
            seq_len,
            density_ffn,
            device
        )
        self.encoder_obj.to(self.device)
        self.decoder_obj.to(self.device)
        
        
    '''
    x has shape [batch, seq_len, entity_idx]
    '''
    def forward(self, seq_data):
        # enc_seq_data is output of the transformer
        # entity_emb is the first layer of embedding
        enc_seq_data, entity_emb = self.encoder_obj(seq_data)
        
        token_density = self.decoder_obj(enc_seq_data, entity_emb )
        return token_density


class ADExp_model_container:
    def __init__(
        self,
        ad_obj,
        model_save_dir = 'model_save_dir',
        LR = 0.001,
        device = None
    ):
        self.device = device
        self.ad_obj = ad_obj
        self.ad_obj.to(self.device)
        # print('[ADExp_model_container]', self.device)
        # print('[ADExp_model_container  ad_obj ]', self.ad_obj.device)
        self.signature = 'ad_{}'.format(self.ad_obj.encoder_obj.emb_dim)
        param_list = self.ad_obj.decoder_obj.parameters()
        self.opt = torch.optim.Adam(param_list, LR)
        self.model_save_dir = model_save_dir
        
        # scheduler1 = CosineAnnealingLR(
        #     self.opt, 
        #     T_max=10,
        #     eta_min=0, 
        #     last_epoch=120
        # )
        # scheduler2 = CyclicLR(
        #     self.opt, 
        #     base_lr=0.001, 
        #     max_lr=0.1,
        #     step_size_up=10,
        #     mode="exp_range",
        #     gamma=0.85)
        # self.scheduler = ChainedScheduler([scheduler1, scheduler2])

        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, 
            T_max = 10,
            eta_min = 0.0001,
            verbose = False
        )
        return
    
    def anomaly_loss_fn(
        self,
        op_density,
        true_labels
    ):
        # Binary cross entropy
        loss_val = F.binary_cross_entropy(op_density, true_labels, reduction='none')
        token_loss = torch.mean(loss_val, dim =-1, keepdim=False)
        seq_label = torch.min(true_labels, dim=-1)[0] 
        return token_loss
    
    def save_model(self):
        Path(self.model_save_dir).mkdir( exist_ok=True, parents=True)
        fpath = os.path.join(self.model_save_dir, 'adExp_decoder_{}.pth'.format(self.signature))
        torch.save(self.ad_obj, fpath) 
    
    def load_model(self):
        fpath = os.path.join(self.model_save_dir, 'adExp_decoder_{}.pth'.format(self.signature))
        if str(self.device)=="cpu":
            self.ad_obj = torch.load(fpath,  map_location=torch.device('cpu'))
        else:
            self.ad_obj = torch.load(fpath)
        self.ad_obj.eval() 
        self.ad_obj.to(self.device)
        return self.ad_obj
    
    def _train_model_(
        self, 
        dataloader_obj,
        num_epochs = 10
    ):
        epoch_loss = []
        for epoch_idx in tqdm(range(num_epochs)):
            cur_epoch_loss = []
            for i,batch_data in enumerate(dataloader_obj):
                self.opt.zero_grad()
                inp = batch_data[0].to(self.device).float()
                labels = batch_data[1].to(self.device).float()
                # calculate loss
                token_prob = self.ad_obj(inp)
                token_loss  = self.anomaly_loss_fn(token_prob, labels)
                token_loss = torch.mean(token_loss)
                
                loss =  token_loss
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                b_loss = loss.cpu().data.numpy()
                cur_epoch_loss.append(b_loss)
                if i%150 == 0:
                    print('[Batch {}], loss: {:.4f}'.format(i+1, b_loss))
            epoch_loss.append(np.mean(cur_epoch_loss))
            print('[Epoch {}], loss: {:.4f}'.format(epoch_idx+1, np.mean(cur_epoch_loss)))
        return
    
    '''
    This is the external facing function
    Input: 
        Dataframe of records
    '''
    def predict_entityProb(
        self, 
        seq_df: pd.DataFrame,
        adjust_id = True # Add 1 to compensate for design
    ): 
        self.ad_obj.eval()
        self.ad_obj.encoder_obj.eval()
        self.ad_obj.decoder_obj.eval()
        # print('[predict_entityProb]',self.device)
        global id_col
        # Adjust id
        try: 
            del seq_df[id_col]
        except:
            pass
        if adjust_id :
            seq_x = seq_df.values + 1
        else:
            seq_x = seq_df.values
        batch_size = 512
        num_batches = len(seq_df)//batch_size+1
        results = []
        with torch.no_grad():
            for b_idx in range(num_batches): 
                _x = seq_x[b_idx*batch_size:(b_idx+1)*batch_size]
                _x1 = LT(_x).to(self.device)
                _y = self.ad_obj(_x1)
                
                results.extend(_y.cpu().data.numpy())
        return results
# -----------------------
# read in config
# -----------------------
def _getConfig_():
    global CONFIG_FILE
    with open(CONFIG_FILE,'rb') as fh:
        config = yaml.safe_load(fh)
    return config


'''
This is to train the anomaly explainer part.
It assumes that the pre-trained encoder is already available.
'''
def train_model(subDIR):
    global id_col, _rel_path_
    config = _getConfig_()
    
    record_data = data_fetcher.get_training_set_data(subDIR)
    del record_data[id_col]
    
    data = record_data
    '''
    Increase the id values by 1 to compensate for the MASK having id=0
    '''
    data = data.values + 1
    domain_dims_df = data_fetcher.get_domain_dims(subDIR)
    cardinality = domain_dims_df['dimension'].tolist()
    domain_dims = domain_dims_df['dimension'].tolist()
    '''
    Increase the cardinality by 1 to compensate for the MASK having id=0
    '''
    cardinality = [ _ + 1 for _ in cardinality]
    seq_len = len(domain_dims)
    model_save_dir = config['model_save_dir']
    model_save_dir = os.path.join(_rel_path_, model_save_dir, subDIR)
    
    
    density_fcn_dims = config['stage2_decoder_densityFCN_dims']
    encoder_num_xformer_layers = config['encoder_num_xformer_layers']
    encoder_xformer_heads = config['encoder_xformer_heads']
    base_emb_dim = config['base_emb_dim']
    batch_size = config['stage_2_batch_size']
    train_epochs = config['stage_2_train_epochs']
    num_workers = MP.cpu_count()//2
    
    enc_model_path = os.path.join(model_save_dir, 'encoder_mlm_{}.pth'.format(encoder_num_xformer_layers))
    # -------------------------------------------
    adExp_obj = xformer_ADExp_v1(
        entity_emb_dim = base_emb_dim,
        seq_len = seq_len,
        cardinality = cardinality,
        encoder_xformer_heads = encoder_xformer_heads,
        encoder_include_PE = True,
        encoder_num_xformer_layers = encoder_num_xformer_layers,
        pretrained_encoder_model_path = enc_model_path,
        density_ffn = density_fcn_dims,
        device=DEVICE
    )
    adExp_obj.to(DEVICE)
    
    dataset_obj = anomaly_dataset(
        data,
        domain_dims,
        alpha_max = 0.3
    )

    dataloader_obj = DataLoader(
        dataset_obj, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True
    )

    adExp_container_obj = ADExp_model_container(
        adExp_obj,
        model_save_dir = model_save_dir,
        device=DEVICE
    )

    adExp_container_obj._train_model_(
        dataloader_obj, 
        num_epochs=train_epochs
    )
    adExp_container_obj.save_model()
    return 


# -------------------------------------------------
# fUNCTION TO INSTANTIATE AND LOAD MODEL 
# Read in hyperparams from config file
# Returns :
# <Object> AD_Explainer type
# -------------------------------------------------
def getTrainedModel(DIR):
    global id_col, _rel_path_, DEVICE
    config = _getConfig_()
    
    domain_dims_df = data_fetcher.get_domain_dims(DIR)
    cardinality = domain_dims_df['dimension'].tolist()
    domain_dims = domain_dims_df['dimension'].tolist()
    
    cardinality = [ _ + 1 for _ in cardinality]
    seq_len = len(domain_dims)
    
    model_save_dir = config['model_save_dir']
    model_save_dir = os.path.join(_rel_path_, 'model_save_dir', DIR)
    density_fcn_dims = config['stage2_decoder_densityFCN_dims']
    encoder_num_xformer_layers = config['encoder_num_xformer_layers']
    encoder_xformer_heads = config['encoder_xformer_heads']
    base_emb_dim = config['base_emb_dim']
    enc_model_path = os.path.join(model_save_dir, 'encoder_mlm_{}.pth'.format(encoder_num_xformer_layers))
    # print('[getTrainedModel] DEVICE ::', DEVICE)
    
    adExp_obj = xformer_ADExp_v1(
        entity_emb_dim = base_emb_dim,
        seq_len = seq_len,
        cardinality = cardinality,
        encoder_xformer_heads = encoder_xformer_heads,
        encoder_include_PE = True,
        encoder_num_xformer_layers = encoder_num_xformer_layers,
        pretrained_encoder_model_path = enc_model_path,
        density_ffn = density_fcn_dims,
        device=DEVICE
    )
    adExp_obj.to(DEVICE)
    # print('[getTrainedModel adExp_obj device::]', adExp_obj.device)
    adExp_container_obj = ADExp_model_container(
        adExp_obj,
        model_save_dir = model_save_dir,
        device=DEVICE
    )
    adExp_container_obj.load_model()
    return adExp_container_obj


# # ------------------------------------------------- # 
# if __name__ == "__main__" :
#     parser = argparse.ArgumentParser(description='Pretrain the encoder through MLM like objective')
#     parser.add_argument(
#         '--dir', 
#         type=str,
#         choices = ['us_import1', 'us_import2', 'us_import3'],
#         help='Which data subset to use ?'
#     )

#     args = parser.parse_args()
#     subdir = args.dir
#     train_model(subDIR=subdir)
    
    




