#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append('./..')
import argparse
import os
import math
import torch
from torch.nn import Module
from tqdm import tqdm
from torch import nn
from typing import *
from torch.nn import functional as F
from torch import LongTensor as LT
import numpy as np
from torch import FloatTensor as FT
from torch import Tensor as T
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from data_masker import get_masked_tokens
except:
    from .data_masker import get_masked_tokens

try:
    import  data_fetcher
except:
    from . import  data_fetcher

try:
    from decoder_MLM_v1 import decoder_MLM_layer
except:
    from .decoder_MLM_v1 import decoder_MLM_layer
try:
    from encoder_v1 import Encoder 
except:
    from .encoder_v1 import Encoder 
# ===========================================================

_rel_path_ = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(_rel_path_,'config.yaml')
id_col = 'PanjivaRecordID'
print(DEVICE)
# -----------------------
# read in config
# -----------------------
def _getConfig_():
    global CONFIG_FILE
    with open(CONFIG_FILE,'rb') as fh:
        config = yaml.safe_load(fh)
    return config

# ============================================================
class MLM_DataSet(torch.utils.data.Dataset):
    def __init__(self, data, cardinality):
        super(MLM_DataSet).__init__()
        
        data, labels, token_weights = get_masked_tokens(
            data, 
            cardinality
        )
        self.data = data
        self.labels = labels
        self.token_weights = np.array(token_weights, dtype=float)
        return 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        label = self.labels[idx]
        token_weight = self.token_weights[idx]
        return record, label, token_weight
    

class tabXformerMLM_obj:
    def __init__(
        self,
        emb_dim:int,
        encoder_xformer_heads:int,
        count_numerical_attr:int,
        cardinality: List,
        encoder_num_xformer_layers:int,
        encoder_include_PE: bool,
        decoder_ffn_layer_dims: List,
        model_save_dir,
        device
    ):
        
        self.device = device
        self.encoder_obj = Encoder(
            emb_dim = emb_dim,
            xformer_heads = encoder_xformer_heads,
            xformer_model_dims =  encoder_xformer_heads * emb_dim,
            cardinality = cardinality,
            count_numerical_attr=0,
            device = device,
            num_xformer_layers = encoder_num_xformer_layers,
            include_PE = encoder_include_PE
        )
        self.model_save_dir = model_save_dir 
        Path(self.model_save_dir).mkdir(exist_ok=True, parents=True)
        self.numXformerLayers = encoder_num_xformer_layers
        self.encoder_obj.to(self.device)
        self.decoder_obj = decoder_MLM_layer(
            emb_dim = emb_dim,
            ffn_layer_dims = decoder_ffn_layer_dims,
            attribute_cardinality=cardinality,
            device = device
        )
        self.decoder_obj.to(self.device)
        self.cardinality = cardinality
        self.CE_loss = nn.ModuleList([ nn.CrossEntropyLoss() for _ in cardinality ])
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.encoder_obj.parameters(), 'lr': 5e-4},
                {'params': self.decoder_obj.parameters(), 'lr': 5e-4}
            ]
        )
        return 
    
    def save_model(self):
        Path(self.model_save_dir).mkdir( exist_ok=True, parents=True)
        fpath = os.path.join(self.model_save_dir, 'encoder_mlm_{}.pth'.format(self.numXformerLayers))
        torch.save(self.encoder_obj, fpath)
        fpath = os.path.join(self.model_save_dir, 'decoder_mlm_{}.pth'.format(self.numXformerLayers))
        torch.save(self.decoder_obj, fpath)
        
    # ======
    # MLM loss function
    # ======
    def mlm_loss(
        self,
        enc_seq_data,
        token_pred,
        labels,
        token_weights
    ):
        cardinality = self.cardinality
        seq_len = enc_seq_data.shape[1]
        
        # Calculate croiss entropy loss
        # preds = torch.split(token_pred, split_size_or_sections = seq_len, dim=0)
        preds = token_pred
        labels_split = torch.chunk(labels, seq_len,  dim =-1)
        token_weights = torch.chunk(token_weights, seq_len,  dim =-1) 
        
        losses = []
        loss = 0
        
        for i in range(seq_len):
            one_hot = F.one_hot(labels_split[i].squeeze(-1), num_classes = cardinality[i])
            ce = (one_hot * torch.log( preds[i] + 1e-7))[one_hot.bool()]
            ce = -1  * token_weights[i].squeeze(-1) * ce
            _loss = ce

            loss += torch.mean(_loss, dim =-1, keepdims=False)
            losses.append(torch.mean(_loss, dim =-1, keepdims=False)) 
        return losses,loss


    def train_model(
        self, 
        data,
        batch_size = 32,
        num_epochs = 10
    ):
        cardinality = self.cardinality
        # Create dataset obj
        data_set_obj = MLM_DataSet(
            data, 
            cardinality 
        )
        
        data_loader = DataLoader(data_set_obj, batch_size=batch_size, shuffle=True)
        epoch_losses = []
        all_losses = []
        for epoch in tqdm(range(num_epochs)):
            cur_epoch_losses = []
            print('[Epoch]', epoch)
            for batch_idx, batch_data in enumerate(data_loader):
                
                self.optimizer.zero_grad()
                seq_data, labels, token_wt  = batch_data[0], batch_data[1], batch_data[2] 
                seq_data = LT(seq_data).to(self.device)
                labels = LT(labels).to(self.device)
                token_wt = torch.tensor(token_wt).float().to(self.device)
                enc_seq_data, _ = self.encoder_obj(seq_data)
                token_pred = self.decoder_obj(enc_seq_data)

                losses, batch_loss = self.mlm_loss(
                    enc_seq_data,
                    token_pred,
                    labels,
                    token_wt
                )
                batch_loss.backward()
                self.optimizer.step()
                if (batch_idx+1)%50 == 0:
                    print('Batch ',batch_idx+1, batch_loss.cpu().data.numpy())
                loss_val = batch_loss.cpu().data.numpy()
                
                cur_epoch_losses.append(loss_val)
                all_losses.append(loss_val)
               
            epoch_losses.append(np.mean(cur_epoch_losses))
            print('[Mean epoch loss] {:4f}'.format(epoch_losses[-1]))
            
        return all_losses
    

def train_xformer_encoder(subDIR):
    print('[train_xformer_encoder]')
    global id_col, _rel_path_
    record_data = data_fetcher.get_training_set_data(subDIR)
    del record_data[id_col]
    config = _getConfig_()
    data = record_data
    data = data.values + 1
    domain_dims_df = data_fetcher.get_domain_dims(subDIR)
    cardinality = domain_dims_df['dimension'].tolist()
    cardinality = [ _ + 1 for _ in cardinality]
    
    density_fcn_dims = config['stage1_decoder_fcn_dims']
    encoder_num_xformer_layers = config['encoder_num_xformer_layers']
    encoder_xformer_heads = config['encoder_xformer_heads']
    base_emb_dim = config['base_emb_dim']
    batch_size = config['stage_1_batch_size']
    train_epochs = config['stage_1_train_epochs']
    model_save_dir = os.path.join(_rel_path_, config['model_save_dir'],subDIR)
    
    xformer_obj = tabXformerMLM_obj(
            emb_dim = base_emb_dim,
            encoder_xformer_heads = encoder_xformer_heads,
            count_numerical_attr = 0 ,
            cardinality = cardinality,
            encoder_num_xformer_layers = encoder_num_xformer_layers,
            encoder_include_PE = True,
            decoder_ffn_layer_dims = density_fcn_dims,
            device = DEVICE,
            model_save_dir = model_save_dir
    )
    print('[Object created]', xformer_obj)
    losses = xformer_obj.train_model(data, batch_size = batch_size, num_epochs = train_epochs)
    xformer_obj.save_model()

# ------------------------------------------------- # 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrain the encoder through MLM like objective')
    parser.add_argument(
        '--dir', 
        type=str,
        choices = ['us_import1', 'us_import2', 'us_import3', 'ecuador_export','colombia_export'],
        help='Train the encoder and the disposable decoder')

    args = parser.parse_args()
    subdir = args.dir
    train_xformer_encoder(subDIR=subdir)


