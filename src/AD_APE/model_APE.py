import torch
import os
import numpy as np
from torch import nn
from tqdm import tqdm, trange
import torch.nn.functional as F
from torch import FloatTensor as FT
from torch import LongTensor as LT
try:
    import tqdm.notebook as tq
except:
    pass
from time import time
from pathlib import Path
'''
https://arxiv.org/pdf/1608.07502.pdf
'''

class APE(nn.Module):
    def __init__(self, emb_dim, domain_dims):
        super(APE, self).__init__()
        self.save_path = None
        self.num_domains = len(domain_dims)
        self.emb_dim = emb_dim

        self.num_entities = sum(domain_dims.values())
     
        self.emb_layer = nn.Embedding(
            num_embeddings=self.num_entities,
            embedding_dim=emb_dim
        )
       
        self.c = nn.Parameter(torch.from_numpy(np.random.random(1)))
        self.mode='train'
        k = 0

        self.pair_W = nn.ParameterDict({})
        for i in range(self.num_domains ):
            for j in range(i+1,self.num_domains ):
                w_k = nn.Parameter(data= FT(np.random.random(1)))
                self.pair_W[str(k)] = w_k
                k+=1
        return

    def obtain_score(self,x):
        x = self.emb_layer(x)
        # Split along axis = 1
        x1 = [ _.squeeze(1) for _ in  torch.chunk( x, self.num_domains, dim=1)]
        k = 0
        _score = []
        for i in range(self.num_domains):
            for j in range(i+1,self.num_domains):
                r = torch.sum(torch.mul(x1[i],x1[j]), dim=-1, keepdims=True)
                r = r * torch.exp(self.pair_W[str(k)])
                _score.append(r)
        _score = torch.cat(_score, dim=-1)
        score = torch.exp(torch.sum(_score,dim=-1) + self.c)
        return score

    # Expect entitty ids to be serialized
    def forward(self, pos_x, neg_x = None):
        pos_score = self.obtain_score(pos_x)
        if self.mode =='test':
            return pos_score
        
        num_neg_samples = neg_x.shape[1]
        neg_x_split = [ _.squeeze(1) for _ in torch.chunk(neg_x,num_neg_samples,dim=1)]
        neg_score = []
        for nsample_i in neg_x_split:
            s = self.obtain_score(nsample_i)
            neg_score.append(s)
        
        neg_score = torch.stack(neg_score, dim=-1)
        return pos_score,neg_score

    def score_records(self, x):
        self.mode = 'test'
        self.eval()
        return self.obtain_score(x)

class APE_container:
    
    def __init__(
        self, 
        emb_dim,
        domain_dims, 
        device, 
        batch_size= 128, 
        LR=0.001,
        model_save_dir = 'ad_model_saved'
    ):
        self.model = APE(emb_dim, domain_dims)
        self.device = device
        self.emb_dim = emb_dim
        self.domain_dims = domain_dims
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.batch_size = batch_size
        self.signature = 'model_{}_{}'.format(emb_dim,int(time()))
        self.epoch_meanLoss_history = []
        self.model_save_dir = model_save_dir
        self.save_path = None
        return

    def train_model(self, pos_x, neg_x, num_epochs = 50, log_interval = 100, tol = 0.025 ):
        self.model.train()
        self.model.mode ='train'
        clip_value = 5
        bs = self.batch_size
        idx = np.arange(pos_x.shape[0])
        num_batches = pos_x.shape[0]//bs +1
        loss_history  = []
        
        for e in range(num_epochs):
            epoch_loss = []
            np.random.shuffle(idx)
            pbar = tqdm(range(num_batches))
            
            for b in pbar:
                self.optimizer.zero_grad()
                _idx = idx[b*bs : (b+1)*bs]
                _x_pos = pos_x[_idx]
                _x_neg = neg_x[_idx]
                _x_pos = LT(_x_pos).to(self.device)
                _x_neg = LT(_x_neg).to(self.device)
                pos_score, neg_score = self.model(_x_pos, _x_neg)
                # Calculate loss
                term1 = torch.log(torch.sigmoid(torch.log(pos_score)))
                term2 = torch.log(torch.sigmoid(-torch.log(neg_score)))
                term2 = torch.sum(term2, dim=-1)
                
                loss = term1 + term2
                loss = torch.mean(loss, dim=0)
                loss = -loss
                # if (b+1) % log_interval == 0 :
                #     print('Epoch {} Batch {} Loss {:.4f}'.format(e,b+1, loss.cpu().data.numpy()))
                    
                epoch_loss.append(np.mean(loss.cpu().data.numpy()))
                tqdm._instances.clear()
                pbar.set_postfix({'Batch ': b+1 ,' Loss ': '{:.4f}'.format(loss.cpu().data.numpy())})
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                self.optimizer.step()
            
            self.epoch_meanLoss_history.append(np.mean(epoch_loss))
            loss_history.extend(epoch_loss)
            print('Mean epoch loss {:.4f}'.format(np.mean(epoch_loss)))
            # ------------------
            # Early stopping
            # ------------------
            if len(self.epoch_meanLoss_history) > 10:
                delta1 = abs(self.epoch_meanLoss_history[-2] - self.epoch_meanLoss_history[-1])
                delta2 = abs(self.epoch_meanLoss_history[-3] - self.epoch_meanLoss_history[-2])

                if  delta2 <= tol and delta1 <= tol:
                    print('Stopping!')
                    break
           
        self.model.mode='test'
        return loss_history

    def predict(self, x_test):
        """
        EValuation (test mode)
        """
        self.model.mode='test'
        self.model.eval()
        bs = 500
        results = []
        
        with torch.no_grad():
            num_batches = x_test.shape[0] // bs + 1
            idx = np.arange(x_test.shape[0])
            for b in range(num_batches):
                b_idx = idx[b * bs:(b + 1) * bs]
                if len(b_idx) == 0:
                    break

                x = LT(x_test[b_idx]).to(self.device)
                score_values = self.model(x)
                vals = score_values.cpu().data.numpy().tolist()
                results.extend(vals)

        return results
    
    def save_model(self):
        """
        Save model
        """
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(Path(self.model_save_dir), self.signature )
        self.save_path = save_path
        torch.save(self.model.state_dict(), save_path)
        return

    def load_model(
        self, 
        path = None
    ):
        """
        Load previously saved model
        """
        if self.save_path is None and path is None:
            print('Error . Null path given to load model ')
            return None
        
        if path is None:
            path = self.save_path 
        
        self.model = APE( 
            self.emb_dim, 
            self.domain_dims
        )
        
        if str(self.device)=="cpu":
            self.model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))
            
        self.model.to(self.device)
        self.model.eval()
        self.model.mode = 'test'
        return
        