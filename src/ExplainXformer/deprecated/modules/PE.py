import torch
import os
import sys
import pandas as pd
import numpy as np
from torch import FloatTensor as FT
from torch import nn, Tensor
from torch import LongTensor as LT
from torch.nn import Module 
from torch.nn import functional as F
from typing import *
import math

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        dropout: float = 0.0, 
        max_len: int = 1000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)