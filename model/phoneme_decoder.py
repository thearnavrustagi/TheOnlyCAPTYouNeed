import torch
import torch.nn as nn

class phoneme_deceoder:
    def __init__(self, hidden_size, dropout):
        self.hidden_size = hidden_size, 
        self.dropout = dropout, 
        self.gru = nn.GRU(hidden_size=hidden_size, dropout=dropout)