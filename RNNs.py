import numpy as np
import sys

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from basic_nodes import *

class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, 
                bias=True, batch_first=True, dropout=0.0001, 
                bidirectional=False, output_layer=False, init_weight=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                         num_layers=num_layers, bias=bias, batch_first=batch_first, 
                         dropout=dropout, bidirectional = bidirectional)
        if init_weight:
	    self.init_weights()
        if output_layer:
            self.linear = LinearBlock(hidden_size, output_size, 'relu')
        else:
            self.linear = None

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x, length):
        batch_size = x.shape[0]
        total_length = x.shape[1]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).type_as(x)
        x = pack_padded_sequence(x, length, batch_first=True)
        output, hn = self.rnn(x, h0)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        if self.linear == None:
            return output
        output = self.linear(output)
        return output
         

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bias=True, batch_first=True, dropout=0.0001, bidirectional=False, output_layer=False, init_weight=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                         num_layers=num_layers, bias=bias, batch_first=batch_first, 
                         dropout=dropout, bidirectional = bidirectional)
        if init_weight:
            print("Xavier init")
            for i in range(num_layers):
                nn.init.xavier_uniform_(self.rnn.all_weights[i][0])
                nn.init.xavier_uniform_(self.rnn.all_weights[i][1])
        if output_layer:
            self.linear = LinearBlock(hidden_size, output_size, 'relu', 
                    dropout_rate=dropout)
        else:
            self.linear = None

    def forward(self, x, h0, length):
        batch_size = x.shape[0]
        total_length = x.shape[1]
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim).type_as(x)
        x = pack_padded_sequence(x, length, batch_first=True)
 
        output, hn = self.rnn(x, h0)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=total_length)
        if self.linear == None:
            return output
        output = self.linear(output)
        return output
