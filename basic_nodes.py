import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import sys

class LinearBlock(nn.Module):
    def __init__(self, pre_dim, dim, activation="none", dropout_rate=0, use_batch_norm=False, use_layer_norm=False):
        self.linear = None
        self.bn = None
        self.ln = None
        self.act = None
        self.dropout_layer = None

        super(LinearBlock, self).__init__()
        if (activation == "relu"):
            self.act = nn.ReLU()
        elif (activation == "tanh"):
            self.act = nn.Tanh()
        elif (activation == "sigmoid"):
            self.act = nn.Sigmoid()
        elif (activation == "normrelu"):
            self.act = normrelu()
        elif (activation == "none"):
            self.act = None
        else:
            print("ERROR: We don't support this kind of activation function yet\n")
        
        if (use_batch_norm):
            self.linear = nn.Linear(pre_dim, dim, bias = False)
            self.bn = nn.BatchNorm1d(dim, momentum=0.05)
        else:
            self.linear = nn.Linear(pre_dim, dim)

        if (use_layer_norm):
            self.ln = LayerNorm(dim)
        if (dropout_rate > 0.0001):
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        

    def forward(self, x):
        if (self.linear != None):
            x = self.linear(x)
        if (self.bn != None):
            x = self.bn(x)
        if (self.ln != None):
            x = self.ln(x)
        if (self.act != None):
            x = self.act(x)
        if (self.dropout_layer != None):
            x = self.dropout_layer(x)
        return x

class CNN_feaproc(nn.Module):
    
    def __init__(self):
       super(CNN_feaproc,self).__init__()
       self.conv1 = nn.Conv2d(1, 100, 3)
       self.conv2 = nn.Conv2d(100, 50, 5)
       
    def forward(self, x):
       steps=x.shape[0]
       batch=x.shape[1]
       x=x.view(x.shape[0]*x.shape[1],1,-1,11)
       out = F.max_pool2d(self.conv1(x), (2, 1))
       out = F.max_pool2d(self.conv2(out), (2, 2))
       out= out.view(steps,batch,-1)
       return out
        

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class normrelu(nn.Module):

    def __init__(self):
        super(normrelu, self).__init__()


    def forward(self, x):
        dim=1
        x=F.relu(x)/(torch.max(x,dim,keepdim=True)[0])
        return x
    
