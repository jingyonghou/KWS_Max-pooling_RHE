import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, 
                groups=1, bias=True):
        super(DilatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding =0 # (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                padding=padding, dilation=dilation, bias=bias)
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            for name, param in self.named_parameters():
#                print("name: %s\n"%name)
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)                              
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.padding != 0 :
            outputs = outputs[:, :, :-self.padding]
        return outputs


class TCN(nn.Module):
    def __init__(self, layer_size, stack_size, in_channels,
            hid_channels, kernel_size=8, dilations=None, dropout=0):
        super(TCN, self).__init__()
        self.num_layers = layer_size * stack_size
        self.preprocess = DilatedConv1d(in_channels, hid_channels, kernel_size=1, dilation=1)

        self.model = torch.nn.ModuleList()
        self.padding = 0
        for stack_num in range(stack_size):
            for layer_num in range(layer_size):
                dilation = 2 ** (layer_num)
                self.padding += dilation * (kernel_size-1)
#                self.model.append(DilatedConv1d(hid_channels, hid_channels, kernel_size=1, dilation=1))
                self.model.append(DilatedConv1d(hid_channels, hid_channels, kernel_size=kernel_size, dilation=dilation))
            if dropout != 0:
                self.model.append(nn.Dropout(float(dropout)))
	print("padding: %d\n"%self.padding)
    def forward(self, inputs, length):
        outputs = F.pad(inputs, (0,0,self.padding,0,0,0), 'constant')
        outputs = outputs.permute(0, 2, 1)
        outputs = F.relu(self.preprocess(outputs))
        for i in range(self.num_layers):
            outputs = F.relu(self.model[i](outputs))
        outputs = outputs.permute(0, 2, 1)
        #if inputs.shape != outputs.shape:
        #    print("Padding Errors\n")
        return outputs
