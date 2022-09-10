import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.functional import softmax

from Networks.TCDF_net import ADDSTCN



class ADDSTCN_validation(ADDSTCN):

    def forward(self, x):
        x = x.to(self.device)
        out_first = self.first_layer(x)
        out_middle = self.middle_layers(out_first)
        out_final = self.final_layer(out_middle)
        out = self.final_conv(out_final)
        return out


