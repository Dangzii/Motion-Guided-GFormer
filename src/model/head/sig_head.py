import torch.nn as nn
from ..layers.sigmodule import SigNet


class sig_head(nn.Module):
    def __init__(self,in_channels, out_dimension, sig_depth, ps=False,):
        super(sig_head,self).__init__()
        self.in_channels = in_channels
        self.out_dimension = out_dimension
        self.sig_depth = sig_depth
        if ps:
            self.head = SigNet(in_channels, out_dimension, sig_depth)
        # else:
        #     self.head = DevNet(n_inputs=self.in_channels, n_outputs= self.out_dimension)
    
    def forward(self,x):
        x = self.head(x)
        return x