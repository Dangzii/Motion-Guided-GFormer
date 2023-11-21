import torch
import torch.nn as nn

class identity_head(nn.Module):
    def __init__(self,**kwags):
        super(identity_head,self).__init__()
        pass
    
    def forward(self,x):
        return x