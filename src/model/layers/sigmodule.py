import torch
import torch.nn as nn
import signatory
import numpy as np



# from .block import to_2tuple



# try:
#     from torch import _assert
# except ImportError:
#     def _assert(condition: bool, message: str):
#         assert condition, message

class SigNet(nn.Module):
    def __init__(self, in_channels, out_dimension, sig_depth, kernel_size=1, include_orignial=True,include_time=True,fc=True, dropout=0):
        super(SigNet, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=kernel_size,
                                         include_original=include_orignial,
                                         include_time=include_time)

        self.signature = signatory.Signature(depth=sig_depth)

        if include_time:
            # +1 because signatory.Augment is used to add time as well
            sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                        depth=sig_depth)
        else:
            sig_channels = signatory.signature_channels(channels=in_channels,
                                                        depth=sig_depth)
        # pdb.set_trace()
        self.fc = fc
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sig_channels,out_dimension)

        # self.replacesig = nn.Sequential(
        # nn.Linear(6600, 40),
        # nn.BatchNorm1d(40),
        # nn.GELU())

        # self.linear = nn.Linear(40, out_dimension)

    def forward(self, inp, basepoint=True):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)

        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the"
                               " signature")
        # x in a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=basepoint)

        # y = x.flatten(1)
        # y = self.replacesig(y)

        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        if self.fc:
            z = self.linear(self.dropout(y))
        # z is a two dimensional tensor of shape (batch, out_dimension)
        else:
            z = y
        # pdb.set_trace()

        return z



