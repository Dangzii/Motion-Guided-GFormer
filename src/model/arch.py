from turtle import forward
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from . import losses as LOSSES
from .encoder import cluster_encoder as ENCODER
from . import head as HEAD



class Finetune(nn.Module):
    def __init__(self,cfg):
        super(Finetune,self).__init__()
        self.cfg = deepcopy(cfg)
        cfg_encoder = self.cfg["model"]["encoder"]
        cfg_losses = self.cfg["model"]["losses"]
        self.encoder = self.build_encoder(cfg_encoder)
        self.losses = self.build_losses(cfg_losses)

        log_dir = self.cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1

    def build_encoder(self,cfg_encoder):
        encoder_type = cfg_encoder.pop('type')
        if hasattr(ENCODER,encoder_type):
            encoder = getattr(ENCODER,encoder_type)(**cfg_encoder)
            return encoder
        else:
            raise KeyError("encoder_type not found. Got {}".format(encoder_type))

    def build_losses(self,cfg_losses):
        loss_type = cfg_losses.pop("type")
        if hasattr(LOSSES,loss_type):
            loss = getattr(LOSSES,loss_type)(**cfg_losses)
            return loss
        else:
            raise KeyError("loss_type not found. Got {}".format(loss_type))
    
    def forward(self,inputs,targets=None):
        pred,binary_mask= self.encoder(inputs)
        if self.training:
            loss = self.losses(binary_mask,pred,targets)
            self.writer.add_scalar("loss",loss.cpu().data.numpy(),self.step)
            loss = torch.unsqueeze(loss,0)
            self.step += 1
            return loss
        else:
            pred, loss = self.losses(binary_mask,pred,targets)
            loss = torch.unsqueeze(loss,0)
            return pred, loss, binary_mask

class Multi_Term(nn.Module):
    def __init__(self,cfg):
        super(Multi_Term,self).__init__()
        self.cfg = deepcopy(cfg)
        cfg_net = self.cfg['model']['net']
        cfg_head = self.cfg['model']['head']
        cfg_losses = self.cfg['model']['losses']
        cfg_aggregation = self.cfg['model']['aggregation']
        self.net = self.build_net(cfg_net)
        self.head = self.build_head(cfg_head)
        self.losses = self.build_losses(deepcopy(cfg_losses))
        self.aggregation = self.build_aggregation(cfg_aggregation)
        
        self.bnneck = nn.BatchNorm2d(cfg['model']['head']['in_channels'])
        nn.init.constant_(self.bnneck.bias, 0.0)
        self.bnneck.bias.requires_grad_(False)  # no shift

        # for log
        log_dir = self.cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1
    
    def build_aggregation(self,cfg_aggregation):
        pool_type = cfg_aggregation.pop('type')
        if hasattr(nn,pool_type):
            pool = getattr(nn,pool_type)(**cfg_aggregation)
            return pool
        else:
            raise KeyError("pool_type not found. Got {}".format(pool_type))

    def build_head(self,cfg_head):
        head_type = cfg_head.pop('type')
        if hasattr(HEAD,head_type):
            head = getattr(HEAD,head_type)(**cfg_head)
            return head
        else:
            raise KeyError("head_type not found. Got {}".format(head_type))

    def build_losses(self,cfg_losses):
        loss_type = cfg_losses.pop("type")
        if hasattr(LOSSES,loss_type):
            loss = getattr(LOSSES,loss_type)(**cfg_losses)
            return loss
        else:
            raise KeyError("loss_type not found. Got {}".format(loss_type))
    
    def build_net(self,cfg_net):
        net_type = cfg_net.pop('type')
        if hasattr(ENCODER,net_type):
            net = getattr(ENCODER,net_type)(**cfg_net)
            return net
        else:
            raise KeyError("net_type not found. Got {}".format(net_type))

    def forward(self,inputs,targets=None):
        pred = self.net(inputs)
        pred = self.bnneck(pred)#[:,:,0,0]
        # import pdb; pdb.set_trace()
        pred = torch.flatten(pred,1)
        # pred = self.bnneck(self.aggregation(self.head(pred)))[:,:,0,0]
        # calu losses
        if self.training:
            # import pdb; pdb.set_trace()
            loss = self.losses(pred,targets)
            self.writer.add_scalar("loss",loss.cpu().data.numpy(),self.step)
            loss = torch.unsqueeze(loss,0)
            self.step += 1
            return loss
        else:
            # inference
            pred = self.losses(pred,targets)
            return pred


class Pretrain(nn.Module):
    def __init__(self,cfg):
        super(Pretrain,self).__init__()
        self.cfg = deepcopy(cfg)
        cfg_encoder = self.cfg["model"]["encoder"]
        self.encoder = self.build_encoder(cfg_encoder)

        log_dir = self.cfg['log_dir']
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 1

    def build_encoder(self,cfg_encoder):
        encoder_type = cfg_encoder.pop('type')
        if hasattr(ENCODER,encoder_type):
            encoder = getattr(ENCODER,encoder_type)(**cfg_encoder)
            return encoder
        else:
            raise KeyError("encoder_type not found. Got {}".format(encoder_type))
        
    def forward(self,inputs):
        loss,pred,mask = self.encoder(inputs)
        if self.training:
            self.writer.add_scalar("loss",loss.cpu().data.numpy(),self.step)
            loss = torch.unsqueeze(loss,0)
            self.step += 1
            return loss








