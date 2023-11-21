from numpy import NaN
from .attention import Attention


import torch.nn as nn
import torch
from itertools import repeat
import collections.abc

from .knn_util import *
import math




def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    # random_tensor = keep_prob + torch.rand(shape, dtype=x.type, device=x.device)
    # random_tensor.floor_()
    # # rand < drop_prob, random_tensor = 0, drop
    # output = x.div(keep_prob) * random_tensor
    # # div for test see discussion:https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    # return output
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self) -> str:
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 L=100):
        super().__init__()
        if norm_layer == nn.BatchNorm1d:
            self.norm1 = norm_layer(L)
            self.norm2 = norm_layer(L)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CTAttention(nn.Module):
    def __init__(self, dim, pdim, num_head=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        # self.dim_out = dim_out
        # self.conv = TokenConv(in_channels=embed_dim, out_channels=dim_out, kernel_size=3, stride=2, padding=1)
        # self.score = nn.Linear(embed_dim, 1)
        assert dim % num_head == 0, 'dim should be divisible by num_heads'
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.scale = self.head_dim ** -0.5
        self.phdim = pdim // num_head
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_pv = nn.Linear(pdim, pdim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pproj = nn.Linear(pdim, pdim)
        
    def init_weights(self):
        nn.init.xavier_normal(self.to_q.weight)


    def forward(self, x_token, x_path, idx_cluster, cluster_num, eps=1e-6):

        B, N, C = x_token.shape
        q, k = self.to_q(x_token), self.to_k(x_token)
        v = self.to_v(x_token).reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        

        q_ = torch.zeros((B,cluster_num,N,C),device=x_token.device)
        k_ = torch.zeros((B,cluster_num,N,C),device=x_token.device)
        idx = idx_cluster[:,None,:,None].repeat(1,cluster_num,1,C)
        q = q[:,None].repeat(1,cluster_num,1,1)
        k = k[:,None].repeat(1,cluster_num,1,1)
        q_ = torch.scatter(q_, 1, idx, q)
        k_ = torch.scatter(k_, 1, idx, k)

        # zeros = torch.zeros((B,N,C), device=x_token.device)
        # ones = torch.ones((B,N,C),device=x_token.device)
        # for i in range(cluster_num):
        #     mask = torch.where(idx_cluster.unsqueeze(-1).repeat(1,1,C)==i, ones, zeros)
        #     q_[:,i] = zeros.masked_scatter(mask==1, q) 
        #     k_[:,i] = zeros.masked_scatter(mask==1, k) 
        
        split_head = lambda v: v.reshape(B, cluster_num, N, self.num_head, self.head_dim).permute(0, 1, 3, 2, 4)
        q_, k_ = map(split_head, (q_, k_))

        attn_map = (q_ @ k_.transpose(-2, -1)) * self.scale
        attn = attn_map.sum(dim=1)  
        attn = torch.masked_fill(attn, attn==0, -1e9) 
        # for stable training
        attn = attn.to(torch.float32).exp_()
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)

        x_token = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_token = self.proj(x_token)
        x_token = self.proj_drop(x_token)

        # pv = self.to_pv(x_path).reshape(B, N, self.num_head, self.phdim).permute(0, 2, 1, 3)
        # x_path = (attn @ pv).transpose(1, 2).reshape(B, N, -1)
        # x_path = self.pproj(x_path)

        # return x_token, x_path, attn_map
        return x_token, attn_map


class CAttention(nn.Module):
    def __init__(self, dim, pdim, num_head=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_head == 0, 'dim should be divisible by num_heads'
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.scale = self.head_dim ** -0.5
        self.phdim = pdim // 1
        
        self.to_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_pv = nn.Linear(pdim, pdim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pproj = nn.Linear(pdim, pdim)
        
        #self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal(self.to_q.weight)


    def forward(self, x_token, x_path, idx_cluster, cluster_num, eps=1e-6):

        B, N, C = x_token.shape
        qk = self.to_qk(x_token).reshape(B, N, 2, -1).permute(2, 0, 1, 3)
        q,k = qk.unbind(0)
        v = self.to_v(x_token)
        pv = self.to_pv(x_path)
        
        idx_shuffle = torch.argsort(idx_cluster)
        idx_restore = torch.argsort(idx_shuffle)

        v = torch.gather(v,dim=1,index=idx_shuffle.unsqueeze(-1).repeat(1,1,C))
        pv = torch.gather(pv,dim=1,index=idx_shuffle.unsqueeze(-1).repeat(1,1,self.phdim))
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.gather(attn_map,dim=1,index=idx_shuffle.unsqueeze(-1).repeat(1,1,N)) 
        attn = torch.gather(attn,dim=1,index=idx_shuffle.unsqueeze(-2).repeat(1,N,1))
        last_len = torch.zeros(B,device=idx_cluster.device,dtype=int)
        mask = torch.zeros((B,N,N),device=attn.device)
        for i in range(cluster_num):          
            cur_len = last_len + torch.where(idx_cluster==i,1,0).sum(dim=-1)
            for j in range(B):
                mask[j,last_len[j]:cur_len[j],last_len[j]:cur_len[j]] = 1
            last_len = cur_len
        
        attn = torch.masked_fill(attn, mask==0, -1e9) 
        # for stable training
        attn = attn.to(torch.float32).exp_()
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)

        x_token = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_token = torch.gather(x_token,dim=1,index=idx_restore.unsqueeze(-1).repeat(1,1,C))
        x_token = self.proj(x_token)
        x_token = self.proj_drop(x_token)

        # reshape(B, N, self.num_head, self.phdim).permute(0, 2, 1, 3)
        # x_path = (attn @ pv).transpose(1, 2).reshape(B, N, -1)
        # x_path = torch.gather(x_path,dim=1,index=idx_restore.unsqueeze(-1).repeat(1,1,self.phdim))
        # x_path = self.pproj(x_path)

        #return x_token, x_path, attn_map
        return x_token, attn_map


class CTAttention2(nn.Module):
    def __init__(self, dim, k, sample_ratio,
                 num_head=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_head == 0, 'dim should be divisible by num_heads'
        self.num_head = num_head
        self.head_dim = dim // num_head
        self.scale = self.head_dim ** -0.5
        self.k = k
        self.sample_ratio = sample_ratio
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def init_weights(self):
        nn.init.xavier_normal(self.to_q.weight)

    def foward_cluster(self, x_token):
        cluster_num = max(math.ceil(x_token.shape[1] * self.sample_ratio), 1)
        idx_cluster, cluster_num = cluster_dpc_knn(
        x_token, cluster_num, self.k)

        return idx_cluster, cluster_num

    def forward(self, x_token, eps=1e-6):

        B, N, C = x_token.shape

        idx_cluster, cluster_num = self.foward_cluster(x_token)

        q, k = self.to_q(x_token), self.to_k(x_token)
        v = self.to_v(x_token).reshape(B, N, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        
        q_ = torch.zeros((B,cluster_num,N,C),device=x_token.device)
        k_ = torch.zeros((B,cluster_num,N,C),device=x_token.device)
        idx = idx_cluster[:,None,:,None].repeat(1,cluster_num,1,C)
        q = q[:,None].repeat(1,cluster_num,1,1)
        k = k[:,None].repeat(1,cluster_num,1,1)
        q_ = torch.scatter(q_, 1, idx, q)
        k_ = torch.scatter(k_, 1, idx, k)
        
        split_head = lambda v: v.reshape(B, cluster_num, N, self.num_head, self.head_dim).permute(0, 1, 3, 2, 4)
        q_, k_ = map(split_head, (q_, k_))

        attn_map = (q_ @ k_.transpose(-2, -1)) * self.scale
        attn = attn_map.sum(dim=1)  
        attn = torch.masked_fill(attn, attn==0, -1e9) 
        # for stable training
        attn = attn.to(torch.float32).exp_()
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)

        x_token = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_token = self.proj(x_token)
        x_token = self.proj_drop(x_token)

        return x_token


class CBlock2(nn.Module):
    def __init__(self, 
                 dim,
                 p_dim,
                 num_head, 
                 sample_ratio,
                 k=5,
                 attn_drop = 0., 
                 drop = 0.,
                 act_layer=nn.GELU,
                 drop_path = 0.5,
                 mlp_ratio = 4,
                 norm_layer=nn.LayerNorm,
                 L=100):
        super().__init__()
        
        if norm_layer == nn.BatchNorm1d:
            self.norm1 = norm_layer(L)
            self.norm2 = norm_layer(L)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
        
        self.path_norm = nn.LayerNorm(p_dim)

        self.sample_ratio = sample_ratio
        self.k = k
        
        self.attn = CTAttention(dim, p_dim, num_head, qkv_bias=False, attn_drop=attn_drop, proj_drop=0.)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)    


    def forward(self, x_token, x_path, idx_cluster=None, cluster_num=None, **kwargs):
        if idx_cluster is None:
            # cluster_num = max(math.ceil(x_token.shape[1] * self.sample_ratio), 1)
            idx_cluster, cluster_num = cluster_dpc_knn(
            x_path, cluster_num, self.k)
        token = self.norm1(x_token) 
        # path = self.path_norm(x_path)       
        #token, x_path, attn_map = self.attn(token, path, idx_cluster, cluster_num)
        token, attn_map = self.attn(token, x_path, idx_cluster, cluster_num)
        x_token = x_token + self.drop_path(token)
        #x_token = x_token + self.drop_path(self.attn(self.norm1(x_token)))
        x_token = x_token + self.drop_path(self.mlp(self.norm2(x_token)))
        return x_token
    
