import sys
from numpy import dtype
import torch.nn as nn
import torch
import torch.distributed as dist


import timm.models.vision_transformer
from ..layers import PatchEmbed,Block
from ..layers.mask import *
from ..layers.block import CBlock2
from ..layers.sigmodule import SigNet
import torch.nn.functional as F
from ..layers.knn_util import *



class GlobalLocalFormer2(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, max_T, decay_alpha, k=5, embed_dims=[16],
                 num_stages=2, sample_ratio=[0.025], depths=[2],
                 num_heads=[4],mlp_ratios=[4], path_dim=12,
                 num_person=2, num_point=25, dimension=3, every_sample=False,
                 kernel_size=1, sig_depth=2, head_dropout=0, sig_head_ratio=8,
                 **kwargs):
        super(GlobalLocalFormer2, self).__init__(**kwargs)

        # patch embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, kwargs['embed_dim']))
        self.patch_embed = PatchEmbed(
            img_size=kwargs['img_size'], patch_size=kwargs['patch_size'], in_chans=dimension,
            embed_dim=kwargs['embed_dim'], norm_layer=kwargs['norm_layer'])
        self.token_num = self.patch_embed.num_patches
        self.pos_embed_local = nn.Parameter(torch.randn(1, self.token_num, kwargs['embed_dim']) * .02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.token_num + 1, kwargs['embed_dim']) * .02)
        
        #data pre
        self.data_bn = nn.BatchNorm1d(num_person * dimension * num_point)
        self.path_conv = nn.Conv1d(dimension * kwargs['patch_size'][0], path_dim, 1, groups=kwargs['patch_size'][0])
        self.path_bn = nn.LayerNorm(path_dim)
        
        # local
        self.path_dim = path_dim
        self.k = k
        self.num_stages = num_stages
        self.every_sample = every_sample
        self.sample_ratio = sample_ratio
        self.channel_embed = nn.ModuleList(nn.Conv1d(embed_dims[i], embed_dims[i+1], 1) for i in range(self.num_stages))
        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            block = nn.ModuleList([CBlock2( L=self.token_num,
                dim=embed_dims[i], p_dim=self.path_dim, sample_ratio=sample_ratio[i], num_head=num_heads[i], k=self.k, mlp_ratio=mlp_ratios[i], 
                drop=kwargs['drop_rate'], attn_drop=kwargs['attn_drop_rate'], 
                drop_path=dpr[cur + j]) for j in range(depths[i])])     
            if kwargs['norm_layer']== nn.BatchNorm1d:
                norm = kwargs['norm_layer'](self.token_num)
            else:
                norm = kwargs['norm_layer'](embed_dims[i])
            cur += depths[i]
            
            setattr(self, f"block{i}", block)
            setattr(self, f"norm{i}", norm)

        # global
        self.encoder_blocks1 = nn.ModuleList([
            Block(embed_dims[-1] ,num_heads[-1], mlp_ratios[-1], 
                  qkv_bias=True, norm_layer=kwargs['norm_layer'],
                  L=self.token_num+1) for i in range(depths[-1])])

        # gumbel mask
        self.sigmoid = nn.Sigmoid()
        self.gumble = MyGumbelSigmoid(max_T,decay_alpha)
        self.cal_score = cal_score2(dim=embed_dims[-1])
        self.cal_attn = cal_soft_origin(soft_dim=self.patch_embed.num_patches, dim=embed_dims[-1] ,qkv_bias=False)

        # head 
        self.catnorm = nn.BatchNorm1d(self.token_num)
        self.reduce_channel = nn.Conv1d(embed_dims[-1]*2 , embed_dims[-1] // sig_head_ratio, 1)
        self.norm_path = nn.LayerNorm(embed_dims[-1]//sig_head_ratio)
        self.head = SigNet(embed_dims[-1]//sig_head_ratio, kwargs['num_classes'], sig_depth, kernel_size, dropout=head_dropout)

    def motion_features(self,x):
        delta_S, delta_T = self.patch_embed.patch_size
        x_motion = x[:,:,:,delta_T-1::delta_T] - x[:,:,:,::delta_T]
        b,c,j,_ = x_motion.shape
        x_motion = x_motion.reshape(b,c,j//delta_S,delta_S,-1).permute(0,1,3,2,4).reshape(b, -1, self.token_num)
        x_motion = x_motion.permute(0,2,1)
        return x_motion

    def forward_global(self, x_embed):
        cls_tokens = self.cls_token.expand(x_embed.shape[0], -1, -1)
        x_token = torch.cat((cls_tokens, x_embed), dim=1)
        x_token = x_token + self.pos_embed
        x_token = self.pos_drop(x_token)

        for blk in self.encoder_blocks1:
            x_token = blk(x_token)
        

        global_mask_soft = self.cal_attn(x_token).unsqueeze(dim=-1)

        return x_token, global_mask_soft 
    
    def forward_local(self, x_path, x_embed):
        
        x_embed = x_embed + self.pos_embed_local

        for i in range(self.num_stages):
            block = getattr(self, f"block{i}")
            norm = getattr(self, f"norm{i}")
            
            if i > 0 :
                x_embed = self.channel_embed[i-1](x_embed.permute(0,2,1))
                x_embed = x_embed.permute(0,2,1)

            cluster_num = max(math.ceil(self.token_num * self.sample_ratio[i]), 1)

            if self.every_sample:
                idx_cluster = None
            else:
                idx_cluster, cluster_num = cluster_dpc_knn(x_path, cluster_num, self.k)

            for j, blk in enumerate(block):
                    x_embed = blk(x_embed, x_path, idx_cluster, cluster_num)

            x_embed = norm(x_embed)
        
        return x_embed 

    def forward_features(self, x):
        if self.patch_embed.padding:
            #padding
            padding = torch.zeros((x.shape[0],x.shape[1],1,x.shape[3])).to(x.device)
            x = torch.cat((padding, x),dim=2)

        # global branch
        x_embed = self.patch_embed(x)
        x_global, global_mask_soft = self.forward_global(x_embed)

        # dynamic branch
        x_embed = self.pos_drop(x_embed)
        x_path = self.motion_features(x) 
        x_local = self.forward_local(x_path, x_embed)
        
        # score prediction module
        mask_soft = self.cal_score(x_local, global_mask_soft)
        mask = self.gumble(mask_soft)

        if not self.training:
            mask_binary = (mask >= 0.5).float()
            mask = mask * mask_binary
                
        binary_mask = torch.where(mask>0.5, torch.ones_like(mask), torch.zeros_like(mask))

        x = x_global[:,1:,:] + x_embed
        x = torch.cat((x, x_local),dim=-1)
        x = self.reduce_channel(x.permute(0,2,1)).permute(0,2,1)
        x = torch.mul(x, mask) 

        ids_store = torch.range(1,self.token_num,1,device=binary_mask.device) 
        ids_store = binary_mask.squeeze(-1) * ids_store    
        ids_store = torch.argsort(ids_store)
        x = torch.gather(x,dim=1,index=ids_store.unsqueeze(-1).repeat(1,1,x.shape[-1]))

        return x, mask