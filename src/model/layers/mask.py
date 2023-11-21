import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from .attention import Attention


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def cal_pearson(x,y):
    #calculate pccs= sum((x_i - x_mean)*(y_i - y_mean)) / sqrt(sum(x_i-x_mean)**2) *  sqrt(sum(y_i-y_mean)**2)
    #x.shape = N,L,D
    #y.shape = N,1,D
    x_mean = torch.mean(x,dim=2,keepdim=True)
    y_mean = torch.mean(y,dim=2,keepdim=True)
    x_std = torch.std(x,dim=2,unbiased=False,keepdim=True)
    y_std = torch.std(y,dim=2,unbiased=False,keepdim=True)
    z_x = (x - x_mean) / x_std
    z_y = ((y - y_mean) / y_std).expand(x.shape[0],x.shape[1],-1)
    r = torch.sum(torch.mul(z_x,z_y),dim=2) / x.shape[2]
    return r

class cal_attn(Attention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dim = kwargs['dim']
        qkv_bias = kwargs['qkv_bias']
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        del self.proj
        del self.proj_drop

    def forward(self, x):
        #x = torch.cat((x,y),dim=1)
        B, N, C = x.shape
        qk = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k= qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        relative = torch.mean(attn[:,:,0,1:],dim=1)

        return relative


class cal_soft_origin(Attention):
    def __init__(self, soft_dim, num_heads=1, sig=False, **kwargs):
        super().__init__(**kwargs)
        dim = kwargs['dim']
        qkv_bias = kwargs['qkv_bias']

        # self.cls = nn.Linear(dim, dim, bias=qkv_bias)
        # self.token = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.sigmoid = nn.Sigmoid()
        self.scale = dim ** -0.5

        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.linear = nn.Linear(soft_dim, soft_dim)
        self.sig = sig
        # self.LN = nn.LayerNorm(380)

        del self.proj
        del self.proj_drop

    def forward(self, x, policy=None):

        B, N, C = x.shape

        qk = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k= qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-1,-2)) * self.scale         
        mask_soft = attn[:,:,0,1:].squeeze(1)   # attn with cls
        mask_soft = self.linear(mask_soft)
        if self.sig:
            mask_soft = self.sigmoid(mask_soft)
  
        return mask_soft


class cal_soft(nn.Module):
    def __init__(self, dim, soft_dim, qkv_bias=True, num_heads=1, **kwargs):
    # def __init__(self, soft_dim, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        # dim = kwargs['dim']
        # qkv_bias = kwargs['qkv_bias']

        self.cls = nn.Linear(dim, dim, bias=qkv_bias)
        self.token = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.sigmoid = nn.Sigmoid()
        self.scale = dim ** -0.5

    def forward(self, x, policy=None):

        B, N, C = x.shape
        cls_token = x[:,:1,]
        other_token = x[:,1:,]
        cls_q = self.cls(cls_token)
        other_k = self.token(other_token)
        v = self.v(other_token)

        attn = (cls_q @ other_k.transpose(-1,-2)) * self.scale  
        # mask_soft = attn @ v
        # mask_soft = self.linear(attn.reshape(B,-1))  
        mask_soft = self.sigmoid(attn.squeeze(1))     
  
        return mask_soft


class cal_score(nn.Module):
    def __init__(self, dim, **kwargs):
    # def __init__(self, soft_dim, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.in_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2),
            # nn.Sigmoid(dim=-1)
        )

    def forward(self, x):
        x = self.in_conv(x)
        return x


class cal_score2(nn.Module):
    def __init__(self, dim, **kwargs):
    # def __init__(self, soft_dim, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.in_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 1),
            nn.GELU(),
        )

        self.out_conv = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, global_score):
        local_score = self.in_conv(x)
        score = torch.cat((local_score,global_score),dim=-1)
        return self.out_conv(score)


class MyGumbelSigmoid(nn.Module):
    def __init__(self, max_T, decay_alpha):
        super(MyGumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.decay_alpha = decay_alpha
        self.softmax = nn.Softmax(dim=-1)
        self.p_value = 1e-8

        self.register_buffer('cur_T', torch.tensor(max_T))

    def forward(self, x):
        if self.training:
            _cur_T = self.cur_T
        else:
            _cur_T = 0.03

        # Shape <x> : [N, L, 1]/[N, C, H, W]
        # Shape <r> : [N, L, 1]/[N, C, H, W]
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (_cur_T + self.p_value)
        r = r + r_N
        r = r / (_cur_T + self.p_value)

        if len(x.shape) > 3: 
            x = torch.cat((x, r), dim=1)
            x = x.permute(0,2,3,1)
            x = self.softmax(x)
            x = x[:, :, :, [0]]
            x = x.permute(0,3,1,2)
        else:
            x = torch.cat((x, r), dim=-1)
            x = self.softmax(x)
            x = x[:, :, [0]]

        if self.training:
            if self.cur_T < 0.03:
                self.cur_T = self.cur_T
            else:
                self.cur_T = self.cur_T * self.decay_alpha
        return x

class GumbelSigmoid(nn.Module):
    def __init__(self, max_T, decay_alpha):
        super(GumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.decay_alpha = decay_alpha
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

        self.register_buffer('cur_T', torch.tensor(max_T))

    def forward(self, x):
        if self.training:
            _cur_T = self.cur_T
        else:
            _cur_T = 0.03

        # Shape <x> : [N, C, H, W]
        # Shape <r> : [N, C, H, W]
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (_cur_T + self.p_value)
        r = r + r_N
        r = r / (_cur_T + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)
        x = x[:, [0], :].squeeze(dim=1)

        if self.training:
            self.cur_T = self.cur_T * self.decay_alpha

        return x


class RegionAttention(nn.Module):
    def __init__(self,dim,qkv_bias,attn_drop,pool_size=[(5,10),(3,6),(2,4)],pool_type='avg_pool'):
        super().__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

        self.scale = dim ** -0.5
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
    
        #self.conv_blocks = nn.ModuleList([nn.Conv2d() for i in range(len(self.pool_size))])
        self.attn_drop = nn.Dropout(attn_drop)

        self.score = nn.Linear(len(self.pool_size), 1)

        self.sigmoid = nn.Sigmoid()
        self.gumbel = MyGumbelSigmoid(0.5, 0.998)
    
    def forward(self,x):
        B, H, W, D = x.size()
        # x = x.permute(0, 3, 1, 2)
        x = self.qk(x.view(B, -1, D)).reshape(B, H, W, 2, D).permute(3, 0, 4, 1, 2) 
        q, k = x.unbind(0)  # B, D, H, W
        for i in range(len(self.pool_size)):
            kernel_size = (math.ceil(H / self.pool_size[i][0]), math.ceil(W / self.pool_size[i][1]))
            stride = (math.ceil(H / self.pool_size[i][0]), math.ceil(W / self.pool_size[i][1]))
            padding = (math.floor((kernel_size[0] * self.pool_size[i][0] - H + 1)/2), math.floor((kernel_size[1] * self.pool_size[i][1] - W + 1)/2))

            referee_conv_q = F.avg_pool2d(q, kernel_size=kernel_size, stride=stride, padding=padding)
            referee_conv_k = F.avg_pool2d(k, kernel_size=kernel_size, stride=stride, padding=padding)
            referee_conv_q = referee_conv_q.repeat_interleave(kernel_size[0],dim=2).repeat_interleave(kernel_size[1],dim=3).unsqueeze(0)
            referee_conv_k = referee_conv_k.repeat_interleave(kernel_size[0],dim=2).repeat_interleave(kernel_size[1],dim=3).unsqueeze(0)

            if i == 0:
                referee_q = torch.cat((q.unsqueeze(0), referee_conv_q), 0)
                referee_k = torch.cat((k.unsqueeze(0), referee_conv_k), 0)
            else:
                referee_q = torch.cat((referee_q, referee_conv_q), 0) #3, B, Region_Num + 1, H, W, D
                referee_k = torch.cat((referee_k, referee_conv_k), 0) 

        referee_q = referee_q.permute(1, 3, 4, 0, 2).reshape(B*H*W, -1, D)
        referee_k = referee_k.permute(1, 3, 4, 0, 2).reshape(B*H*W, -1, D)
        # q, k = referee.unbind(0)
        attn = (referee_q @ referee_k.transpose(-2, -1)) * self.scale #(B*H*W, R, R)
        #attn = attn[:,0,:].sum(-1).reshape(B, H*W)
        attn = attn[:, 0, 1:]
        attn = self.score(attn).reshape(B, H*W, 1)
        attn_soft = self.sigmoid(attn)
        mask = self.gumbel(attn_soft)
                
        return mask


class MultiAttention(nn.Module):
    def __init__(self,max_T,alpha,dim,qkv_bias,attn_drop,region_kernel=[(5,10),(3,6),(2,4)],):
        super().__init__()

        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.region_kernel = region_kernel
  
        # self.conv0 = nn.Conv2d(dim, dim, region_kernel[0], region_kernel[0])
        # self.conv1 = nn.Conv2d(dim, dim, region_kernel[1], region_kernel[1])
        # self.conv2 = nn.Conv2d(dim, dim, region_kernel[2], region_kernel[2])
        
        self.conv0 = nn.AvgPool2d(kernel_size=region_kernel[0], stride=region_kernel[0])
        self.conv1 = nn.AvgPool2d(kernel_size=region_kernel[1], stride=region_kernel[1])
        self.conv2 = nn.AvgPool2d(kernel_size=region_kernel[2], stride=region_kernel[2])
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.score = nn.Linear(len(region_kernel), 1)


        self.sigmoid = nn.Sigmoid()
        self.gumbel = MyGumbelSigmoid(max_T,alpha)
    
    def forward(self,x):
        #B, H, W, D = x.size()
        x = x.permute(0, 3, 1, 2)
        # x = self.qk(x.view(B, -1, D)).reshape(B, H, W, 2, D).permute(3, 0, 4, 1, 2) 
        # q, k = x.unbind(0)  # B, D, H, W
        x_conv0 = self.conv0(x).repeat_interleave(self.region_kernel[0][0],dim=2).repeat_interleave(self.region_kernel[0][1],dim=3).unsqueeze(1)
        x_conv1 = self.conv1(x).repeat_interleave(self.region_kernel[1][0],dim=2).repeat_interleave(self.region_kernel[1][1],dim=3).unsqueeze(1)
        x_conv2 = self.conv2(x).repeat_interleave(self.region_kernel[2][0],dim=2).repeat_interleave(self.region_kernel[2][1],dim=3).unsqueeze(1)
        x_conv = torch.cat((x_conv0, x_conv1, x_conv2),dim=1)
        #x_multi = torch.cat((x.unsqueeze(1), x_conv),dim=1).permute(0, 1, 3, 4, 2)
        B, N, D, H, W = x_conv.size()
        q = self.q(x.reshape(B, D, H*W).permute(0, 2, 1)).reshape(B*H*W, 1, D) # B*L ,1, D
        k = self.k(x_conv.view(B*N, -1, D)).reshape(B, N, H*W, D).permute(0, 2, 1, 3).reshape(B*H*W, N, D) #B*L, N, D
        #qk = self.qk(x_multi.view(B*N, -1, D)).reshape(B, N, H, W, 2, D).permute(4, 0, 2, 3, 1, 5) 
        #q, k = qk.reshape(2, B*H*W, N, D).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale #(B*H*W, N, N)
        attn = attn.reshape(B, H, W, -1)
        # attn = attn[:, 0, 1:].reshape(B, H*W, -1).softmax(1)
        for i in range(N):
            attn_softmax = F.unfold(attn[:,:,:,i].unsqueeze(1),self.region_kernel[i],stride=self.region_kernel[i]).softmax(1)
            # min, max = torch.min(attn_softmax,dim=1,keepdim=True)[0], torch.max(attn_softmax,dim=1,keepdim=True)[0]
            # attn_softmax = (attn_softmax - min) / (max - min)
            attn_softmax = F.fold(attn_softmax,(H, W),self.region_kernel[i],stride=self.region_kernel[i])
            attn[:,:,:,i] = attn_softmax.squeeze(1)
        
        attn = attn.reshape(B, -1, N)
        # attn = self.alpha * attn
        # attn = torch.sum(attn, dim=-1, keepdim=True)
        attn = self.score(attn)
        attn_soft = self.sigmoid(attn)
        mask = self.gumbel(attn_soft)
                
        return mask