import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficentLePE(nn.Module):
    def __init__(self, dim, res, idx, split_size=7, num_heads=8, qk_scale=None) -> None:
        super().__init__()
        self.res = res
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if idx == -1:
            H_sp, W_sp = self.res, self.res
        elif idx == 0:
            H_sp, W_sp = self.res, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.res, self.split_size
        else:
            print("ERROR MODE : ",idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.get_v = nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim)

    def im2cswin(self,x):
        B,L,C = x.shape
        H = W = int(np.sqrt(L))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x
    
    def forward(self,qkv):
        q,k,v = qkv[0],qkv[1],qkv[2]
        B,L,C = q.shape
        H = W = self.res
        assert(L == H*W)
        k = self.im2cswin(k)
        v = self.im2cswin(v)

        # get lepe start
        q = q.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        q = q.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = self.get_v(q) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        q = q.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        # get lepe end

        att = k.transpose(-2,-1)@v * self.scale
        att = nn.functional.softmax(att,dim=-1,dtype=att.dtype)

        x = q@att +lepe
        x = x.transpose(1,2).reshape(-1,self.H_sp*self.W_sp,C)

        x = windows2img(x,self.H_sp,self.W_sp,H,W).view(B,-1,C)
        return x

class CSWinAttention(nn.Module):
    def __init__(self, dim, res, num_heads, split_size,
                 qkv_bias=False, qk_scale=None, switch=False,
                 norm_layer=nn.LayerNorm,last_stage=False) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_res = res
        self.split_size = split_size
        self.to_qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.norm = norm_layer(dim)
        self.switch = switch

        if self.patch_res == split_size:
            last_stage = True
        
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        if last_stage:
            self.attns = nn.ModuleList([
                EfficentLePE(dim, res, -1, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])
        else:
            self.attns = nn.ModuleList([
                EfficentLePE(dim//2, res, i, split_size, num_heads, qk_scale)
            for i in range(self.branch_num)
            ])

    def forward(self,x):
        """
        Args:
            x: B H*W C
        Returns:
            x: B H*W C
        """
        H = W = self.patch_res
        B,L,C = x.shape
        assert(H*W == L)
        x = self.norm(x)

        qkv = self.to_qkv(x).reshape(B,-1,3,C).permute(2,0,1,3)
        if self.branch_num == 2:
            if self.switch:
                x1 = self.attns[0](qkv[:,:,:,:C//2])
                x2 = self.attns[1](qkv[:,:,:,C//2:])
            else:
                x1 = self.attns[0](qkv[:,:,:,C//2:])
                x2 = self.attns[1](qkv[:,:,:,:C//2])
            att = torch.cat([x1,x2],dim=2)
        else:
            att = self.attns[0](qkv)
        
        return att

class TokenMixer(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2*dim)
        self.norm2 = nn.LayerNorm(2*dim)
        self.mixer1 = nn.Conv1d(2*dim,dim,1,1)
        self.mixer2 = nn.Conv1d(2*dim,dim,1,1)
    
    def forward(self,att1,x1,att2,x2):
        m1 = self.norm1(torch.cat([att1,x2],dim=-1)).transpose(-1,-2)
        m1 = self.mixer1(m1).transpose(-1,-2) + att1
        m2 = self.norm2(torch.cat([att2,x1],dim=-1)).transpose(-1,-2)
        m2 = self.mixer2(m2).transpose(-1,-2) + att2

        return m1,m2

class ChannelMixer(nn.Module):
    def __init__(self,dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(2*dim,dim)
        self.mixer1 = nn.Linear(2*dim,dim)
        self.norm2 = nn.LayerNorm(2*dim,dim)
        self.mixer2 = nn.Linear(2*dim,dim)

    def forward(self,t1,att1,t2,att2):
        c1 = self.norm1(torch.cat([t1,att2],dim=-1))
        c1 = self.mixer1(c1) + t1
        c2 = self.norm2(torch.cat([t2,att1],dim=-1))
        c2 = self.mixer2(c2) + t2

        return c1,c2


class DualAttentionFusionBlock(nn.Module):
    def __init__(self,dim, res, split_size_h,split_size_l,
                 num_heads_h, num_heads_l, qkv_bias=False, qk_scale=None, act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,last_stage=False) -> None:
        super().__init__()
        self.block_h = CSWinAttention(dim,res,num_heads_h,split_size_h,qkv_bias,qk_scale,False,norm_layer,last_stage)
        self.block_l = CSWinAttention(dim,res,num_heads_l,split_size_l,qkv_bias,qk_scale,True,norm_layer,last_stage)
        self.token_mixer = TokenMixer(dim)
        self.channel_mixer = ChannelMixer(dim)
        self.last_stage = last_stage
        self.res = res
        if last_stage:
            self.merge = nn.Sequential(
                Rearrange("b (h w) c -> b c h w", h=res,w=res),
                nn.Conv2d(2*dim,dim,3,1,2),
                Rearrange("b c h w -> b (h w) c"),
                nn.LayerNorm(dim)
            )
    def forward(self,x1, x2):
        """
        Args:
            x_h: B L C (high resolution features)
            x_l: B L C (low resolution features)
        Returns:
            x_h: B L C
            x_l: B L C 
        or 
            x : B L C (if last stage)
        """
        # print(self.res**2, x_h.shape[1])
        assert(self.res*self.res == x1.shape[1]) # H*W == L
        assert(self.res*self.res == x2.shape[1]) # H*W == L
        att1 = self.block_h(x1)
        att2 = self.block_l(x2)
        t1,t2 = self.token_mixer(att1,x1,att2,x2)
        x1,x2 = self.channel_mixer(t1,att1,t2,att2)
        
        if self.last_stage:
            x = torch.cat([x1,x2], dim=-1)
            x = self.merge(x)
            return x
        return x1, x2

class SymmetricFusionPatchMerge(nn.Module):
    def __init__(self, dim_in, res) -> None:
        super().__init__()
        dim_out = 2*dim_in
        self.conv_mixer = nn.Conv2d(dim_in,dim_out,3,1,1)
        self.norm = nn.LayerNorm([dim_out,res,res])
        self.conv_l = nn.Conv2d(dim_out,dim_in,3,2,1)
        self.act_l = nn.GELU()
        self.conv_h = nn.Conv2d(dim_out,dim_in,3,2,1)
        self.act_h  = nn.GELU()
    
    def forward(self,x):
        """
        Args:
            x : B C H W
        Returns:
            x_h,x_l = B 2C H/2, W/2
        """
        x = self.conv_mixer(x)
        x = self.norm(x)
        x_l = self.conv_l(x)
        x_l = self.act_l(x_l)
        x = self.conv_h(x)
        x = self.act_h(x)

        return x_l,x
    

