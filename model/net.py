import torch
import torch.nn as nn
import numpy as np
from .Blocks import CSWinBlock, PatchMerging, TIF, Up_conv, PatchEmbedLarge, PatchEmbedSmall
from thop import profile, clever_format
from einops import rearrange
from mmcv.cnn import ConvModule

"""
-PatchEmbed
-Subnet
--Encoder
---EncoderStage
----EncoderAttnPart
-----CSWinBlock
----EncoderMerge
----EncoderFusion
-----LE
-----TIF
--Decoder
---DecoderStage
----DecoderUpSample
----DecoderLinear
----DecoderCat
-LinearPredictingLayer
"""


class EncoderAttnPart(nn.Module):
    def __init__(self,
                 curr_small_dim=64,
                 curr_large_dim=32,
                 num_heads_l=[4, 8, 16, 32],
                 num_heads_s=[6, 12, 24, 48],
                 depth_s=[2, 2, 18, 2],
                 depth_l=[2, 2, 6, 2],
                 stage_num=1,
                 patches_resolution_l=224 // 2,
                 patches_resolution_s=224 // 4,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 split_size=[1, 2, 2, 7]
                 ):
        super(EncoderAttnPart, self).__init__()

        self.large_scale = nn.ModuleList([
            CSWinBlock(dim=curr_large_dim,
                       num_heads=num_heads_l[stage_num - 1],
                       patches_resolution=patches_resolution_l,
                       mlp_ratio=mlp_ratio,
                       drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       split_size=split_size[stage_num - 1]
                       )
            for i in range(depth_l[stage_num - 1])]
        )

        self.small_scale = nn.ModuleList([
            CSWinBlock(dim=curr_small_dim,
                       num_heads=num_heads_s[stage_num - 1],
                       patches_resolution=patches_resolution_s,
                       mlp_ratio=mlp_ratio,
                       drop=drop_rate,
                       attn_drop=attn_drop_rate,
                       split_size=split_size[stage_num - 1]
                       )
            for i in range(depth_s[stage_num - 1])],
        )

    def forward(self, x_l, x_s):

        for f in self.large_scale:
            x_l = f(x_l)

        for f in self.small_scale:
            x_s = f(x_s)

        return x_l, x_s


class EncoderMerge(nn.Module):
    def __init__(self,
                 curr_large_dim=24,
                 curr_small_dim=12,
                 patches_resolution_l=224 // 4,
                 patches_resolution_s=224 // 8
                 ):
        super(EncoderMerge, self).__init__()
        self.large_scale_merge = PatchMerging(dim=curr_large_dim, res=patches_resolution_l)
        self.small_scale_merge = PatchMerging(dim=curr_small_dim, res=patches_resolution_s)

    def forward(self, x_l, x_s):
        x_l = self.large_scale_merge(x_l)
        x_s = self.small_scale_merge(x_s)
        return x_l, x_s


class EncoderFusion(nn.Module):
    def __init__(self,
                 curr_small_dim=64,
                 curr_large_dim=32,
                 ):
        super(EncoderFusion, self).__init__()
        self.local_emphasis_l_conv = nn.Conv2d(curr_large_dim, curr_large_dim, 3, padding=1, bias=False)
        self.local_emphasis_l_norm = nn.LayerNorm(curr_large_dim)
        self.local_emphasis_s_conv = nn.Conv2d(curr_small_dim, curr_small_dim, 3, padding=1, bias=False)
        self.local_emphasis_s_norm = nn.LayerNorm(curr_small_dim)
        self.act = nn.GELU()
        self.tif = TIF(dim_s=curr_small_dim, dim_l=curr_large_dim)

    def forward(self, x_l, x_s):
        Bl, Ll, Cl = x_l.shape
        Hl = Wl = int(np.sqrt(Ll))
        x_l = rearrange(x_l, 'b (h w) c -> b c h w', h=Hl, w=Wl)  # (B, L, C) -> (B, C, H, W)

        x_l = self.local_emphasis_l_conv(x_l)
        x_l = x_l.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x_l = self.local_emphasis_l_norm(x_l)
        x_l = x_l.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x_l = self.local_emphasis_l_conv(x_l)
        x_l = self.act(x_l)
        x_l = self.local_emphasis_l_conv(x_l)

        Bs, Ls, Cs = x_s.shape
        Hs = Ws = int(np.sqrt(Ls))
        x_s = rearrange(x_s, 'b (h w) c -> b c h w', h=Hs, w=Ws)

        x_s = self.local_emphasis_s_conv(x_s)
        x_s = x_s.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x_s = self.local_emphasis_s_norm(x_s)
        x_s = x_s.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x_s = self.local_emphasis_s_conv(x_s)
        x_s = self.act(x_s)
        x_s = self.local_emphasis_s_conv(x_s)

        x = self.tif(x_l, x_s)
        return x


class EncoderStage(nn.Module):

    def __init__(self,
                 curr_small_dim=64,
                 curr_large_dim=32,
                 num_heads_l=[4, 8, 16, 32],
                 num_heads_s=[4, 8, 16, 32],
                 depth_s=[2, 2, 18, 2],
                 depth_l=[2, 2, 6, 2],
                 stage_num=0,
                 patches_resolution_l=224 // 2,
                 patches_resolution_s=224 // 4,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 split_size=[1, 2, 2, 7]
                 ):
        super().__init__()
        self.att_stage = EncoderAttnPart(curr_small_dim, curr_large_dim, num_heads_l, num_heads_s, depth_s, depth_l,
                                         stage_num, patches_resolution_l, patches_resolution_s, mlp_ratio, drop_rate,
                                         attn_drop_rate, split_size)
        self.fusion_stage = EncoderFusion(curr_small_dim=curr_small_dim, curr_large_dim=curr_large_dim)
        self.merge = EncoderMerge(curr_large_dim=curr_large_dim, curr_small_dim=curr_small_dim,
                                  patches_resolution_l=patches_resolution_l, patches_resolution_s=patches_resolution_s)

    def forward(self, x_l, x_s):
        x_l, x_s = self.att_stage(x_l, x_s)
        x_f = self.fusion_stage(x_l, x_s)
        x_l, x_s = self.merge(x_l, x_s)
        return x_l, x_s, x_f


class Encoder(nn.Module):
    def __init__(self,
                 curr_small_dim=64,
                 curr_large_dim=32,
                 num_heads_l=[4, 8, 16, 32],
                 num_heads_s=[4, 8, 16, 32],
                 patches_resolution_l=224 // 2,
                 patches_resolution_s=224 // 4,
                 split_size=[1, 2, 2, 7],
                 depth_s=[2, 2, 18, 2],
                 depth_l=[2, 2, 6, 2],
                 ):
        super(Encoder, self).__init__()
        self.encoderStage1 = EncoderStage(curr_small_dim=curr_small_dim,
                                          curr_large_dim=curr_large_dim,
                                          num_heads_l=num_heads_l,
                                          num_heads_s=num_heads_s,
                                          stage_num=1,
                                          patches_resolution_l=patches_resolution_l,
                                          patches_resolution_s=patches_resolution_s,
                                          split_size=split_size, depth_l=depth_l, depth_s=depth_s)
        self.encoderStage2 = EncoderStage(curr_small_dim=curr_small_dim * 2,
                                          curr_large_dim=curr_large_dim * 2,
                                          num_heads_l=num_heads_l,
                                          num_heads_s=num_heads_s,
                                          stage_num=2,
                                          patches_resolution_l=patches_resolution_l // 2,
                                          patches_resolution_s=patches_resolution_s // 2,
                                          split_size=split_size, depth_l=depth_l, depth_s=depth_s)
        self.encoderStage3 = EncoderStage(curr_small_dim=curr_small_dim * 4,
                                          curr_large_dim=curr_large_dim * 4,
                                          num_heads_l=num_heads_l,
                                          num_heads_s=num_heads_s,
                                          stage_num=3,
                                          patches_resolution_l=patches_resolution_l // 4,
                                          patches_resolution_s=patches_resolution_s // 4, split_size=split_size,
                                          depth_l=depth_l, depth_s=depth_s)
        self.encoderStage4 = EncoderStage(curr_small_dim=curr_small_dim * 8,
                                          curr_large_dim=curr_large_dim * 8,
                                          num_heads_l=num_heads_l,
                                          num_heads_s=num_heads_s,
                                          stage_num=4,
                                          patches_resolution_l=patches_resolution_l // 8,
                                          patches_resolution_s=patches_resolution_s // 8, split_size=split_size,
                                          depth_l=depth_l, depth_s=depth_s)

    def forward(self, x_l, x_s):
        x_l, x_s, x_f1 = self.encoderStage1(x_l, x_s)

        x_l, x_s, x_f2 = self.encoderStage2(x_l, x_s)

        x_l, x_s, x_f3 = self.encoderStage3(x_l, x_s)

        _, _, x_f4 = self.encoderStage4(x_l, x_s)

        # x_f1 = rearrange(x_f1, 'b c h w -> b (h w) c')
        # x_f2 = rearrange(x_f2, 'b c h w -> b (h w) c')
        # x_f3 = rearrange(x_f3, 'b c h w -> b (h w) c')
        # x_f4 = rearrange(x_f4, 'b c h w -> b (h w) c')
        return x_f1, x_f2, x_f3, x_f4


class Decoder(nn.Module):
    def __init__(self,
                 curr_dim=256,
                 patches_resolution=14,
                 ):
        super(Decoder, self).__init__()
        self.up_4 = Up_conv(in_ch=curr_dim, out_ch=curr_dim // 2)
        self.up_34 = Up_conv(in_ch=curr_dim // 2, out_ch=curr_dim // 4)
        self.up_23 = Up_conv(in_ch=curr_dim // 4, out_ch=curr_dim // 8)
        self.linear_fuse34 = ConvModule(in_channels=curr_dim, out_channels=curr_dim // 2, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse23 = ConvModule(in_channels=curr_dim // 2, out_channels=curr_dim // 4, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse12 = ConvModule(in_channels=curr_dim // 4, out_channels=curr_dim // 8, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))

    def forward(self, x_f4, x_f3, x_f2, x_f1):
        u4 = self.up_4(x_f4)
        f_34 = self.linear_fuse34(torch.cat([u4, x_f3], dim=1))
        u_34 = self.up_34(f_34)
        f_23 = self.linear_fuse23(torch.cat([u_34, x_f2], dim=1))
        up_23 = self.up_23(f_23)
        f_12 = self.linear_fuse12(torch.cat([up_23, x_f1], dim=1))
        return f_12


class SubNet(nn.Module):
    def __init__(self,
                 embed_dim=32,
                 img_size=224,
                 num_heads_l=[4, 8, 16, 32],
                 num_heads_s=[4, 8, 16, 32],
                 depth_s=[2, 2, 18, 2],
                 depth_l=[2, 2, 6, 2],
                 split_size=[1, 2, 2, 7], patch_size=4):
        super(SubNet, self).__init__()
        self.encoder = Encoder(curr_small_dim=embed_dim * 2, curr_large_dim=embed_dim, num_heads_s=num_heads_s,
                               num_heads_l=num_heads_l, depth_l=depth_l, depth_s=depth_s, split_size=split_size,
                               patches_resolution_l=img_size // patch_size,
                               patches_resolution_s=img_size // (patch_size * 2))
        self.decoder = Decoder(curr_dim=embed_dim * 8, patches_resolution=img_size // 16)

    def forward(self, x_l, x_s):
        x_f1, x_f2, x_f3, x_f4 = self.encoder(x_l, x_s)
        x_out = self.decoder(x_f4, x_f3, x_f2, x_f1)
        return x_out


class PatchEmbed(nn.Module):
    def __init__(self, dim_in=1, embed_dim=32, img_size=224, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.patch_embed_large = PatchEmbedLarge(dim_in, embed_dim, patch_resolution=img_size // patch_size)
        self.patch_embed_small = PatchEmbedSmall(dim_in, 2 * embed_dim, patch_resolution=img_size // (patch_size * 2))

    def forward(self, input):
        input_large = self.patch_embed_large(input)
        input_large = rearrange(input_large, 'b c h w -> b (h w) c')
        input_small = self.patch_embed_small(input)
        input_small = rearrange(input_small, 'b c h w -> b (h w) c')
        return input_large, input_small


class DownSamplePart_1(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(DownSamplePart_1, self).__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, image_input):
        down1 = self.down(image_input)
        return down1


class UNet(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_chans=1,
                 embed_dim=32,
                 num_classes=9,
                 num_heads_l=[4, 8, 16, 32],
                 num_heads_s=[8, 16, 32, 64],
                 depth_s=[2, 4, 18, 2],
                 depth_l=[2, 2, 6, 2],
                 split_size=[1, 2, 2, 7],
                 patch_size=4):
        super(UNet, self).__init__()
        self.embed = PatchEmbed(dim_in=in_chans, embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.subnet = SubNet(embed_dim=embed_dim, img_size=img_size, num_heads_s=num_heads_s, num_heads_l=num_heads_l,
                             depth_l=depth_l, depth_s=depth_s, split_size=split_size, patch_size=patch_size)
        self.predict_layer = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, input):
        x_l, x_s = self.embed(input)  # B L C, B L C
        x = self.subnet(x_l, x_s)
        x = self.predict_layer(x)
        up = nn.UpsamplingBilinear2d(scale_factor=4)  # directly 4 * UpSample
        output = up(x)
        return output


if __name__ == '__main__':
    x = torch.randn(2, 1, 256, 256).cuda()

    model = UNet(img_size=256, embed_dim=32, patch_size=4, num_classes=9, in_chans=1).cuda()

    flops, params = profile(model=model, inputs=(x,))
    flops, params = clever_format([flops, params], '%.3f')
    print("flops per sample: ", flops)
    print("model parameters: ", params)
