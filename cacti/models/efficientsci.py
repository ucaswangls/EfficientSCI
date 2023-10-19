from torch import nn 
import torch 
import einops
from .builder import MODELS

class TimesAttention3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
      
        self.qkv = nn.Linear(dim, (dim//2) * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        C = C//2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class CFormerBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.scb = nn.Sequential(
            nn.Conv3d(dim, dim, (1,3,3), padding=(0,1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3), padding=(0,1,1)),
        )
        self.tsab = TimesAttention3D(dim,num_heads=4)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim,dim,1)
        )
    def forward(self,x):
        _,_,_,h,w = x.shape
        scb_out = self.scb(x)
        tsab_in = einops.rearrange(x,"b c d h w->(b h w) d c")
        tsab_out = self.tsab(tsab_in)
        tsab_out = einops.rearrange(tsab_out,"(b h w) d c->b c d h w",h =h,w=w)
        ffn_in = scb_out+tsab_out+x
        ffn_out = self.ffn(ffn_in)+ffn_in
        return ffn_out

class ResDNetBlock(nn.Module):
    def __init__(self,dim,group_num):
        super().__init__()
        self.cformer_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim//group_num
        self.dense_conv = nn.ModuleList()
        for i in range(group_num):
            self.cformer_list.append(CFormerBlock(group_dim))
            if i > 0:
                self.dense_conv.append(
                    nn.Sequential(
                        nn.Conv3d(group_dim*(i+1),group_dim,1),
                        nn.LeakyReLU(inplace=True)
                    )
                )
        self.last_conv = nn.Conv3d(dim,dim,1)

    def forward(self, x):
        input_list = torch.chunk(x,chunks=self.group_num,dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.cformer_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1,self.group_num):
            in_list = out_list.copy()
            in_list.append(input_list[i])
            cf_in = torch.cat(in_list,dim=1)
            cf_in = self.dense_conv[i-1](cf_in)
            cf_out = self.cformer_list[i](cf_in)
            out_list.append(cf_out)
        out = torch.cat(out_list,dim=1)
        out = self.last_conv(out)
        out = x + out
        return out

@MODELS.register_module 
class EfficientSCI(nn.Module):
    def __init__(self,in_ch=64, units=8,group_num=4,color_ch=1):
        super().__init__()
        self.color_ch = color_ch
        self.fem = nn.Sequential(
            nn.Conv3d(1, in_ch, kernel_size=(3,7,7), stride=1,padding=(1,3,3)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch*2, in_ch*4, kernel_size=3, stride=(1,2,2), padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up_conv = nn.Conv3d(in_ch*4,in_ch*8,1,1)
        self.up = nn.PixelShuffle(2)
        self.vrm = nn.Sequential(
            nn.Conv3d(in_ch*2, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch*2, in_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, color_ch, kernel_size=3, stride=1, padding=1),
        )
        self.resdnet_list = nn.ModuleList()
        for i in range(units):
            self.resdnet_list.append(ResDNetBlock(in_ch*4,group_num=group_num))

    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,1,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,:,ba[0]::2,ba[1]::2]
        y_bayer = einops.rearrange(y_bayer,"b f h w ba->(b ba) f h w")
        Phi_bayer = einops.rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer,"b f h w ba->(b ba) f h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = einops.rearrange(x,"(b ba) f h w->b f h w ba",b=b)
        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)): 
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x
    def forward(self, y,Phi,Phi_s):
        out_list = []
        if self.color_ch==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            # meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            x = x.unsqueeze(1)
    
        out = self.fem(x)
        for resdnet in self.resdnet_list:
            out = resdnet(out)

        out = self.up_conv(out)
        out = einops.rearrange(out,"b c t h w-> b t c h w")
        out = self.up(out)
        out = einops.rearrange(out,"b t c h w-> b c t h w")
        out = self.vrm(out)

        if self.color_ch!=3:
            out = out.squeeze(1)
        out_list.append(out)
        return out_list
