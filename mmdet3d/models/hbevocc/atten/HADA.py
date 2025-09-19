import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
import einops
from timm.models.layers import to_2tuple, trunc_normal_
import math
    

class HeightAwareDeformableAttention(nn.Module):

    def __init__(
        self, q_size, q_in_dim, in_dim, dim, out_dim, n_heads, points,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe,
        no_off, ksize
    ):

        super().__init__()
        self.n_heads = n_heads
        self.points = points
        self.q_h, self.q_w = q_size
        self.q_in_dim = q_in_dim
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.scale = self.dim ** -0.5
        self.nc = in_dim * n_heads
        self.use_pe = use_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.q_in_dim, self.q_in_dim, kk, stride, pad_size),
            nn.GroupNorm(1, self.q_in_dim),
            nn.ReLU(),
            nn.Conv2d(self.q_in_dim, 2 * self.n_heads*self.points, 1, 1, 0, groups=self.n_heads, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.q_in_dim, self.n_heads * self.dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_k = nn.Conv2d(
            self.in_dim, self.n_heads*self.dim,
            kernel_size=1, stride=1, padding=0, groups=self.n_heads, bias=False
        )

        self.proj_v = nn.Conv2d(
            self.in_dim, self.n_heads*self.dim,
            kernel_size=1, stride=1, padding=0, groups=self.n_heads, bias=False
        )
        
        self.proj_x = nn.Conv2d(
            self.in_dim, self.n_heads*self.dim,
            kernel_size=1, stride=1, padding=0, bias=False
        )

        self.proj_out = nn.Sequential(nn.Conv2d(
            self.n_heads * self.dim, self.out_dim,
            kernel_size=1, stride=1, padding=0, groups=self.n_heads
        ))
        
        self.att_points = 2
        self.proj_k_h = nn.Conv2d(
            self.in_dim, self.n_heads*self.att_points * self.dim,
            kernel_size=1, stride=1, padding=0, groups=1, bias=False
        )

        self.proj_v_h = nn.Conv2d(
            self.in_dim, self.n_heads*self.att_points * self.dim,
            kernel_size=1, stride=1, padding=0, groups=1, bias=False
        )
        radio = 1
        self.ca_xh = ChannelAttention(channel=self.n_heads*self.dim, radio=radio, groups=self.att_points)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm1 = nn.GroupNorm(self.n_heads, self.out_dim)
        self.norm2 = nn.GroupNorm(1, self.out_dim)
        self.ffn = nn.Sequential(nn.Conv2d(self.out_dim, self.out_dim*2,kernel_size=1, stride=1, padding=0, groups=self.n_heads),
                                 nn.ReLU(),
                                 nn.Dropout(proj_drop),
                                 nn.Conv2d(self.out_dim*2, self.out_dim,kernel_size=1, stride=1, padding=0))
        
        max_len = self.q_h * self.q_w
        pe = torch.zeros(max_len, in_dim)  # max_len 是解码器生成句子的最长的长度，假设是 10
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, in_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / in_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1).reshape(1, in_dim, self.q_h, self.q_w)
        self.pe = nn.Parameter(pe, requires_grad=False)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1) # B H W 2

        return ref
    
    def forward(self, q, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        residual1 = x

        q = q + self.pe
        offset = self.conv_offset(q).contiguous()  # B * g 2 Hg Wg
        q = self.proj_q(q)
        offset = einops.rearrange(offset, 'b (n c) h w -> (b n) c h w', n=self.n_heads*self.points, c=2)
        n_sample = H * W

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (H - 1.0), 1.0 / (W - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b c h w -> b h w c')
        reference = self._get_ref_points(H, W, B * self.n_heads*self.points, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)
        pos = einops.rearrange(pos, '(b n) h w c -> b (n h) w c', b=B, n=self.n_heads*self.points)

        x = x + self.pe
        x = self.proj_x(x)
        x_h = x + self.ca_xh(x)
        kh = self.proj_k_h(x_h)
        kh = kh.reshape(self.n_heads*B, self.att_points*self.dim, H, W)
        kh = kh.reshape(self.n_heads*B, self.att_points*self.dim, n_sample).reshape(self.n_heads*B, self.att_points, self.dim, n_sample)
        vh = self.proj_v_h(x_h)
        vh = vh.reshape(self.n_heads*B, self.att_points*self.dim, H, W)
        vh = vh.reshape(self.n_heads*B, self.att_points*self.dim, n_sample).reshape(self.n_heads*B, self.att_points, self.dim, n_sample)
        x_sampled_list = []
        for i in range(self.n_heads):
            x_sampled = F.grid_sample(input=x[:,i*self.dim:(i+1)*self.dim,:,:], 
                grid=pos[:,i*self.points*H:(i+1)*self.points*H,:,:][..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            x_sampled_list.append(x_sampled)
        x_sampled = torch.cat(x_sampled_list, dim=1)
        x_sampled = einops.rearrange(x_sampled, 'b c (n h) w -> (b n) c h w', n=self.points, h=H)

        q = q.reshape(B * self.n_heads, self.dim, n_sample)
        k = self.proj_k(x_sampled).reshape(B*self.points, self.n_heads*self.dim, n_sample).reshape(B, self.points, self.n_heads*self.dim, n_sample)
        k = einops.rearrange(k, 'b p (h c) m -> (b h) p c m', h=self.n_heads, c=self.dim)
        k = torch.cat([k,kh], dim=1)
        v = self.proj_v(x_sampled).reshape(B*self.points, self.n_heads*self.dim, n_sample).reshape(B, self.points, self.n_heads*self.dim, n_sample)
        v = einops.rearrange(v, 'b p (h c) m -> (b h) p c m', h=self.n_heads, c=self.dim)
        v = torch.cat([v,vh],dim=1)

        attn = torch.einsum('b c m, b p c m -> b p m', q, k) # B*n_head 1 n_sample
        attn = attn.mul(self.scale)

        attn = F.softmax(attn, dim=1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b p m, b p c m -> b c m', attn, v)
        out = out.reshape(B, self.n_heads*self.dim, H, W)
        out = self.proj_out(out)
        out = residual1 + out
        out = self.norm1(out)
        residual2 = out
        
        y = self.ffn(out)
        y = y + residual2
        y = self.norm2(y)

        return y
    
class ChannelAttention(nn.Module):
    def __init__(self,channel,radio=1,groups=8):
        super().__init__()
        self.g = groups
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel*radio,1,bias=False),
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        b,c,h,w = avg_result.shape
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output*x