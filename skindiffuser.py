import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class NoiseScheduler:

    def __init__(self, steps=300):
        self.steps = steps
        self.betas = NoiseScheduler.beta_schedule(steps)
        self.alphas = 1. - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def beta_schedule(steps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, steps)
    
    def get_index_from_list(vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_sample(self, x, t, device="cpu"):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = NoiseScheduler.get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = NoiseScheduler.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        return sqrt_alphas_cumprod_t.to(device) * x.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

class Block(nn.Module):
    
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, down=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        self.up = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) if up else None
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1) if down else None

    def forward(self, x, time):
        h = self.conv1(x)
        h = self.bnorm1(h)
        h += self.time_mlp(time)[:, :, None, None]
        h = self.relu(h)
        h = self.bnorm2(self.conv2(h))
        h = self.relu(h + self.residual_conv(x))

        if self.down:
            h = self.down(h)
        elif self.up:
            h = self.up(h)
        return h

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[: , None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        normed = self.norm(x)
        qkv = self.qkv(normed)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, c, -1).transpose(1, 2) # B, HW, C
        k = k.reshape(b, c, -1) # B, C, HW
        v = v.reshape(b, c, -1).transpose(1, 2) # B, HW, C

        attn = torch.softmax(q @ k / (c ** 0.5), dim=-1) # B, HW, HW
        out = attn @ v # B, HW, C
        out = out.transpose(1, 2).reshape(b, c, h, w) # B, C, H, W
        return self.proj(out + x)

class SkinUnet(nn.Module):

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        mid_channels = 1024
        out_dim = 3
        time_embedding_dim = 64

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_embedding_dim, down=True) for i in range(len(down_channels) - 1)])

        # Upsample
        self.ups = nn.ModuleList([
            Block(up_channels[0] + down_channels[-1], up_channels[1], time_embedding_dim, up=True),
            Block(up_channels[1] + down_channels[-2], up_channels[2], time_embedding_dim, up=True),
            Block(up_channels[2] + down_channels[-3], up_channels[3], time_embedding_dim, up=True),
            Block(up_channels[3] + down_channels[-4], up_channels[4], time_embedding_dim, up=True)
        ])

        # Bottleneck
        self.bottleneck_block1 = Block(down_channels[-1], mid_channels, time_embedding_dim)
        self.bottleneck_block2 = Block(mid_channels, up_channels[0], time_embedding_dim)
        self.selfattention = SelfAttention(mid_channels)


        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, time):
        # Embedd time
        t = self.time_mlp(time)
        # Initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []

        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        # Bottleneck
        x = self.bottleneck_block1(x, t)
        x = self.selfattention(x)
        x = self.bottleneck_block2(x, t)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


