"""
GANomaly v2 — Improved architecture for DR screening.
Changes from v1:
  - 128×128 resolution (was 64×64)
  - Self-Attention at 16×16 feature maps
  - Residual blocks in encoder
  - Spectral Normalization in discriminator
  - Deeper encoder/decoder (5 downsample layers + final conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ── Building Blocks ──────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Self-attention mechanism for spatial feature refinement."""
    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 8, 1)
        self.query = nn.Conv2d(ch, mid, 1, bias=False)
        self.key   = nn.Conv2d(ch, mid, 1, bias=False)
        self.value = nn.Conv2d(ch, ch,  1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x).view(B, -1, N)               # B × C' × N
        k = self.key(x).view(B, -1, N)                  # B × C' × N
        v = self.value(x).view(B, -1, N)                # B × C  × N

        attn = torch.bmm(q.permute(0, 2, 1), k)         # B × N × N
        attn = F.softmax(attn / (q.size(1) ** 0.5), dim=-1)
        out  = torch.bmm(v, attn.permute(0, 2, 1))      # B × C × N
        out  = out.view(B, C, H, W)
        return self.gamma * out + x


class ResidualDownBlock(nn.Module):
    """Conv block with residual connection and optional downsampling."""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Shortcut: match channels and spatial dims
        if in_ch != out_ch or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


# ── Encoder ──────────────────────────────────────────────────────

class Encoder(nn.Module):
    """128×128 → latent_dim×1×1 with residual blocks and self-attention.

    Path: 128 → 64 → 32 → 16(attn) → 8 → 4 → 1
    Channels: 3 → nf → 2nf → 4nf → 4nf(attn) → 8nf → 8nf → latent
    """
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            # Block 1: 128→64, first block no BN on input
            nn.Conv2d(ch, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 64→32 (residual)
            ResidualDownBlock(nf, nf * 2, downsample=True),

            # Block 3: 32→16 (residual)
            ResidualDownBlock(nf * 2, nf * 4, downsample=True),

            # Self-Attention at 16×16
            SelfAttention(nf * 4),

            # Block 4: 16→8 (residual)
            ResidualDownBlock(nf * 4, nf * 8, downsample=True),

            # Block 5: 8→4 (residual)
            ResidualDownBlock(nf * 8, nf * 8, downsample=True),

            # Final: 4→1
            nn.Conv2d(nf * 8, latent_dim, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x)


# ── Decoder ──────────────────────────────────────────────────────

class Decoder(nn.Module):
    """latent_dim×1×1 → 128×128 image.

    Path: 1 → 4 → 8 → 16 → 32 → 64 → 128
    """
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 1→4
            nn.ConvTranspose2d(latent_dim, nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),

            # 4→8
            nn.ConvTranspose2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),

            # 8→16
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),

            # 16→32
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),

            # 32→64
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf), nn.ReLU(True),

            # 64→128
            nn.ConvTranspose2d(nf, ch, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)


# ── Generator (Enc1 → Dec → Enc2) ───────────────────────────────

class Generator(nn.Module):
    """GANomaly generator: Encoder1 → Decoder → Encoder2.
    Returns (reconstructed_image, z_enc1, z_enc2).
    """
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.encoder1 = Encoder(ch, latent_dim, nf)
        self.decoder  = Decoder(ch, latent_dim, nf)
        self.encoder2 = Encoder(ch, latent_dim, nf)

    def forward(self, x):
        z     = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


# ── Discriminator with Spectral Normalization ────────────────────

def _sn_conv(in_ch, out_ch, k=4, s=2, p=1):
    """Spectrally-normalized conv layer."""
    return spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p, bias=False))


class Discriminator(nn.Module):
    """PatchGAN-style discriminator with spectral normalization.
    Returns (real/fake probability, intermediate features for feature matching).
    128×128 → 4×4 feature map → scalar.
    """
    def __init__(self, ch=3, nf=64):
        super().__init__()
        self.features = nn.Sequential(
            # 128→64
            _sn_conv(ch, nf),
            nn.LeakyReLU(0.2, inplace=True),

            # 64→32
            _sn_conv(nf, nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32→16
            _sn_conv(nf * 2, nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16→8
            _sn_conv(nf * 4, nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8→4
            _sn_conv(nf * 8, nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = spectral_norm(nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False))

    def forward(self, x):
        feat = self.features(x)
        pred = self.classifier(feat).view(-1)
        return pred, feat
