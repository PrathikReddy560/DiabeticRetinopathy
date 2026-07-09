"""
GANomaly model architecture.
Generator: Encoder1 → Decoder → Encoder2
Anomaly score = distance between z (from Enc1) and z_hat (from Enc2).
"""

import torch.nn as nn


def _encoder_block(in_ch, out_ch, first=False):
    """Single encoder conv block: Conv → [BatchNorm] → LeakyReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if not first:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def _decoder_block(in_ch, out_ch, last=False):
    """Single decoder conv block: ConvTranspose → BatchNorm → ReLU (or Tanh)."""
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if last:
        layers.append(nn.Tanh())
    else:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(True)]
    return layers


class Encoder(nn.Module):
    """Compresses 64×64 image → latent vector (latent_dim × 1 × 1)."""
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            *_encoder_block(ch, nf, first=True),       # 64→32
            *_encoder_block(nf, nf*2),                  # 32→16
            *_encoder_block(nf*2, nf*4),                # 16→8
            *_encoder_block(nf*4, nf*8),                # 8→4
            nn.Conv2d(nf*8, latent_dim, 4, 1, 0, bias=False),  # 4→1
        )

    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    """Reconstructs latent vector → 64×64 image."""
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, nf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf*8), nn.ReLU(True),       # 1→4
            *_decoder_block(nf*8, nf*4),                # 4→8
            *_decoder_block(nf*4, nf*2),                # 8→16
            *_decoder_block(nf*2, nf),                  # 16→32
            *_decoder_block(nf, ch, last=True),         # 32→64
        )

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """Encoder1 → Decoder → Encoder2. Returns (reconstructed, z, z_hat)."""
    def __init__(self, ch=3, latent_dim=100, nf=64):
        super().__init__()
        self.encoder1 = Encoder(ch, latent_dim, nf)
        self.decoder = Decoder(ch, latent_dim, nf)
        self.encoder2 = Encoder(ch, latent_dim, nf)

    def forward(self, x):
        z = self.encoder1(x)
        x_hat = self.decoder(z)
        z_hat = self.encoder2(x_hat)
        return x_hat, z, z_hat


class Discriminator(nn.Module):
    """Returns (real/fake probability, feature maps) for adversarial + feature matching loss."""
    def __init__(self, ch=3, nf=64):
        super().__init__()
        self.features = nn.Sequential(
            *_encoder_block(ch, nf, first=True),
            *_encoder_block(nf, nf*2),
            *_encoder_block(nf*2, nf*4),
            *_encoder_block(nf*4, nf*8),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(nf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.features(x)
        pred = self.classifier(feat).view(-1)
        return pred, feat
