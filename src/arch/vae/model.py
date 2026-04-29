import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .encdec import Motion1Dto2D, Motion2Dto1D, Encoder, Decoder
from src.utils import capture_init_kwargs


class VAE(nn.Module):
    @capture_init_kwargs
    def __init__(
        self, num_joints, pose_dim, width=64, depth=3,
        latent_dim=16, activation='relu', dataset_name='t2m',
        scale_factor=None, shift_factor=None, ch_mult=None,
    ):
        super().__init__()

        self.m_enc = Motion1Dto2D(pose_dim, width, activation, dataset_name)
        self.m_dec = Motion2Dto1D(pose_dim, width, activation, dataset_name)
        self.encoder = Encoder(num_joints=num_joints, width=width, depth=depth, activation=activation, ch_mult=ch_mult)
        self.decoder = Decoder(num_joints=num_joints, width=width, depth=depth, activation=activation, ch_mult=ch_mult)
        self.gaussian = DiagonalGaussian(chunk_dim=-1)

        self.proj_in = nn.Sequential(
            Rearrange('b d t j -> b t j d'),
            nn.Linear(width, latent_dim * 2)
        )
        self.proj_out = nn.Sequential(
            nn.Linear(latent_dim, width),
            Rearrange('b t j d -> b d t j'),
        )

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        x = self.m_enc(x)
        x = self.encoder(x)
        x = self.proj_in(x)

        # latent space
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        if self.scale_factor is not None and self.shift_factor is not None:
            z = (z - self.shift_factor) * self.scale_factor
        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)

        return z, {"loss_kl": loss_kl}

    def decode(self, x):
        if self.scale_factor is not None and self.shift_factor is not None:
            x = x / self.scale_factor + self.shift_factor
        x = self.proj_out(x)
        x = self.decoder(x)
        x = self.m_dec(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, return_latent=False, down=1.0):
        """
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """
        # encode
        x = x.detach().float()
        z_encoded, loss_dict = self.encode(x)
        out = self.decode(z_encoded)

        if return_latent:
            return out, loss_dict, z_encoded
        return out, loss_dict

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = -1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean