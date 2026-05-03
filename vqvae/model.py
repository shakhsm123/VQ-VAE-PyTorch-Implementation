import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderModule
from .quantizer import VectorQuantizer
from .decoder import Decoder


class VQ_VAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder   = EncoderModule(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder   = Decoder(config)

    def forward(self, x):
        z_e              = self.encoder(x)
        z_q, commit_loss, indices = self.quantizer(z_e)
        x_recon          = self.decoder(z_q)
        return x_recon, commit_loss, indices


def vq_vae_loss(x_recon, x, commit_loss):
    recon_loss = F.mse_loss(x_recon, x)
    total_loss = recon_loss + commit_loss
    return total_loss, recon_loss, commit_loss
