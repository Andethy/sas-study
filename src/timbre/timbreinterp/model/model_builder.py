from .encoder import Encoder
from .decoder import Decoder

import torch
import torch.nn as nn


class VanillaAE(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.encoder = Encoder(model_configs["encoder"])
        self.decoder = Decoder(model_configs["decoder"])

    
    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)

        return out