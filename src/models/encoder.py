from functools import partial
import torch
from torch import nn
from models.vdm_unet import get_timestep_embedding
from diffusers.models.vae import Encoder as E


class Encoder(nn.Module):
    def __init__(self, image_shape, cfg):
        super().__init__()
        self._image_shape = image_shape
        self._cfg = cfg
        in_channels, h, w = image_shape
        self.model = E(
            in_channels=in_channels,
            out_channels=1,
            block_out_channels=(cfg.block_out_channels,),
            layers_per_block=cfg.layers_per_block,
            norm_num_groups=cfg.norm_num_groups,
            double_z=False,
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(h * w, 2 * cfg.w_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
    def clone_with_random_weights(self):
        return Encoder(self._image_shape, self._cfg)


class EncoderTime(nn.Module):
    def __init__(self, w_dim, t_embedding_dim) -> None:
        """Input: (28x28x1, 1)"""
        super().__init__()
        # encode x
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        x_embedding_dim = 3 * 3 * 32

        # encode t
        self.t_embed = partial(get_timestep_embedding, embedding_dim=t_embedding_dim)
        self.embed_conditioning = nn.Sequential(
            nn.Linear(t_embedding_dim, t_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(t_embedding_dim * 4, t_embedding_dim * 4),
            nn.SiLU(),
        )

        # combine x and t into z
        self.lin = nn.Sequential(
            nn.Linear(x_embedding_dim + t_embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * w_dim),
        )

    def forward(self, x, t):
        x = self.cnn(x)
        x = self.flatten(x)

        t = self.t_embed(t)
        t = self.embed_conditioning(t)
        x = torch.cat((x, t), dim=1)

        x = self.lin(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = torch.exp(log_std)
        return mean, std
