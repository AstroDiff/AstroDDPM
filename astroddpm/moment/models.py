## Standard imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
import torch
import astroddpm
import tqdm.notebook as tqdm

from astroddpm.diffusion.models.network import DownResBlock, MidResBlock, NormConv2d, sinusoidal_embedding, gaussian_fourier_embedding, SineCosine
from torch.nn import functional as F
from torch import nn
import torch.linalg as la

## Relative imports
from astroddpm.diffusion.stochastic.sde import get_sde, DiscreteSDE, ContinuousSDE
from astroddpm.datahandler.dataset import get_dataset_and_dataloader
from astroddpm.diffusion.power_spectra.powerspec_sampler import get_ps_sampler
from astroddpm.utils.scheduler import get_optimizer_and_scheduler

class TparamMomentNetwork(nn.Module):
    def __init__(self, in_channels, in_size, dim_param, order, num_blocks, first_channels = 10, time_embed_dim = 100,padding_mode="circular", 
        activation=None, normalize="GN", group_c=1, skiprescale = True, discretization = "continuous", embedding_mode = None, n_steps = 1000,dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.first_channels = first_channels
        self.dim_param = dim_param
        self.order = order
        assert self.order <= 2, "Only implemented up to second order moments"
        self.num_blocks = num_blocks
        self.time_embed_dim = time_embed_dim
        self.padding_mode = padding_mode
        self.activation = activation
        self.normalize = normalize
        self.group_c = group_c
        self.skiprescale = skiprescale
        self.discretization = discretization
        self.embedding_mode = embedding_mode
        self.n_steps = n_steps
        self.dropout = dropout

        self.config = {"in_channels": self.in_channels, "in_size": self.in_size, "dim_param": self.dim_param, "order": self.order, "num_blocks": self.num_blocks, 
            "first_channels": self.first_channels, "time_embed_dim": self.time_embed_dim, "padding_mode": self.padding_mode, 
            "activation": self.activation, "normalize": self.normalize, "group_c": self.group_c, "skiprescale": self.skiprescale, "discretization": self.discretization, 
            "embedding_mode": self.embedding_mode, "n_steps": self.n_steps, "dropout": self.dropout}

        if discretization == "discrete":
            self.time_embed = nn.Embedding(n_steps, time_embed_dim)
            self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_embed_dim)
            self.time_embed.requires_grad_(False)
        
        elif discretization == "continuous" or discretization == "default":
            if (embedding_mode is None) or (embedding_mode == "fourier"):
                assert time_embed_dim % 2 == 0, "time_emb_dim must be even for fourier embedding"
                linear1 = nn.Linear(1, time_embed_dim//2)
                linear1.weight.data = gaussian_fourier_embedding(time_embed_dim//2).unsqueeze(1)
                linear1.bias.data = torch.zeros(time_embed_dim//2)
                self.time_embed = nn.Sequential(linear1, SineCosine())
                self.time_embed.requires_grad_(False)
            elif embedding_mode == "forward":
                self.time_embed = nn.Linear(1, time_embed_dim)
        
        self.blocks = nn.ModuleList()
        self.current_channels = self.in_channels
        if self.in_channels != self.first_channels:
            self.head = NormConv2d((self.in_channels, self.in_size, self.in_size),self.in_channels, self.first_channels, kernel_size=1, padding_mode=self.padding_mode, 
                normalize=self.normalize, group_c=self.group_c)
            self.current_channels = self.first_channels
        else:
            self.head = nn.Identity()
        for i in range(self.num_blocks):
            self.blocks.append(DownResBlock(self.in_size, self.current_channels, self.current_channels, padding_mode=self.padding_mode, 
                 normalize=self.normalize, group_c=self.group_c, skiprescale=self.skiprescale))
            
        ## Output dimension will be dim_param ** order
        self.out_dim = self.dim_param ** self.order
        self.fc_tail = nn.Linear(self.current_channels, self.out_dim)

    def forward(self, x, t):
        t = self.time_embed(t)
        x = self.head(x)
        for block in self.blocks:
            x = block(x, t)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=self.dropout)
        x = self.fc_tail(x)

        if self.order == 1:
            return x
        elif self.order == 2:
            x = x.view(x.size(0), self.dim_param, self.dim_param)
            x = (x + x.transpose(1,2))/2 ## Symmetrize
            return x

class ResBlock(nn.Module):
    def __init__(self, size, in_c, out_c, padding_mode="circular", normalize="GN", group_c=1, skiprescale=True, eps_norm=1e-5, dropout=0.0):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            NormConv2d((in_c, size, size), in_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((out_c, size, size), out_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((out_c, size, size), out_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

    def forward(self, x):
        h = self.block(x)
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2)
        else:
            h = h + x
        return h

class SigmaMomentNetwork(nn.Module):
    def __init__(self, in_channels, in_size, dim_param, order, num_blocks, first_channels = 10,padding_mode="circular", 
        activation=None, normalize="GN", group_c=1, skiprescale = True,dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.first_channels = first_channels
        self.dim_param = dim_param
        self.order = order
        assert self.order <= 2, "Only implemented up to second order moments"
        self.num_blocks = num_blocks
        self.padding_mode = padding_mode
        self.activation = activation
        self.normalize = normalize
        self.group_c = group_c
        self.skiprescale = skiprescale
        self.dropout = dropout

        self.config = {"in_channels": self.in_channels, "in_size": self.in_size, "dim_param": self.dim_param, "order": self.order, "num_blocks": self.num_blocks, 
            "first_channels": self.first_channels, "padding_mode": self.padding_mode, 
            "activation": self.activation, "normalize": self.normalize, "group_c": self.group_c, "skiprescale": self.skiprescale,"dropout": self.dropout}
        
        self.blocks = nn.ModuleList()
        self.current_channels = self.in_channels
        if self.in_channels != self.first_channels:
            self.head = NormConv2d((self.in_channels, self.in_size, self.in_size),self.in_channels, self.first_channels, kernel_size=1, padding_mode=self.padding_mode, 
                normalize=self.normalize, group_c=self.group_c)
            self.current_channels = self.first_channels
        else:
            self.head = nn.Identity()
        for i in range(self.num_blocks):
            self.blocks.append(ResBlock(self.in_size, self.current_channels, self.current_channels, padding_mode=self.padding_mode, 
                 normalize=self.normalize, group_c=self.group_c, skiprescale=self.skiprescale))
            
        ## Output dimension will be dim_param ** order
        self.out_dim = (self.dim_param+1) ** self.order
        self.fc_tail = nn.Sequential(nn.Linear(self.current_channels, self.current_channels), nn.ReLU(), nn.Linear(self.current_channels, self.out_dim))
        print(self.current_channels, self.out_dim)

    def forward(self, x):
        x = self.head(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x,p=self.dropout)
        x = self.fc_tail(x)

        if self.order == 1:
            return x
        elif self.order == 2:
            x = x.view(x.size(0), self.dim_param, self.dim_param)
            x = (x + x.transpose(1,2))/2 ## Symmetrize
            return x

class MomentModel(nn.Module):
    def __init__(self, network, sde, ps, conetwork = None, exp_matr = True):
        super().__init__()
        self.network = network
        self.sde = sde
        self.ps = ps
        self.order = self.network.order
        self.dim_param = self.network.dim_param
        self.exp_matr = exp_matr
        self.config = {"network": self.network.config, "sde": self.sde.config, "ps": self.ps.config, "order": self.order, "exp_matr": self.exp_matr}
        self.conetwork = conetwork
        for param in self.ps.parameters():
            param.requires_grad_(False) ## Freeze power spectrum parameters
    def loss(self, batch):
        if isinstance(self.sde, DiscreteSDE):
            timesteps = torch.randint(0, self.sde.N, (batch.shape[0],1), device=batch.device)
        else:
            timesteps = torch.rand((batch.shape[0],1), device=batch.device)*(self.sde.tmax - self.sde.tmin) + self.sde.tmin
        ps_tensor, phi = self.ps.sample_ps(batch.shape[0])
        batch_tilde, _ , _ = self.sde.sampling(batch, timesteps, torch.sqrt(ps_tensor))
        rphi = self.ps.rescale_phi(phi)
        out = self.network(batch_tilde, timesteps)
        if self.order == 1:
            loss = F.mse_loss(out, rphi)
        elif self.order == 2:
            rphi_mod = rphi - self.conetwork(batch_tilde, timesteps)
            goal = torch.einsum("BD,BC->BDC", rphi_mod, rphi_mod)
            if self.exp_matr:
                goal = la.matrix_exp(out)
            loss = F.mse_loss(out, goal)
        return loss

class SigmaMomentModel(nn.Module):
    def __init__(self, network, sde, ps, log_noise_level = False):
        super().__init__()
        self.network = network
        self.sde = sde
        self.ps = ps
        self.dim_param = self.network.dim_param
        self.config = {"network": self.network.config, "sde": self.sde.config, "ps": self.ps.config, 'log_noise_level': log_noise_level}
        for param in self.ps.parameters():
            param.requires_grad_(False)
        
    def loss(self, batch):
        if isinstance(self.sde, DiscreteSDE):
            timesteps = torch.randint(0, self.sde.N, (batch.shape[0],1), device=batch.device)
        else:
            timesteps = torch.rand((batch.shape[0],1), device=batch.device)*(self.sde.tmax - self.sde.tmin) + self.sde.tmin
        noise_levels = self.sde.noise_level(timesteps)
        log_noise_levels = torch.log(noise_levels)
        ps_tensor, phi = self.ps.sample_ps(batch.shape[0])
        batch_tilde, _ , _ = self.sde.sampling(batch, timesteps, torch.sqrt(ps_tensor))
        rphi = self.ps.rescale_phi(phi)
        out = self.network(batch_tilde)
        target = torch.cat((rphi, log_noise_levels), dim = 1)
        loss = F.mse_loss(out, target)
        return loss
        


def get_moment_network(config):
    if "sde" not in config.keys():
        config["sde"] = {}
    if "ps" not in config.keys():
        config["ps"] = {}
    if "conetwork" not in config.keys():
        config["conetwork"] = {None}
    sde = get_sde(config["sde"])