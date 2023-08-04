import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

##TODO ajouter des flags ou une gestion des constantes à l'échelle du dossier ddpm


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor(
        [[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)]
    )
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


class NormConv2d(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding="same", padding_mode="circular", 
        activation=None, normalize="LN", group_c=1, eps_norm=1e-5,):
        super(NormConv2d, self).__init__()

        if normalize == "LN":
            self.norm = nn.LayerNorm(shape, eps=eps_norm)
        elif normalize == "BN":
            self.norm = nn.BatchNorm2d(in_c, eps=eps_norm)
        elif normalize == "GN":
            self.norm = nn.GroupNorm(in_c // group_c, in_c, eps=eps_norm)
        else:
            self.norm = nn.Identity()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, padding_mode=padding_mode)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = not (normalize is None)

    def forward(self, x):
        return self.activation(self.conv1(self.norm(x)))


def make_te(dim_in, dim_out):
    return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))

def gaussian_fourier_embedding(d, sigma=0.1):
    ## Returns the random Fourier features embedding matrix
    embedding = torch.randn(d) * sigma * 2 * np.pi
    return embedding

class SineCosine(nn.Module):
    def __init__(self,):
        super(SineCosine, self).__init__()
    def forward(self, x):
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

#####################################


class DownResBlock(nn.Module):
    def __init__(self, size, in_c, out_c, time_emb_dim=100, normalize="LN", group_c=1,
        padding_mode="circular", dropout=0, eps_norm=1e-5, skiprescale=False, all_norm=True,):
        super(DownResBlock, self).__init__()

        self.block = nn.Sequential(
            NormConv2d((in_c, size, size), in_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((out_c, size, size), out_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((out_c, size, size), out_c, out_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
        )
        self.skip = nn.Conv2d(in_c, out_c, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2.0)
        else:
            h = h + x
        return h


class UpResBlock(nn.Module):
    def __init__(self, size, in_c, time_emb_dim=100, normalize="LN", group_c=1, 
        padding_mode="circular", dropout=0, eps_norm=1e-5, skiprescale=False,):
        super(UpResBlock, self).__init__()

        self.block = nn.Sequential(
            NormConv2d((in_c, size, size), in_c, in_c // 2, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((in_c // 2, size, size), in_c // 2, in_c // 4, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((in_c // 4, size, size), in_c // 4, in_c // 4, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
        )
        self.skip = nn.Conv2d(in_c, in_c // 4, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2.0)
        else:
            h = h + x
        return h


class MidResBlock(nn.Module):
    def __init__(self, size, in_c, time_emb_dim=100, normalize="LN", group_c=1, 
        padding_mode="circular", dropout=0, eps_norm=1e-5,skiprescale=False,):
        super(MidResBlock, self).__init__()

        self.block = nn.Sequential(
            NormConv2d((in_c, size, size), in_c, in_c // 2, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((in_c // 2, size, size), in_c // 2,in_c // 2, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
            NormConv2d((in_c // 2, size, size),in_c // 2, in_c, normalize=normalize, group_c=group_c, padding_mode=padding_mode, eps_norm=eps_norm,),
        )
        self.skip = nn.Conv2d(in_c, in_c, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2.0)
        else:
            h = h + x
        return h


class ResUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, first_c=10, sizes=[256, 128, 64, 32], num_blocks=1, n_steps=1000, time_emb_dim=100, 
        dropout=0, attention=[], normalisation="default", padding_mode="circular", eps_norm=1e-5, skiprescale=False, discretization = "discrete", embedding_mode = None
    ):
        super(ResUNet, self).__init__()
        ## TODO add attention

        if normalisation == "default":
            normalisation = "GN"
        if discretization == "continuous" and embedding_mode is None:
            embedding_mode = "fourier"
        self.config = { "in_c": in_c, "out_c": out_c, "first_c": first_c, "sizes": sizes, "num_blocks": num_blocks, "n_steps": n_steps, "time_emb_dim": time_emb_dim,
        "dropout": dropout, "attention": attention, "normalisation": normalisation, "padding_mode": padding_mode, "eps_norm": eps_norm, "skiprescale": skiprescale, "type" : "ResUNet", 
        "discretization": discretization, "embedding_mode": embedding_mode}

        self.normalisation = normalisation
        self.in_c = in_c
        self.out_c = out_c
        self.first_c = first_c
        self.sizes = sizes
        self.num_blocks = num_blocks
        self.n_steps = n_steps
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        self.attention = attention
        self.padding_mode = padding_mode
        self.eps_norm = eps_norm
        self.discretiation = discretization
        self.embedding_mode = embedding_mode

        
        # Time embedding
        if discretization == "discrete" or discretization == "default":
            self.time_embed = nn.Embedding(n_steps, time_emb_dim)
            self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
            self.time_embed.requires_grad_(False)
        
        elif discretization == "continuous":
            if embedding_mode == "fourier":
                assert time_emb_dim % 2 == 0, "time_emb_dim must be even for fourier embedding"
                linear1 = nn.Linear(1, time_emb_dim//2)
                linear1.weight.data = gaussian_fourier_embedding(time_emb_dim//2).unsqueeze(1)
                linear1.bias.data = torch.zeros(time_emb_dim//2)
                self.time_embed = nn.Sequential(linear1, SineCosine())
                self.time_embed.requires_grad_(False)
            elif embedding_mode == "forward":
                self.time_embed = nn.Linear(1, time_emb_dim)
            

        # First Half

        curr_c = in_c
        ## Blocks with downsample
        self.downblocks = nn.ModuleList()

        tot_groups = len(sizes)

        for i, size in enumerate(sizes):
            for j in range(num_blocks):
                if i == tot_groups - 1 and j == num_blocks - 1:
                    block_out_c = curr_c
                    self.downblocks.append(
                        MidResBlock(size,curr_c,time_emb_dim=time_emb_dim,normalize=normalisation,
                        group_c=first_c // 2,padding_mode=padding_mode,dropout=dropout,eps_norm=eps_norm,skiprescale=skiprescale,)
                    )
                    pass
                elif i == 0 and j == 0:
                    block_out_c = first_c
                    self.downblocks.append(
                            DownResBlock(size,curr_c,block_out_c,time_emb_dim=time_emb_dim,normalize=normalisation,
                            padding_mode=padding_mode,dropout=dropout,eps_norm=eps_norm,skiprescale=skiprescale,)
                    )
                else:
                    block_out_c = 2 * curr_c
                    self.downblocks.append(
                        DownResBlock(size,curr_c,block_out_c,time_emb_dim=time_emb_dim,normalize=normalisation,
                        group_c=first_c // 2,padding_mode=padding_mode,dropout=dropout,eps_norm=eps_norm,skiprescale=skiprescale,)
                    )
                curr_c = block_out_c

            if i != tot_groups - 1:
                self.downblocks.append(
                    nn.Conv2d(curr_c, curr_c, kernel_size=4, stride=2, padding=1)
                )

        # Second Half

        ## Blocks with upsample

        self.upblocks = nn.ModuleList()

        for i, size in enumerate(sizes[:0:-1]):
            self.upblocks.append(
                nn.ConvTranspose2d(curr_c, curr_c, kernel_size=4, stride=2, padding=1)
            )
            for j in range(num_blocks):
                if i == len(sizes) - 2 and j == num_blocks - 1:
                    self.upblocks.append(
                        DownResBlock(size * 2,2 * curr_c,out_c=curr_c,normalize=normalisation,group_c=first_c // 2,
                        padding_mode=padding_mode,dropout=dropout,eps_norm=eps_norm,skiprescale=skiprescale,)
                    )
                else:
                    self.upblocks.append(
                        UpResBlock(size * 2,2 * curr_c,time_emb_dim,normalize=normalisation,group_c=first_c // 2,
                        padding_mode=padding_mode,dropout=dropout,eps_norm=eps_norm,skiprescale=skiprescale,)
                    )
                    curr_c = curr_c // 2

        if normalisation == None:
            norm_l = nn.Identity()
        elif normalisation == "BN":
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)
        elif normalisation == "LN":
            norm_l = nn.LayerNorm(
                (out_c, sizes[0], sizes[0]), elementwise_affine=False, eps=eps_norm
            )
        else:
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)

        self.tail = nn.Sequential(
            nn.Conv2d(curr_c, out_c, kernel_size=3, stride=1, padding="same", padding_mode=padding_mode,),
            norm_l,
        )

    def forward(self, x, t):

        t = self.time_embed(t)
        h = x

        h_list = [h]

        for block in self.downblocks:
            if isinstance(block, DownResBlock) or isinstance(block, MidResBlock):
                h = block(h, t)
                h_list.append(h)
            else:
                h = block(h)

        # h=self.middleblocks(h,t)
        # h_list.append(h)

        h_list.pop()

        for block in self.upblocks:
            if isinstance(block, UpResBlock) or isinstance(block, DownResBlock):
                h = torch.cat([h, h_list.pop()], dim=1)
                h = block(h, t)
            else:
                h = block(h)
        h = self.tail(h)
        return h

def get_network(config): ## TODO add more networks, TODO be careful when I will change how ResUNet works!!!
    if "type" not in config.keys():
        config["type"] = "ResUNet"
    if config["type"].lower() == "resunet":
        if "in_c" not in config.keys():
            config["in_c"] = 1
        if "out_c" not in config.keys():
            config["out_c"] = 1
        if "first_c" not in config.keys():
            config["first_c"] = 10
        if "sizes" not in config.keys():
            config["sizes"] = [256, 128, 64, 32]
        if "num_blocks" not in config.keys():
            config["num_blocks"] = 1
        if "n_steps" not in config.keys():
            print("Warning: n_steps not specified, defaulting")
            config["n_steps"] = 1000
        if "time_emb_dim" not in config.keys():
            config["time_emb_dim"] = 100
        if "dropout" not in config.keys():
            print("Warning: dropout not specified, defaulting")
            config["dropout"] = 0
        if "attention" not in config.keys():
            print("Warning: attention not specified, defaulting")
            config["attention"] = []
        if "normalisation" not in config.keys():
            print("Warning: normalisation not specified, defaulting")
            config["normalisation"] = "default"
        if "padding_mode" not in config.keys():
            config["padding_mode"] = "circular"
        if "eps_norm" not in config.keys():
            config["eps_norm"] = 1e-5
        if "skiprescale" not in config.keys():
            config["skiprescale"] = True
        if "discretization" not in config.keys():
            config["discretization"] = "discrete"
        if "embedding_mode" not in config.keys():
            config["embedding_mode"] = None
        return ResUNet(in_c=config["in_c"], out_c=config["out_c"], first_c=config["first_c"], sizes=config["sizes"], num_blocks=config["num_blocks"], n_steps=config["n_steps"], time_emb_dim=config["time_emb_dim"], dropout=config["dropout"], attention=config["attention"], normalisation=config["normalisation"], padding_mode=config["padding_mode"], eps_norm=config["eps_norm"], skiprescale=config["skiprescale"],)
    else:
        raise NotImplementedError(f"Network type {config['type']} not implemented")
