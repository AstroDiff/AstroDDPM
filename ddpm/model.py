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


class DoubleNorm(nn.Module):
    def __init__(self, in_c, shape, eps=1e-5):
        super(DoubleNorm, self).__init__()
        self.shape = shape
        self.in_c = in_c
        self.bn = nn.BatchNorm2d(in_c // 2, eps=eps)
        i, s, s = shape
        self.ln = nn.LayerNorm((i // 2, s, s), eps=eps)

    def forward(self, x):
        x1, x2 = torch.split(x, self.in_c // 2, dim=1)  ### split along dim 1
        x1 = self.bn(x1)
        x2 = self.ln(x2)
        return torch.cat((x1, x2), 1)  ####Concatenate at the channel dim


class NormalizedConvolution(nn.Module):
    def __init__(
        self,
        shape,
        in_c,
        out_c,
        kernel_size=3,
        stride=1,
        padding="same",
        padding_mode="circular",
        activation=None,
        normalize="LN",
        group_c=1,
        eps_norm=1e-5,
    ):
        super(NormalizedConvolution, self).__init__()

        if normalize == "LN":
            self.ln = nn.LayerNorm(shape, eps=eps_norm)
        elif normalize == "BN":
            self.ln = nn.BatchNorm2d(in_c, eps=eps_norm)
        elif normalize == "GN":
            self.ln = nn.GroupNorm(in_c // group_c, in_c, eps=eps_norm)
        elif normalize == "DN":
            self.ln = DoubleNorm(in_c, shape, eps=eps_norm)

        self.conv1 = nn.Conv2d(
            in_c, out_c, kernel_size, stride, padding, padding_mode=padding_mode
        )
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = not (normalize is None)

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out


def make_te(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
    )


## TODO make that clearer and smarter
def normalization_parameters(normalisation, length):
    if normalisation == "LN" or normalisation == "default":
        return (length * ["LN"], length * ["LN"], "LN")
    if normalisation == "BN":
        return length * ["BN"], (length - 1) * ["BN"], "BN"
    if normalisation == "GN":
        return length * ["GN"], (length - 1) * ["GN"], "GN"
    if normalisation == "None":
        return length * [None], (length - 1) * [None], None
    if normalisation == "DN":
        return (length * ["DN"], (length - 1) * ["DN"], "LN")

    if normalisation == "LN-D":
        return [None] + (length - 1) * ["LN"], length * ["LN"], "LN"
    if normalisation == "LN-F":
        return length * ["LN"], (length - 1) * ["LN"], None
    if normalisation == "LN-F+":
        return length * ["LN"], (length - 2) * ["LN"] + [None], None
    if normalisation == "LN-DF":
        return [None] + (length - 1) * ["LN"], (length - 2) * ["LN"] + [None], None
    if normalisation == "BN/LN":
        return length * ["BN"], (length - 1) * ["LN"], "LN"
    if normalisation == "BN/FLN":
        return length * ["BN"], (length - 1) * ["BN"], "LN"
    if normalisation == "BN/F+LN":
        return length * ["BN"], (length - 2) * ["BN"] + ["LN"], "LN"
    if normalisation == "DBN/LN":
        return length * ["LN"], (length - 1) * ["LN"], "LN"


def normalization_parameters_parse(normalization, length):
    return


class DownBlock(nn.Module):
    def __init__(
        self,
        size,
        in_c,
        out_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
    ):
        super(DownBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (out_c, size, size),
                out_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (out_c, size, size),
                out_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
        )
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        return self.block(x + self.te(t).reshape(n, -1, 1, 1))


class UpBlock(nn.Module):
    def __init__(
        self,
        size,
        in_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
    ):
        super(UpBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c // 4,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 4, size, size),
                in_c // 4,
                in_c // 4,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
        )
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        return self.block(x + self.te(t).reshape(n, -1, 1, 1))


class MidBlock(nn.Module):
    def __init__(
        self,
        size,
        in_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
    ):
        super(MidBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
        )
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        return self.block(x + self.te(t).reshape(n, -1, 1, 1))


class UNet(nn.Module):
    def __init__(
        self,
        in_c=1,
        out_c=1,
        first_c=10,
        sizes=[256, 128, 64, 32],
        num_blocks=1,
        n_steps=1000,
        time_emb_dim=100,
        dropout=0,
        attention=[],
        normalisation="default",
        padding_mode="circular",
        eps_norm=1e-5,
    ):
        super(UNet, self).__init__()

        norm_down, norm_up, norm_tail = normalization_parameters(
            normalisation, len(sizes) * num_blocks
        )

        self.normalisation = normalisation

        # Sinusoidal embedding

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First Half

        curr_c = in_c
        if normalisation == "DN":
            self.head = nn.Conv2d(
                in_c,
                4,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode,
                padding="same",
            )
            curr_c = 4
        ## Blocks with downsample
        self.downblocks = nn.ModuleList()

        tot_groups = len(sizes)

        for i, size in enumerate(sizes):
            for j in range(num_blocks):
                if i == tot_groups - 1 and j == num_blocks - 1:
                    block_out_c = curr_c
                    self.downblocks.append(
                        MidBlock(
                            size,
                            curr_c,
                            time_emb_dim=time_emb_dim,
                            normalize=norm_down[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                        )
                    )
                    pass
                elif i == 0 and j == 0:
                    block_out_c = first_c
                    if normalisation == "DN":
                        self.downblocks.append(
                            DownBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                group_c=1,
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                            )
                        )
                    elif normalisation == "GN":
                        self.downblocks.append(
                            DownBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                group_c=1,
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                            )
                        )
                        self.downblocks[-1].block[1].ln = nn.GroupNorm(2, first_c)
                        self.downblocks[-1].block[2].ln = nn.GroupNorm(2, first_c)
                    else:
                        self.downblocks.append(
                            DownBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                            )
                        )
                else:
                    block_out_c = 2 * curr_c
                    self.downblocks.append(
                        DownBlock(
                            size,
                            curr_c,
                            block_out_c,
                            time_emb_dim=time_emb_dim,
                            normalize=norm_down[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                        )
                    )

                curr_c = block_out_c

            if i != tot_groups - 1:
                self.downblocks.append(
                    nn.Conv2d(curr_c, curr_c, kernel_size=4, stride=2, padding=1)
                )

        # Middleblocks

        # self.middleblocks= DownBlock(sizes[-1],curr_c,curr_c,time_emb_dim,normalize=norm_mid)

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
                        DownBlock(
                            size * 2,
                            2 * curr_c,
                            out_c=curr_c,
                            normalize=norm_up[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                        )
                    )
                else:
                    self.upblocks.append(
                        UpBlock(
                            size * 2,
                            2 * curr_c,
                            time_emb_dim,
                            normalize=norm_up[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                        )
                    )
                    curr_c = curr_c // 2

        if norm_tail == None:
            norm_l = nn.Identity()
        elif norm_tail == "BN":
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)
        elif norm_tail == "LN":
            norm_l = nn.LayerNorm(
                (out_c, sizes[0], sizes[0]), elementwise_affine=False, eps=eps_norm
            )
        else:
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)

        self.tail = nn.Sequential(
            nn.Conv2d(
                curr_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            norm_l,
        )

    def forward(self, x, t):
        t = self.time_embed(t)

        h = x

        if self.normalisation == "DN":
            h = self.head(x)
        h_list = [h]

        for block in self.downblocks:
            if isinstance(block, DownBlock) or isinstance(block, MidBlock):
                h = block(h, t)
                h_list.append(h)
            else:
                h = block(h)

        # h=self.middleblocks(h,t)
        # h_list.append(h)

        h_list.pop()

        for block in self.upblocks:
            if isinstance(block, UpBlock) or isinstance(block, DownBlock):
                h = torch.cat([h, h_list.pop()], dim=1)
                h = block(h, t)
            else:
                h = block(h)
        h = self.tail(h)
        return h


#####################################


class DownResBlock(nn.Module):
    def __init__(
        self,
        size,
        in_c,
        out_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
        skiprescale=False,
    ):
        super(DownResBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (out_c, size, size),
                out_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (out_c, size, size),
                out_c,
                out_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
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
    def __init__(
        self,
        size,
        in_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
        skiprescale=False,
    ):
        super(UpResBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c // 4,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 4, size, size),
                in_c // 4,
                in_c // 4,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
        )
        self.skip = nn.Conv2d(in_c, in_c // 4, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2.0)
        else:
            h = h + x
        return h


class MidResBlock(nn.Module):
    def __init__(
        self,
        size,
        in_c,
        time_emb_dim=100,
        normalize="LN",
        group_c=1,
        padding_mode="circular",
        dropout=0,
        eps_norm=1e-5,
        skiprescale=False,
    ):
        super(MidResBlock, self).__init__()
        self.block = nn.Sequential(
            NormalizedConvolution(
                (in_c, size, size),
                in_c,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c // 2,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
            NormalizedConvolution(
                (in_c // 2, size, size),
                in_c // 2,
                in_c,
                normalize=normalize,
                group_c=group_c,
                padding_mode=padding_mode,
                eps_norm=eps_norm,
            ),
        )
        self.skip = nn.Conv2d(in_c, in_c, 1)
        self.skiprescale = skiprescale
        if dropout > 0:
            self.block.append(nn.Dropout(dropout))

        self.te = make_te(time_emb_dim, in_c)

    def forward(self, x, t):
        n = len(x)
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        h = self.block(x + self.te(t).reshape(n, -1, 1, 1))
        x = self.skip(x)
        if self.skiprescale:
            h = (h + x) / np.sqrt(2.0)
        else:
            h = h + x
        return h


class ResUNet(nn.Module):
    def __init__(
        self,
        in_c=1,
        out_c=1,
        first_c=10,
        sizes=[256, 128, 64, 32],
        num_blocks=1,
        n_steps=1000,
        time_emb_dim=100,
        dropout=0,
        attention=[],
        normalisation="default",
        padding_mode="circular",
        eps_norm=1e-5,
        skiprescale=False,
    ):
        super(ResUNet, self).__init__()

        norm_down, norm_up, norm_tail = normalization_parameters(
            normalisation, len(sizes) * num_blocks
        )

        self.normalisation = normalisation

        # Sinusoidal embedding

        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First Half

        curr_c = in_c
        if normalisation == "DN":
            self.head = nn.Conv2d(
                in_c,
                4,
                kernel_size=3,
                stride=1,
                padding_mode=padding_mode,
                padding="same",
            )
            curr_c = 4
        ## Blocks with downsample
        self.downblocks = nn.ModuleList()

        tot_groups = len(sizes)

        for i, size in enumerate(sizes):
            for j in range(num_blocks):
                if i == tot_groups - 1 and j == num_blocks - 1:
                    block_out_c = curr_c
                    self.downblocks.append(
                        MidResBlock(
                            size,
                            curr_c,
                            time_emb_dim=time_emb_dim,
                            normalize=norm_down[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                            skiprescale=skiprescale,
                        )
                    )
                    pass
                elif i == 0 and j == 0:
                    block_out_c = first_c
                    if normalisation == "DN":
                        self.downblocks.append(
                            DownResBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                group_c=1,
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                                skiprescale=skiprescale,
                            )
                        )
                    elif normalisation == "GN":
                        self.downblocks.append(
                            DownResBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                group_c=1,
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                                skiprescale=skiprescale,
                            )
                        )
                        self.downblocks[-1].block[1].ln = nn.GroupNorm(2, first_c)
                        self.downblocks[-1].block[2].ln = nn.GroupNorm(2, first_c)
                    else:
                        self.downblocks.append(
                            DownResBlock(
                                size,
                                curr_c,
                                block_out_c,
                                time_emb_dim=time_emb_dim,
                                normalize=norm_down[i * num_blocks + j],
                                padding_mode=padding_mode,
                                dropout=dropout,
                                eps_norm=eps_norm,
                                skiprescale=skiprescale,
                            )
                        )
                else:
                    block_out_c = 2 * curr_c
                    self.downblocks.append(
                        DownResBlock(
                            size,
                            curr_c,
                            block_out_c,
                            time_emb_dim=time_emb_dim,
                            normalize=norm_down[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                            skiprescale=skiprescale,
                        )
                    )

                curr_c = block_out_c

            if i != tot_groups - 1:
                self.downblocks.append(
                    nn.Conv2d(curr_c, curr_c, kernel_size=4, stride=2, padding=1)
                )

        # Middleblocks

        # self.middleblocks= DownBlock(sizes[-1],curr_c,curr_c,time_emb_dim,normalize=norm_mid)

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
                        DownResBlock(
                            size * 2,
                            2 * curr_c,
                            out_c=curr_c,
                            normalize=norm_up[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                            skiprescale=skiprescale,
                        )
                    )
                else:
                    self.upblocks.append(
                        UpResBlock(
                            size * 2,
                            2 * curr_c,
                            time_emb_dim,
                            normalize=norm_up[i * num_blocks + j],
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            eps_norm=eps_norm,
                            skiprescale=skiprescale,
                        )
                    )
                    curr_c = curr_c // 2

        if norm_tail == None:
            norm_l = nn.Identity()
        elif norm_tail == "BN":
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)
        elif norm_tail == "LN":
            norm_l = nn.LayerNorm(
                (out_c, sizes[0], sizes[0]), elementwise_affine=False, eps=eps_norm
            )
        else:
            norm_l = nn.BatchNorm2d(out_c, eps=eps_norm)

        self.tail = nn.Sequential(
            nn.Conv2d(
                curr_c,
                out_c,
                kernel_size=3,
                stride=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            norm_l,
        )

    def forward(self, x, t):
        t = self.time_embed(t)

        h = x

        if self.normalisation == "DN":
            h = self.head(x)
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
    

class ManifoldResNetClassifier(nn.Module):
    def __init__(self,
            size,
            in_c=3,  
            first_c=10,
            num_blocks=6,
            n_steps=1000,
            time_emb_dim=100,
            dropout=0,
            attention=[],
            normalisation="default",
            padding_mode="circular",
            skiprescale=True,):
        super(ManifoldResNetClassifier, self).__init__()
        ## Parameters
        self.n_steps=n_steps
        self.num_blocks=num_blocks
        self.first_c=first_c
        self.in_c=in_c
        self.normalisation = normalisation
        self.dropout=dropout
        self.skiprescale=skiprescale

        ## Time Embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.head=NormalizedConvolution(size, in_c, first_c,3, 1, "same", "circular", normalize="GN", group_c=in_c)

        self.block_list=nn.ModuleList()
        curr_size=size

        curr_c=first_c
        for idx in range(num_blocks):
            self.block_list.append(DownResBlock(
                            curr_size,
                            curr_c,
                            out_c=curr_c*2,
                            normalize=normalisation,
                            group_c=first_c // 2,
                            padding_mode=padding_mode,
                            dropout=dropout,
                            skiprescale=skiprescale,
                        ))
            curr_size=curr_size//2
            curr_c=2*curr_c
        
        self.min_size=curr_size

        self.tail=nn.Linear(curr_c,1)

    def forward(self, z, t):
        t = self.time_embed(t)

        h = self.head(z)

        for block in self.block_list:
            h = block(h, t)
            h=nn.AvgPool2d(2)(h)
        
        h = nn.AvgPool2d(self.min_size)(h)

        out = self.tail(nn.Flatten()(h))

        return out.squeeze(1)


