import torch
from torch import nn
from torch.nn import functional as F
from dataHandler.dataset import MHDProjDataset,LogNormalTransform
import tqdm
import matplotlib.pyplot as plt
import os


MODEL_ID='VAE_1.0'

SOURCE_DIR='/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj'

SAMPLE_FOLDER='/mnt/home/dheurtel/ceph/20_samples/ddpm_initial/'  #For periodic samples
CKPT_FOLDER='/mnt/home/dheurtel/ceph/10_checkpoints/VAE_FID/' #For checkpoints and losses



SIZE=256
SAMPLE_BATCH_SIZE=8
BATCH_SIZE=64
RESULT_SAMPLE_SIZE=256


SAMPLE_STEP_EPOCH=100

NUM_EPOCHS=500

LR=1e-3


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NormalizedConvolution(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding='same', padding_mode='circular', activation=None, normalize=True):
        super(NormalizedConvolution, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding,padding_mode=padding_mode)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out
    
def TinyBlock(size, in_c, out_c):
    return nn.Sequential(NormalizedConvolution((in_c, size, size), in_c, out_c), 
                         NormalizedConvolution((out_c, size, size), out_c, out_c), 
                         NormalizedConvolution((out_c, size, size), out_c, out_c))

def TinyUp(size, in_c):
    return nn.Sequential(NormalizedConvolution((in_c, size, size), in_c, in_c//2), 
                         NormalizedConvolution((in_c//2, size, size), in_c//2, in_c//4), 
                         NormalizedConvolution((in_c//4, size, size), in_c//4, in_c//4))



class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=512, z_dim=256):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            TinyBlock(SIZE, 1, 8),
            nn.Conv2d(8, 8, 4, 4, 1),
            TinyBlock(SIZE//4,8,32),
            nn.Conv2d(32, 32, 4, 4, 1),
            TinyBlock(SIZE//16, 32, 128),
            nn.Conv2d(128,128, 4, 4, 1),
            TinyBlock(SIZE//64, 128, 512),
            nn.Conv2d(512, 512, 4, 4, 1),
            nn.Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc=nn.Linear(h_dim,h_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 4, 0),
            TinyUp(SIZE//64, 512),
            nn.ConvTranspose2d(128, 128, 4, 4, 0),
            TinyUp(SIZE//16, 128),
            nn.ConvTranspose2d(32, 32, 4, 4, 0),
            TinyUp(SIZE//4, 32),
            nn.ConvTranspose2d(8, 8, 4, 4, 0),
            TinyBlock(SIZE, 8, 4),
            nn.Conv2d(4, 1,1, 1, 'same'),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    
    def forward(self, x):
        h = self.encoder(x)
        #z = self.fc1(self.fc3(h))
        z=self.fc(h).unsqueeze(-1).unsqueeze(-1)
        return self.decoder(z)
    # def forward(self, x):
    #     h = self.encoder(x)
    #     z, mu, logvar = self.bottleneck(h)
    #     z = self.fc3(z).unsqueeze(-1).unsqueeze(-1)
    #     return self.decoder(z), mu, logvar