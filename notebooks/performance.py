## Standard imports
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import tqdm

## Relative imports
import astroddpm
from astroddpm.runners import Diffuser, config_from_id, get_samples
from astroddpm.separation import double_loader, method1_algo2, check_separation1score, separation1score, load_everything
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE
from astroddpm.diffusion.models.network import ResUNet
from astroddpm.utils import colormap_custom

from astroddpm.analysis.validationMetrics import powerSpectrum
from astroddpm.analysis.validationMetrics.powerSpectrum import compare_separation_power_spectrum_iso
import astroddpm.analysis.validationMetrics.powerSpectrum as powerspectrum
from scipy.stats import wasserstein_distance

import bm3d
from bm3d import bm3d, BM3DStages
MODEL_ID_1= 'DiscreteSBM_MultiSigmaVPSDE_I_BPROJ_N_1000_bottleneck_16_firstc_6'

amin,amax=(-3, 3)
bins = torch.linspace(0, np.pi, 100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffuser_1 = load_everything(MODEL_ID_1)

mean_theta = diffuser_1.diffmodel.ps.sample_theta(10000).mean(dim = 0, keepdim = True).to(device)


batch = next(iter(diffuser_1.train_dataloader)).unsqueeze(1).to(device)

power_spectra_dataset, mean_dataset, _, _ = powerSpectrum.set_power_spectrum_iso2d(batch, bins=bins, only_stat=False, use_gpu=True)
log_mean_data = torch.log(mean_dataset).cpu().detach()


## Theta distribution : torch.tensor([40, 49.2e-3])*torch.rand(n_samples, 2) + torch.tensor([50, 7.5e-3])

theta_0_min = 50
theta_0_max = 90
theta_1_min = 7.5e-3
theta_1_max = 56.7e-3

n_pixels = 41

theta_grid = torch.stack(torch.meshgrid(torch.linspace(theta_0_min, theta_0_max, n_pixels), torch.linspace(theta_1_min, theta_1_max, n_pixels)), dim = -1).reshape(-1, 2)
coordinates = torch.stack(torch.meshgrid(torch.linspace(0, 1, n_pixels), torch.linspace(0, 1, n_pixels)), dim = -1).reshape(-1, 2)

diff_multi_data_theta = torch.zeros(n_pixels, n_pixels, 100)
wass_dist_multi_data_theta = np.zeros((n_pixels, n_pixels, 100))
## Compute the diff of mean power spectrum for each theta in the grid with data (batch)

def wass_func_ps(power_spectra1, power_spectra2):
    assert power_spectra1.shape[1] == power_spectra2.shape[1]
    all_wass = np.zeros(power_spectra1.shape[1])
    for i in range(power_spectra1.shape[1]):
        wass = wasserstein_distance(torch.log(power_spectra1)[:,i].cpu().detach().numpy(), torch.log(power_spectra2)[:,i].cpu().detach().numpy())
        all_wass[i] = wass
    return all_wass

progress_bar = tqdm.tqdm(total = n_pixels**2)
for i in range(n_pixels):
    for j in range(n_pixels):
        theta = theta_grid[i*n_pixels + j].unsqueeze(0).to(device)
        sample = diffuser_1.diffmodel.generate_image(64, thetas = theta.repeat(64,1), verbose=False)
        power_spectra_sample, mean, _, _ = powerSpectrum.set_power_spectrum_iso2d(sample, bins=bins, only_stat=False, use_gpu=True)
        diff_multi_data_theta[i,j] = (log_mean_data - torch.log(mean).cpu().detach()).abs()
        wass_dist_multi_data_theta[i,j] = wass_func_ps(power_spectra_dataset, power_spectra_sample)
        progress_bar.update(1)
progress_bar.close()
diff_multi_data_theta.abs().mean(dim = -1)
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(diff_multi_data_theta.abs().mean(dim = -1))
fig.colorbar(im, ax=ax)
fig.savefig('dist_mono_data_theta_large.pdf')


## Wasserstein distance GIF
images = []
for i in range(100):
    im = Image.fromarray(np.uint8(cm.viridis(wass_dist_multi_data_theta[:,:,i])*255))
    draw = ImageDraw.Draw(im)
    images.append(im)

## Generate a GIF with all the images
images[0].save('wass_multi_data_theta_large.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

## Save diff_multi_data_theta and wass_dist_multi_data_theta to a .pt file named 'diff_multi_data_theta.pt' and 'wass_dist_multi_data_theta.pt'

torch.save(diff_multi_data_theta, 'diff_multi_data_theta_large.pt')
torch.save(wass_dist_multi_data_theta, 'wass_dist_multi_data_theta_large.pt')
