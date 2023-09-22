## Similar as inference_0.py but with argparse to chose the target phi. 

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import astroddpm
import tqdm

## Relative imports
from astroddpm.runners import Diffuser, config_from_id, get_samples
from astroddpm.analysis.validationMetrics import powerSpectrum, minkowskiFunctional, basics
from astroddpm.analysis import overfitting_check
from astroddpm.utils.plot import check_nearest_epoch, plot_losses, check_training_samples, plot_comparaison
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE
from astroddpm.diffusion.models.network import ResUNet
import astroddpm.utils.colormap_custom 
from cmb_hmc.cmb_ps import CMBPS, unnormalize_phi, normalize_phi
from cmb_hmc.hmc_torch import HMC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amin = - 3
amax = 3

MODEL_ID = 'DiscreteSBM_MultiSigmaVPSDE_MHD_BPROJ_N_1000_bottleneck_32_firstc_20_invsqrt'
NUM_CHAIN = 16

parser = argparse.ArgumentParser(description='Inference with HMC')
parser.add_argument('--n_gibbs', type=int, default=4000,
                    help='Number of Gibbs steps')
parser.add_argument('--phi_target', type=float, nargs=2, default=[0.5, 0.5],
                    help='Initial phi')
parser.add_argument('--burnin_hmc', type=int, default=0,help='burnin for HMC')

args = parser.parse_args()
N_GIBBS = args.n_gibbs
BURNIN_HMC = args.burnin_hmc
phi_target = torch.tensor(args.phi_target).to(device).unsqueeze(0)

placeholder_dm = DiscreteSBM(DiscreteVPSDE(1000), ResUNet())
diffuser = Diffuser(placeholder_dm)
diffuser.load(config=config_from_id(MODEL_ID), also_ckpt=True, for_training=True)
ps_model = CMBPS(norm_phi=True).to(device)

batch = next(iter(diffuser.test_dataloader))
image = batch.unsqueeze(1).to(device)[:1]

theta_target = unnormalize_phi(phi_target)
ps_target = diffuser.diffmodel.ps(theta_target)
sq_ps = torch.sqrt(ps_target).to(device)
theta_target = theta_target.to(device)

noisy, mean, noise = diffuser.diffmodel.sde.sampling(image, 100, sq_ps)
batch = image.repeat(NUM_CHAIN, 1, 1, 1)
ps_0, thetas_0 = diffuser.diffmodel.ps.sample_ps(NUM_CHAIN)

noisy_batch = noisy.repeat(NUM_CHAIN, 1, 1, 1)

timesteps = 100 * torch.ones(NUM_CHAIN, dtype=torch.int32).to(device)

def sample_prior(n):
    """
    Sample from the (normalized) prior distribution.
    phi = (H0, Obh2) with H0 ~ U(0, 1), Obh2 ~ U(0, 1)
    (unnormalized prior is H0 ~ U(50, 90), Obh2 ~ U(0.0075, 0.0567))
    """
    phi = torch.rand(n, 2).to(device)
    return phi

def log_likelihood(phi, x):
    """
    Compute the log likelihood of the Gaussian model.
    """
    x_dim = x.shape[-1]*x.shape[-2]

    ps = ps_model(phi)
    xf = torch.fft.fft2(x)

    term_pi = -(x_dim/2) * np.log(2*np.pi)
    term_logdet = -1/2*torch.sum(torch.log(ps), dim=(-1, -2, -3)) # The determinant is the product of the diagonal elements of the PS
    term_x = -1/2*torch.sum(1/ps*torch.abs(xf)**2, dim=(-1, -2, -3))/x_dim # We divide by x_dim because of the normalization of the FFT

    return term_pi + term_logdet + term_x

def log_prior(phi):
    """
    Compute the log (normalized) prior of the parameters.
    """
    H0, Obh2 = phi[..., 0], phi[..., 1]
    term_H0 = torch.log(torch.logical_and(H0 >= 0.0, H0 <= 1.0).float())
    term_Obh2 = torch.log(torch.logical_and(Obh2 >= 0.0, Obh2 <= 1.0).float())
    return term_H0 + term_Obh2

def log_posterior(phi, x):
    """
    Compute the log posterior of the parameters (not normalized by the evidence).
    """
    return log_likelihood(phi, x) + log_prior(phi)

### Gibbs sampling

theta_list = []
phi_list = []

rescaled_batch = diffuser.diffmodel.sde.rescale_preserved_to_additive(noisy_batch, timesteps)

thetas = thetas_0.clone().to(device)

progress_bar = tqdm.tqdm(range(N_GIBBS))
for n in range(N_GIBBS):
    X_0, _ = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, initial_timestep=100, verbose=False, thetas = thetas)
    ps = diffuser.diffmodel.ps(thetas)
    sq_ps = torch.sqrt(ps).to(device)
    _, mean, _ = diffuser.diffmodel.sde.sampling(X_0, 100, sq_ps)
    epsilon_hat = (rescaled_batch - mean)/diffuser.diffmodel.sde.noise_level(timesteps)
    log_prob = lambda phi: log_posterior(phi, epsilon_hat)

    def log_prob_grad(phi):
        """
        Compute the log posterior and its gradient.
        """
        log_prob = log_posterior(phi, epsilon_hat)
        grad_log_prob = torch.autograd.grad(log_prob, phi, grad_outputs=torch.ones_like(log_prob))[0]
        return log_prob, grad_log_prob
    hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)

    kwargs = {'nsamples': 1,
          'burnin': 0,
          'step_size': 5e-6,
          'nleap': 20}

    phi_0 = sample_prior(NUM_CHAIN).requires_grad_()
    phi_0 = phi_list[-1].requires_grad_() if len(phi_list) > 0 else phi_0
    sampler = hmc.sample(phi_0, verbose = False, **kwargs)
    phi = sampler.samples[:,-1]
    phi_list.append(phi) 
    thetas = unnormalize_phi(phi)
    theta_list.append(thetas)
    thetas.detach()
    thetas.requires_grad_()
    progress_bar.update(1)

tensor_phi_list = [phi.unsqueeze(0) for phi in phi_list]
tensor_phi_list = torch.cat(tensor_phi_list, dim=0)

torch.save(tensor_phi_list, 'phi_list_{}_{}.pt'.format(f'{phi_target[0,0].cpu().item():.2f}', f'{phi_target[0,1].cpu().item():.2f}'))