## Standard imports
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
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE, DiscreteSDE, ContinuousSDE, ContinuousVPSDE
from astroddpm.diffusion.stochastic.solver import get_schedule
from astroddpm.diffusion.models.network import ResUNet
import astroddpm.utils.colormap_custom 
import arviz as az

from cmb_hmc.hmc_torch import HMC

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run MCMC on a given model to calibrate the posterior')

parser.add_argument('--model_id', type=str, default='ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_16_firstc_6_phi_beta_cosine')
parser.add_argument('--num_chain', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=0.1)
parser.add_argument('--save_path', type=str, default='/mnt/home/dheurtel/ceph/04_inference/dummy_results.pt')

args = parser.parse_args()

MODEL_ID = args.model_id
NUM_CHAIN = args.num_chain
NUM_SAMPLES = args.num_samples
NOISE_LEVEL = args.noise_level
BURNIN_MCMC = 40

CKPT_FREQ = 50

placeholder_dm = DiscreteSBM(DiscreteVPSDE(1000), ResUNet())
diffuser = Diffuser(placeholder_dm)
diffuser.load(config=config_from_id(MODEL_ID), also_ckpt=True, for_training=True)

TIME_STEP = diffuser.diffmodel.sde.get_closest_timestep(torch.tensor(NOISE_LEVEL))
print(f"Time step chosen for noise level {NOISE_LEVEL}: {TIME_STEP}")

TIME_STEP = TIME_STEP.to(device)

ps_model = diffuser.diffmodel.ps

def sample_prior(n):
    """
    Sample from the (normalized) prior distribution.
    phi = (H0, Obh2) with H0 ~ U(-1, 1), Obh2 ~ U(-1, 1)
    (unnormalized prior is H0 ~ U(50, 90), Obh2 ~ U(0.0075, 0.0567))
    """
    phi = 2*torch.rand(n, 2).to(device)-1
    return phi

def log_likelihood(rphi, x):
    """
    Compute the log likelihood of the Gaussian model.
    """
    x_dim = x.shape[-1]*x.shape[-2]

    ps = ps_model(rphi, to_rescale = False)
    xf = torch.fft.fft2(x)

    term_pi = -(x_dim/2) * np.log(2*np.pi)
    term_logdet = -1/2*torch.sum(torch.log(ps), dim=(-1, -2, -3)) # The determinant is the product of the diagonal elements of the PS
    term_x = -1/2*torch.sum(1/ps*torch.abs(xf)**2, dim=(-1, -2, -3))/x_dim # We divide by x_dim because of the normalization of the FFT

    return term_pi + term_logdet + term_x

def log_prior(rphi):
    """
    Compute the log (normalized) prior of the parameters.
    """
    H0, Obh2 = rphi[..., 0], rphi[..., 1]
    term_H0 = torch.log(torch.logical_and(H0 >= -1.0, H0 <= 1.0).float()/2.0)
    term_Obh2 = torch.log(torch.logical_and(Obh2 >= -1.0, Obh2 <= 1.0).float()/2.0)
    return term_H0 + term_Obh2

def log_posterior(rphi, x):
    """
    Compute the log posterior of the parameters (not normalized by the evidence).
    """
    return log_likelihood(rphi, x) + log_prior(rphi)


timesteps = TIME_STEP * torch.ones(NUM_CHAIN, dtype=torch.int32).to(device)

phi_list = []
rphi_list = []

batch = next(iter(diffuser.test_dataloader))

if len(batch.shape) == 3:
    batch = batch.unsqueeze(1)

batch = batch.to(device)

if len(batch)<NUM_CHAIN:
    batch = batch.repeat((NUM_CHAIN//len(batch)+1, 1, 1, 1))
batch = batch[:NUM_CHAIN]

ps_true, phi_true = ps_model.sample_ps(NUM_CHAIN)
sq_ps_true = torch.sqrt(ps_true).to(device)
phi_true = phi_true.to(device)
rphi_true = ps_model.rescale_phi(phi_true).to(device)

noisy_batch, mean_batch, noise_batch = diffuser.diffmodel.sde.sampling(batch, timesteps, sq_ps_true)

rescaled_batch = diffuser.diffmodel.sde.rescale_preserved_to_additive(noisy_batch, timesteps)

phi = ps_model.sample_phi(NUM_CHAIN).to(device)
rphi = ps_model.rescale_phi(phi).to(device)

progress_bar = tqdm.tqdm(range(NUM_SAMPLES+BURNIN_MCMC))

for n in range(NUM_SAMPLES+BURNIN_MCMC):
    if isinstance(diffuser.diffmodel.sde, ContinuousSDE):
        if diffuser.diffmodel.sde.beta_schedule == 'cosine':
            schedule = get_schedule('power_law', t_min = diffuser.diffmodel.sde.tmin, t_max = TIME_STEP.item(), n_iter = 200, power = 1.5)
            X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, schedule = schedule, verbose=False, phi = phi)
        else:
            schedule = get_schedule('power_law', t_min = diffuser.diffmodel.sde.tmin, t_max = TIME_STEP.item(), n_iter = 200, power = 2)
            X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, schedule = schedule, verbose=False, phi = phi)
    else:
        X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, initial_timestep=TIME_STEP, verbose=False, phi = phi)

    ps = diffuser.diffmodel.ps(phi)
    sq_ps = torch.sqrt(ps).to(device)
    _, mean, _ = diffuser.diffmodel.sde.sampling(X_0, TIME_STEP, sq_ps)
    epsilon_hat = (rescaled_batch - mean)/diffuser.diffmodel.sde.noise_level(timesteps).reshape(-1, 1, 1, 1)
    log_prob = lambda rphi: log_posterior(rphi, epsilon_hat)

    def log_prob_grad(rphi):
        """
	    Compute the log posterior and its gradient.
		"""
        log_prob = log_posterior(rphi, epsilon_hat)
        grad_log_prob = torch.autograd.grad(log_prob, rphi, grad_outputs=torch.ones_like(log_prob))[0]
        return log_prob, grad_log_prob

    hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
    kwargs = {'nsamples': 1,
			'burnin': 10,
			'step_size':3e-3,
			'nleap': 5}
    epsadapt = 0

    rphi_0 = sample_prior(NUM_CHAIN).requires_grad_().to(device)
    rphi_0 = rphi_list[-1].requires_grad_() if len(rphi_list) > 0 else rphi_0
    sampler = hmc.sample(rphi_0, epsadapt = epsadapt, verbose = False, **kwargs)
    rphi = sampler.samples[:,-1]
    rphi_list.append(rphi) 
    phi = ps_model.unscale_phi(rphi)
    phi_list.append(phi)
    phi.detach()
    phi.requires_grad_()
    phi = phi.to(device)
    progress_bar.update(1)

    if n>0 and (n % CKPT_FREQ == 0 or n == NUM_SAMPLES+BURNIN_MCMC-1):
        tensor_rphi_list = [rphi.unsqueeze(0) for rphi in rphi_list]
        tensor_rphi_list = torch.cat(tensor_rphi_list, dim=0)
        dict_to_save = {'phi_true': phi_true, 'rphi_true': rphi_true, 'tensor_rphi_list' : tensor_rphi_list}
        torch.save(dict_to_save, args.save_path)
progress_bar.close()
