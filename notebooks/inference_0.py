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
N_GIBBS = 100
N_HMC = 100
BURNIN_GIBBS = 400

placeholder_dm = DiscreteSBM(DiscreteVPSDE(1000), ResUNet())
diffuser = Diffuser(placeholder_dm)
diffuser.load(config=config_from_id(MODEL_ID), also_ckpt=True, for_training=True)
ps_model = CMBPS(norm_phi=True).to(device)

batch = next(iter(diffuser.test_dataloader))
image = batch.unsqueeze(1).to(device)[:1]

ps, theta_target = diffuser.diffmodel.ps.sample_ps(1)
rtheta_target = diffuser.diffmodel.ps.rescale_theta(theta_target)
sq_ps = torch.sqrt(ps).to(device)
theta_target = theta_target.to(device)

noisy, mean, noise = diffuser.diffmodel.sde.sampling(image, 100, sq_ps)
batch = image.repeat(NUM_CHAIN, 1, 1, 1)
ps_0, thetas_0 = diffuser.diffmodel.ps.sample_ps(NUM_CHAIN)
batch_mean = mean.repeat(NUM_CHAIN, 1, 1, 1)
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

thetas = thetas_0.clone()

progress_bar = tqdm.tqdm(range(N_GIBBS + BURNIN_GIBBS))
for n in range(N_GIBBS+ BURNIN_GIBBS):

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

    kwargs = {'nsamples': 2,
          'burnin': N_HMC,
          'step_size': 5e-6,
          'nleap': 10}

    phi_0 = sample_prior(NUM_CHAIN).requires_grad_()
    phi_0 = phi_list[-1].requires_grad_() if len(phi_list) > 0 else phi_0
    sampler = hmc.sample(phi_0, verbose = False, **kwargs)
    phi = sampler.samples[0]
    thetas = unnormalize_phi(phi)
    phi_list.append(phi)
    theta_list.append(thetas)
    thetas.detach()
    thetas.requires_grad_()
    progress_bar.update(1)
    
phi_list = [phi.unsqueeze(0) for phi in phi_list]
phi_list = torch.cat(phi_list, dim=0)
phi_list.shape
phi_test = normalize_phi(theta_target)
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
for i in range(phi_list.shape[1]):
    ax.scatter(phi_list[BURNIN_GIBBS:, i, 0].detach().cpu().numpy(), phi_list[BURNIN_GIBBS:,i, 1].detach().cpu().numpy(), s=4, alpha=0.5)
ax.axvline(phi_test[0,0].detach().cpu().numpy(), color='red', linestyle='--')
ax.axhline(phi_test[0,1].detach().cpu().numpy(), color='red', linestyle='--')
plt.xlabel(r"$H_0$")
plt.ylabel(r"$\Omega_b h^2$")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('gibbs.pdf')

torch.save(phi_list, 'phi_list.pt')