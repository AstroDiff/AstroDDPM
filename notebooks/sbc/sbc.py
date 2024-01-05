## Standard imports
import os
import torch
import tqdm

## Relative imports
from astroddpm.runners import Diffuser, config_from_id, get_samples
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE, ContinuousSDE
from astroddpm.diffusion.stochastic.solver import get_schedule
from astroddpm.diffusion.models.network import ResUNet

from inference.cmb_ps import CMBPS
from inference.utils import unnormalize_phi, normalize_phi, log_prior_phi, sample_prior_phi, log_likelihood_eps_phi, get_phi_bounds
from inference.hmc import HMC

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
BURNIN_HEURISTIC = 1

CKPT_FREQ = 20

save_path_dir = os.path.dirname(args.save_path)
os.makedirs(save_path_dir, exist_ok=True)

norm_phi_mode = 'compact'               # Normalization mode for phi among ['compact', 'inf', None]
phi_min, phi_max = get_phi_bounds()     # Bounds on phi (unnormalized)

placeholder_dm = DiscreteSBM(DiscreteVPSDE(1000), ResUNet())
diffuser = Diffuser(placeholder_dm)
diffuser.load(config=config_from_id(MODEL_ID), also_ckpt=True, for_training=True)

TIME_STEP = diffuser.diffmodel.sde.get_closest_timestep(torch.tensor(NOISE_LEVEL))
print(f"Time step chosen for noise level {NOISE_LEVEL}: {TIME_STEP}")

TIME_STEP = TIME_STEP.to(device)

ps_model = CMBPS(norm_input_phi=norm_phi_mode).to(device)

#
# Prior, likelihood, and posterior functions
#

sample_prior = lambda n: sample_prior_phi(n, norm=norm_phi_mode, device=device)
log_likelihood = lambda phi, x: log_likelihood_eps_phi(phi, x, ps_model)
log_prior = lambda phi: log_prior_phi(phi, norm=norm_phi_mode)

def log_posterior(phi, x):
    """
    Compute the log posterior of the parameters (not normalized by the evidence).
    """
    return log_likelihood(phi, x) + log_prior(phi)

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

rphi_true = sample_prior(NUM_CHAIN).to(device)
phi_true = unnormalize_phi(rphi_true, mode=norm_phi_mode)
ps_true = ps_model(rphi_true).unsqueeze(1)
sq_ps_true = torch.sqrt(ps_true).to(device).to(torch.float32)

noisy_batch, mean_batch, noise_batch = diffuser.diffmodel.sde.sampling(batch, timesteps, sq_ps_true)

rescaled_batch = diffuser.diffmodel.sde.rescale_preserved_to_additive(noisy_batch, timesteps)

rphi = sample_prior(NUM_CHAIN).to(device)
phi = unnormalize_phi(rphi, mode=norm_phi_mode)

progress_bar = tqdm.tqdm(range(NUM_SAMPLES+BURNIN_MCMC + BURNIN_HEURISTIC))

for n in range(NUM_SAMPLES+BURNIN_MCMC+BURNIN_HEURISTIC):
	if isinstance(diffuser.diffmodel.sde, ContinuousSDE):
		schedule = get_schedule('power_law', t_min = diffuser.diffmodel.sde.tmin, t_max = TIME_STEP.item(), n_iter = 600, power = 2)
		X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, schedule = schedule.to(device), verbose=False, phi = phi)
	else:
		X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, initial_timestep=TIME_STEP, verbose=False, phi = phi)
	_, mean, _ = diffuser.diffmodel.sde.sampling(X_0, TIME_STEP)
	epsilon_hat = (rescaled_batch - mean)/diffuser.diffmodel.sde.noise_level(timesteps).reshape(-1, 1, 1, 1)
	epsilon_hat = epsilon_hat[:, 0, :, :]

	log_prob = lambda rphi: log_posterior(rphi, epsilon_hat)
	def log_prob_grad(rphi):
		""" Compute the log posterior and its gradient."""
		rphi.requires_grad_(True)
		log_prob = log_posterior(rphi, epsilon_hat)
		grad_log_prob = torch.autograd.grad(log_prob, rphi, grad_outputs=torch.ones_like(log_prob))[0]
		return log_prob.detach(), grad_log_prob

	rphi = normalize_phi(phi, mode=norm_phi_mode)
	if n < BURNIN_HEURISTIC:
		hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad)
		samples = hmc.sample(rphi, nsamples=1, burnin=100, step_size=1e-6, nleap = 10, epsadapt=100, verbose = False, ret_side_quantities=False)
		step_size = hmc.step_size
		inv_mass_matrix = hmc.mass_matrix_inv
	else:
		hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad, inv_mass_matrix=inv_mass_matrix)
		samples = hmc.sample(rphi, nsamples=1, burnin=10, step_size=step_size, nleap = 10, epsadapt=0, verbose = False)

	rphi = samples[:,0,:]
	rphi_list.append(rphi)
	phi = unnormalize_phi(rphi, mode=norm_phi_mode)
	phi_list.append(phi)
	phi.detach()
	rphi.detach()
	phi.requires_grad_()
	phi = phi.to(device)
	if n > 0 and (n % CKPT_FREQ == 0 or n == NUM_SAMPLES + BURNIN_MCMC - 1):
		tensor_rphi_list = [rphi.unsqueeze(0) for rphi in rphi_list]
		tensor_rphi_list = torch.cat(tensor_rphi_list, dim=0)
		dict_to_save = {'phi_true': phi_true, 'rphi_true': rphi_true, 'tensor_rphi_list': tensor_rphi_list}
		torch.save(dict_to_save, args.save_path)
	progress_bar.update(1)

progress_bar.close()