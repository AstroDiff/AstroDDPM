## Standard imports
import os
import torch
import tqdm
import argparse

## Relative imports
from astroddpm.runners import Diffuser, config_from_id, get_samples
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE, ContinuousSDE, ContinuousVPSDE
from astroddpm.diffusion.stochastic.solver import get_schedule
from astroddpm.diffusion.models.network import ResUNet
from astroddpm.moment.models import MomentModel, TparamMomentNetwork

from inference.cmb_ps import CMBPS
from inference.utils import unnormalize_phi, normalize_phi, log_prior_phi, sample_prior_phi, log_likelihood_eps_phi, get_phi_bounds
from inference.hmc import HMC

## Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Parser
parser = argparse.ArgumentParser(description='Run MCMC on a given model to calibrate the posterior')

parser.add_argument('--model_id', type=str, default='ContinuousSBM_ContinuousVPSDE_I_BPROJ_bottleneck_32_firstc_10_phi_beta_cosine_betamax_0.5_betamin_0.01')
parser.add_argument('--moment_model_id', type=str, default = 'MomentModel_ContinuousVPSDE_I_BPROJ_beta_0_0.01_beta_T_0.5_beta_schedule_cosine')
parser.add_argument('--num_chain', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=0.1)
parser.add_argument('--save_path', type=str, default='/mnt/home/dheurtel/ceph/04_inference/dummy_results.pt')

args = parser.parse_args()

## Constants
MODEL_ID = args.model_id
MOMENT_MODEL_ID = args.moment_model_id
NUM_CHAIN = args.num_chain
NUM_SAMPLES = args.num_samples
NOISE_LEVEL = args.noise_level
BURNIN_MCMC = 5
BURNIN_HEURISTIC = 1
CKPT_FREQ = 20

save_path_dir = os.path.dirname(args.save_path)
os.makedirs(save_path_dir, exist_ok=True)

### Loading the diffusion model and ps sampler 

norm_phi_mode = 'compact'               # Normalization mode for phi among ['compact', 'inf', None]
phi_min, phi_max = get_phi_bounds()     # Bounds on phi (unnormalized)

placeholder_dm = DiscreteSBM(DiscreteVPSDE(1000), ResUNet())
diffuser = Diffuser(placeholder_dm)
diffuser.load(config=config_from_id(MODEL_ID), also_ckpt=True, for_training=True)

TIME_STEP = diffuser.diffmodel.sde.get_closest_timestep(torch.tensor(NOISE_LEVEL))
print(f"Time step chosen for noise level {NOISE_LEVEL}: {TIME_STEP}")

TIME_STEP = TIME_STEP.to(device)

ps_model = CMBPS(norm_input_phi=norm_phi_mode).to(device)

### Loading the moment model
ckpt_moment_model = torch.load(os.path.join('/mnt/home/dheurtel/ceph/02_checkpoints/', MOMENT_MODEL_ID, 'ckpt.pt'))

config_moment_model = ckpt_moment_model['config']
moment_network = TparamMomentNetwork(**config_moment_model['network']).to(device)
config_moment_model['sde'].pop('type')
## SDE was poorly saved
sde_moment_model = ContinuousVPSDE(**config_moment_model['sde'])
sde_moment_model.beta_0 = 0.01
sde_moment_model.beta_T = 0.5

moment_model = MomentModel(moment_network, sde_moment_model, ps = diffuser.diffmodel.ps).to(device)
## Ps will be disregarded
moment_model.load_state_dict(ckpt_moment_model['model'])

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

## Helpers for HMC bounday management

phi_min_norm, phi_max_norm = get_phi_bounds(device=device)
phi_min_norm, phi_max_norm = normalize_phi(phi_min_norm, mode=norm_phi_mode), normalize_phi(phi_max_norm, mode=norm_phi_mode)

def collision_manager(q, p, p_nxt):
    p_ret = p_nxt
    nparams = q.shape[-1]
    for i in range(nparams):
        crossed_min_boundary = q[..., i] < phi_min_norm[i]
        crossed_max_boundary = q[..., i] > phi_max_norm[i]

        # Reflecting boundary conditions
        p_ret[..., i][crossed_min_boundary] = -p[..., i][crossed_min_boundary]
        p_ret[..., i][crossed_max_boundary] = -p[..., i][crossed_max_boundary]

    return p_ret

timesteps = TIME_STEP * torch.ones(NUM_CHAIN, dtype=torch.int32).to(device)

## Building the artificial superposition of CMB and Dust

batch = next(iter(diffuser.test_dataloader))

if len(batch.shape) == 3:
    batch = batch.unsqueeze(1)

batch = batch.to(device)

# Repeat the batch if necessary
if len(batch)<NUM_CHAIN:
    batch = batch.repeat((NUM_CHAIN//len(batch)+1, 1, 1, 1))
batch = batch[:NUM_CHAIN]

# Targets
rphi_true = sample_prior(NUM_CHAIN).to(device)
phi_true = unnormalize_phi(rphi_true, mode=norm_phi_mode)
ps_true = ps_model(rphi_true).unsqueeze(1)
sq_ps_true = torch.sqrt(ps_true).to(device).to(torch.float32)

noisy_batch, mean_batch, noise_batch = diffuser.diffmodel.sde.sampling(batch, timesteps, sq_ps_true)

rescaled_batch = diffuser.diffmodel.sde.rescale_preserved_to_additive(noisy_batch, timesteps)


## Initialization of the MCMC chains

phi_list = []
rphi_list = []

with torch.no_grad():
	TIME_STEP_MOMENT = sde_moment_model.get_closest_timestep(torch.tensor(NOISE_LEVEL))
	timesteps_moment = TIME_STEP_MOMENT * torch.ones((NUM_CHAIN,1), dtype=torch.int32).to(device)
	rphi = (moment_network(noisy_batch, timesteps_moment)+1)/2
	phi = unnormalize_phi(rphi, mode=norm_phi_mode).to(device)

	phi_list.append(phi)
	rphi_list.append(rphi)

#rphi = sample_prior(NUM_CHAIN).to(device)
#phi = unnormalize_phi(rphi, mode=norm_phi_mode)

progress_bar = tqdm.tqdm(range(NUM_SAMPLES+BURNIN_MCMC + BURNIN_HEURISTIC))
timesteps_min = torch.tensor([diffuser.diffmodel.sde.tmin]).to(device).repeat(NUM_CHAIN)
for n in range(NUM_SAMPLES+BURNIN_MCMC+BURNIN_HEURISTIC):
	if isinstance(diffuser.diffmodel.sde, ContinuousSDE):
		schedule = get_schedule('power_law', t_min = timesteps_min, t_max = timesteps, n_iter = 600, power = 2)
		X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, schedule = schedule.to(device), verbose=False, phi = phi)
	else:
		X_0 = diffuser.diffmodel.generate_image(NUM_CHAIN, sample = noisy_batch, initial_timestep=TIME_STEP, verbose=False, phi = phi)
	#_, mean, _ = diffuser.diffmodel.sde.sampling(X_0, TIME_STEP)
	epsilon_hat = (rescaled_batch - X_0)/diffuser.diffmodel.sde.noise_level(timesteps).reshape(-1, 1, 1, 1)
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
		hmc.set_collision_fn(collision_manager)

		samples = hmc.sample(rphi, nsamples=1, burnin=100, step_size=1e-6, nleap = (5,15), epsadapt=100, verbose = False, ret_side_quantities=False)
		step_size = hmc.step_size
		inv_mass_matrix = hmc.mass_matrix_inv
	else:
		hmc = HMC(log_prob, log_prob_and_grad=log_prob_grad, inv_mass_matrix=inv_mass_matrix)
		hmc.set_collision_fn(collision_manager)
		samples = hmc.sample(rphi, nsamples=1, burnin=10, step_size=step_size, nleap = (5,15), epsadapt=0, verbose = False)

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