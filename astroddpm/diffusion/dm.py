import torch
from torch import nn
import tqdm
import numpy as np

from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import stochastic
from . import models

## TODO add __repr__ to all models
## TODO add DDIM


class DiffusionModel(nn.Module):
    def __init__(self, sde, network):
        super(DiffusionModel, self).__init__()
        self.sde = sde
        self.network = network
    def loss(self, batch, timesteps):
        raise NotImplementedError
    def step(self, model_output, timestep, sample): ## TODO remove that method?
        raise NotImplementedError
    def generate_image(self, sample_size, channel, size, sample=None, initial_timestep=None):
        raise NotImplementedError
    def ddim(self, sample_size, channel, size, schedule):
        raise NotImplementedError

class DiscreteSBM(DiffusionModel):
    def __init__(self, sde, network):
        super(DiscreteSBM, self).__init__(sde, network)
        self.sde = sde
        self.network = network

        self.config = { "sde" : self.sde.config, "network" : self.network.config, "type" : "DiscreteSBM"}
    def loss(self, batch, timesteps):
        batch_tilde, _ , rescaled_noise = self.sde.sampling(batch, timesteps)
        rescaled_noise_pred = self.network(batch_tilde, timesteps)
        return F.mse_loss(rescaled_noise_pred, rescaled_noise)
    def step(self, model_output, timestep, sample):
        drift, brownian = self.sde.reverse(sample, timestep, model_output)
        return sample + drift + brownian
    def ode_step(self, model_output, timestep, sample):
        drift = self.sde.ode_drift(sample, timestep, model_output)
        return sample + drift 
    def generate_image(self, sample_size, sample=None, initial_timestep=None, verbose=True):
        self.eval()

        channel, size = self.network.in_c, self.network.sizes[0]
        
        if initial_timestep is None:
            tot_steps = self.sde.N
        else:
            tot_steps = initial_timestep
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            if sample is None:
                sample = self.sde.prior_sampling((sample_size, channel, size, size))
            progress_bar = tqdm.tqdm(total=tot_steps, disable=not verbose)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.network(sample, time_tensor)
                sample = self.step(residual, time_tensor[0], sample)
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        return sample
    def ode_sampling(self, sample_size, sample = None, initial_timestep = None, verbose=True): ## TODO scheduler ? 
        self.eval()

        channel, size = self.network.in_c, self.network.sizes[0]

        if initial_timestep is None:
            tot_steps = self.sde.N
        else:
            tot_steps = initial_timestep
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            if sample is None:
                sample = self.sde.prior_sampling((sample_size, channel, size, size))
            progress_bar = tqdm.tqdm(total=tot_steps, disable=not verbose)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.network(sample, time_tensor)
                sample = self.ode_step(residual, time_tensor[0], sample)
                progress_bar.update(1)
            progress_bar.close()

        return sample
    
    def log_likelihood(self, batch, initial_timestep = None, verbose=True, repeat = 1):
        '''Sample in forward time the ODE and compute the log likelihood of the batch, see [REF]'''
        self.eval()
        log_likelihood = torch.zeros(len(batch)).to(device)
        with torch.no_grad():
            N = self.sde.N
            progress_bar = tqdm.tqdm(total=N, disable=not verbose)
            for i in range(N):
                timesteps = torch.tensor([i]).repeat(len(batch)).to(device)
                modified_score = self.network(gen, timesteps)
                gen -= self.sde.ode_drift(gen, timesteps, modified_score)
                progress_bar.update(1)
                log_likelihood_increase = 0
                with torch.enable_grad():
                    for _ in range(repeat):
                        epsilon = torch.randn_like(batch)
                        gen.requires_grad = True
                        reduced = torch.sum(modified_score * epsilon)
                        grad = torch.autograd.grad(reduced, gen, create_graph=True)[0]
                        gen.requires_grad = False
                        log_likelihood_increase += torch.sum(grad * epsilon, dim=(1, 2, 3))
                log_likelihood_increase /= repeat
                log_likelihood += log_likelihood_increase/(N-initial_timestep)
            progress_bar.close()
            log_likelihood += self.sde.prior_log_likelihood(batch)
        self.train()
        return log_likelihood

        
        

class ContinuousSBM(DiffusionModel):
    def __init__(self, sde, network):
        super().__init__(sde, network)
        raise NotImplementedError

class DiscreteLatentSBM(DiffusionModel):
    def __init__(self, sde, network):
        super().__init__(sde, network)
        raise NotImplementedError


compatibility_sde_network = {"DiscreteSDE" : ["resunet"] }

def get_diffusion_model(config):
    '''Returns the diffusion model from the config dict'''

    if "sde" not in config.keys():
        config["sde"] = {}
    sde_config = config["sde"]

    sto_diff_eq = stochastic.sde.get_sde(sde_config)

    if "network" not in config.keys():
        config["network"] = {}
    network_config = config["network"]

    if isinstance(sto_diff_eq, stochastic.sde.DiscreteSDE):
        if "type" in network_config.keys():
            if network_config["type"].lower() not in compatibility_sde_network["DiscreteSDE"]:
                raise ValueError("Network {} not compatible with discrete SDE".format(network_config["type"]))
        else:
            network_config["type"] = "resunet"
        network_config["n_steps"] = sto_diff_eq.N
    
    ## TODO continuous diffusion model

    network = models.network.get_network(network_config)

    if "type" not in config.keys():
        config["type"] = "DiscreteSBM"
    if config["type"].lower() == "discretesbm":
        return DiscreteSBM(sto_diff_eq, network)
    else:
        raise NotImplementedError("Diffusion model {} not implemented for now".format(config["type"]))