import torch
from torch import nn
import tqdm
import numpy as np

from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from . import stochastic
from . import models
from .power_spectra import powerspec_sampler

## TODO add __repr__ to all models
## TODO add DDIM


class DiffusionModel(nn.Module):
    def __init__(self, sde, network):
        super(DiffusionModel, self).__init__()
        self.sde = sde
        self.network = network

    def loss(self, batch, timesteps): ## TODO exclude timesteps from the args
        raise NotImplementedError

    def step(self, model_output, timestep, sample): ## TODO remove that method?
        raise NotImplementedError

    def generate_image(self, sample_size, channel, size, sample=None, initial_timestep=None):
        raise NotImplementedError

    def ddim(self, sample_size, channel, size, schedule):
        raise NotImplementedError

class DiscreteSBM(DiffusionModel):
    def __init__(self, sde, network, ps = None):
        super(DiscreteSBM, self).__init__(sde, network)
        self.sde = sde
        self.network = network
        self.N = self.sde.N
        self.ps = ps
        if ps is not None:
            self.has_thetas = self.ps.has_thetas
        else:
            self.has_thetas = False
        self.config = { "sde" : self.sde.config, "network" : self.network.config, "type" : "DiscreteSBM"}
        if ps is None:
            self.config["ps"] = {}
        else:
            self.config["ps"] = self.ps.config

    def loss(self, batch):
        timesteps = (torch.randint(0, self.N, (batch.shape[0],)).long().to(device))
        if self.ps is None:
            batch_tilde, _ , rescaled_noise = self.sde.sampling(batch, timesteps)
            rescaled_noise_pred = self.network(batch_tilde, timesteps)
        else:
            if self.has_thetas:
                ps_tensor, thetas = self.ps.sample_ps(batch.shape[0])
                batch_tilde, _ , rescaled_noise = self.sde.sampling(batch, timesteps, torch.sqrt(ps_tensor))
                thetas = self.ps.rescale_theta(thetas)
                rescaled_noise_pred = self.network(batch_tilde, timesteps, thetas)
            else:
                ps_tensor = self.ps.sample_ps(batch.shape[0])
                batch_tilde, _ , rescaled_noise = self.sde.sampling(batch, timesteps, torch.sqrt(ps_tensor))
                rescaled_noise_pred = self.network(batch_tilde, timesteps)  
        return F.mse_loss(rescaled_noise_pred, rescaled_noise)

    def step(self, model_output, timestep, sample, sq_ps = None):
        drift, brownian = self.sde.reverse(sample, timestep, model_output, sq_ps = sq_ps)
        return sample + drift + brownian

    def ode_step(self, model_output, timestep, sample):
        drift = self.sde.ode_drift(sample, timestep, model_output)
        return sample + drift 

    def generate_image(self, sample_size, sample=None, initial_timestep=None, verbose=True, thetas = None, return_thetas = False):
        self.eval()
        ## Get the power spectrum of the noise
        if self.ps is None:
            sq_ps = None
        else:
            if self.has_thetas:
                if thetas is None:
                    ps, thetas = self.ps.sample_ps(sample_size)
                    ps, thetas = ps.to(device), thetas.to(device)
                    sq_ps = torch.sqrt(ps).to(device)
                    thetas = self.ps.rescale_theta(thetas)
                else:
                    ps = self.ps(thetas).to(device)
                    sq_ps = torch.sqrt(ps).to(device)
                    thetas = self.ps.rescale_theta(thetas)
            else:
                ps = self.ps.sample_ps(sample_size).to(device)
                sq_ps = torch.sqrt(ps).to(device)

        channel, size = self.network.in_c, self.network.sizes[0]
        ## Allow for different initial timesteps
        if initial_timestep is None:
            tot_steps = self.sde.N
        else:
            tot_steps = initial_timestep

        ## Generating loop
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            ## Initial sample = seed
            if sample is None:
                sample = self.sde.prior_sampling((sample_size, channel, size, size), sq_ps = sq_ps)

            progress_bar = tqdm.tqdm(total=tot_steps, disable=not verbose)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)

                if self.has_thetas:
                    residual = self.network(sample, time_tensor, thetas)
                else:
                    residual = self.network(sample, time_tensor)

                sample = self.step(residual, time_tensor[0], sample, sq_ps = sq_ps)
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        if self.has_thetas and return_thetas:
            return sample, thetas
        return sample

    def ode_sampling(self, sample_size, sample = None, initial_timestep = None, verbose=True, thetas = None, return_thetas = False): 
        ## TODO at least offer option to use RK (which would mean jumping over a few steps because we are already discretized)
        self.eval()
        ## Get the power spectrum of the noise
        if self.ps is None:
            sq_ps = None
        else:
            if self.has_thetas:
                if thetas is None:
                    ps, thetas = self.ps.sample_ps(sample_size)
                    ps, thetas = ps.to(device), thetas.to(device)
                    sq_ps = torch.sqrt(ps)
                    thetas = self.ps.rescale_theta(thetas)
                else:
                    ps = self.ps(thetas).to(device)
                    sq_ps = torch.sqrt(ps)
                    thetas = self.ps.rescale_theta(thetas).to(device)
            else:
                sq_ps = self.ps.sample_ps(sample_size).to(device)
        
        channel, size = self.network.in_c, self.network.sizes[0]
        ## Allow for different initial timesteps
        if initial_timestep is None:
            tot_steps = self.sde.N
        else:
            tot_steps = initial_timestep
        
        ## Generating loop
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            ## Initial sample = seed
            if sample is None:
                sample = self.sde.prior_sampling((sample_size, channel, size, size), sq_ps = sq_ps)

            progress_bar = tqdm.tqdm(total=tot_steps, disable=not verbose)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                if self.has_thetas:
                    residual = self.network(sample, time_tensor, thetas)
                else:
                    residual = self.network(sample, time_tensor)
                sample = self.ode_step(residual, time_tensor[0], sample)
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        if self.has_thetas and return_thetas:
            return sample, thetas
        return sample
    
    def forward_ode_sampling(self, sample, initial_timestep = None, verbose=True, thetas = None, return_thetas = False):
        self.eval()
        ## Get the power spectrum of the noise
        if self.ps is not None:
            if self.has_thetas:
                if thetas is None:
                    raise ValueError("Computing a latent code require the value of theta")
                else:
                    thetas = self.ps.rescale_theta(thetas).to(device)
        gen = sample.to(device).clone()
        if initial_timestep is None:
            initial_timestep = 0
        with torch.no_grad():
            N = self.sde.N
            progress_bar = tqdm.tqdm(total=N - initial_timestep, disable=not verbose)
            for i in range(initial_timestep,N):
                time_tensor = torch.tensor([i]).repeat(len(sample)).to(device)
                if self.has_thetas:
                    residual = self.network(sample, time_tensor, thetas)
                else:
                    residual = self.network(sample, time_tensor)
                gen -= self.sde.ode_drift(gen, time_tensor, residual)
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        if self.has_thetas and return_thetas:
            return gen, thetas
        return gen
    
    def log_likelihood(self, batch, initial_timestep = None, verbose=True, repeat = 1, thetas = None):
        '''Sample in forward time the ODE and compute the log likelihood of the batch, see [REF]'''
        self.eval()
        ## Get the power spectrum of the noise
        if self.ps is None:
            sq_ps = None
            ps = None
        else:
            if self.has_thetas:
                if thetas is None:
                    raise ValueError("Values to compute the power spectrum must be provided in order to compute log likelihoods")
                    ps, thetas = self.ps.sample_ps(batch.shape[0])
                    ps, thetas = ps.to(device), thetas.to(device)
                    sq_ps = torch.sqrt(ps)
                    thetas = self.ps.rescale_theta(thetas)
                else:
                    ps = self.ps(thetas).to(device)
                    sq_ps = torch.sqrt(ps)
                    thetas = self.ps.rescale_theta(thetas).to(device)
            else:
                ps = self.ps.sample_ps(batch.shape[0]).to(device)
                sq_ps = torch.sqrt(ps).to(device)
        
        if initial_timestep is None:
            initial_timestep = 0
        N = self.sde.N
        progress_bar = tqdm.tqdm(total=(N-initial_timestep), disable=not verbose)

        log_likelihood = torch.zeros(batch.shape[0]).to(device)
        gen = batch
        lls = []

        gen.requires_grad = True

        for i in range(initial_timestep, N):
            timesteps = torch.tensor([i]).repeat(len(gen)).to(device)
            if self.has_thetas:
                modified_score = self.network(gen, timesteps, thetas)
            else:
                modified_score = self.network(gen, timesteps)

            drift = - self.sde.ode_drift(gen, timesteps, modified_score)

            log_likelihood_increase = 0
            for j in range(repeat): ## Vmaping that would be nice but not possible as of now and what I can do (potentional vmaps are actually slower because of the need to recompute stuff)
                epsilon = torch.randn_like(gen)
                reduced = torch.sum(drift * epsilon)
                grad = torch.autograd.grad(reduced, gen, retain_graph= (j!=repeat-1))[0] ## Retain graph except for the last iteration
                log_likelihood_increase += torch.sum(grad * epsilon, dim=(1, 2, 3))
            
            # ODE step and updates
            gen = gen + drift
            log_likelihood_increase /= repeat
            log_likelihood += log_likelihood_increase
            lls.append(log_likelihood_increase)
            progress_bar.update(1)

        progress_bar.close()
        pre_prior_ll = log_likelihood.clone()
        log_likelihood += self.sde.prior_log_likelihood(gen, ps = ps)
        self.train()
        return log_likelihood, pre_prior_ll, lls, gen

class ContinuousSBM(DiffusionModel):
    def __init__(self, sde, network):
        raise NotImplementedError("Not implemented yet")
        super(ContinuousSBM, self).__init__(sde, network)
        self.sde = sde
        self.network = network
        self.config = { "sde" : self.sde.config, "network" : self.network.config, "type" : "ContinuousSBM"}
        self.tmin = self.sde.tmin
        self.tmax = self.sde.tmax

    def loss(self, batch):
        timesteps = (torch.rand(batch.shape[0]).to(device) * (self.tmax - self.tmin) + self.tmin).long()
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
        raise NotImplementedError("Not implemented yet")
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

    def ode_sampling(self, sample_size, sample = None, initial_timestep = None, verbose=True): ## TODO scheduler maybe only in continuous time? 
        self.eval()
        raise NotImplementedError("Not implemented yet")
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
        raise NotImplementedError("Not implemented yet")
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
                        ## TODO add zero grad
                log_likelihood_increase /= repeat
                log_likelihood += log_likelihood_increase/(N-initial_timestep)
            progress_bar.close()
            log_likelihood += self.sde.prior_log_likelihood(batch)
        self.train()
        return log_likelihood

class DiscreteLatentSBM(DiffusionModel):
    def __init__(self, sde, network):
        super().__init__(sde, network)
        raise NotImplementedError


compatibility_sde_network = {"DiscreteSDE" : ["resunet"] , "ContinuousSDE" : ["resunet"]}

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

    if "ps" not in config.keys():
        config["ps"] = {}
    ps_config = config["ps"]

    ps_sampler = powerspec_sampler.get_ps_sampler(ps_config)

    if "type" not in config.keys():
        config["type"] = "DiscreteSBM"
    if config["type"].lower() == "discretesbm":
        return DiscreteSBM(sto_diff_eq, network, ps = ps_sampler)
    else:
        raise NotImplementedError("Diffusion model {} not implemented for now".format(config["type"]))