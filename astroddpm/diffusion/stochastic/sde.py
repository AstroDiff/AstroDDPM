"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np
import os
import warnings

## TODO: add a link to Yang Song's paper, code and to our papers
## We have made the following changes to the code/score formula:
## 1. The output of the network will be - Sigma^2 score instead of score
##      This allow easier implementation of the loss and reverse SDE...
##      at no cost (up to a time dependent rescaling of the loss). 
##      Additionally, this coincides with the initial DDPM formulation 
##      while also making it easier to use matrices for Sigma, including 
##      matrices that are diagonal in the Fourrier space and that would take 
##      a large amount of memory if we were to use the original formulation.
## 2. reverse is no longer a class-wide method, it is an abstract method, that
##      needs to be implemented by the child class. This is because the reverse
##      SDE can take harder forms than just a simple change of sign and rescaling
##      of the score.
## 3. TODO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiscreteSDE(abc.ABC):
    def __init__(self,N):
        super(DiscreteSDE, self).__init__()
        self.N = N
    @abc.abstractmethod
    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the Discretized SDE
        pass
    @abc.abstractmethod
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the noise terms
        pass
    @abc.abstractmethod
    def reverse(self, x, t, modified_score):
        ## Returns the drift and brownian term of the reverse Discretized SDE for a given modified score
        ## x_i = x_{i+1} + discretized_drift + discretized_brownian
        pass
    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass
    @abc.abstractmethod
    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        pass
    @abc.abstractmethod
    def prior_log_likelihood(self, z):
        pass
    @abc.abstractmethod
    def rescale_additive_to_preserved(self, x, t):
        pass
    @abc.abstractmethod
    def rescale_preserved_to_additive(self, x, t):
        pass
    @abc.abstractmethod
    def noise_level(self, t):
        pass
    @abc.abstractmethod
    def tweedie_reverse(self, x, t, modified_score):
        pass

class DiscreteVPSDE(DiscreteSDE):
    def __init__(self, N, beta_min = 0.1, beta_max = 20.0, beta_schedule = 'linear'):
        super(DiscreteVPSDE, self).__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.beta_schedule = beta_schedule

        if beta_schedule == 'linear':
            self.beta = torch.linspace(self.beta_min/N, self.beta_max/N, N, dtype=torch.float32).to(device) ## Discretized beta(t) (for the Euler Maruyama scheme)
            self.t = torch.linspace(1/N, 1, N, dtype=torch.float32).to(device) ## Discretized t
            self.Beta = (beta_min * self.t + 1/2*(beta_max - beta_min) * self.t**2) ## Primitive of beta(t) then discretized
        elif beta_schedule == 'cosine':
            self.t = torch.linspace(1/N, 1, N, dtype=torch.float32).to(device)
            self.beta = self.beta_min/N + (self.beta_max - self.beta_min)*(1 - torch.cos(np.pi*self.t))/(2*N)
            self.Beta = beta_min * self.t + (self.beta_max - self.beta_min)*(self.t - torch.sin(np.pi*self.t)/np.pi)/2
        else:
            raise NotImplementedError('The beta schedule {} is not implemented'.format(beta_schedule))
        self.config = {'type' : 'DiscreteVPSDE', 'beta_min':self.beta_min, 'beta_max':self.beta_max, 'N':self.N, 'beta_schedule':beta_schedule}

    def foward(self, x_t, t, sq_ps = None):
        ## Returns the drift and brownian term of the Discretized SDE
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        drift = - beta_t/2* x_t
        if sq_ps is None:
            brownian = torch.sqrt(beta_t) * torch.randn_like(x_t)
        else:
            brownian = torch.sqrt(beta_t) * torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x_t))).real
        return drift, brownian
    
    def sampling(self, x, t, sq_ps = None):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms
        mean = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)*x
        if sq_ps is None:
            seed = torch.randn_like(x)
        else:
            seed = torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
        x_tilde = mean + torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*seed
        return x_tilde, mean, seed

    def reverse(self, x, t, modified_score, sq_ps = None):
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
        drift = (beta_t/2 ) * x - (beta_t/sq_1_expB_t)*modified_score ## modified_score is the noise estimate so we will have x_i = some term - estimated noise + discretized_brownian
        if sq_ps is None:
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.randn_like(x)
        else:
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
        return drift, brownian

    def tweedie_reverse(self, x, t, modified_score):
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
        sq_expBt = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)
        return (x - sq_1_expB_t*modified_score)/sq_expBt

    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
        drift = (beta_t/2 ) * x - 1/2*(beta_t/sq_1_expB_t)*modified_score
        return drift

    def prior_sampling(self, shape, sq_ps = None):
        if sq_ps is None:
            return torch.randn(shape).to(device)
        else:
            return torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn(shape).to(device))).real.to(device)

    def prior_log_likelihood(self, z, ps = None):
        ''' Assumes the image z is a B C H W image (and, in the case of non white noise, that channels are independent)'''
        if ps is None:
            ## Assumes it is a B C, H, W image
             return -1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z**2, dim=[1, 2, 3])) ## +log(1)
        else:
            return -1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z*torch.fft.ifft2(ps**-1 * torch.fft.fft2(z)).real, dim=(1, 2,3)) + torch.sum(torch.log(ps), dim=(1, 2, 3))) ## Assuming independent channels (zero cross spectra)

    def rescale_additive_to_preserved(self, x, t):
        return x * torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)## TODO view?

    def rescale_preserved_to_additive(self, x, t):
        return x / torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)

    def noise_level(self, t):
        return torch.sqrt(1-torch.exp(-self.Beta[t]))/torch.exp(-self.Beta[t]/2)
    
    def get_closest_timestep(self, noise_level):
        all_time_steps = torch.arange(self.N).long().to(device)
        return torch.argmin(torch.abs(self.noise_level(all_time_steps) - noise_level))

class ContinuousSDE(abc.ABC):
    def __init__(self):
        super(ContinuousSDE, self).__init__()
    @abc.abstractmethod
    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the SDE
        pass
    @abc.abstractmethod
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the noise terms
        pass
    @abc.abstractmethod
    def reverse(self, x, t, modified_score):
        ## Returns the drift and brownian term of the reverse SDE for a given modified score
        pass
    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass
    @abc.abstractmethod
    def prior_log_likelihood(self, z):
        pass
    @abc.abstractmethod
    def rescale_additive_to_preserved(self, x, t):
        pass
    @abc.abstractmethod
    def rescale_preserved_to_additive(self, x, t):
        pass
    @abc.abstractmethod
    def noise_level(self, t):
        pass
    @abc.abstractmethod
    def tweedie_reverse(self, x, t, modified_score):
        pass
    @abc.abstractmethod
    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        pass

class ContinuousVPSDE(ContinuousSDE):
    def __init__(self, beta_min = 0.1, beta_max = 20.0, t_min = 1e-4, t_max = 1.0, beta_schedule = 'linear'):
        super(ContinuousVPSDE, self).__init__()
        self.beta_0 = beta_min
        self.beta_T = beta_max
        self.tmin = t_min
        self.tmax = t_max
        if self.tmax < self.tmin:
            raise ValueError('t_max must be greater than t_min')
        if self.tmax != 1.0:
            warnings.warn('t_max != 1.0 is not implemented yet and behavior is not guaranteed')
        self.beta_schedule = beta_schedule
        if beta_schedule == 'linear':
            self.beta = lambda t: (self.beta_0 +  (self.beta_T - self.beta_0) * t)
            self.Beta = lambda t: (self.beta_0 * t + 1/2*(self.beta_T - self.beta_0) * t**2)
        elif beta_schedule == 'cosine':
            self.beta = lambda t: self.beta_0 + (self.beta_T - self.beta_0)*(1 - torch.cos(np.pi*t))/(2)
            self.Beta = lambda t: self.beta_0 * t + (self.beta_T - self.beta_0)*(t - torch.sin(np.pi*t)/np.pi)/2

        self.config = {'type' : 'ContinuousVPSDE', 'beta_min':beta_min, 'beta_max':beta_max, 't_min':t_min, 't_max':t_max, 'beta_schedule':beta_schedule}

    def foward(self, x_t, t, sq_ps = None):
        ## Returns the drift and brownian term of the forward SDE
        beta_t = self.beta(t).reshape(-1, 1, 1, 1)
        drift = (- beta_t/2)* x_t
        if sq_ps is None:
            brownian = torch.sqrt(beta_t)*torch.randn_like(x_t)
        else:
            brownian = torch.sqrt(beta_t)*torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x_t))).real
        return drift, brownian
    
    def sampling(self, x, t, sq_ps = None):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms
        Beta_t = self.Beta(t).reshape(-1, 1, 1, 1)
        mean = torch.exp(-Beta_t/2)*x
        if sq_ps is None:
            seed = torch.randn_like(x)
        else:
            seed = torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
        noise = torch.sqrt(1 - torch.exp(-Beta_t))*seed
        x_tilde = mean + noise
        return x_tilde, mean, seed

    def reverse(self, x, t, modified_score, sq_ps = None):
        beta_t = self.beta(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1, 1, 1)
        drift = -(beta_t/2 ) * x + (beta_t/sq_1_expB_t)*modified_score 
        if sq_ps is None:
            brownian = torch.sqrt(beta_t)*torch.randn_like(x)
        else:
            brownian = torch.sqrt(beta_t)*torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
        return drift, brownian

    def tweedie_reverse(self, x, t, modified_score):
        t = torch.clamp(t, self.tmin, self.tmax)
        Beta_t = self.Beta(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-Beta_t))
        sq_expBt = torch.exp(-Beta_t/2)
        return (x - sq_1_expB_t*modified_score)/sq_expBt

    def ode_drift(self, x, t, modified_score):
        beta_t = self.beta(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1, 1, 1)
        drift = (beta_t/2 ) * x - (1/2 * beta_t/sq_1_expB_t)*modified_score 
        return drift

    def prior_sampling(self, shape, sq_ps = None):
        if sq_ps is None:
            return torch.randn(shape).to(device)
        else:
            return torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(sq_ps))).real.to(device)

    def prior_log_likelihood(self, z, ps = None):
        if ps is None:
            return 1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z**2, dim=[1, 2, 3]))
        else:
            return -1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z*torch.fft.ifft2(ps**-1 * torch.fft.fft2(z)).real, dim=(1, 2,3)) + torch.sum(torch.log(ps), dim=(1, 2, 3)))

    def rescale_additive_to_preserved(self, x, t):
        return x * torch.exp(-self.Beta(t)/2).reshape(-1, 1, 1, 1)

    def rescale_preserved_to_additive(self, x, t):
        return x / torch.exp(-self.Beta(t)/2).reshape(-1, 1, 1, 1)

    def noise_level(self, t):
        return torch.sqrt(1 - torch.exp(-self.Beta(t)))/torch.exp(-self.Beta(t)/2)
    
    def get_closest_timestep(self, noise_level, n_step_method = 20, method = 'newton'):
        ## Solves the equation noise_level = self.noise_level(t) for t
        ## This is done analytically, noting that noise_level = sqrt(1 - exp(-Beta))/exp(-Beta/2)
        ## We first substitute x = exp(-Beta/2) and solve for x, then we solve for t (simple polynomial)
        if self.beta_schedule == 'linear':
            delta = self.beta_0**2 + 2*(self.beta_T - self.beta_0)*torch.log(1+noise_level**2)
            timesteps = (-self.beta_0 + torch.sqrt(delta))/(self.beta_T - self.beta_0)
            return torch.clamp(timesteps, self.tmin, self.tmax) ## TODO: check if this is correct
        elif self.beta_schedule == 'cosine':
            if type(noise_level) == float or type(noise_level) == int:
                noise_level = torch.tensor(noise_level)
            t_guess = 1/2*torch.ones_like(noise_level)
            if method == 'implicit':
                def implicit_func(t):
                    return (torch.log(1+noise_level**2)+(self.beta_T - self.beta_0)*torch.sin(np.pi*t)/(2*np.pi))*(2/(self.beta_T + self.beta_0))
                ## Solve implicit_func(t) = t
                timesteps = t_guess
                for _ in range(n_step_method):
                    timesteps = implicit_func(timesteps)
            elif method == 'newton':
                def newton_func(t):
                    return (self.beta_T + self.beta_0)/2*t - (torch.log(1+noise_level**2)+(self.beta_T - self.beta_0)*torch.sin(np.pi*t)/(2*np.pi))
                def newton_func_prime(t):
                    return (self.beta_T + self.beta_0)/2 - (self.beta_T - self.beta_0)*torch.cos(np.pi*t)/2
                timesteps = t_guess
                for _ in range(n_step_method):
                    timesteps = timesteps - newton_func(timesteps)/newton_func_prime(timesteps)
            return torch.clamp(timesteps, self.tmin, self.tmax)
        else:
            raise NotImplementedError('The beta schedule {} is not implemented'.format(self.beta_schedule))

def get_sde(config): ## TODO factorize the code
    """Returns an SDE object from a config dictionary."""
    sto_diff_eq = None
    if "type" not in config.keys():
        config["type"] = "DiscreteVPSDE"
    if config["type"].lower() == "discretevpsde":
        if "beta_min" not in config.keys():
            config["beta_min"] = 0.1
        if "beta_max" not in config.keys():
            config["beta_max"] = 20
        if "N" not in config.keys():
            config["N"] = 1000
        if "beta_schedule" not in config.keys():
            config["beta_schedule"] = 'linear'

        sto_diff_eq = DiscreteVPSDE(beta_min=config["beta_min"], beta_max=config["beta_max"], N=config["N"], beta_schedule=config["beta_schedule"])

    elif config["type"].lower() == "continuousvpsde":
        if "beta_min" not in config.keys():
            config["beta_min"] = 0.1
        if "beta_max" not in config.keys():
            config["beta_max"] = 20
        if 't_max' not in config.keys():
            config['t_max'] = 1.0
        if "t_min" not in config.keys():
            config["t_min"] = 1e-4
        if "beta_schedule" not in config.keys():
            config["beta_schedule"] = 'linear'

        sto_diff_eq = ContinuousVPSDE(beta_min=config["beta_min"], beta_max=config["beta_max"], t_min=config["t_min"], t_max=config["t_max"], beta_schedule=config["beta_schedule"])

    if sto_diff_eq is None:
        warnings.warn("There was a problem with the SDE config, using default DiscreteVPSDE")
        sto_diff_eq = DiscreteVPSDE()
    return sto_diff_eq