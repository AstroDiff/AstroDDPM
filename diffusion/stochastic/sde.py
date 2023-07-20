"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
from typing import Any
import torch
import numpy as np

######################
## TODO put a disclaimer that it is Yang Song's code
######################

## TODO: add a link to the paper
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
    def prior_log_likelihood(self, z):
        pass

class DiscreteVPSDE(DiscreteSDE):
    def __init__(self, N, beta_min = 0.1, beta_max = 20.0, ddpm_math = True):
        super(DiscreteVPSDE, self).__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = max(1e-4, beta_min/N)
        self.beta_T = beta_max/N
        self.betas = torch.linspace(self.beta_0, self.beta_T, N, dtype=torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.t = torch.linspace(0, 1, N, dtype=torch.float32).to(device)
        self.Beta = (beta_min * self.t + (beta_max - beta_min) * self.t**2) ## Primitive of beta(t) then discretized

        self.ddpm_math = ddpm_math

    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the Discretized SDE
        if not self.ddpm_math:
            beta_t = self.betas[t]
            drift = - beta_t/2* x_t
            brownian = torch.sqrt(beta_t)*torch.randn_like(x_t)
            return drift, brownian
        else:
            beta_t = self.betas[t]
            drift = (1 - np.sqrt(1-beta_t))*x_t
            brownian = torch.sqrt(beta_t)*torch.randn_like(x_t)
            return drift, brownian
    
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms ## TODO 
        if not self.ddpm_math:
            mean = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)*x
            noise = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*torch.randn_like(x)
            x_tilde = mean + noise
            return x_tilde, mean, noise
        else:
            sqrt_alpha_barre_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_barre_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            mean = sqrt_alpha_barre_t.reshape(-1, 1, 1, 1)*x
            noise = sqrt_one_minus_alpha_barre_t*torch.randn_like(x)
            x_tilde = mean + noise
            return x_tilde, mean, noise/sqrt_one_minus_alpha_barre_t

    def reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            raise NotImplementedError
            beta_t = self.betas[t]
            drift = - (beta_t/2 ).reshape(-1, 1, 1, 1) * x - modified_score ## modified_score is the noise estimate so we will have x_i = some term - estimated noise + discretized_brownian
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.randn_like(x)
            return drift, brownian
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (coef_first_t*coef_eps_t)*modified_score
            brownian = (self.betas[t] ** 0.5).reshape(-1, 1, 1, 1) * torch.randn_like(x)*(t > 0).reshape(-1, 1, 1, 1)
            return drift, brownian

    def prior_sampling(self, shape):
        return torch.randn(shape).to(device)
    def prior_log_likelihood(self, z):
        raise NotImplementedError
        return 0.0
    
class DiscreteSigmaVPSDE(DiscreteSDE):
    def __init__(self, N, power_spectrum = 'cmb', beta_min = 0.1, beta_max = 20.0, ddpm_math = True):
        super(DiscreteSigmaVPSDE, self).__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = max(1e-4, beta_min/N)
        self.beta_T = beta_max/N
        self.betas = torch.linspace(self.beta_0, self.beta_T, N, dtype=torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.t = torch.linspace(0, 1, N, dtype=torch.float32).to(device)
        self.Beta = (beta_min * self.t + (beta_max - beta_min) * self.t**2)

        self.ddpm_math = ddpm_math

        if type(power_spectrum)==str:
            try:
                pass
            except:
                raise ValueError('Power spectrum not recognized')
        elif type(power_spectrum)==np.ndarray:
            self.power_spectrum = torch.from_numpy(power_spectrum).to(device).type(torch.float32)
        elif type(power_spectrum)==torch.Tensor:
            self.power_spectrum = power_spectrum.to(device)
        else:
            raise ValueError('''Argument power_spectrum must be one of:
             - a string -> a common power spectrum stored in the library see add_power_spectrum NOT IMPLEMENTED
             - a numpy array or a torch tensor -> the power spectrum itself (cross spectrum not implemented)''')
        self.sqrt_ps = torch.sqrt(self.power_spectrum)
    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the Discretized SDE
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            drift = - beta_t/2* x_t
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x_t))).real
            return drift, brownian
        else:
            beta_t = self.betas[t]
            drift = (1 - torch.sqrt(1-beta_t)).reshape(-1, 1, 1, 1)*x_t
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x_t))).real
            return drift, brownian

    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms 
        if not self.ddpm_math:
            mean = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)*x
            noise = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + noise
            return x_tilde, mean, noise
        else:
            sqrt_alpha_barre_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_barre_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            mean = sqrt_alpha_barre_t.reshape(-1, 1, 1, 1)*x
            noise = sqrt_one_minus_alpha_barre_t*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + noise
            return x_tilde, mean, noise/sqrt_one_minus_alpha_barre_t
    def reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            raise NotImplementedError
            beta_t = self.betas[t]
            drift = - (beta_t/2 ).reshape(-1, 1, 1, 1) * x - modified_score
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            return drift, brownian
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (coef_first_t*coef_eps_t)*modified_score
            brownian = (self.betas[t] ** 0.5).reshape(-1, 1, 1, 1) * torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real*(t > 0).reshape(-1, 1, 1, 1)
            return drift, brownian
    def prior_sampling(self, shape):
        return torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn(shape)).to(device)).real
    def prior_log_likelihood(self, z):
        raise NotImplementedError ## TODO
        return 0.0
    

