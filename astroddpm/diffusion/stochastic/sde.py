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
    def __init__(self, N, beta_min = 0.1, beta_max = 20.0, ddpm_math = True):
        super(DiscreteVPSDE, self).__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = beta_min/N
        self.beta_T = beta_max/N
        self.betas = torch.linspace(self.beta_0, self.beta_T, N, dtype=torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.t = torch.linspace(1/N, 1, N, dtype=torch.float32).to(device)
        self.Beta = (beta_min * self.t + 1/2*(beta_max - beta_min) * self.t**2) ## Primitive of beta(t) then discretized

        self.ddpm_math = ddpm_math

        self.config = {'type' : 'DiscreteVPSDE', 'beta_min':self.beta_min, 'beta_max':self.beta_max, 'ddpm_math':self.ddpm_math, 'N':self.N}

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
            seed = torch.randn_like(x)
            x_tilde = mean + torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*seed
            return x_tilde, mean, seed
        else:
            sqrt_alpha_barre_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_barre_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            mean = sqrt_alpha_barre_t.reshape(-1, 1, 1, 1)*x
            seed = torch.randn_like(x)
            x_tilde = mean + sqrt_one_minus_alpha_barre_t*seed
            return x_tilde, mean, seed

    def reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x - (beta_t/sq_1_expB_t)*modified_score ## modified_score is the noise estimate so we will have x_i = some term - estimated noise + discretized_brownian
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
    def tweedie_reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            sq_expBt = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)
            return (x - sq_1_expB_t*modified_score)/sq_expBt
        else:
            s1 = self.sqrt_alphas_cumprod[t] # bs
            s2 = self.sqrt_one_minus_alphas_cumprod[t] # bs
            s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
            s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)

            batch_denoised= (x- s2* modified_score)/s1
            return batch_denoised
    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        if not self.ddpm_math:
            raise NotImplementedError
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (1/2*coef_first_t*coef_eps_t)*modified_score
            return drift

    def prior_sampling(self, shape):
        return torch.randn(shape).to(device)

    def prior_log_likelihood(self, z):
        shape = z.shape
        ndim = z.ndim
        if ndim == 2:
            return -1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z**2)) ## +log(1)
        elif ndim == 3: ## Assumes it is a C, H, W image
            return -1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z**2)) ## +log(1)
        elif ndim == 4: ## Assumes it is a B C, H, W image
            return -1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z**2, dim=[1, 2, 3])) ## +log(1)

    def rescale_additive_to_preserved(self, x, t):
        return x * self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def rescale_preserved_to_additive(self, x, t):
        return x / self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def noise_level(self, t):
        return self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)/self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    
class DiscreteSigmaVPSDE(DiscreteSDE):
    def __init__(self, N, power_spectrum = 'cmb', beta_min = 0.1, beta_max = 20.0, ddpm_math = True): ## TODO change default ps
        super(DiscreteSigmaVPSDE, self).__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = beta_min/N
        self.beta_T = beta_max/N
        self.betas = torch.linspace(self.beta_0, self.beta_T, N, dtype=torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.t = torch.linspace(1/N, 1, N, dtype=torch.float32).to(device)
        self.Beta = (self.beta_min * self.t + 1/2*(self.beta_max - self.beta_min) * self.t**2)

        self.ddpm_math = ddpm_math

        if type(power_spectrum)==str:
            self.power_spectrum_name = power_spectrum
            try:
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'power_spectra')
                self.power_spectrum = torch.from_numpy(np.load(os.path.join(folder, power_spectrum + '.npy'))).to(device).type(torch.float32)
            except:
                raise ValueError('Power spectrum not recognized')
        elif type(power_spectrum)==np.ndarray:
            self.power_spectrum = torch.from_numpy(power_spectrum).to(device).type(torch.float32)
            self.power_spectrum_name = 'custom'
        elif type(power_spectrum)==torch.Tensor:
            self.power_spectrum = power_spectrum.to(device)
            self.power_spectrum_name = 'custom'
        else:
            raise ValueError('''Argument power_spectrum must be one of:
             - a string -> a common power spectrum stored in the library see add_power_spectrum NOT IMPLEMENTED
             - a numpy array or a torch tensor -> the power spectrum itself (cross spectrum not implemented)''')
        self.sqrt_ps = torch.sqrt(self.power_spectrum)

        self.config = {'type' : 'DiscreteSigmaVPSDE', 'power_spectrum_name':self.power_spectrum_name, 'beta_min':self.beta_min, 'beta_max':self.beta_max, 'ddpm_math':self.ddpm_math, 'N':self.N}

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
            seed = torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*seed
            return x_tilde, mean, seed
        else:
            sqrt_alpha_barre_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_barre_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            mean = sqrt_alpha_barre_t.reshape(-1, 1, 1, 1)*x
            seed = torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + sqrt_one_minus_alpha_barre_t*seed
            return x_tilde, mean, seed
        
    def reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x - (beta_t/sq_1_expB_t)*modified_score
            brownian = torch.sqrt(beta_t)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
            return drift, brownian
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (coef_first_t*coef_eps_t)*modified_score
            brownian = (self.betas[t] ** 0.5).reshape(-1, 1, 1, 1) * torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real*(t > 0).reshape(-1, 1, 1, 1)
            return drift, brownian

    def tweedie_reverse(self, x, t, modified_score):
        if not self.ddpm_math:
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            sq_expBt = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)
            return (x - sq_1_expB_t*modified_score)/sq_expBt
        else:
            s1 = self.sqrt_alphas_cumprod[t] # bs
            s2 = self.sqrt_one_minus_alphas_cumprod[t] # bs
            s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
            s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)

            batch_denoised= (x- s2* modified_score)/s1
            return batch_denoised

    def prior_sampling(self, shape):
        return torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn(shape)).to(device)).real

    def prior_log_likelihood(self, z):
        shape = z.shape
        ndim = len(shape)
        if ndim == 2:
            return -1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z*torch.fft.ifft2(self.power_spectrum**-1 * torch.fft.fft2(z)).real) + torch.sum(torch.log(self.power_spectrum)))
        elif ndim == 3:
            return -1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z*torch.fft.ifft2(self.power_spectrum**-1 * torch.fft.fft2(z)).real) + torch.sum(torch.log(self.power_spectrum))) ## Assuming independent channels (zero cross spectra)
        elif ndim == 4:
            return -1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z*torch.fft.ifft2(self.power_spectrum**-1 * torch.fft.fft2(z)).real, dim=(1, 2,3)) + torch.sum(torch.log(self.power_spectrum)))

    def rescale_additive_to_preserved(self, x, t):
        return x * self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def rescale_preserved_to_additive(self, x, t):
        return x / self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def noise_level(self, t):
        return self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)/self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x - 1/2*(beta_t/sq_1_expB_t)*modified_score
            return drift
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)
            drift = (coef_first_t - 1) * x - (1/2*coef_first_t*coef_eps_t)*modified_score
            return drift

class DiscreteAllSigmaVPSDE(DiscreteSDE):
    def __init__(self, N = 1000, beta_min = 0.1, beta_max = 20.0, ddpm_math = True):
        super(DiscreteAllSigmaVPSDE, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_0 = beta_min/N
        self.beta_T = beta_max/N
        self.betas = torch.linspace(self.beta_0, self.beta_T, N, dtype=torch.float32).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        ## For non DDPM math.
        self.t = torch.linspace(1/N, 1, N, dtype=torch.float32).to(device)
        self.Beta = (self.beta_min * self.t + 1/2*(self.beta_max - self.beta_min) * self.t**2)

        self.ddpm_math = ddpm_math
        self.config = {'type' : 'DiscreteAllSigmaVPSDE', 'beta_min':self.beta_min, 'beta_max':self.beta_max, 'N':self.N, 'ddpm_math':self.ddpm_math}

    def foward(self, x_t, t, sq_ps):
        ## Returns the drift and brownian term of the SDE
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x_t
            brownian = torch.sqrt(beta_t) * torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn(x_t.shape)).to(device)).real
            return drift, brownian
        else:
            beta_t = self.betas[t]
            drift = (1 - torch.sqrt(1-beta_t)).reshape(-1, 1, 1, 1)*x_t
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x_t))).real
            return drift, brownian
    def sampling(self, x, t, sq_ps):
        if not self.ddpm_math:
            mean = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)*x
            seed = torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)*seed
            return x_tilde, mean, seed
        else:
            sqrt_alpha_barre_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_barre_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
            mean = sqrt_alpha_barre_t.reshape(-1, 1, 1, 1)*x
            seed = torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
            x_tilde = mean + sqrt_one_minus_alpha_barre_t*seed
            return x_tilde, mean, seed
    def reverse(self, x, t, modified_score, sq_ps):
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x - 1/2*(beta_t/sq_1_expB_t)*modified_score
            brownian = torch.sqrt(beta_t) * torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn(x.shape)).to(device)).real
            return drift, brownian
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (1/2*coef_first_t*coef_eps_t)*modified_score
            brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn_like(x))).real
            return drift, brownian
    def tweedie_reverse(self, x, t, modified_score): ## TODO use heritage on this whole .py file. There has to be a way to factorize this.
        if not self.ddpm_math:
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            sq_expBt = torch.exp(-self.Beta[t]/2).reshape(-1, 1, 1, 1)
            return (x - sq_1_expB_t*modified_score)/sq_expBt
        else:
            s1 = self.sqrt_alphas_cumprod[t] # bs
            s2 = self.sqrt_one_minus_alphas_cumprod[t] # bs
            s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
            s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)

            batch_denoised= (x- s2* modified_score)/s1
            return batch_denoised
    def ode_drift(self, x, t, modified_score):
        ## Used for the ODE solver as well as likelihood computations
        if not self.ddpm_math:
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta[t])).reshape(-1, 1, 1, 1)
            drift = (beta_t/2 ) * x - 1/2*(beta_t/sq_1_expB_t)*modified_score
            return drift
        else:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)

            drift = (coef_first_t - 1) * x - (1/2*coef_first_t*coef_eps_t)*modified_score
            return drift
    def prior_sampling(self, shape, sq_ps):
        return torch.fft.ifft2(sq_ps*torch.fft.fft2(torch.randn(shape))).real
    def prior_log_likelihood(self, z, ps):
        raise NotImplementedError()
        shape = z.shape
        ndim = len(shape)
        if ndim == 2:
            return 1/2*(z.numel()*np.log(2*np.pi) + torch.sum(ps*torch.fft.fft2(z)).real )
        elif ndim == 3:
            raise NotImplementedError("Not implemented for 3D multichannel likelihood computations are a pain if there is cross spectrum in the noise at time T")
        elif ndim == 4:
            return 1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(ps*torch.fft.fft2(z), dim=(1, 2,3)).real )
    def rescale_additive_to_preserved(self, x, t):
        return x * self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    def rescale_preserved_to_additive(self, x, t):
        return x / self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    def noise_level(self, t):
        return self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)/self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        


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
    def __init__(self, beta_min = 0.1, beta_max = 20.0, t_min = 1e-4):
        super(ContinuousVPSDE, self).__init__()
        self.beta_0 = beta_min
        self.beta_T = beta_max
        self.tmin = t_min
        self.tmax = 1.0
        self.beta = lambda t: ( (self.beta_T - self.beta_0) * t)
        self.Beta = lambda t: ( 1/2*(self.beta_T - self.beta_0) * t**2)

        self.config = {'type' : 'ContinuousVPSDE', 'beta_min':beta_min, 'beta_max':beta_max}

    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the SDE
        beta_t = self.beta(t)
        drift = (- beta_t/2)* x_t
        brownian = torch.sqrt(beta_t)*torch.randn_like(x_t)
        return drift, brownian
    
    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms
        Beta_t = self.Beta(t).reshape(-1, 1, 1, 1)
        mean = torch.exp(-Beta_t/2)*x
        seed = torch.randn_like(x)
        noise = torch.sqrt(1 - torch.exp(-Beta_t))*seed
        x_tilde = mean + noise
        return x_tilde, mean, seed

    def reverse(self, x, t, modified_score):
        beta_t = self.betas(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1, 1, 1)
        drift = (beta_t/2 ) * x - (beta_t/sq_1_expB_t)*modified_score 
        brownian = torch.sqrt(beta_t).reshape(-1, 1, 1, 1)*torch.randn_like(x)
        return drift, brownian

    def tweedie_reverse(self, x, t, modified_score):
        t = torch.clamp(t, self.tmin, self.tmax)
        Beta_t = self.Beta(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-Beta_t))
        sq_expBt = torch.exp(-Beta_t/2)
        return (x - sq_1_expB_t*modified_score)/sq_expBt

    def ode_drift(self, x, t, modified_score):
        beta_t = self.betas(t).reshape(-1, 1, 1, 1)
        sq_1_expB_t = torch.sqrt(1 - torch.exp(-self.Beta(t))).reshape(-1, 1, 1, 1)
        drift = (beta_t/2 ) * x - (1/2 * beta_t/sq_1_expB_t)*modified_score 
        return drift

    def prior_sampling(self, shape):
        return torch.randn(shape)

    def prior_log_likelihood(self, z):
        shape = z.shape
        ndim = z.ndim
        if ndim == 2:
            return 1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z**2)) ## +log(1)
        elif ndim == 3: ## Assumes it is a C, H, W image
            return 1/2*(z.numel()*np.log(2*np.pi) + torch.sum(z**2))
        elif ndim == 4: ## Assumes it is a B C, H, W image
            return 1/2*(z[0].numel()*np.log(2*np.pi) + torch.sum(z**2, dim=[1, 2, 3]))

    def rescale_additive_to_preserved(self, x, t):
        return x * torch.exp(-self.Beta(t)/2)

    def rescale_preserved_to_additive(self, x, t):
        return x / torch.exp(-self.Beta(t)/2)

    def noise_level(self, t):
        return torch.sqrt(1 - torch.exp(-self.Beta(t)))/torch.exp(-self.Beta(t)/2)
    
class ContinuousSigmaVPSDE(ContinuousSDE): ## TODO change default ps
    def __init__(self, power_spectrum = 'cmb', beta_min = 0.1, beta_max = 20.0):
        super(ContinuousSigmaVPSDE, self).__init__()
        self.beta_0 = beta_min
        self.beta_T = beta_max
        self.beta = lambda t: (self.beta_0 + (self.beta_T - self.beta_0) * t)
        self.Beta = lambda t: (self.beta_0 * t + 1/2*(self.beta_T - self.beta_0) * t**2)

        if type(power_spectrum)==str:
            self.power_spectrum_name = power_spectrum
            try:
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),'power_spectra')
                self.power_spectrum = torch.from_numpy(np.load(os.path.join(folder, power_spectrum + '.npy'))).to(device).type(torch.float32)
            except:
                raise ValueError('Power spectrum not recognized')
        elif type(power_spectrum)==np.ndarray:
            self.power_spectrum = torch.from_numpy(power_spectrum).to(device).type(torch.float32)
            self.power_spectrum_name = 'custom'
        elif type(power_spectrum)==torch.Tensor:
            self.power_spectrum = power_spectrum.to(device)
            self.power_spectrum_name = 'custom'
        else:
            raise ValueError('''Argument power_spectrum must be one of:
             - a string -> a common power spectrum stored in the library see add_power_spectrum NOT IMPLEMENTED
             - a numpy array or a torch tensor -> the power spectrum itself (cross spectrum not implemented)''')
        self.sqrt_ps = torch.sqrt(self.power_spectrum)

        self.config = {'type' : 'ContinuousSigmaVPSDE', 'power_spectrum_name':self.power_spectrum_name, 'beta_min':beta_min, 'beta_max':beta_max}

    def foward(self, x_t, t):
        ## Returns the drift and brownian term of the SDE
        beta_t = self.beta(t)
        drift = (- beta_t/2)* x_t
        brownian = torch.sqrt(beta_t)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x_t))).real
        return drift, brownian

    def sampling(self, x, t):
        ## Samples from the SDE at time t and returns x_tilde, the mean and the rescaled noise terms
        mean = torch.exp(-self.Beta(t)/2)*x
        noise = torch.sqrt(1 - torch.exp(-self.Beta(t)))*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
        x_tilde = mean + noise
        return x_tilde, mean, noise

    def reverse(self, x, t, modified_score):
        raise NotImplementedError
        beta_t = self.beta(t)
        drift = - (beta_t/2) * x - modified_score
        brownian = torch.sqrt(beta_t)*torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn_like(x))).real
        return drift, brownian
    def tweedie_reverse(self, x, t, modified_score):
        raise NotImplementedError
    def ode_drift(self, x, t, modified_score):
        raise NotImplementedError
        ## Used for the ODE solver as well as likelihood computations
        beta_t = self.beta(t)
        drift = - (beta_t/2) * x - (beta_t/2)*modified_score
        return drift

    def prior_sampling(self, shape):
        return torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(torch.randn(shape)).to(device)).real
    def prior_log_likelihood(self, z):
        raise NotImplementedError
        return 0.0
    def rescale_additive_to_preserved(self, x, t):
        return x * torch.exp(-self.Beta(t)/2)
    def rescale_preserved_to_additive(self, x, t):
        return x / torch.exp(-self.Beta(t)/2)
    def noise_level(self, t):
        return torch.sqrt(1 - torch.exp(-self.Beta(t)))/torch.exp(-self.Beta(t)/2)


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
        if "ddpm_math" not in config.keys():
            config["ddpm_math"] = True
        sto_diff_eq = DiscreteVPSDE(beta_min=config["beta_min"], beta_max=config["beta_max"], N=config["N"], ddpm_math=config["ddpm_math"])
    elif config["type"].lower() == "discretesigmavpsde":
        if "beta_min" not in config.keys():
            config["beta_min"] = 0.1
        if "beta_max" not in config.keys():
            config["beta_max"] = 20
        if "N" not in config.keys():
            config["N"] = 1000
        if "ddpm_math" not in config.keys():
            config["ddpm_math"] = True
        if "power_spectrum_name" not in config.keys():
            config["power_spectrum_name"] = "cmb_256_8arcmippixel"
        sto_diff_eq = DiscreteSigmaVPSDE(beta_min=config["beta_min"], beta_max=config["beta_max"], N=config["N"], ddpm_math=config["ddpm_math"], power_spectrum=config["power_spectrum_name"])
    elif config["type"].lower() == "continuousvpsde":
        if "beta_min" not in config.keys():
            config["beta_min"] = 0.1
        if "beta_max" not in config.keys():
            config["beta_max"] = 20
        if 't_max' not in config.keys():
            config['t_max'] = 1.0
        if "t_min" not in config.keys():
            config["t_min"] = 1e-4
        sto_diff_eq = ContinuousVPSDE(beta_min=config["beta_min"], beta_max=config["beta_max"], t_min=config["t_min"], t_max=config["t_max"])
    if sto_diff_eq is None:
        warnings.warn("There was a problem with the SDE config, using default DiscreteVPSDE")
        sto_diff_eq = DiscreteVPSDE()
    return sto_diff_eq