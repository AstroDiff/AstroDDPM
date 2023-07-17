import torch
from torch import nn
import tqdm
import numpy as np

from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPM(nn.Module):
    def __init__(
        self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device
    ) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod
        ) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]  # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # bs
        s1 = s1.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise, x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)

    def loss(self, t, x_start, x_noise):
        noisy, _ = self.add_noise(x_start, x_noise, t)
        noise_pred = self.reverse(noisy, t)

        loss = F.mse_loss(noise_pred, x_noise)
        return loss

    def step(self, model_output, timestep, sample, noise=None):
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
        coef_first = 1 / self.alphas**0.5
        coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)
        pred_prev_sample = coef_first_t * (sample - coef_eps_t * model_output)

        variance = 0
        if t > 0:
            if noise is None:
                noise = torch.randn_like(model_output)
            variance = (self.betas[t] ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    def generate_image(
        self, sample_size, channel, size, sample=None, initial_timestep=None
    ):
        """Generate the image from the Gaussian noise, only work with ddpm"""
        self.eval()
        if initial_timestep is None:
            tot_steps = self.num_timesteps
        else:
            tot_steps = initial_timestep
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            if sample is None:
                sample = torch.randn(sample_size, channel, size, size).to(device)
            progress_bar = tqdm.tqdm(total=tot_steps)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.reverse(sample, time_tensor)
                sample = self.step(residual, time_tensor[0], sample)

                progress_bar.update(1)
            progress_bar.close()
        self.train()
        return sample
    def ddim(self, sample_size, channel, size, schedule):
        self.eval()
        with torch.no_grad():
            timesteps = schedule[::-1]
            sample = torch.randn(sample_size, channel, size, size).to(device)
            progress_bar = tqdm.tqdm(total=len(schedule))
            for i, t in enumerate(timesteps):
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.reverse(sample, time_tensor)

                sq_a_t_1 = (self.sqrt_alphas_cumprod[timesteps[i + 1]] if t > 0 else 1)
                sq_1_min_a_t_1 = (self.sqrt_one_minus_alphas_cumprod[timesteps[i + 1]] if t > 0 else 0)
                sq_a_t = self.sqrt_alphas_cumprod[t]
                sq_1_min_a_t = (self.sqrt_one_minus_alphas_cumprod[t])

                sample = (sq_a_t_1 / sq_a_t) * (sample) + (sq_1_min_a_t_1 - sq_1_min_a_t * sq_a_t_1 /sq_a_t) * residual
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        return sample
    
class SigmaDDPM(nn.Module):
    def __init__(
        self, network, num_timesteps, power_spectrum, beta_start=0.0001, beta_end=0.02, device=device
    ) -> None:
        super(SigmaDDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod
        ) ** 0.5  # used in add_noise and step
        self.power_spectrum = power_spectrum.to(device)
        self.sqrt_ps = torch.sqrt(power_spectrum).to(device)

    def add_noise(self, x_start, x_noise, timesteps):
        # Integrated forward SDE process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        grf_noise = torch.fft.ifft2(self.sqrt_ps*torch.fft.fft2(x_noise)).real.to(device)
        s1 = self.sqrt_alphas_cumprod[timesteps]  # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # bs
        s1 = s1.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1)
        return s1 * x_start + s2 * grf_noise, grf_noise 

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)

    def loss(self, t, x_start, x_noise):
  
        noisy, grf_noise = self.add_noise(x_start, x_noise, t)
        noise_pred = self.reverse(noisy, t)
        loss = F.mse_loss(noise_pred, grf_noise)
        return loss

    def step(self, model_output, timestep, sample, ancestral = True):
        # one step of sampling
        # timestep (1)
        t = timestep
        if ancestral:
            coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
            coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
            coef_first = 1 / self.alphas**0.5
            coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)
            pred_prev_sample = coef_first_t * (sample - coef_eps_t * model_output.real)
        else:
            coef_x = 1 - self.alphas[t]**0.5
            coef_score = self.betas[t]/self.sqrt_one_minus_alphas_cumprod[t]
            pred_prev_sample = coef_x * sample - coef_score * model_output
        
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.betas[t] ** 0.5) * torch.fft.ifft2(self.sqrt_ps * torch.fft.fft2(noise)).real

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    def generate_image(
        self, sample_size, channel, size, sample=None, initial_timestep=None
    ):
        """Generate the image from the Gaussian noise, only work with ddpm"""
        self.eval()
        if initial_timestep is None:
            tot_steps = self.num_timesteps
        else:
            tot_steps = initial_timestep
        with torch.no_grad():
            timesteps = list(range(tot_steps))[::-1]
            if sample is None:
                sample = torch.randn(sample_size, channel, size, size).to(device)
            progress_bar = tqdm.tqdm(total=tot_steps)
            for t in timesteps:
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.reverse(sample, time_tensor)
                sample = self.step(residual, time_tensor[0], sample)

                progress_bar.update(1)
            progress_bar.close()
        self.train()
        return sample
    def ddim(self, sample_size, channel, size, schedule):
        self.eval()
        with torch.no_grad():
            timesteps = schedule[::-1]
            sample = torch.randn(sample_size, channel, size, size).to(device)
            progress_bar = tqdm.tqdm(total=len(schedule))
            for i, t in enumerate(timesteps):
                time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
                residual = self.reverse(sample, time_tensor)

                sq_a_t_1 = (self.sqrt_alphas_cumprod[timesteps[i + 1]] if t > 0 else 1)
                sq_1_min_a_t_1 = (self.sqrt_one_minus_alphas_cumprod[timesteps[i + 1]] if t > 0 else 0)
                sq_a_t = self.sqrt_alphas_cumprod[t]
                sq_1_min_a_t = (self.sqrt_one_minus_alphas_cumprod[t])

                sample = (sq_a_t_1 / sq_a_t) * (sample) + (sq_1_min_a_t_1 - sq_1_min_a_t * sq_a_t_1 /sq_a_t) * residual
                progress_bar.update(1)
            progress_bar.close()
        self.train()
        return sample


def generate_image(
    ddpm, sample_size, channel, size, sample=None, initial_timestep=None
):
    """Generate the image from the Gaussian noise, only work with ddpm"""
    ddpm.eval()
    frames = []
    frames_mid = []
    if initial_timestep is None:
        tot_steps = ddpm.num_timesteps
    else:
        tot_steps = initial_timestep
    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(tot_steps))[::-1]
        if sample is None:
            sample = torch.randn(sample_size, channel, size, size).to(device)
        progress_bar = tqdm.tqdm(total=tot_steps)
        for t in timesteps:
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor)
            sample = ddpm.step(residual, time_tensor[0], sample)

            if t == tot_steps // 2:
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())
            progress_bar.update(1)
        progress_bar.close()
        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    ddpm.train()
    return frames, frames_mid

def ddim(model, sample_size, channel, size, schedule):
    model.eval()
    with torch.no_grad():
        timesteps = schedule[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        progress_bar = tqdm.tqdm(total=len(schedule))
        for i, t in enumerate(timesteps):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = model.reverse(sample, time_tensor)

            sq_a_t_1 = (model.sqrt_alphas_cumprod[timesteps[i + 1]] if t > 0 else 1)
            sq_1_min_a_t_1 = (model.sqrt_one_minus_alphas_cumprod[timesteps[i + 1]] if t > 0 else 0)
            sq_a_t = model.sqrt_alphas_cumprod[t]
            sq_1_min_a_t = (model.sqrt_one_minus_alphas_cumprod[t])

            sample = sq_a_t_1 / sq_a_t * (sample) + (sq_1_min_a_t_1 - sq_1_min_a_t * sq_a_t_1 /sq_a_t) * residual
            progress_bar.update(1)
        progress_bar.close()
    model.train()
    return sample

class NCSN(nn.Module):
    def __init__(
        self,
        network,
        num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        device=device,
        sampling="ancestral",
    ) -> None:
        super(NCSN, self).__init__()
        self.num_timesteps = num_timesteps
        self.sigmas = torch.exp(
            torch.linspace(
                np.log(beta_start), np.log(beta_end), num_timesteps, dtype=torch.float32
            )
        ).to(device)
        self.network = network
        self.device = device

        self.sampling = sampling

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        sigmas = self.sigmas[timesteps] ** 2 - self.sigmas[0] ** 2  # bs
        sigmas = torch.sqrt(sigmas.reshape(-1, 1, 1, 1))  # (bs, 1, 1, 1)
        return x_start + sigmas * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)

    def loss(self, t, x_start, x_noise):
        noisy = self.add_noise(x_start, x_noise, t)
        noise_pred = self.reverse(noisy, t)

        loss = F.mse_loss(noise_pred * self.sigmas[t].reshape(-1, 1, 1, 1), x_noise)
        return loss

    def step(self, model_output, timestep, sample):
        # one step of sampling
        # timestep (1)
        t = timestep
        variance = 0
        if t > 0:
            sigmai_1, sigmai = self.sigmas[t], self.sigmas[t - 1]
        else:
            sigmai_1, sigmai = self.sigmas[t], self.sigmas[t]

        deltasigma = sigmai_1**2 - sigmai**2

        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((deltasigma) ** 0.5) * noise

        pred_prev_sample = sample - deltasigma * model_output + variance

        return pred_prev_sample



class ClassifierCondDDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device):
        super(ClassifierCondDDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, dtype=torch.float32
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (
            1 - self.alphas_cumprod
        ) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]  # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # bs
        s1 = s1.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise
    
    
    def loss(self, t, x_start, x_noise, targets, loss_fn=F.binary_cross_entropy_with_logits, activation=nn.Identity()):
        noisy = self.add_noise(x_start, x_noise, t)
        pred = self.network(noisy, t)
        loss = loss_fn(activation(pred), targets)
        return loss

    def predict(self, z,t):
        return self.network(z,t)
    def grad_model(): ###Will need autodiff or something to be made more efficient
        return
    def conditionnal_residual(z,t):
        return



