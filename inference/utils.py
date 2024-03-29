import torch
import numpy as np


def get_phi_bounds(device=None):
    phi_min = torch.tensor([50, 7.5e-3]).to(device)
    phi_max = torch.tensor([90, 56.7e-3]).to(device)
    return phi_min, phi_max

def get_phi_bounds_sigma(device=None):
    phi_min = torch.tensor([50, 7.5e-3,1e-3]).to(device)
    phi_max = torch.tensor([90, 56.7e-3,1.0]).to(device)
    return phi_min, phi_max

def normalize_phi(phi, mode='compact'):
    """ Normalize phi from its bounded domain to 
    - [0, 1]x[0, 1] for mode=='compact'
    - [-inf, inf]x[-inf, inf] for mode=='inf' """
    phi_min, phi_max = get_phi_bounds(device=phi.device)
    dphi = phi_max - phi_min
    norm_phi = (phi - phi_min) / dphi
    if mode == 'compact':
        return norm_phi
    elif mode == 'inf':
        return torch.tan((norm_phi - 0.5)*np.pi)
    elif mode is None:
        return phi
    else:
        raise ValueError(f"Unknown normalization mode {mode}")

def unnormalize_phi(phi, mode='compact'):
    """ Unnormalize phi according to the prescribed mode."""
    phi_min, phi_max = get_phi_bounds(device=phi.device)
    dphi = phi_max - phi_min
    if mode == 'compact':
        return phi * dphi + phi_min
    elif mode == 'inf':
        return (torch.arctan(phi)/np.pi + 0.5) * dphi + phi_min
    elif mode is None:
        return phi
    else:
        raise ValueError(f"Unknown normalization mode {mode}")
    
def gen_x(phi, ps_model, device=None):
    """ Generate a CMB map from the parameters phi."""
    ps = ps_model(phi)
    return torch.fft.ifft2(torch.fft.fft2(torch.randn(ps.shape, device=device))*torch.sqrt(ps)).real

def sample_prior_phi(n, norm='compact', device=None):
    """
    Sample from the prior distribution on phi.
    """
    phi_min, phi_max = get_phi_bounds(device=device)
    phi = torch.rand(n, 2).to(device) * (phi_max - phi_min) + phi_min
    return normalize_phi(phi, mode=norm)

def log_prior_phi(phi, norm='compact'):
    """
    Compute the log prior of the parameters.
    """
    if norm == 'compact':
        logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
        for i in range(1, phi.shape[-1]):
            logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())
    elif norm is None:
        phi_min, phi_max = get_phi_bounds(device=phi.device)
        logp = torch.log(torch.logical_and(phi[..., 0] >= phi_min[0], phi[..., 0] <= phi_max[0]).float())
        for i in range(1, phi.shape[-1]):
            logp += torch.log(torch.logical_and(phi[..., i] >= phi_min[i], phi[..., i] <= phi_max[i]).float())
        logp += torch.log(1.0 / torch.prod(phi_max - phi_min)) # Constant term
    elif norm == 'inf': # Log density of tan(U[-pi/2, pi/2])
        logp = -torch.log(1.0 + phi[..., 0]**2)
        for i in range(1, phi.shape[-1]):
            logp -= torch.log(1.0 + phi[..., i]**2)
        logp -= np.log(1.0 / np.pi**phi.shape[-1]) # Constant term
    return logp

def log_likelihood_eps_phi(phi, eps, ps_model):
    """
    Compute the log likelihood of the Gaussian model (epsilon | phi).
    """
    eps_dim = eps.shape[-1]*eps.shape[-2]
    ps = ps_model(phi)
    xf = torch.fft.fft2(eps)

    term_pi = -(eps_dim/2) * np.log(2*np.pi)
    term_logdet = -0.5 * torch.sum(torch.log(ps), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / ps, dim=(-1, -2))/eps_dim # We divide by eps_dim because of the normalization of the FFT
    return term_pi + term_logdet + term_x

def log_prior_phi_sigma(phi, sigma, sigma_min=1e-3, sigma_max=1.0,norm='compact'):
    """
    Compute the log prior of the parameters.
    the sigma range should be restricted to a reasonable range
    """
    if norm == 'compact':
        logp = torch.log(torch.logical_and(phi[..., 0] >= 0.0, phi[..., 0] <= 1.0).float()) #gives either 0 or -inf
        for i in range(1, phi.shape[-1]):
            logp += torch.log(torch.logical_and(phi[..., i] >= 0.0, phi[..., i] <= 1.0).float())
    elif norm is None:
        phi_min, phi_max = get_phi_bounds(device=phi.device)
        logp = torch.log(torch.logical_and(phi[..., 0] >= phi_min[0], phi[..., 0] <= phi_max[0]).float())
        for i in range(1, phi.shape[-1]):
            logp += torch.log(torch.logical_and(phi[..., i] >= phi_min[i], phi[..., i] <= phi_max[i]).float())
        logp += torch.log(1.0 / torch.prod(phi_max - phi_min)) # Constant term
    elif norm == 'inf': # Log density of tan(U[-pi/2, pi/2])
        logp = -torch.log(1.0 + phi[..., 0]**2)
        for i in range(1, phi.shape[-1]):
            logp -= torch.log(1.0 + phi[..., i]**2)
        logp -= np.log(1.0 / np.pi**phi.shape[-1]) # Constant term
    ## Add prior on sigma
    logp += torch.log(torch.logical_and(sigma >= sigma_min, sigma <= sigma_max).float())
    return logp

def log_likelihood_eps_phi_sigma(phi, sigma, eps, ps_model):
    """
    Compute the log likelihood of the Gaussian model (epsilon | phi).
    """
    eps_dim = eps.shape[-1]*eps.shape[-2]
    ps = ps_model(phi)
    xf = torch.fft.fft2(eps)
    sigma = sigma.reshape(-1,1,1)

    term_pi = -(eps_dim/2) * np.log(2*np.pi)
    term_logdet = -0.5 * torch.sum(torch.log(sigma**2*ps), dim=(-1, -2)) # The determinant is the product of the diagonal elements of the PS
    term_x = -0.5 * torch.sum((torch.abs(xf).pow(2)) / (sigma**2*ps), dim=(-1, -2))/eps_dim # We divide by eps_dim because of the normalization of the FFT
    return term_pi + term_logdet + term_x
