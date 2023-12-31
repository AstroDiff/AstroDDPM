import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchcubicspline
import os
import numpy as np
import warnings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConstantPs(nn.Module):
    """
    Constant power spectrum sampler. Always returns the same power spectrum. Has no additional parameters.
    """
    def __init__(self, ps_path):
        super(ConstantPs, self).__init__()
        self.ps = torch.from_numpy(np.load(ps_path, allow_pickle=True)).to(device).type(torch.float32)
        if len(self.ps.shape) == 2:
            self.ps = self.ps.unsqueeze(0).unsqueeze(0)
        elif len(self.ps.shape) == 3:
            self.ps = self.ps.unsqueeze(0)
        self.ps = self.ps.to(device)
        self.has_phi = False
        self.config = {'type': 'constant', 'ps_path': ps_path}
    def forward(self):
        '''
        Necessary method to have the same interface as other power spectrum samplers. Should not be used.
        Returns:
            ps: tensor of shape (batch, channels, height, width)
        '''
        warnings.warn('Calling forward method of ConstantPs. This method should not be called. Will return a constant power spectrum of shape (1 , channels, height, width). To get a power spectrum of shape (batch, channels, height, width), use sample_ps method.')
        return self.ps
    def sample_ps(self, n_samples):
        '''
        Returns:
            ps: tensor of shape (n_samples, channels, height, width)
        '''
        return self.ps.repeat(n_samples, 1, 1, 1)
    

class ConstantPsFromTensor(nn.Module):
    def __init__(self, ps):
        super(ConstantPsFromTensor, self).__init__()
        self.ps = ps
        if len(self.ps.shape) == 2:
            self.ps = self.ps.unsqueeze(0).unsqueeze(0)
        elif len(self.ps.shape) == 3:
            self.ps = self.ps.unsqueeze(0)
        self.ps = self.ps.to(device)
        self.has_phi = False
        self.config = {'type': 'constant', 'nature' : 'custom'}
    def forward(self):
        return self.ps
    def sample_ps(self, n_samples):
        return self.ps.repeat(n_samples, 1, 1, 1)

class MLP(nn.Module):
    """
    Simple MLP with ReLU activation functions. Will be used to emulate the diagonal values of the log power spectrum."""
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for i in range(self.n_hidden_layers + 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x

class CMB_H_OMBH2(nn.Module):
    """
    Emulator for the power spectrum of the CMB. The emulator is trained to reproduce said power spectrum on a patch of size 256x256 pixels of the sky.
    Each pixel is 8 arcminutes wide. 
    The emulator is a function of two cosmological parameters: H0 and ombh2. Others are fixed to the TODO values.
    """
    def __init__(self):
        super(CMB_H_OMBH2, self).__init__()

        CKPT_FOLDER = '/mnt/home/dheurtel/ceph/02_checkpoints/SIGMA_EMULATOR'
        MODEL_ID = 'Emulator_H0_ombh2_1'
        ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID + '.pt'))

        self.emulator = MLP(2, 100, 128, 2).to(device)
        self.emulator.load_state_dict(ckpt['network'])
        for param in self.emulator.parameters():
            param.requires_grad = False
        wn = (256*np.fft.fftfreq(256, d=1.0)).reshape((256,) + (1,) * (2 - 1))
        wn_iso = np.zeros((256,256))
        for i in range(2):
            wn_iso += np.moveaxis(wn, 0, i) ** 2
        wn_iso = np.sqrt(wn_iso)
        indices = np.fft.fftshift(wn_iso).diagonal()[128:] ## The value of the wavenumbers along which we have the power spectrum diagonal

        self.torch_wn_iso = torch.tensor(wn_iso, dtype=torch.float32).to(device)
        self.torch_indices = torch.tensor(indices).to(device)
        self.has_phi = True

        self.config = {'type': 'cmb_h_ombh2'}
    
    def rescale_phi(self, phi):
        """
        Rescale the cosmological parameters to be in the range [-1, 1] for the emulator.
        Args:
            phi: tensor of shape (batch_size, 2), the cosmological parameters
        Returns:
            rphi: tensor of shape (batch_size, 2), the rescaled cosmological parameters"""
        return (phi - torch.tensor([70, 32e-3]).to(device))/torch.tensor([20,25e-3]).to(device)

    def unscale_phi(self, rphi):
        """
        Unscale the cosmological parameters from the range [-1, 1] to their physical range.
        Args:
            rphi: tensor of shape (batch_size, 2), the rescaled cosmological parameters
        Returns:
            phi: tensor of shape (batch_size, 2), the cosmological parameters"""
        return rphi*torch.tensor([20,25e-3]).to(device) + torch.tensor([70, 32e-3]).to(device)

    def forward(self, phi, to_rescale=True):
        """
        Returns the power spectrum at values of the cosmological parameters phi.
        Args:
            phi: tensor of shape (batch_size, 2), the cosmological parameters
            to_rescale (optional): bool, whether to rescale the cosmological parameters to the range [-1, 1] for the emulator. Default: True (parameters are to be rescaled)
        Returns:
            ps: tensor of shape (batch_size, 1, 128), the diagonal of the power spectrum
        """
        if to_rescale:
            phi = self.rescale_phi(phi) ## Shape (batch_size, 2) (H0, ombh2) are the 2 cosmological parameters
        torch_diagonals = self.emulator(phi) ## Shape (batch_size, 128) (128 is the number of wavenumbers along which we have the power spectrum diagonal)
        torch_diagonals = torch.moveaxis(torch_diagonals, -1, 0) ## Shape (128, batch_size) to be able to use torchcubicspline
        spline = torchcubicspline.NaturalCubicSpline(torchcubicspline.natural_cubic_spline_coeffs(self.torch_indices, torch_diagonals))
        return torch.exp(torch.moveaxis(spline.evaluate(self.torch_wn_iso), -1, 0)).unsqueeze(1).type(torch.float32)/12734 ## Normalization factor to have a power spectrum of order 1

    def sample_phi(self, n_samples):
        """
        Sample values of the cosmological parameters phi according to the prior p_phi (uniform distribution in the range [50, 90] for H0 and [7.5e-3, 56.7e-3] for ombh2)
        Args:
            n_samples: int, number of samples to return
        Returns:
            phi: tensor of shape (n_samples, 2), the cosmological parameters
        """
        return torch.tensor([40, 49.2e-3])*torch.rand(n_samples, 2) + torch.tensor([50, 7.5e-3])

    def sample_ps(self, n_samples):
        """
        Sample values of the cosmological parameters and then return these values and the associated power spectrum.
        Args:
            n_samples: int, number of samples to return
        Returns:
            ps: tensor of shape (n_samples, 1, 128), the diagonal of the power spectrum
            phi: tensor of shape (n_samples, 2), the cosmological parameters
        """
        phi = self.sample_phi(n_samples).to(device)
        return self.forward(phi), phi



def get_ps_sampler(config):
    """
    Returns a power spectrum sampler according to the config.
    Args:
        config: dict, configuration of the power spectrum sampler
    Returns:
        ps_sampler: power spectrum sampler
    """
    if not(config):
        return None
    elif 'type' in config.keys():
        if config['type'].lower() == 'constant':
            if 'ps_path' in config.keys():
                return ConstantPs(config['ps_path'])
            else:
                warnings.warn('No ps_path provided for constant power spectrum sampler. Returning None.')
                return None
        elif config['type'].lower() == 'cmb_h_ombh2':
            return CMB_H_OMBH2()
        else:
            warnings.warn('Power spectrum sampler type not recognized. Returning None.')
            return None
