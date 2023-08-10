import torch 
import os
import numpy as np
import warnings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def constant_ps_sampler(ps_path):
    return torch.from_numpy(np.load(ps_path, allow_pickle=True)).to(device).type(torch.float32)

class MLP(nn.Module):
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
    def __init__(self, ps_path):
        super(CMB_H_OMBH2, self).__init__()

        CKPT_FOLDER = '/mnt/home/dheurtel/ceph/02_checkpoints/SIGMA_EMULATOR'
        MODEL_ID = 'Emulator_H0_ombh2_1'
        ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID + '.pt'))

        self.emulator = MLP(2, 100, 128, 2).to(device)
        self.emulator.load_state_dict(ckpt['network'])

        wn = (256*np.fft.fftfreq(256, d=1.0)).reshape((256,) + (1,) * (2 - 1))
        wn_iso = np.zeros((256,256))
        for i in range(2):
            wn_iso += np.moveaxis(wn, 0, i) ** 2
        indices = np.fft.fftshift(wn_iso).diagonal()[128:] ## The value of the wavenumbers along which we have the power spectrum diagonal

        self.torch_wn_iso = torch.tensor(np.sqrt(wn_iso), dtype=torch.float32).to(device)
        self.torch_indices = torch.tensor(indices).to(device)

    def forward(self, theta):
        theta = rescale_theta(theta) ## Shape (batch_size, 2) (H0, ombh2) are the 2 cosmological parameters
        torch_diagonals = emulator(theta) ## Shape (batch_size, 128) (128 is the number of wavenumbers along which we have the power spectrum diagonal)
        torch_diagonals = torch_diagonals.reshape((128, -1)) ## Shape (128, batch_size) to be able to use torchcubicspline
        spline = torchcubicspline.NaturalCubicSpline(torchcubicspline.natural_cubic_spline_coeffs(self.torch_indices, torch_diagonals))
        return torch.moveaxis(spline.evaluate(self.torch_wn_iso), -1, 0)

    def sample_theta(self, n_samples):
        return torch.tensor([40, 49.2e-3])*torch.rand(n_samples, 2) + torch.tensor([50, 7.5e-3])

    def sample_ps(self, n_samples):
        theta = self.sample_theta(n_samples).to(device)
        return self.forward(theta), theta



def get_ps_sampler(config):
    if not(config):
        return None
    elif 'type' in config.keys():
        if config['type'].lower() == 'constant':
            if 'ps_path' in config.keys():
                return constant_ps_sampler(config['ps_path'])
            else:
                warnings.warn('No ps_path provided for constant power spectrum sampler. Returning None.')
                return None
        elif config['type'].lower() == 'cmb_h_ombh2':
            return CMB_H_OMBH2()
        else:
            warnings.warn('Power spectrum sampler type not recognized. Returning None.')
            return None
