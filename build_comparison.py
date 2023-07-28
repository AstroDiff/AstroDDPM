import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import tqdm

import bm3d
import astroddpm
from astroddpm.runners import Diffuser, config_from_id
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE
from astroddpm.separation import double_loader, method1_algo2, check_separation1score, separation1score, load_everything
from astroddpm.diffusion.models.network import ResUNet
from astroddpm.diffusion.stochastic import sde
import astroddpm.analysis.validationMetrics.powerSpectrum as powerspectrum
import argparse

import bm3d
from bm3d import bm3d, BM3DProfile, BM3DStages

class PowerSpectrumIso2d(nn.Module):
    '''Torch Module wrapper to allow for parallel computation of the isotropic power spectrum of a batch of images'''
    def __init__(self):
        super().__init__()
    def forward(self, batch):
        bins, power_spectra, _ = powerspectrum.power_spectrum_iso2d(batch , bins = torch.linspace(0, np.pi, 100), use_gpu=True)
        return bins, power_spectra



def build_comparison(model_id, ckpt_folder = None, verbose = True, save_path_ = 'default', num_to_denoise=-1, num_samples=1, noise_min=0.0, noise_max=1.0, num_levels=10, noise_interp='linear', methods='all', batch_size=-1, gpu_for_spectrum = False, parallel = False):
    ## Load the diffuser
    diffuser = load_everything(model_id, ckpt_folder)

    if save_path_.lower() == 'default':
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'comparison',model_id,)
    else:
        save_path = save_path_
    ## Create the file where to save the comparison
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'comparison.pt')
    save_dict = {}

    ## Load the test set
    test_set = diffuser.test_dataloader.dataset
    if batch_size == -1:
        batch_size = diffuser.test_dataloader.batch_size

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    noise_min = max(noise_min, diffuser.diffmodel.sde.noise_level(0).reshape(-1).cpu().numpy())
    noise_max = min(noise_max, diffuser.diffmodel.sde.noise_level(-1).reshape(-1).cpu().numpy())
    bins = torch.linspace(0, np.pi, 100)

    ## Create the noise levels
    if noise_interp == 'linear':
        noise_levels = np.linspace(noise_min, noise_max, num_levels)
    elif noise_interp == 'log':
        noise_levels = np.logspace(np.log10(noise_min), np.log10(noise_max), num_levels)
    else:
        raise NotImplementedError("noise_interp {} not implemented".format(noise_interp))

    ## Create the list of times from the noise levels by chosing the first time just above the noise level
    diff_sde = diffuser.diffmodel.sde
    if isinstance(diff_sde, sde.DiscreteSDE):
        N = diff_sde.N
        all_times = np.arange(N)
        all_levels = diff_sde.noise_level(all_times).reshape(-1).cpu().numpy()
        times = np.zeros(num_levels, dtype=np.int64)
        for i, level in enumerate(noise_levels):
            times[i] = all_times[all_levels >= level][0]
    elif isinstance(diff_sde, sde.ContinuousSDE):
        raise NotImplementedError("ContinuousSDE not implemented yet")
    
    if gpu_for_spectrum:
        ps_computer = PowerSpectrumIso2d().to(device)
        if parallel:
            ps_computer = nn.DataParallel(ps_computer)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    diffuser.diffmodel.to(device)

    if 'sbm' in methods.lower() or methods=='all':

        signed_relative_error_sbm = torch.zeros((num_levels, len(bins)-1))
        if parallel:
            diffuser.diffmodel = nn.DataParallel(diffuser.diffmodel)
        for i, time in enumerate(times):
            with torch.no_grad():
                m = len(test_loader.dataset)
                n = len(test_loader)
                progress_bar = tqdm.tqdm(total=n)
                progress_bar.set_description(f"Noise level n° {i}")
                for _, testbatch in enumerate(test_loader):
                    if testbatch.ndim == 3:
                        testbatch = testbatch.to(device).unsqueeze(1)
                    else:
                        testbatch = testbatch.to(device)
                    
                    timesteps=torch.full((testbatch.shape[0],), time).long().to(device)
                    observation , _, _ = diffuser.diffmodel.sde.sampling(testbatch, timesteps)
                    observation = diffuser.diffmodel.sde.rescale_preserved_to_additive(observation, timesteps)

                    _ , denoised_list = separation1score(diffuser, observation, noise_step=time, NUM_SAMPLES=num_samples, tweedie=False, rescale_observation=True, verbose = False)
                    denoised_list = torch.cat(denoised_list, dim=0)

                    if not gpu_for_spectrum:
                        _, power_spectra, _= powerspectrum.power_spectrum_iso2d(denoised_list, bins = bins, use_gpu=gpu_for_spectrum)
                        _, power_spectra_true, _ = powerspectrum.power_spectrum_iso2d(testbatch, bins = bins, use_gpu=gpu_for_spectrum)
                    else:
                        power_spectra, power_spectra_true = ps_computer(denoised_list), ps_computer(testbatch)

                    power_spectra = torch.split(power_spectra, num_samples, dim=0)
                    power_spectra = torch.cat([ps.mean(dim=0, keepdim=True) for ps in power_spectra], dim=0)

                    signed_relative_error_sbm[i] += ((power_spectra - power_spectra_true) / power_spectra_true).sum(dim=0)[:-1]
                    progress_bar.update(1)
                progress_bar.close()
                signed_relative_error_sbm[i] /= m
            save_dict['sbm'] = signed_relative_error_sbm
            torch.save(save_dict, save_path)

    if 'bm3d' in methods.lower() or methods=='all':

        signed_relative_error_bm3d = torch.zeros((len(times), len(bins)-1))

        noise_level_cpu = noise_levels
        if isinstance(diffuser.diffmodel.sde, DiscreteVPSDE):
            power_spectrum = 1
        else:
            power_spectrum = diffuser.diffmodel.sde.power_spectrum.detach().cpu().numpy()

        for i, time in enumerate(times[1:]):

            with torch.no_grad():
                n = len(test_set)
                progress_bar = tqdm.tqdm(total=n)
                progress_bar.set_description(f"Noise level n° {i}")
                for _, testimg in enumerate(test_set):
                    testbatch = testimg.unsqueeze(0).unsqueeze(0).to(device)
                    _, ps_true, _ = powerspectrum.power_spectrum_iso2d(testbatch, bins = bins, use_gpu=True)
                    if ps_true.ndim == 2:
                        ps_true = ps_true.reshape(-1)
                    testbatch = testbatch.repeat(num_samples, 1, 1, 1)
                    timesteps = torch.full((testbatch.shape[0],), time).long().to(device)
                    observation , _, _ = diffuser.diffmodel.sde.sampling(testbatch, timesteps)
                    observation = diffuser.diffmodel.sde.rescale_preserved_to_additive(observation, timesteps)
                    denoised_bm3d = torch.zeros_like(observation)
                    for j in range(num_samples):
                        img = observation[j][0].cpu().numpy()
                        denoised_bm3d[j,0, :, :] = torch.from_numpy(bm3d(img, sigma_psd = noise_level_cpu[i+1]*power_spectrum, stage_arg=BM3DStages.HARD_THRESHOLDING,)).to(device)
                    if not gpu_for_spectrum:
                        _, ps_denoised, _= powerspectrum.power_spectrum_iso2d(denoised_bm3d, bins = bins, use_gpu=gpu_for_spectrum)
                    else:
                        ps_denoised = ps_computer(denoised_bm3d)

                    ps_denoised = ps_denoised.mean(dim=0,)
                    signed_relative_error_bm3d[i+1] += ((ps_denoised.reshape(-1).cpu() - ps_true.reshape(-1).cpu()) / ps_true.reshape(-1).cpu())[:-1]
                    progress_bar.update(1)
                signed_relative_error_bm3d[i+1] /= n
                progress_bar.close()
            save_dict['bm3d'] = signed_relative_error_bm3d
            torch.save(save_dict, save_path)


    if 'tweedie' in methods.lower() or methods=='all':
        pass


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='Build comparison of denoising methods for a given diffuser')
    parser.add_argument("--model_id", type=str, help="model id of the diffuser")
    parser.add_argument("--ckpt_folder", type=str, help="ckpt folder of the diffuser if different from the default one in MODELS.json")
    parser.add_argument("--verbose" , type=str, default="Reduced", help="verbose")
    parser.add_argument("--save_path" , type=str, default="default", help="Path on which to save the comparison")

    parser.add_argument("--num_to_denoise" , type=int, default=-1, help="Number of samples from the test set to use for the comparison, -1 for all")
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to draw from the diffuser for each field in the test set, each noise level and each method')
    parser.add_argument('--noise_min', type=float, default=0.0, help='min noise level to use for the comparison')
    parser.add_argument('--noise_max', type=float, default=1.0, help='max noise level to use for the comparison')
    parser.add_argument('--num_levels', type=int, default=10, help='number of noise levels to use for the comparison')
    parser.add_argument('--noise_interp', type=str, default='linear', help='interpolation method for the noise levels')

    parser.add_argument('--methods', type=str, default='all', help='methods to use for the comparison, methods name separated by a coma, default is all')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size for the comparison, -1 for the default one in the diffuser')

    parser.add_argument('--gpu_for_spectrum', type=bool, default=False, help='gpu to use for the power spectra computations')
    parser.add_argument('--parallel', type=bool, default=False, help='whether to use parallel computations for all the computations')

    args = parser.parse_args()

    build_comparison(args.model_id, args.ckpt_folder, args.verbose, args.save_path, args.num_to_denoise, args.num_samples, args.noise_min, args.noise_max, args.num_levels, args.noise_interp, args.methods, args.batch_size, args.gpu_for_spectrum, args.parallel)


