#!/usr/bin/env python
# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt
import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import tqdm

from astroddpm.datahandler.dataset import NPDataset, get_dataset_and_dataloader

from astroddpm.diffusion.dm import DiscreteSBM, DiffusionModel
from astroddpm.diffusion.stochastic.sde import DiscreteSDE, DiscreteVPSDE, DiscreteSigmaVPSDE
from astroddpm.diffusion.models.network import ResUNet

from astroddpm.runners import Diffuser



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_everything(model_id, ckpt_folder = None):
    default_diffmodel = DiscreteSBM(DiscreteVPSDE(1000),ResUNet())
    diffuser = Diffuser(default_diffmodel)
    if ckpt_folder is None:
        print("Looking for the diffuser corresponding to model_id {} in the MODELS.json all config file".format(model_id)) 
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config' ,"MODELS.json")) as f:
            total_models = json.load(f)
        config = total_models[model_id]
    else:
        print("Looking for the diffuser config file corresponding to model_id {} in the ckpt folder provided".format(model_id)) 
        with open(os.path.join(ckpt_folder,'config.json')) as f:
            config = json.load(f)
    diffuser.load(config, also_ckpt=True)
    return diffuser

def double_loader(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2):
    diffuser1 = load_everything(MODEL_ID1, CKPT_FOLDER1)
    diffuser2 = load_everything(MODEL_ID2, CKPT_FOLDER2)
    return diffuser1, diffuser2

## TODO allow batches? 
def separation2score_loaded(model1, model2, testbatch1, testbatch2, NUM_SAMPLES=32, method='M1A2'):
    raise NotImplementedError("This method is not adapted yet")

    return


def separation2score(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2,NUM_SAMPLES=16, method='M1A2'):
    raise NotImplementedError("This method is not adapted yet")
    model1, model2, testbatch1, testbatch2 = double_loader(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2)
    return separation2score_loaded(model1, model2, testbatch1, testbatch2, NUM_SAMPLES=NUM_SAMPLES, method = method)


def method1_algo2(model1, model2, observation, NUM_SAMPLES=32):
    raise NotImplementedError("This method is not adapted yet")
    assert len(observation.shape)==4 , "Image should be provided following the BS, CH, H, W standard in pytorch"
    model1.eval()
    model2.eval()
    superpos = observation.repeat(NUM_SAMPLES, 1, 1, 1).to(device)
    with torch.no_grad():
        tot_steps=model1.num_timesteps
        timesteps=list(range(tot_steps))[::-1]
        sample1=torch.randn(superpos.shape).to(device)

        progress_bar = tqdm.tqdm(total=tot_steps)

        for t in timesteps:
            time_tensor = (torch.ones(superpos.shape[0], 1)* t).long().to(device)
            residual1 = model1.reverse(sample1, time_tensor)

            c_t1=model1.sqrt_alphas_cumprod[time_tensor[0]]*superpos

            conditional_residual1 = - model2.reverse(c_t1 - sample1 , time_tensor) ## sqrt(1-alpha_barre) score y_t (c_t - x_t)

            sample1 = model1.step(residual1+conditional_residual1, time_tensor[0], sample1)
            progress_bar.update(1)
    return sample1, observation


def check_separation1score(diffuser, ckpt_dir = None, noise_step = 500, num_to_denoise = 1, NUM_SAMPLES = 16, tweedie = False):
    '''This function is used to check the separation1score function on a subset of the test batch provided with the diffuser. It returns the truth, the noisy and the denoised batch for the first num_to_denoise images of the test dataset of the diffuser corresponding to model_id. If tweedie is True, the function uses the one_step_denoising function, otherwise it uses the multi_step_denoising function. If ckpt_dir is None, the function will look for the diffuser corresponding to model_id in the MODELS.json file. If ckpt_dir is provided, the function will look for the diffuser config file in the ckpt_dir folder.'''
    if isinstance(diffuser, str):
        diffuser = load_everything(diffuser, ckpt_dir)
    elif not isinstance(diffuser, Diffuser):
        raise ValueError("The diffuser you provided is not a Diffuser object or a model_id. Please provide a Diffuser object or a model_id.")
    if not hasattr(diffuser, 'test_dataloader'):
        raise ValueError("The diffuser you loaded does not have a test dataloader. If you still want to use this function, you should provide a test dataloader to the diffuser or use manually one_step_denoising or multi_step_denoising with the diffusion model you want to use and the batch you want to denoise.")
    test_dataloader = diffuser.test_dataloader

    diffmodel = diffuser.diffmodel

    test_batch1 = next(iter(test_dataloader))

    if len(test_batch1.shape) == 3:
        test_batch1 = test_batch1.unsqueeze(1)

    partial_batch = torch.split(test_batch1[:num_to_denoise], 1, dim = 0)

    partial_batch = [elt.repeat(NUM_SAMPLES, 1, 1, 1) for elt in partial_batch]

    partial_batch = torch.cat(partial_batch, dim = 0)

    if tweedie:
        noisy_batch, batch_denoised = one_step_denoising(diffmodel, partial_batch, noise_step)
    else:
        noisy_batch, batch_denoised = multi_step_denoising(diffmodel, partial_batch, noise_step)

    truth_list = torch.split(test_batch1[:num_to_denoise], 1, dim = 0)
    noisy_list = torch.split(noisy_batch, NUM_SAMPLES, dim = 0)
    denoised_list = torch.split(batch_denoised, NUM_SAMPLES, dim=0)

    return truth_list, noisy_list, denoised_list
    

def separation1score(diffuser, observation, CKPT_FOLDER1 = None, noise_step = 500, NUM_SAMPLES = 16, tweedie = False, rescale_observation = True, verbose = True):
    '''This function is used to compute the separation1score of the diffuser corresponding to model_id1 on the observation provided. If tweedie is True, the function uses the one_step_denoising function, otherwise it uses the multi_step_denoising function. If ckpt_dir is None, the function will look for the diffuser corresponding to model_id in the MODELS.json file. If ckpt_dir is provided, the function will look for the diffuser config file in the ckpt_dir folder.'''
    if isinstance(diffuser, str):
        diffuser = load_everything(diffuser, CKPT_FOLDER1)
    elif not isinstance(diffuser, Diffuser):
        raise ValueError("The diffuser you provided is not a Diffuser object or a model_id. Please provide a Diffuser object or a model_id.")
    diffmodel = diffuser.diffmodel
    t = noise_step
    if rescale_observation:
        partial_batch = torch.split(diffmodel.sde.rescale_additive_to_preserved(observation, t).to(device), 1, dim = 0)
    else:
        partial_batch = torch.split(observation.to(device), 1, dim = 0)

    partial_batch = [elt.repeat(NUM_SAMPLES, 1, 1, 1) for elt in partial_batch]
    partial_batch = torch.cat(partial_batch, dim = 0)
    if tweedie:
        noisy_batch, batch_denoised = one_step_denoising(diffmodel, partial_batch, noise_step, is_observation = True)
    else:
        noisy_batch, batch_denoised = multi_step_denoising(diffmodel, partial_batch, noise_step, is_observation = True, verbose = verbose)

    noisy_list = torch.split(noisy_batch, NUM_SAMPLES, dim = 0)
    denoised_list = torch.split(batch_denoised, NUM_SAMPLES, dim=0)

    return noisy_list, denoised_list


def one_step_denoising(model, batch, t, is_observation = False):
    batch=batch.to(device)
    if not(is_observation):
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch, _, _ =model.sde.sampling(batch, timesteps)
    else:
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch = batch

    modified_score = model.network(noisy_batch, timesteps)

    batch_denoised= model.sde.tweedie_reverse(noisy_batch, timesteps, modified_score)
    
    rescaled_noisy_batch = model.sde.rescale_preserved_to_additive(noisy_batch, timesteps)

    return rescaled_noisy_batch, batch_denoised

def multi_step_denoising(model, batch, t, is_observation = False, verbose = True):
    batch = batch.to(device)
    if not(is_observation):
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch, _, _=model.sde.sampling(batch, timesteps)
    else:
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch = batch

    batch_denoised = model.generate_image(noisy_batch.shape[0],sample=noisy_batch,initial_timestep=t, verbose = verbose)

    rescaled_noisy_batch = model.sde.rescale_preserved_to_additive(noisy_batch, timesteps)

    return rescaled_noisy_batch, batch_denoised