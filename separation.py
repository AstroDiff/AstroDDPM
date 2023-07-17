#!/usr/bin/env python
# coding: utf-8

import numpy as np

import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import tqdm

from dataHandler.dataset import MHDProjDataset, LogNormalTransform
from ddpm.model import UNet, ResUNet
from ddpm.diffusion import DDPM, NCSN, SigmaDDPM, generate_image

from absl import app, flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_flag_name(str):
    pref='--'
    suff='='
    try:
        idx1 = str.index(pref)
        idx2 = str.index(suff)
    except:
        try:
            idx1 = str.index(pref)
            idx2=len(str)
        except:
            return None
    return str[idx1 + len(pref): idx2]

def load_everything(MODEL_ID, CKPT_FOLDER):
    FLAGS=flags.FLAGS
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)
    try:
        flags.DEFINE_string('model_id','MHD_DDPM_forget',help= 'ID of the model either trained, finetuned, evaluated....')

        ## Data & transforms
        flags.DEFINE_string('source_dir','/mnt/home/dheurtel/ceph/00_exploration_data/density/b_proj',help= 'Source dir containing a list of npy files')
        flags.DEFINE_bool('random_rotate', True, help='')
        flags.DEFINE_bool('no_lognorm', False, help='apply a lognormal transformation to the dataset')

        ## Network & diffusion parameters

        flags.DEFINE_enum('diffusion_mode', 'ddpm', ['ddpm', 'smld', 'VE', 'VD', 'sub_VP', 'SigmaDDPM'], help='Type of diffusion SDE used during training and inference')
        ## TODO if we want to fully use the SDE/ score based framework (and have custom/off the shelf SDE solvers, we will nedd to use runners/differentiate the training loop)
        # For DDPM
        flags.DEFINE_integer('n_steps', 1000, help= 'Diffusion total time see Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models')
        flags.DEFINE_float('beta_start', 1e-4, help = 'Beta at time 0 see Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models')
        flags.DEFINE_float('beta_T',0.02, help= 'Beta at time T=n_steps Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models')
        flags.DEFINE_string("power_spectrum", '/mnt/home/dheurtel/ceph/00_exploration_data/power_spectra/power2.npy', help = "Power Spectrum file")

        #Unet
        flags.DEFINE_enum('network', 'unet', ['unet', 'ResUNet'], help='DUNet not yet implemented')
        flags.DEFINE_integer('size',256, help='height and width of the images')
        flags.DEFINE_integer('in_channel', 1, help= 'number of channel on input and output images')
        flags.DEFINE_enum('normalization', 'LN' , ['LN' , 'default', 'LN-D', 'LN-F', 'LN-F+', 'LN-DF', 'BN', 'BN/LN', 'BN/FLN', 'BN/F+LN', 'DBN/LN', 'GN', 'DN','None'], help= 'type of normalization applied') ## TODO upcoming cleaning of these options based on elimination and perceived redundancies
        flags.DEFINE_float('eps_norm', 1e-5, help= 'epsilon value added to the variance in the normalization layer to ensure numerical stability')
        flags.DEFINE_integer('size_min', 32, help= 'size at the bottleneck')
        flags.DEFINE_integer('num_blocks', 1, help= 'num of blocks per size on descent')
        flags.DEFINE_enum('padding_mode', 'circular' ,['zeros', 'reflect', 'replicate','circular'], help='Conv2d padding mode')
        flags.DEFINE_bool('muP', False, help= 'Use mu Parametrisation for initialisation and training') ## TODO 
        flags.DEFINE_float('dropout', 0, help= 'Probability for dropout, we did not find any impact because our models tend not to overfit')
        flags.DEFINE_integer('first_c_mult', 10, help= 'Multiplier between in_c and out_c for the first block')
        flags.DEFINE_bool('skip_rescale', False, help='Rescale skip connections (see Score Based Generative Modelling paper)')

        ## Training parameters 
        flags.DEFINE_integer('batch_size', 64, help='Dataloader batch size')
        flags.DEFINE_integer('num_sample', 8, help='Number of sample for an epoch in the middle')
        flags.DEFINE_integer('num_result_sample', 256, help='Number of sample for an epoch in the middle')
        flags.DEFINE_float('lr', 1e-3, help= 'Learning rate')
        flags.DEFINE_enum('lr_scheduler', 'None', ['None','stepLR'], help='scheduler, if any used in training')
        flags.DEFINE_integer('warmup', 100, help='Length of warmup, if 0 then no warmup')
        flags.DEFINE_integer('test_set_len', 95, help='')
        flags.DEFINE_integer('num_epochs', 500, help='Number of epochs')
        flags.DEFINE_enum('optimizer', 'Adam', ['AdamW', 'Adam', 'MoMo'], help='MoMo not implemented in particular for now') ## TODO MoMo 
        flags.DEFINE_float('weight_decay', 0.0, help= 'Weight decay hyper parameter')
        flags.DEFINE_float('ema', 0.0, help='Exponentially moving average momentum, if 0 then no ema applied NOT IMPLEMENTED yet ')  ## TODO

        ## Sampling and checkpointing
        flags.DEFINE_integer('save_step_epoch', 100, help='Period in nb of epochs for saving ckpt & losses')
        flags.DEFINE_integer('sample_step_epoch', 100, help='Period in nb of epoch for generating a few npy samples')
        flags.DEFINE_string('sample_folder','/mnt/home/dheurtel/ceph/20_samples/artificial_architecture_exps',help= 'directory where generated samples (in the middle of training) or results are stored')
        flags.DEFINE_string('ckpt_folder','/mnt/home/dheurtel/ceph/10_checkpoints/artificial_architecture_exps',help= 'Directory for ckpt & loss storage (as well as some training specs)')

    except:
        print("Flags already defined but you can still re-parse them with next few cells")

    with open(os.path.join(CKPT_FOLDER, MODEL_ID, 'flagfile.txt')) as f:
        lines = f.read().splitlines()

    lines=[s for s in lines if extract_flag_name(s) in FLAGS]

    FLAGS(lines)

    if FLAGS.network == "unet":
        log2sizes = list(
            range(int(np.log2(FLAGS.size_min)), int(np.log2(FLAGS.size)) + 1)
        )[::-1]
        sizes = [2**i for i in log2sizes]

        network = UNet(
            in_c=FLAGS.in_channel,
            out_c=FLAGS.in_channel,
            first_c=FLAGS.first_c_mult * FLAGS.in_channel,
            sizes=sizes,
            num_blocks=1,
            n_steps=FLAGS.n_steps,
            time_emb_dim=100,
            dropout=FLAGS.dropout,
            attention=[],
            normalisation=FLAGS.normalization,
            padding_mode=FLAGS.padding_mode,
            eps_norm=FLAGS.eps_norm,
        )
        network = network.to(device)
    if FLAGS.network == "ResUNet":
        log2sizes = list(
            range(int(np.log2(FLAGS.size_min)), int(np.log2(FLAGS.size)) + 1)
        )[::-1]
        sizes = [2**i for i in log2sizes]

        network = ResUNet(
            in_c=FLAGS.in_channel,
            out_c=FLAGS.in_channel,
            first_c=FLAGS.first_c_mult * FLAGS.in_channel,
            sizes=sizes,
            num_blocks=1,
            n_steps=FLAGS.n_steps,
            time_emb_dim=100,
            dropout=FLAGS.dropout,
            attention=[],
            normalisation=FLAGS.normalization,
            padding_mode=FLAGS.padding_mode,
            eps_norm=FLAGS.eps_norm,
            skiprescale=FLAGS.skip_rescale,
        )
        network = network.to(device)
    beta_T=FLAGS.beta_T*1000/FLAGS.n_steps
    if FLAGS.diffusion_mode == "ddpm":
        model = DDPM(
            network,
            FLAGS.n_steps,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    elif FLAGS.diffusion_mode == "smld":
        model = NCSN(
            network,
            FLAGS.n_steps,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    elif FLAGS.diffusion_mode == "SigmaDDPM":
        power_spectrum = torch.from_numpy(np.load(FLAGS.power_spectrum, allow_pickle=True).astype(np.float32))
        model  = SigmaDDPM(
            network,
            FLAGS.n_steps,
            power_spectrum=power_spectrum,
            beta_start=FLAGS.beta_start,
            beta_end=beta_T,
            device=device,
        )
    ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID, 'ckpt.pt'),map_location=torch.device('cpu'))

    file_list=ckpt['test_set']
    np.random.shuffle(file_list)
    model.load_state_dict(ckpt['ddpm_model'])

    SOURCE_DIR=FLAGS.source_dir

    if FLAGS.no_lognorm:
        transforms = None
    else:
        transforms = LogNormalTransform()  

    dataset=MHDProjDataset(SOURCE_DIR,random_rotate=FLAGS.random_rotate,transforms=transforms,test_batch_length=FLAGS.test_set_len, test_file_list=file_list)

    test_batch=dataset.test_batch()
    return test_batch, model

def double_loader(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2):
    test_batch1, model1 = load_everything(MODEL_ID1, CKPT_FOLDER1)
    test_batch2, model2 = load_everything(MODEL_ID2, CKPT_FOLDER2)
    return model1, model2, test_batch1, test_batch2

## TODO allow batches? 
def separation2score_loaded(model1, model2, testbatch1, testbatch2, NUM_SAMPLES=32, method='M1A2'):

    return


def separation2score(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2,NUM_SAMPLES=16, method='M1A2'):
    model1, model2, testbatch1, testbatch2 = double_loader(MODEL_ID1, CKPT_FOLDER1, MODEL_ID2, CKPT_FOLDER2)
    return separation2score_loaded(model1, model2, testbatch1, testbatch2, NUM_SAMPLES=NUM_SAMPLES, method = method)


def method1_algo2(model1, model2, observation, NUM_SAMPLES=32):
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


def check_separation1score(MODEL_ID1, CKPT_FOLDER1, noise_step = 500, num_to_denoise = 1, NUM_SAMPLES = 16, tweedie = False):
    test_batch1, model1 = load_everything(MODEL_ID1, CKPT_FOLDER1)
    partial_batch = torch.split(test_batch1[:num_to_denoise], 1, dim = 0)

    partial_batch = [elt.repeat(NUM_SAMPLES, 1, 1, 1) for elt in partial_batch]

    partial_batch = torch.cat(partial_batch, dim = 0)

    if tweedie:
        noisy_batch, batch_denoised = one_step_denoising(model1, partial_batch, noise_step)
    else:
        noisy_batch, batch_denoised = multi_step_denoising(model1, partial_batch, noise_step)

    truth_list = torch.split(test_batch1[:num_to_denoise], 1, dim = 0)
    noisy_list = torch.split(noisy_batch, NUM_SAMPLES, dim = 0)
    denoised_list = torch.split(batch_denoised, NUM_SAMPLES, dim=0)

    return truth_list, noisy_list, denoised_list
    

def separation1score(MODEL_ID1, CKPT_FOLDER1, observation, noise_step = 500, NUM_SAMPLES = 16, tweedie = False, rescale_observation = True):
    _ , model = load_everything(MODEL_ID1, CKPT_FOLDER1)
    t = noise_step
    if rescale_observation:
        partial_batch = torch.split(model.sqrt_alphas_cumprod[t]*observation.to(device), 1, dim = 0)
    else:
        partial_batch = torch.split(observation.to(device), 1, dim = 0)

    partial_batch = [elt.repeat(NUM_SAMPLES, 1, 1, 1) for elt in partial_batch]
    partial_batch = torch.cat(partial_batch, dim = 0)

    if tweedie:
        noisy_batch, batch_denoised = one_step_denoising(model, partial_batch, noise_step, is_observation = True)
    else:
        noisy_batch, batch_denoised = multi_step_denoising(model, partial_batch, noise_step, is_observation = True)

    noisy_list = torch.split(noisy_batch, NUM_SAMPLES, dim = 0)
    denoised_list = torch.split(batch_denoised, NUM_SAMPLES, dim=0)

    return noisy_list, denoised_list


def one_step_denoising(model, batch, t, is_observation = False):
    batch=batch.to(device)

    if not(is_observation):
        noise=torch.randn(batch.shape).to(device)
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch, _=model.add_noise(batch, noise, timesteps)
    else:
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch = batch

    residual=model.reverse(noisy_batch, timesteps)

    s1 = model.sqrt_alphas_cumprod[timesteps] # bs
    s2 = model.sqrt_one_minus_alphas_cumprod[timesteps] # bs
    s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
    s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)

    batch_denoised= (noisy_batch- s2* residual)/s1
        
    return noisy_batch/s1, batch_denoised

def multi_step_denoising(model, batch, t, is_observation = False):
    batch = batch.to(device)
    if not(is_observation):
        noise=torch.randn(batch.shape).to(device)
        timesteps=torch.full((batch.shape[0],), t).long().to(device)
        noisy_batch, _=model.add_noise(batch, noise, timesteps)
    else:
        noisy_batch = batch

    batch_denoised = model.generate_image(noisy_batch.shape[0],noisy_batch.shape[1],noisy_batch.shape[2],sample=noisy_batch,initial_timestep=t)

    return noisy_batch/model.sqrt_alphas_cumprod[t], batch_denoised