#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torch.optim.lr_scheduler as lr_scheduler
import argparse

import tqdm

from astroddpm.runners import Diffuser
from diffusion.dm import DiscreteSBM ## Placeholder
from diffusion.stochastic.sde import DiscreteVPSDE ## Placeholder
from diffusion.models.network import ResUNet ## Placeholder


## Constants and folders through flag parsing
####################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Training
####################################################################################

def main(args):

    if args.ckpt_folder is None:
        print("Looking for the diffuser corresponding to model_id {} in the MODELS.json all config file".format(args.model_id)) 
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config' ,"MODELS.json")) as f:
            total_models = json.load(f)
        config = total_models[args.model_id]
    else:
        print("Looking for the diffuser config file corresponding to model_id {} in the ckpt folder provided".format(args.model_id)) 
        with open(os.path.join(args.ckpt_folder,'config.json')) as f:
            all_config = json.load(f)
        config = all_config[args.model_id]
    diffuser = Diffuser(DiscreteSBM(DiscreteVPSDE(1000),ResUNet()), verbose = args.verbose) ##Placeholder
    if args.resume_training or args.finetune:
        diffuser.load(config, also_ckpt=True)
        raise NotImplementedError("Finetuning and resume training are not implemented yet")
    else:
        diffuser.load(config, also_ckpt=False)

    ## check that config is complete for ckpint and sampling?

    diffuser.train(verbose = args.verbose, save_all_models = args.all_models)

if __name__ == "__main__":
        ## argparse first
    parser = argparse.ArgumentParser(description='Train a diffuser')
    parser.add_argument('--model_id', type=str, default='DiffuserForget', help='model id')
    parser.add_argument('--ckpt_folder', type=str, default=None, help='ckpt folder')
    parser.add_argument('--all_models', type=str, default=None, help='config file')
    parser.add_argument('--finetune', type=bool, default=False, help='finetune')
    parser.add_argument('--finetune_ckpt_path', type=str, default=None, help='finetune ckpt path')
    parser.add_argument('--resume_training', type=bool, default=False, help='resume training')
    parser.add_argument('--verbose', type=str, default="Reduced", help='verbose')

    args = parser.parse_args()
    main(args)
