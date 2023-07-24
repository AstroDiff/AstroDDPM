import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import json
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import warnings
import tqdm

import datahandler.dataset as dataset
from diffusion import dm
from diffusion.stochastic import sde
from utils.scheduler import InverseSquareRootScheduler, CosineAnnealingScheduler, LinearScheduler, WarmUp, get_optimizer_and_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## List of knwargs for Diffuser class: use_gpu, device, config, diffusion_model, dataset, dataloader, optimizer, learning rate, scheduler, logging stuff (weight and biases, tensorboard, etc.)
## transforms (for the dataset) -> or maybe just pass the dataset and dataloader as kwargs or put dataset kwargs as a subdict of config
## TODO: add __repr__ to all classes and docstrings to all functions
## TODO add likelihood computations? are is it to hands off for the user?


## List of methods and attributes for Diffuser class: train (step, loop, with test losses, ability to see samples ie test and save ckpt as well as resume training)
## , test (generate), save, load (with options to add to a json dict), plot, log, etc.


class Diffuser(nn.Module):
    def __init__(self, diffusion_model, **kwargs):
        super(Diffuser, self).__init__()

        self.diffmodel = diffusion_model

        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "config"):
            print("Existing attributes may be overwritten.\n Loading config from {}".format(self.config))
            self.load(config=self.config)
        elif hasattr(self, "model_id"):
            print("Model id was provided at initialization. Attributes provided along it as args may be overwritten. Loading config (and possibly ckpt) corresponding to the model id: {}\n".format(self.model_id))
            self.load(model_id=self.model_id)
        elif hasattr(self, "ckpt"):
            warnings.warn("Initializing with a checkpoint is discouraged to avoid loading a cheackpoint on an incompatible model. Define the model architecture (if not already done) and use the load method instead.")

        ## Syncing the config with the attributes
        self.config = self.get_config() 
        
        if hasattr(self, "diffmodel"):
            self.diffmodel = self.diffmodel.to(device)

        ## logging stuff (weight and biases, tensorboard, etc.)
        ## transforms (for the dataset) -> or maybe just pass the dataset and dataloader as kwargs or put dataset kwargs as a subdict of config
    def get_config(self):
        res_config = {}
        if hasattr(self, "diffmodel"):
            diffmodel_config = self.diffmodel.config
            res_config["diffusion_model"] = diffmodel_config
        if hasattr(self, "train_dataloader"):
            try:
                train_dataloader_config = self.train_dataloader.config
                res_config["dataloaders"] = train_dataloader_config
            except:
                res_config["dataloaders"] = {}
        if hasattr(self, "model_id"):
            model_id = self.model_id
            res_config["model_id"] = model_id
        if hasattr(self, "optimizer"):
            optimizer_config = self.optimizer.config
            res_config["optimizer"] = optimizer_config
        if hasattr(self, "scheduler"):
            scheduler_config = self.scheduler.config
            res_config["scheduler"] = scheduler_config
        if hasattr(self, "ckpt_epoch"):
            ckpt_epoch = self.ckpt_epoch
            res_config["ckpt_epoch"] = ckpt_epoch
        if hasattr(self, "ckpt_dir"):
            ckpt_dir = self.ckpt_dir
            res_config["ckpt_dir"] = ckpt_dir
        if hasattr(self, "sample_epoch"):
            sample_epoch = self.sample_epoch
            res_config["sample_epoch"] = sample_epoch
        if hasattr(self, "sample_dir"):
            sample_dir = self.sample_dir
            res_config["sample_dir"] = sample_dir
        if hasattr(self, "sample_size"):
            sample_size = self.sample_size
            res_config["sample_size"] = sample_size
        return res_config
        

    def set_model_id(self, model_id):
        if hasattr(self, "model_id"):
            previous_model_id_attribute = self.model_id
        if hasattr(self, "config"):
            if "model_id" in self.config.keys():
                previous_model_id_config = self.config["model_id"]
        self.model_id = model_id
        self.config["model_id"] = model_id
        if previous_model_id_attribute is not None:
            if previous_model_id_attribute != model_id:
                print("Model id was changed from {} to {}.".format(previous_model_id_attribute, self.model_id))
        elif previous_model_id_config is not None:
            if previous_model_id_config != model_id:
                print("Model id was changed from {} to {}.".format(previous_model_id_config, self.model_id))
        
    def set_ckpt_sample(self, ckpt_dir = None, sample_dir = None, ckpt_epoch = None, sample_epoch = None, sample_size = None):
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.config["ckpt_dir"] = ckpt_dir
            if ckpt_epoch is not None:
                self.ckpt_epoch = ckpt_epoch
                self.config["ckpt_epoch"] = ckpt_epoch
            else:
                warnings.warn("No checkpoint step provided, using the default value (1000).")
                self.ckpt_epoch = 100
                self.config["ckpt_epoch"] = 100
        if sample_dir is not None:
            self.sample_dir = sample_dir
            self.config["sample_dir"] = sample_dir
            if sample_epoch is not None:
                self.sample_epoch = sample_epoch
                self.config["sample_epoch"] = sample_epoch
            else:
                warnings.warn("No sample step provided, using the default value (100).")
                self.sample_epoch = 100
                self.config["sample_epoch"] = 100
            if sample_size is not None:
                self.sample_size = sample_size
                self.config["sample_size"] = sample_size
            else:
                warnings.warn("No sample size provided, using the default value (16).")
                self.sample_size = 16
                self.config["sample_size"] = 16

    def load(self, **kwargs):
        ## Load a model from a checkpoint, or from a MODEL ID, linked to the json with the link to the ckpt, with option to load the optimizer state, etc.
        ## Option to load parameters, name, ID and stuff from a json dict in the config folder (or elsewhere) for ease of reuse

        if "ckpt" in kwargs.keys():
            print("""            Loading model weights and possibly optimizer state from a checkpoint.
            This method will not change the model architecture, nor the optimizer, dataset. It can only change the state they were left in.
            If no model (or the DiffusionModel() dummy model) was provided or the architecture is not a match, this will result in an error. 
            If this load is part of resuming training (or finetuning), good practices recommend having the optimizer state in the checkpoint.""")
            if type(kwargs["ckpt"]) == str:
                self.load_ckpt(ckpt_path=kwargs["ckpt"])
            elif type(kwargs["ckpt"]) == dict:
                self.load_ckpt(dict=kwargs["ckpt"])
            self.load_ckpt()
        elif "model_id" in kwargs.keys():
            print("Loading a whole diffuser given a model id. The MODELS.json dict value associated to the id should contain all the config elements. This method will change the model architecture, optimizer, dataset, etc. to match the saved model then load the corresponding states.")
            with open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config","MODELS.json")
            ) as f:
                all_models = json.load(f)
            if kwargs["model_id"] in all_models.keys():
                self.config = all_models[kwargs["model_id"]]
                self.load(config=self.config)
            else:
                raise ValueError("Model ID not found in MODELS.json")
        elif "config" in kwargs.keys():
            print("Loading the diffuser from a config dict or json file.")
            def load_dict(config_dict):
                if "diffusion_model" in self.config.keys():
                    self.diffmodel = dm.get_diffusion_model(self.config["diffusion_model"]).to(device)
                else:
                    warnings.warn("No diffusion model found in the config, the model will be initialized with the default parameters.")
                    self.diffmodel = dm.get_diffusion_model({}).to(device)
                if "dataloaders" in self.config.keys():
                    self.train_dataset, self.test_dataset, self.train_dataloader, self.test_dataloader = dataset.get_dataset_and_dataloader(self.config["dataloaders"])
                else:
                    warnings.warn("No dataloaders found in the config, the dataloaders will be initialized with the default parameters.")
                    self.train_dataset, self.test_dataset, self.train_dataloader, self.test_dataloader = dataset.get_dataset_and_dataloader({})
                if "optimizer" in self.config.keys():
                    self.optimizer, self.scheduler = get_optimizer_and_scheduler(self.config, self.diffmodel.parameters())
                else:
                    warnings.warn("No optimizer found in the config, the optimizer will be initialized with the default parameters.")
                    self.optimizer, self.scheduler = get_optimizer_and_scheduler({}, self.diffmodel.parameters())
                if "model_id" in self.config.keys():
                    self.model_id = self.config["model_id"]
                else:
                    if hasattr(self, "model_id"):
                        pass
                    else:
                        warnings.warn("No model id found in the config use the set_model_id method.")
                if "ckpt_dir" in self.config.keys():
                    self.ckpt_dir = self.config["ckpt_dir"]
                else:
                    if hasattr(self, "ckpt_dir"):
                        pass
                    else:
                        warnings.warn("No ckpt dir found in the config use the set_ckpt_sample method.")
                if "sample_dir" in self.config.keys():
                    self.sample_dir = self.config["sample_dir"]
                else:
                    if hasattr(self, "sample_dir"):
                        pass
                    else:
                        warnings.warn("No sample dir found in the config use the set_ckpt_sample method.")
                if "ckpt_epoch" in self.config.keys():
                    self.ckpt_epoch = self.config["ckpt_epoch"]
                else:
                    if hasattr(self, "ckpt_epoch"):
                        pass
                    else:
                        warnings.warn("No ckpt step found in the config use the set_ckpt_sample method.")
                if "sample_epoch" in self.config.keys():
                    self.sample_epoch = self.config["sample_epoch"]
                else:
                    if hasattr(self, "sample_epoch"):
                        pass
                    else:
                        warnings.warn("No sample step found in the config use the set_ckpt_sample method.")
                if "sample_size" in self.config.keys():
                    self.sample_size = self.config["sample_size"]
                else:
                    if hasattr(self, "sample_size"):
                        pass
                    else:
                        warnings.warn("No sample size found in the config use the set_ckpt_sample method.")

            def find_and_load_ckpt():
                if "ckpt_dir" in self.config.keys():
                    ckpt_dir = self.config["ckpt_dir"]
                else:
                    print("No ckpt dir path provided, trying to find the ckpt in the default ckpt dir (./ckpt).")
                    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
                if "model_id" in self.config.keys():
                    ckpt_name = self.config["model_id"]
                else:
                    print("No model id provided, trying to find the ckpt with the default model id (DefaultDDPM).")
                    ckpt_name = "DefaultDDPM"
                if os.path.isfile(os.path.join(ckpt_dir, ckpt_name+'.pt')):
                    if "for_training" in self.config.keys():
                        print("Loading the checkpoint for training, as specified in the config.")
                        self.load_ckpt(path = os.path.join(ckpt_dir, ckpt_name), for_training = self.config["for_training"])
                    else:
                        print("Loading the checkpoint without optimizer and training information, as specified in the config.")
                        self.load_ckpt(path = os.path.join(ckpt_dir, ckpt_name))
                else:
                    warnings.warn("No checkpoint found in the ckpt_dir, the model will be initialized with the default torch layer parameters.")
            if type(kwargs["config"]) == str:
                with open(kwargs["config"]) as f:
                    self.config = json.load(f)
                load_dict(self.config)
                find_and_load_ckpt()

            elif type(kwargs["config"]) == dict:
                self.config = kwargs["config"]
                load_dict(self.config)
                find_and_load_ckpt()

                ## Try and find a checkpoint to load the weights from there, otherwise say it + warn the user that if he want's to finetune, he should change the model id. if he wants to resume training, he doesn"t have to
            else:
                raise ValueError("Config should be a dict or a path to a json file.")
        else:
            raise ValueError("No checkpoint (path or dict), no model id and no config provided.")


    def load_ckpt(self, path = None, ckpt_dict = None, for_training = False, url = None):
        ## Getting the checkpoint if not provided
        if path is None and ckpt_dict is None and url is None:
            raise ValueError("No checkpoint path or dict provided.")
        elif ckpt_dict is None and path is not None:
            try:
                ckpt = torch.load(path,map_location=torch.device('cpu'))
            except:
                raise ValueError("Could not load the checkpoint from the provided path.")
        elif ckpt_dict is None and url is not None:
            try:
                ckpt = torch.hub.load_state_dict_from_url(url,map_location=torch.device('cpu')) ## TODO actually so cool
            except:
                raise ValueError("Could not load the checkpoint from the provided url.")
        ## Loading the checkpoint
        ckpt = ckpt_dict
        if "diffusion_model" in ckpt.keys():
            self.diffmodel.load_state_dict(ckpt["diffusion_model"]) 
        else:
            warnings.warn("No diffusion model found in the checkpoint, the model will be initialized with the default parameters as model weights will not be loaded.")
        if for_training:
            print("Loading the optimizer and scheduler states from the checkpoint, as well as previous training info. You should change the model id if you want to finetune the model.")
            if "optimizer" in ckpt.keys():
                self.optimizer.load_state_dict(ckpt["optimizer"])
            else:
                warnings.warn("No optimizer found in the checkpoint, the optimizer will be initialized with the default parameters. It could result in instability (in the case of Adam for example).")
            if "scheduler" in ckpt.keys():
                self.scheduler.load_state_dict(ckpt["scheduler"])
            else:
                warnings.warn("No scheduler found in the checkpoint, the scheduler will be initialized with the default parameters. Learning rate may jump.")
            if "epoch" in ckpt.keys():
                self.epoch = ckpt["epoch"]
            else:
                warnings.warn("No epoch found in the checkpoint, the epoch will be initialized to 0.")
                self.epoch = 0
            if "epochs" in ckpt.keys():
                self.epochs = ckpt["epochs"]
            else:
                warnings.warn("No epochs found in the checkpoint, the epochs will be initialized to 100.")
                self.epochs = 100
            if "loss" in ckpt.keys():
                self.losses = ckpt["loss"]
            else:
                warnings.warn("No loss found in the checkpoint, the loss will be initialized to an empty list.")
            if "test_losses" in ckpt.keys():
                self.test_losses = ckpt["test_losses"]
            else:
                warnings.warn("No test loss found in the checkpoint, the test loss will be initialized to an empty list.")
        else:
            print(" Not loading the optimizer and scheduler states from the checkpoint, as well as previous training info, only the weights.")

    def save(self, **kwargs):
        ## Save the model config to json dict in ckpt folder plus to the MODELS.json (if asked) and then save the ckpt
        ## Save the model to a checkpoint, with option to save the optimizer state, etc.  
        ## Option to save parameters, name, ID and stuff to a json dict in the config folder (or elsewhere) for ease of reuse
        ## TODO add MODEL_ID attribute to the class and save it to the json dict (how to generate coherent model IDs when not provided...)
        pass

    def save_ckpt(self, ckpt_folder, ckpt_name,verbose = False, ):
        ## Save the model to a checkpoint
        diffmodel_state_dict = self.diffmodel.state_dict()

        raise NotImplementedError("Saving checkpoints is not implemented yet.") ## TODO
        print("Successfully saved checkpoint to {}".format(os.path.join(ckpt_folder, ckpt_name)))
        return os.path.join(ckpt_folder, ckpt_name)

    def train_parser(self,**kwargs):
        ## Parse the arguments for training
        if "resume_training" in kwargs.keys(): 
            if kwargs["resume_training"]:
                print("Resuming training from the checkpoint path, as specified in the config.")
                self.load_ckpt(path = os.path.join(self.ckpt_dir, self.model_id), for_training = True)
                ## this should also load self.losses, self.test_losses, self.epoch, self.epochs, etc.
                if self.epoch == self.epochs-1:
                    warnings.warn("The model has already been trained for the specified number of epochs. The checkpoint has been loaded but no training will take place. You can change the number of epochs either by diffuser.epochs = ... then calling train() or by specifying a higher number of epoch in the kwargs of train as epochs = ...")
            else:
                resume_training = False
        else:
            resume_training = False
    
        if "finetune" in kwargs.keys(): ## TODO
            if kwargs["finetune"]:
                raise NotImplementedError("Finetuning is not implemented yet.")
            else:
                finetune = False
        else:
            finetune = False
        
        if hasattr(self, "optimizer"):
            optimizer = self.optimizer
        else:
            print("No optimizer provided, using Adam with lr=1e-3 and setting the corresponding attr to it.")
            optimizer = optim.Adam(self.diffmodel.parameters(), lr=1e-3)
            self.optimizer = optimizer
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler
        else:
            scheduler = None
        if "epochs" in kwargs.keys():
            epochs = kwargs["epochs"]
            print("Epochs provided as argument, using it instead of a possible one as attribute and setting the epochs attribute to it.")
            self.epochs = epochs
        else:   
            if hasattr(self, "epochs"):
                print("Epochs provided as attribute, using it. Be warned the model will start training at self.epoch until self.epochs-1.")
                epochs = self.epochs
            else:
                print("No epochs provided, either in the constructor of the diffuser or as argument, training for 100 epochs.")
                epochs = 100
        if "train_dataloader" in kwargs.keys() and finetune:
            train_dataloader = kwargs["train_dataloader"]
            print("Train dataloader provided as argument, using it instead of a possible one as attribute. This only works if finetuning is set to True.")
        else:
            if hasattr(self, "train_dataloader"):
                train_dataloader = self.train_dataloader
            else:
                raise ValueError("No dataloader provided, either in the constructor of the diffuser or as argument.")
        if "test_dataloader" in kwargs.keys():
            test_dataloader = kwargs["test_dataloader"]
            print("Test dataloader provided as argument, using it instead of a possible one as attribute.")
        else:
            if hasattr(self, "test_dataloader"):
                test_dataloader = self.test_dataloader
            else:
                test_dataloader = None
                print("No test dataloader provided, either in the constructor of the diffuser or as argument, not using any test dataloader.")
        if hasattr(self, "losses") and (resume_training or finetune):
            print("Losses provided as attribute, will use those to keep continuity.")
            losses = self.losses
        else:
            losses = []
        if hasattr(self, "test_losses") and (resume_training or finetune):
            print("Test losses provided as attribute, will use those to keep continuity.")
            test_losses = self.test_losses
        else:
            test_losses = []

        if hasattr(self, "epoch") and (resume_training or finetune):
            print("Epoch provided as attribute, will start training from epoch.")
            epoch = self.epoch
        else:
            epoch = 0
        return optimizer, scheduler, train_dataloader, test_dataloader, losses, test_losses, epochs, epoch, resume_training, finetune

    def train(self, **kwargs):
        ## Train the model, option resume training after loading a ckpt, finetuning a pretrained model, etc.
        

        ## TODO add option to train on a subset of the dataset (for example for debugging)
        ## TODO add checkpoint management (save, load, etc.) (BUT DO NOT MODIFY THE CKPT GIVEN AS LOADING WEIGHT....only use ckpt_dir and model_id)

        optimizer, scheduler, train_dataloader, test_dataloader, losses, test_losses, epochs, epoch, resume_training, finetune = self.train_parser(**kwargs)
        ## TODO if training without ckpting or saving, add a warning that the training will not be saved and that the user should save the model manually if he wants to keep it, or start again. 
        global_step = 0
        losses = []
        test_losses_epoch = []
        steps_per_epoch = len(train_dataloader)
        num_timesteps = self.diffmodel.sde.N

        ## TODO if discrete only!!!
        for epoch in range(epochs):
            self.diffmodel.train()

            progress_bar = tqdm.tqdm(total=steps_per_epoch)
            progress_bar.set_description(f"Epoch {epoch}")

            for _, batch in enumerate(train_dataloader):
                if len(batch.shape)==3:
                    batch = batch.to(device).unsqueeze(1)
                else:
                    batch=batch.to(device)

                timesteps = (
                    torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)
                )
                loss = self.diffmodel.loss(batch, timesteps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1
                if not (scheduler is None):
                    scheduler.step()
            progress_bar.close()

            

            if not (test_dataloader is None):
                self.diffmodel.eval()
                with torch.no_grad():
                    loss = 0
                    tot_len = 0
                    for _ , test_loss_batch in enumerate(test_dataloader):
                        if len(test_loss_batch.shape)==3:
                            test_loss_batch = test_loss_batch.to(device).unsqueeze(1)
                        else:
                            test_loss_batch = test_loss_batch.to(device)
                        batch = test_loss_batch.to(device)
                        timesteps = (
                            torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)
                        )
                        loss += self.diffmodel.loss(batch, timesteps).detach().cpu().item()*len(test_loss_batch)
                        tot_len += len(test_loss_batch)

                test_losses_epoch.append(loss/tot_len)
                self.diffmodel.train()
        return losses, test_losses_epoch



        ## Modify those if resuming training or finetuning TODO...
        if "resume_training" in kwargs.keys():
            if kwargs["resume_training"]:
                raise NotImplementedError("Resume training is not implemented yet.")
        if "finetune" in kwargs.keys():
            if kwargs["finetune"]:
                raise NotImplementedError("Finetuning is not implemented yet.")
        
    
    def generate(self, **kwargs):
        ## Generate samples from the model with option to chose DDIM (or other) sampling method, number of samples, etc.
        ## can also be used to generate samples from a trained model (after loading it of course).
        ## Shouldn't have to be modified to support latent diffusion models/score based models
        pass




    def plot(self, **kwargs):
        ## Plot training curves, samples, etc. TODO change the method name and duplicate it for every needed plot
        pass

    def log(self, **kwargs):
        ## Retrieve training curves, samples, and logs then print them OR give the link towards the tensorboard/weight and biases project/server session
        pass



### Add general config (default stuff) function (put that where?)
### Add dataset function