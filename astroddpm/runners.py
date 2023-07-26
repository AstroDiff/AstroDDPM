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

from astroddpm.datahandler.dataset import get_dataset_and_dataloader
from astroddpm.diffusion import dm
from astroddpm.diffusion.stochastic import sde
from astroddpm.utils.scheduler import InverseSquareRootScheduler, CosineAnnealingScheduler, LinearScheduler, WarmUp, get_optimizer_and_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## List of knwargs for Diffuser class: use_gpu, device, config, diffusion_model, dataset, dataloader, optimizer, learning rate, scheduler, logging stuff (weight and biases, tensorboard, etc.)
## transforms (for the dataset) -> or maybe just pass the dataset and dataloader as kwargs or put dataset kwargs as a subdict of config
## TODO: add __repr__ to all classes and docstrings to all functions
## TODO add likelihood computations? are is it to hands off for the user?
## TODO add EMA, DDIM, DDRM, PIGDM...


## List of methods and attributes for Diffuser class: train (step, loop, with test losses, ability to see samples ie test and save ckpt as well as resume training)
## , test (generate), save, load (with options to add to a json dict), plot, log, etc.


class Diffuser(nn.Module):
    def __init__(self, diffusion_model, **kwargs):
        super(Diffuser, self).__init__()

        self.diffmodel = diffusion_model

        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "verbose"):
            self.verbose = True ## TODO implement!!!
        if hasattr(self, "config"):
            if type(self.config) == str:
                print("Existing attributes may be overwritten: loading config from {}".format(self.config))
            else:
                print("Existing attributes may be overwritten: loading config from a dict.")
            self.load(config=self.config)

        ## Syncing the config with the attributes
        self.config = self.get_config() 
        
        if hasattr(self, "diffmodel"): ## TODO add an option for cpu?
            self.diffmodel = self.diffmodel.to(device)
        ## TODO check that model id are coherent with the config
        ## logging stuff (weight and biases, tensorboard, etc.)

    def get_config(self):
        res_config = {}
        if hasattr(self, "diffmodel"):
            res_config["diffusion_model"] = self.diffmodel.config
        if hasattr(self, "train_dataloader"):
            try:
                res_config["dataloaders"] = self.train_dataloader.config
            except:
                res_config["dataloaders"] = {}
        
        if hasattr(self, "optimizer"):
            try:
                res_config["optimizer"] = self.optimizer.config
            except:
                res_config["optimizer"] = {}
        if hasattr(self, "scheduler"):
            try:
                res_config["scheduler"] = self.scheduler.config
            except:
                res_config["scheduler"] = {}
        if hasattr(self, "ckpt_epoch"):
            res_config["ckpt_epoch"] = self.ckpt_epoch
        if hasattr(self, "separate_ckpt"):
            res_config["separate_ckpt"] = self.separate_ckpt
        if hasattr(self, "ckpt_dir"):
            res_config["ckpt_dir"] = self.ckpt_dir
        if hasattr(self, "sample_epoch"):
            res_config["sample_epoch"] = self.sample_epoch
        if hasattr(self, "sample_dir"):
            res_config["sample_dir"] = self.sample_dir
        if hasattr(self, "sample_size"):
            res_config["sample_size"] = self.sample_size
        if hasattr(self, "results_size"):
            res_config["results_size"] = self.results_size
        if hasattr(self, "epochs"):
            res_config["epochs"] = self.epochs
        if hasattr(self, "model_id"):
            res_config["model_id"] = self.model_id
        else:
            print("No model id found")
        return res_config
        

    def set_model_id(self, model_id):
        ## TODO check conflicts
        if hasattr(self, "model_id"):
            previous_model_id_attribute = self.model_id
        else:
            previous_model_id_attribute = None
        if hasattr(self, "config"):
            if "model_id" in self.config.keys():
                previous_model_id_config = self.config["model_id"]
            else:
                previous_model_id_config = None
        else:
            previous_model_id_config = None
        self.model_id = model_id
        self.config["model_id"] = model_id
        if previous_model_id_attribute is not None:
            if previous_model_id_attribute != model_id:
                print("Model id was changed from {} to {}.".format(previous_model_id_attribute, self.model_id))
        elif previous_model_id_config is not None:
            if previous_model_id_config != model_id:
                print("Model id was changed from {} to {}.".format(previous_model_id_config, self.model_id))
        
    def set_ckpt_sample(self, ckpt_dir = None, sample_dir = None, ckpt_epoch = None, sample_epoch = None, sample_size = None, results_size = None, separate_ckpt = False):
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.config["ckpt_dir"] = ckpt_dir
            if ckpt_epoch is not None:
                self.ckpt_epoch = ckpt_epoch
                self.config["ckpt_epoch"] = ckpt_epoch
            else:
                warnings.warn("No checkpoint epoch provided, using the default value (100).")
                self.ckpt_epoch = 100
                self.config["ckpt_epoch"] = 100
            self.separate_ckpt = separate_ckpt
        if sample_dir is not None:
            self.sample_dir = sample_dir
            self.config["sample_dir"] = sample_dir
            if sample_epoch is not None:
                self.sample_epoch = sample_epoch
                self.config["sample_epoch"] = sample_epoch
            else:
                warnings.warn("No sample step epoch, using the default value (100).")
                self.sample_epoch = 100
                self.config["sample_epoch"] = 100
            if sample_size is not None:
                self.sample_size = sample_size
                self.config["sample_size"] = sample_size
            else:
                warnings.warn("No sample size provided, using the default value (16).")
                self.sample_size = 16
                self.config["sample_size"] = 16
            if results_size is not None:
                self.results_size = results_size
                self.config["results_size"] = results_size
            else:
                warnings.warn("No results size provided, using the default value (64).")
                self.results_size = 64
                self.config["results_size"] = 64

    def load(self, config, for_training = False, also_ckpt = True, new_model_id = None, **kwargs):
        ## Load a model from a checkpoint, or from a MODEL ID, linked to the json with the link to the ckpt, with option to load the optimizer state, etc.
        ## Option to load parameters, name, ID and stuff from a json dict in the config folder (or elsewhere) for ease of reuse
        ## TODO model id be carful not to overwrite stuff
        print("Loading the diffuser from a config dict.")
        def load_dict():
            if "diffusion_model" in self.config.keys():
                self.diffmodel = dm.get_diffusion_model(self.config["diffusion_model"]).to(device)
            else:
                warnings.warn("No diffusion model found in the config, the model will be initialized with the default parameters.")
                self.diffmodel = dm.get_diffusion_model({}).to(device)
            if "dataloaders" in self.config.keys():
                self.train_dataset, self.test_dataset, self.train_dataloader, self.test_dataloader = get_dataset_and_dataloader(self.config["dataloaders"])
            else:
                warnings.warn("No dataloaders found in the config, the dataloaders will be initialized with the default parameters.")
                self.train_dataset, self.test_dataset, self.train_dataloader, self.test_dataloader = get_dataset_and_dataloader({})
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
            if "separate_ckpt" in self.config.keys():
                self.separate_ckpt = self.config["separate_ckpt"]
            else:
                if hasattr(self, "separate_ckpt"):
                    pass
                else:
                    self.separate_ckpt = False
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
            if "results_size" in self.config.keys():
                self.results_size = self.config["results_size"]
            else:
                if hasattr(self, "results_size"):
                    pass
                else:
                    warnings.warn("No results size found in the config use the set_ckpt_sample method.")
            if "epochs" in self.config.keys(): ## Will get overwritten if epochs is provided as an argument or in a ckpt
                self.epochs = self.config["epochs"]
            else:
                if hasattr(self, "epochs"): 
                    pass
                else:
                    warnings.warn("No epochs found in the config.")

        def find_and_load_ckpt():
            if "ckpt_dir" in self.config.keys():
                ckpt_dir = self.config["ckpt_dir"]
            elif hasattr(self, "ckpt_dir"):
                ckpt_dir = self.ckpt_dir
                print("No ckpt dir path provided, using ckpt dir find in the attributes of the diffuser.")
            else:
                print("No ckpt dir path provided, trying to find the ckpt in the default ckpt dir (./ckpt).")
                ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
            if "model_id" in self.config.keys():
                ckpt_name = self.config["model_id"]
            elif hasattr(self, "model_id"):
                ckpt_name = self.model_id
                print("No model id provided, using model id find in the attributes of the diffuser.")
            else:
                print("No model id provided, trying to find the ckpt with the default model id (DefaultDDPM).")
                ckpt_name = "DefaultDDPM"
            if os.path.isfile(os.path.join(ckpt_dir, ckpt_name,'ckpt.pt')):
                self.load_ckpt(path = os.path.join(ckpt_dir, ckpt_name, 'ckpt.pt'), for_training = for_training)
            else:
                warnings.warn("No checkpoint found in the ckpt_dir, the model will be initialized with the default torch layer parameters.")
        
        if type(config) == str:
            with open(kwargs["config"]) as f:
                self.config = json.load(f)
            load_dict()
            if new_model_id is not None:
                self.set_model_id(new_model_id)
            if also_ckpt:
                try:
                    find_and_load_ckpt()
                except:
                    warnings.warn("Error when loading the checkpoint, load it manually using load_ckpt(path = ..., for_training = ...) and check the arguments you provide.")

        elif type(config) == dict:
            self.config = config
            load_dict()
            if new_model_id is not None:
                self.set_model_id(new_model_id)
            if also_ckpt:
                #try:
                find_and_load_ckpt()
                #except:
                #    warnings.warn("Error when loading the checkpoint, load it manually using load_ckpt(path = ..., for_training = ...) and check the arguments you provide.")   
        else:
            raise ValueError("Config should be a dict or a path to a json file.")

    def load_ckpt(self, path = None, ckpt_dict = None, for_training = False, url = None):
        ## Getting the checkpoint if not provided
        if path is None and ckpt_dict is None and url is None:
            raise ValueError("No checkpoint path or dict provided.")
        elif ckpt_dict is None and path is not None:
            try:
                ckpt_dict = torch.load(path,map_location=torch.device('cpu'))
            except:
                raise ValueError("Could not load the checkpoint from the provided path.")
        elif ckpt_dict is None and url is not None:
            try:
                ckpt_dict = torch.hub.load_state_dict_from_url(url,map_location=torch.device('cpu')) ## TODO actually so cool
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
            if "losses" in ckpt.keys():
                self.losses = ckpt["losses"]
            else:
                warnings.warn("No losses found in the checkpoint, the loss will be initialized to an empty list.")
                self.losses = []
            if "test_losses" in ckpt.keys():
                self.test_losses = ckpt["test_losses"]
            else:
                warnings.warn("No test loss found in the checkpoint, the test loss will be initialized to an empty list.")
                self.test_losses = []
        else:
            print("Loading only the weights, no optimizer or scheduler.")


    def save(self, all_models = False, new_model_id = None, also_ckpt = True, for_training = True):
        ## Save the model config to json dict in ckpt folder plus to the MODELS.json (if asked) and then save the ckpt
        ## Save the model to a checkpoint, with option to save the optimizer state, etc.  
        ## Option to save parameters, name, ID and stuff to a json dict in the config folder (or elsewhere) for ease of reuse
        ## TODO add MODEL_ID attribute to the class and save it to the json dict (how to generate coherent model IDs when not provided...)
        config = self.get_config()
        if new_model_id is not None:
            config["model_id"] = new_model_id
        if all_models:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config' ,"MODELS.json")) as f:
                total_models = json.load(f)
            total_models[config["model_id"]] = config
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', "MODELS.json"), 'w') as f:
                json.dump(total_models, f,indent=4)
        if hasattr(self, "ckpt_dir") and hasattr(self, "model_id"):
            ## Create the dir if needed:
            if not os.path.isdir(os.path.join(self.ckpt_dir, self.model_id)):
                os.makedirs(os.path.join(self.ckpt_dir, self.model_id))
            with open(os.path.join(self.ckpt_dir, self.model_id,'config.json'), 'w') as f:
                json.dump(config, f,indent=4)
        else:
            warnings.warn("No ckpt dir or model id found ==> not saving the config.")
        if also_ckpt:
            try:
                self.save_ckpt(for_training = for_training)
            except:
                warnings.warn("Error when saving the checkpoint, save it manually using save_ckpt(for_training = ...) and check the arguments you provide.")
        pass

    def save_ckpt(self, ckpt_dir = None, ckpt_name = None,verbose = False, for_training = True, separate = False):
        ## Save the model to a checkpoint
        total_ckpt = {}
        diffmodel_state_dict = self.diffmodel.state_dict()
        total_ckpt["diffusion_model"] = diffmodel_state_dict
        if ckpt_name is None:
            if hasattr(self, "model_id"):
                ckpt_name = self.model_id
            else:
                raise ValueError("No model id (or ckpt name) provided, please provide one.")
        if ckpt_dir is None:
            if hasattr(self, "ckpt_dir"):
                ckpt_dir = self.ckpt_dir
            else:
                raise ValueError("No ckpt dir provided, please provide one.")
        if for_training:
            if hasattr(self, "optimizer"):
                optimizer_state_dict = self.optimizer.state_dict()
                total_ckpt["optimizer"] = optimizer_state_dict
            else:
                if verbose:
                    warnings.warn("No optimizer found, not saving the optimizer state.")
            if hasattr(self, "scheduler"):
                scheduler_state_dict = self.scheduler.state_dict()
                total_ckpt["scheduler"] = scheduler_state_dict
            else:
                if verbose:
                    warnings.warn("No scheduler found, not saving the scheduler state.")
            if hasattr(self, "epoch"):
                epoch = self.epoch
                total_ckpt["epoch"] = epoch
            else:
                if verbose:
                    warnings.warn("No epoch found, not saving the epoch.")
            if hasattr(self, "epochs"):
                epochs = self.epochs
                total_ckpt["epochs"] = epochs
            else:
                if verbose:
                    warnings.warn("No epochs found, not saving the epochs.")
            if hasattr(self, "losses"):
                losses = self.losses
                total_ckpt["losses"] = losses
            else:
                if verbose:
                    warnings.warn("No losses found, not saving the losses.")
            if hasattr(self, "test_losses"):
                test_losses = self.test_losses
                total_ckpt["test_losses"] = test_losses
            else:
                if verbose:
                    warnings.warn("No test losses found, not saving the test losses.")            
        else:
            if verbose:
                warnings.warn("Not saving the optimizer and scheduler states, as well as previous training info.")
        if not separate:
            if not os.path.isdir(os.path.join(ckpt_dir, ckpt_name)):
                os.makedirs(os.path.join(ckpt_dir, ckpt_name))
            torch.save(total_ckpt, os.path.join(ckpt_dir, ckpt_name, 'ckpt.pt'))
            print("Successfully saved checkpoint to {}".format(os.path.join(ckpt_dir, ckpt_name)))
            return os.path.join(ckpt_dir, ckpt_name)
        else:
            if not os.path.isdir(os.path.join(ckpt_dir, ckpt_name)):
                os.makedirs(os.path.join(ckpt_dir, ckpt_name))
            str_epoch = str(self.epoch).zfill(np.floor(np.log10(self.epochs)).astype(int)+1)
            torch.save(total_ckpt, os.path.join(ckpt_dir, ckpt_name, 'ckpt_epoch_{}.pt'.format(str_epoch)))
            print("Successfully saved checkpoint to {}".format(os.path.join(ckpt_dir, ckpt_name)))
            return os.path.join(ckpt_dir)

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
            self.optimizer.config = {"type": "Adam", "lr": 1e-3}
        if hasattr(self, "scheduler"):
            scheduler = self.scheduler
        else:
            scheduler = None
            self.scheduler = None
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
                self.epochs = 100
        if "train_dataloader" in kwargs.keys() and finetune:
            train_dataloader = kwargs["train_dataloader"]
            self.train_dataloader = train_dataloader
            print("Train dataloader provided as argument, using it instead of a possible one as attribute. This only works if finetuning is set to True.")
        else:
            if hasattr(self, "train_dataloader"):
                train_dataloader = self.train_dataloader
            else:
                raise ValueError("No dataloader provided, either in the constructor of the diffuser or as argument.")
        if "test_dataloader" in kwargs.keys():
            test_dataloader = kwargs["test_dataloader"]
            self.test_dataloader = test_dataloader
            print("Test dataloader provided as argument, using it instead of a possible one as attribute.")
        else:
            if hasattr(self, "test_dataloader"):
                test_dataloader = self.test_dataloader
            else:
                test_dataloader = None
                self.test_dataloader = None
                print("No test dataloader provided, either in the constructor of the diffuser or as argument, not using any test dataloader.")
        if hasattr(self, "losses") and (resume_training or finetune):
            print("Losses provided as attribute, will use those to keep continuity.")
            losses = self.losses
        else:
            losses = []
            self.losses = []
        if hasattr(self, "test_losses") and (resume_training or finetune):
            print("Test losses provided as attribute, will use those to keep continuity.")
            test_losses = self.test_losses
        else:
            test_losses = []
            self.test_losses = []

        if hasattr(self, "epoch") and (resume_training or finetune):
            print("Epoch provided as attribute, will start training from epoch.")
            epoch = self.epoch
        else:
            epoch = 0
            self.epoch = 0
        if not hasattr(self, 'sample_dir') or not hasattr(self, 'model_id'):
            warnings.warn("Either model_id or sample dir not provided ==> no sample will we generated either during training or at the end.")
            sampling = False
        else:
            sampling = True
            if not hasattr(self, 'sample_size'):
                warnings.warn("No sample size provided, 8 samples will be generated when asked, self.sample_size updated accordingly.")
                self.sample_size = 8
            if not hasattr(self, 'results_size'):
                warnings.warn("No results size provided, samples at the end will be generated in the same quantity as during training, self.results_size updated accordingly.")
                self.results_size = self.sample_size
            if not hasattr(self, "sample_epoch"):
                warnings.warn("No sample epoch provided, using the default value (10 percent of all epochs).")
                self.sample_epoch = self.epochs//10
        if not hasattr(self, "ckpt_dir"):
            warnings.warn("No ckpt dir provided ==> no checkpoint will be saved.")
            ckpting = False
        else:
            ckpting = True
            if not hasattr(self, "ckpt_epoch"):
                warnings.warn("No ckpt epoch provided, using the default value (10 percent of all epochs).")
                self.ckpt_epoch = self.epochs//10
        if "verbose" in kwargs.keys():
            verbose = kwargs["verbose"]
        else:
            verbose = True
        if "save_all_models" in kwargs.keys():
            save_all_models = kwargs["save_all_models"] ##whether or not to save to the MODELS.json file
        else:
            save_all_models = False
        if "separate_ckpt" in kwargs.keys():
            separate_ckpt = kwargs["separate_ckpt"]
            self.separate_ckpt = separate_ckpt
        else:
            if hasattr(self, "separate_ckpt"):
                separate_ckpt = self.separate_ckpt
            separate_ckpt = False
        return finetune, resume_training, ckpting, sampling, verbose, save_all_models, separate_ckpt


    def train(self, **kwargs):
        ## Train the model, option resume training after loading a ckpt, finetuning a pretrained model, etc.
        ## TODO add option to train on a subset of the dataset (for example for debugging)

        finetune, resume_training, ckpting, sampling, verbose, save_all_models, separate_ckpt = self.train_parser(**kwargs)

        if save_all_models:
            self.save(all_models = True)
        else:
            self.save()

        global_step = 0
        steps_per_epoch = len(self.train_dataloader)
        num_timesteps = self.diffmodel.sde.N
            
        ## TODO if discrete only!!!
        for current_epoch in range(self.epoch,self.epochs): ## or epoch+1?
            self.epoch = current_epoch
            self.diffmodel.train()

            progress_bar = tqdm.tqdm(total=steps_per_epoch, disable=not verbose)
            progress_bar.set_description(f"Epoch {self.epoch}")

            for _, batch in enumerate(self.train_dataloader):
                if len(batch.shape)==3:
                    batch = batch.to(device).unsqueeze(1)
                else:
                    batch=batch.to(device)

                timesteps = (
                    torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)
                )
                loss = self.diffmodel.loss(batch, timesteps)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                self.losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1
                if not (self.scheduler is None):
                    self.scheduler.step()
            progress_bar.close()

            if not (self.test_dataloader is None):
                self.diffmodel.eval()
                with torch.no_grad():
                    loss = 0
                    tot_len = 0
                    for _ , test_loss_batch in enumerate(self.test_dataloader):
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

                self.test_losses.append(loss/tot_len)
                self.diffmodel.train()
            if ckpting and (self.epoch % self.ckpt_epoch == 0):
                if separate_ckpt:
                    try:
                        self.save_ckpt(separate=True, verbose=False, for_training = True)
                    except:
                        raise ValueError("Error when trying to save the checkpoint separately, maybe try again without the separate_ckpt option (in train method and attribute).")
                else:
                    self.save_ckpt(verbose=False, for_training = True)

            if sampling and (self.epoch % self.sample_epoch) == 0:
                gen = self.generate(self.sample_size)
                str_epoch = str(self.epoch).zfill(np.floor(np.log10(self.epochs)).astype(int)+1)
                ndigit_samples = np.floor(np.log10(self.sample_size)).astype(int)+1
                if not os.path.isdir(os.path.join(self.sample_dir, self.model_id)):
                    os.makedirs(os.path.join(self.sample_dir, self.model_id))
                gen = np.split(gen.detach().cpu().numpy(),len(gen), axis = 0)
                for i, img in enumerate(gen):
                    file_path = os.path.join(self.sample_dir, self.model_id,"epoch_{}_{}".format(str_epoch, str(i).zfill(ndigit_samples)))
                    np.save(file_path, img)
            if self.epoch == self.epochs-1:
                print("Training finished. Final sampling and checkpointing.")
                if sampling:
                    gen = self.generate(self.results_size)
                    gen = np.split(gen.detach().cpu().numpy(), len(gen), axis = 0)
                    ndigit_samples = np.floor(np.log10(self.results_size)).astype(int)+1
                    for i, img in enumerate(gen):
                        file_path = os.path.join(self.sample_dir, self.model_id, "results_{}".format(str(i).zfill(ndigit_samples)))
                        np.save(file_path, img)
                if ckpting:
                    self.save_ckpt(verbose=False, for_training = True)

        return self.losses, self.test_losses



        ## Modify those if resuming training or finetuning TODO...
        if "resume_training" in kwargs.keys():
            if kwargs["resume_training"]:
                raise NotImplementedError("Resume training is not implemented yet.")
        if "finetune" in kwargs.keys():
            if kwargs["finetune"]:
                raise NotImplementedError("Finetuning is not implemented yet.")
        
    
    def generate(self, sample_size = None, batch_size = None ):
        if sample_size is None:
            if hasattr(self, "sample_size"):
                sample_size = self.sample_size
            else:
                sample_size = 16
        if batch_size is None:
            if hasattr(self, "batch_size"):
                batch_size = self.batch_size
            else:
                batch_size = sample_size
        generated = []
        for _ in range(sample_size//batch_size):
            generated.append(self.diffmodel.generate_image(batch_size))
        if sample_size%batch_size != 0:
            generated.append(self.diffmodel.generate_image(sample_size%batch_size))
        generated = torch.cat(generated, dim=0)
        return generated


    def plot(self, **kwargs):
        ## Plot training curves, samples, etc. TODO change the method name and duplicate it for every needed plot
        pass

    def log(self, **kwargs):
        ## Retrieve training curves, samples, and logs then print them OR give the link towards the tensorboard/weight and biases project/server session
        pass

def config_from_id(model_id, json_path = None):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'config' ,"MODELS.json")) as f:
        total_models = json.load(f)
    if model_id in total_models.keys():
        return total_models[model_id]
    else:
        raise ValueError("Model id {} not found in the MODELS.json file.".format(model_id))
### Add general config (default stuff) function (put that where?)
### Add dataset function