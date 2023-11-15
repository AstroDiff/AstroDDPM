# -*- coding: utf-8 -*-
################################################################################
# Imports
import os
import random
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import transforms
from PIL import Image
from astropy.io import fits
import warnings


################################################################################
## Downloads

## TODO add __repr__ to all classes

def json_load(file):
    with open(file) as f:
        obj = json.load(f)
    return obj


def download_dataset(dataset_name):
    locations_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "locations.json"
    )
    local_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local.json")
    locations_dict = json_load(locations_file)
    local_dict = json_load(local_file)
    if dataset_name in local_dict:
        return local_dict[dataset_name]
    elif dataset_name in locations_dict:
        return web_download(dataset_name, locations_dict[dataset_name])
    else:
        raise ValueError("Dataset name not found in locations.json or local.json")


def web_download(dataset_name, locations_dict):
    ## Download the dataset from the web, depends on the dataset type

    ## CATS-MHD TODO simplify the download process (one tar file for all datasets)
    if dataset_name == "CATS_MHD_BPROJ_DENSITY":
        url = locations_dict["url"]
        print("Downloading dataset {} from {}".format(dataset_name, url))
    if dataset_name == "CATS_MHD_OrthBPROJ_DENSITY":
        url = locations_dict["url"]
        print("Downloading dataset {} from {}".format(dataset_name, url))
    if dataset_name == "CATS_MHD_BPROJ_IQU":
        url = locations_dict["url"]
        print("Downloading dataset {} from {}".format(dataset_name, url))
    if dataset_name == "CATS_MHD_OrthBPROJ_IQU":
        url = locations_dict["url"]
        print("Downloading dataset {} from {}".format(dataset_name, url))



################################################################################
## Transforms

class RandomRotate90(torch.nn.Module):
    '''Randomly rotate the image by a multiple 90 degrees'''
    def __init__(self):
        super().__init__()
    def forward(self, x):
        ''''
        Rotate the image by a multiple of 90 degrees randomly
        Args:
            x: tensor of shape (batch, channels, height, width)
        Returns:
            x: tensor of shape (batch, channels, height, width)'''
        if len(x.shape)==4:
            return torch.rot90(x, random.randint(0, 3), [2, 3])
        elif len(x.shape)==3:
            return torch.rot90(x, random.randint(0, 3), [1, 2])
        elif len(x.shape)==2:
            return torch.rot90(x, random.randint(0, 3), [0, 1])
        else:
            raise ValueError('Invalid shape for image tensor')

def transform_parser(transforms_list, config=None):
    if transforms_list==[] or transforms_list is None:
        return None
    else:
        transforms_to_apply = []
        for transform_str in transforms_list:
            if transform_str.lower() == 'totensor':
                transforms_to_apply.append(transforms.ToTensor())
            elif transform_str.lower() == 'random_rotate90':
                transforms_to_apply.append(RandomRotate90())
            elif transform_str.lower() == 'random_horizontal_flip':
                transforms_to_apply.append(transforms.RandomHorizontalFlip())
            elif transform_str.lower() == 'random_vertical_flip':
                transforms_to_apply.append(transforms.RandomVerticalFlip())
            else:
                warnings.warn('Transform {} not implemented'.format(transform_str))
                pass
        return transforms.Compose(transforms_to_apply)

################################################################################
## Datasets


def make_dataset(dataset_name, transformations=None, seed=33, split=0.1):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_config.json")) as f:
        configs = json.load(f)
    is_downloaded = dataset_name in configs.keys()

    if not is_downloaded:
        print("Dataset {} not found in dataset_config.json, if you have it, you should use add_dataset(dataset_name, dir, transforms) (after having downloaded it).".format(dataset_name))
        return None 
    config = configs[dataset_name]

    if config["type"] == "npy files":
        file_list = os.listdir(config["dir"])
        file_list.sort()
        np.random.seed(seed)
        np.random.shuffle(file_list)
        train_list = file_list[: int((1 - split) * len(file_list))]
        test_list = file_list[int((1 - split) * len(file_list)) :]
        train_list.sort()
        test_list.sort()

        if transformations=="None":
            transformations = None
        elif transformations is None:
            try:
                transformations = transform_parser(config["transforms"])
            except:
                transformations = None
        else:
            pass

        config["name"] = dataset_name

        train_dataset = NPDataset(config, train_list, transforms=transformations)
        test_dataset = NPDataset(config, test_list, transforms=transformations)

        return train_dataset, test_dataset
    elif config["type"] == "fits files":
        return None  # FITSDataset(config, transforms=transforms, seed=seed, split=split) TODO...

def add_dataset(dataset_name, dataset_dir, transformations = None):
    ## Adds the dataset to the list of available datasets in the dataset_config.json file, and downloads it if necessary and if url is provided.
    pass

class NPDataset(Dataset):
    '''Custom dataset for npy files'''
    def __init__(self, config = None, file_list = None, dir = None, transforms=None):
        super(NPDataset).__init__()
        self.config = config
        try:
            self.dir = config["dir"]
        except:
            if dir is None:
                raise ValueError("No directory specified in config")
            else:
                self.dir = dir
        self.transforms = transforms
        if file_list is None:
            self.file_list = os.listdir(self.dir)
            self.file_list.sort()
        else:
            self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = torch.from_numpy(np.load(os.path.join(self.dir, self.file_list[index])))
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_dataset_and_dataloader(config):
    '''Returns the dataset and dataloader from the config dict'''
    dataloaders_config = config
    if "name" not in dataloaders_config.keys():
        dataloaders_config["name"] = "CATS_MHD_BPROJ_DENSITY"
    if "seed" not in dataloaders_config.keys():
        dataloaders_config["seed"] = 33
    if "split" not in dataloaders_config.keys():
        dataloaders_config["split"] = 0.1
    
    train_dataset, test_dataset = make_dataset(dataset_name=dataloaders_config["name"], seed=dataloaders_config["seed"], split=dataloaders_config["split"])

    if "train_batch_size" not in dataloaders_config.keys():
        dataloaders_config["train_batch_size"] = 32
    if "test_batch_size" not in dataloaders_config.keys():
        dataloaders_config["test_batch_size"] = 32
    if "num_workers" not in dataloaders_config.keys():
        dataloaders_config["num_workers"] = 8
    
    train_dataloader = DataLoader(train_dataset, batch_size=dataloaders_config["train_batch_size"], shuffle=True, num_workers=dataloaders_config["num_workers"], pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=min(dataloaders_config["test_batch_size"],len(test_dataset)), shuffle=True, num_workers=dataloaders_config["num_workers"], pin_memory=True, drop_last=True)

    train_dataloader.config = dataloaders_config
    return train_dataset, test_dataset, train_dataloader, test_dataloader
