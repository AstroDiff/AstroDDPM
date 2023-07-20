# -*- coding: utf-8 -*-
################################################################################
# This file, part of AstroDDPM, is distributed under an MIT License. It defines several functions and classes
# that are used to build a dataset and separate it into training and test sets (possibly with a validation set).
# It relies on standard python librairies (os, glob, random, etc.) and on the pytorch library.
# Datasets defined here are designed to parse and load numpy arrays of images or physical fields but can be easily
# adapted to other formats (e.g. FITS files, etc.).
# It is not yet optimized for large datasets/parallelization.

################################################################################
# Imports
import os
import random
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision import transforms
from PIL import Image
from astropy.io import fits


################################################################################
## Downloads


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
## Datasets


def make_dataset(dataset_name, transforms=None, seed=33, split=0.1):
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    ) as f:
        configs = json.load(f)
    is_downloaded = dataset_name in configs.keys()

    if not is_downloaded:
        download_dataset(dataset_name)
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

        train_dataset = NPDataset(config, train_list, transforms=transforms)
        test_dataset = NPDataset(config, test_list, transforms=transforms)

        return train_dataset, test_dataset
    elif config["type"] == "fits files":
        return None  # FITSDataset(config, transforms=transforms, seed=seed, split=split) TODO...


class NPDataset(Dataset):
    def __init__(self, config, file_list, transforms=None):
        super(NPDataset).__init__()
        try:
            self.dir = config["dir"]
        except:
            raise ValueError("No directory specified in config")
        self.transforms = transforms
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = torch.from_numpy(np.load(os.path.join(self.dir, self.file_list[index])))
        if self.transforms is not None:
            img = self.transforms(img)
        return img
