import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import torch
import re


def plot_hist_samples_dataset(diffuser, samples = None, title = None, legend = True, max_num_samples = 100, savefig = None):
    if samples is None:
        ## Get results from the sample_dir corresponding to the diffuser
        samples = get_samples(diffuser)
    label_samp = diffuser.config["model_id"]
    try:
        label_dataset = diffuser.config["dataloaders"]["dataset"]["name"]
    except:
        label_dataset = "Dataset"
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    datapoints = torch.cat([dataset[i] for i in range(min(len(dataset), max_num_samples))])
    ## Plot histogram of samples
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.hist([samples.flatten(), datapoints.flatten()] , bins = 100, density = True, histtype='step', label=[label_samp, label_dataset])
    plt.legend()
    plt.show()
    if title is not None:
        plt.title(title)
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()

def wasserstein(diffuser, samples = None, max_num_samples = 100):
    if samples is None:
        ## Get results from the sample_dir corresponding to the diffuser
        samples = get_samples(diffuser)
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    datapoints = torch.cat([dataset[i] for i in range(min(len(dataset), max_num_samples))])
    ## Compute wasserstein distance
    wasserstein = stats.wasserstein_distance(samples.flatten(), datapoints.flatten())
    return wasserstein



def get_samples(diffuser):
    pattern_results = r'results_(\d+)'
    l = os.listdir(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"]))
    results = [file for file in l if re.match(pattern_results, file)]
    ## Collect all the np img in results
    samples = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],file)) for file in results]
    ## Reshape samples so that they are (N_samples, C, H, W)
    if len(samples[0].shape) == 2:
        samples = [np.reshape(sample, (1, 1, sample.shape[0], sample.shape[1])) for sample in samples]
    elif len(samples[0].shape) == 3:
        samples = [np.reshape(sample, (1, sample.shape[0], sample.shape[1], sample.shape[2])) for sample in samples]
    samples = np.concatenate(samples, axis = 0)
    return samples
