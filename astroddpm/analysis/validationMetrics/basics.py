import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from astroddpm.runners import get_samples

def plot_hist_samples_dataset(diffuser, samples = None, title = None, legend = True, max_num_samples = 100, savefig = None):
    """
    Plots the histogram of the samples and the dataset elements.
    Args:
        diffuser: Diffuser object
        samples: tensor of shape (batch, channels, height, width), if None, the samples are retrieved from the sample_dir corresponding to the diffuser
        title: str, title of the plot
        legend: bool, whether to show the legend
        max_num_samples: int, maximum number of samples to plot
        savefig: str, path to save the plot
    """
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
    """
    Computes the wasserstein distance between the samples and the dataset elements.
    Args:
        diffuser: Diffuser object
        samples: tensor of shape (batch, channels, height, width), if None, the samples are retrieved from the sample_dir corresponding to the diffuser
        max_num_samples: int, maximum number of samples to use
    Returns:
        wasserstein: float, wasserstein distance between the samples and the dataset elements (marginally over the pixel intensities)
    """
    if samples is None:
        ## Get results from the sample_dir corresponding to the diffuser
        samples = get_samples(diffuser)
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    datapoints = torch.cat([dataset[i] for i in range(min(len(dataset), max_num_samples))])
    ## Compute wasserstein distance
    wasserstein = stats.wasserstein_distance(samples.flatten(), datapoints.flatten())
    return wasserstein