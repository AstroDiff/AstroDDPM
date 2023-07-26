import torch 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from astroddpm.utils.plot import plot_and_save_lines
from astroddpm.runners import get_samples

def correl_fourrier(batch1, batch2 = None, use_gpu = True):
    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    ## Only returns the real part of the correlation
    if batch2 is None:
        batch1 = batch1.to(device)
        fft1 = torch.fft.fft2(batch1)
        return torch.fft.ifft2(torch.einsum('iabc,jabc->ijabc',fft1,fft1.conj())).real
    else:
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        fft1 = torch.fft.fft2(batch1)
        fft2 = torch.fft.fft2(batch2)
        return torch.fft.ifft2(torch.einsum('iabc,jabc->ijabc',fft1,fft2.conj())).real


def closest_pairs(batch1, batch2 = None, use_gpu = True, subset_of_pairs = None):
    if subset_of_pairs is not None and isinstance(subset_of_pairs, int):
        subset_of_pairs = np.random.choice(batch1.shape[0], size = subset_of_pairs, replace = False)
    else:
        if batch2 is None:
            subset_of_pairs = np.arange(batch1.shape[0])
        else:
            subset_of_pairs = np.arange(batch2.shape[0])
    if batch2 is None:
        batch2_ = batch1[subset_of_pairs]
    else:
        batch2_ = batch2[subset_of_pairs]
    corr = correl_fourrier(batch1, batch2_, use_gpu)
    assert len(corr.shape) == 5, "Correlation should be of shape (batch1, batch2, channels, height, width)"

    ## Get the indices of the closest pairs
    B1, B2, C, H, W = corr.shape
    corr = corr.sum(dim = 2)
    corr = corr.reshape(corr.shape[0], corr.shape[1], -1)
    shift_maximas = torch.max(corr, dim = 2)
    closest_pairs = []
    for i in range(B1): ## TODO torchify possible?
        max_i=0
        argmax_i=0
        idx=0
        for j in range(B2):
            if batch2 is None and i == j:
                continue
            temp_max=shift_maximas[0][i][j]
            temp_idx=shift_maximas[1][i][j]
            if temp_max>max_i:
                max_i=temp_max
                argmax_i=j
                idx=temp_idx.cpu()
        closest_pairs.append([i, argmax_i, idx//H, idx%H])
    return closest_pairs

def plot_closest_pairs(batch1, batch2 = None, use_gpu = True, subset_of_pairs = None, num_to_plot = 8, save_file = None, label_list = None, title = None):
    closest_pairs_ = np.array(closest_pairs(batch1, batch2, use_gpu, subset_of_pairs))
    if subset_of_pairs is not None and isinstance(subset_of_pairs, int):
        subset_of_pairs = np.random.choice(batch1.shape[0], size = subset_of_pairs, replace = False)
    else:
        if batch2 is None:
            subset_of_pairs = np.arange(batch1.shape[0])
        else:
            subset_of_pairs = np.arange(batch2.shape[0])
    if batch2 is None:
        batch2_ = batch1[subset_of_pairs]
    else:
        batch2_ = batch2[subset_of_pairs]
    
    batch1_to_plot = batch1[:num_to_plot]
    batch2_to_plot = batch2_[closest_pairs_[:num_to_plot, 1]]
    for i in range(num_to_plot):
        batch2_to_plot[i] = batch2_to_plot[i].roll(shifts = (+closest_pairs_[i ,2], +closest_pairs_[i , 3]), dims = (1,2))
    plot_and_save_lines([batch1_to_plot, batch2_to_plot], save_file = save_file, label_list=label_list, title=title)

def plot_closest_pairs_samples_dataset(diffuser, num_to_plot = 8, save_file = None, label_list = None, title_list = None):
    ## Get results from the sample_dir corresponding to the diffuser
    samples = get_samples(diffuser)
    samples = torch.from_numpy(samples)
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(1)
    elif len(samples.shape) > 4:
        raise ValueError("Samples should be of shape (batch, channels, height, width)")
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    if dataset[0].ndim == 2:
        datapoints = torch.cat([dataset[i].unsqueeze(0).unsqueeze(1) for i in range(min(len(dataset), num_to_plot))])
    elif dataset[0].ndim == 3:
        datapoints = torch.cat([dataset[i].unsqueeze(0) for i in range(min(len(dataset), num_to_plot))])
    try:
        title_0 = title_list[0]
    except:
        title_0 = "Closest generated samples/dataset samples pairs"
    try:
        title_1 = title_list[1]
    except:
        title_1 = "Closest generated/generated pairs"
    plot_closest_pairs(samples, datapoints, num_to_plot = num_to_plot, save_file = save_file, label_list = label_list, title = title_0 )
    plot_closest_pairs(samples, num_to_plot=num_to_plot, save_file = save_file, label_list = label_list, title = title_1)

def plot_closest_pairs_sanity_check(samples):
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(1)
    elif len(samples.shape) > 4:
        raise ValueError("Samples should be of shape (batch, channels, height, width)")
    plot_closest_pairs(samples, samples, num_to_plot = 8, save_file = None, label_list = None, title = None)



