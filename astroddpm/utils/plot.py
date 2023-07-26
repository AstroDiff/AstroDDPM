import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re
from astroddpm.datahandler import dataset
import astroddpm.diffusion.dm as diffmodels
from astroddpm.runners import Diffuser, config_from_id
from astroddpm.analysis.validationMetrics import powerSpectrum, minkowskiFunctional
from astroddpm.diffusion.dm import DiscreteSBM
from astroddpm.diffusion.stochastic.sde import DiscreteVPSDE, DiscreteSigmaVPSDE
from astroddpm.diffusion.models.network import ResUNet


amin,amax=(-6.232629, 7.390278)
def plot_epoch():
    return 

def check_nearest_epoch(diffuser, epoch, max_to_plot = 8, amin = -6 ,amax = 6, save_file = None):
    pattern = r'epoch_(\d+)_'
    l = os.listdir(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"]))
    l_epochs = []
    for file in l:
        match = re.search(pattern, file)
        if match:
            l_epochs.append(int(match.group(1)))
    l_epochs = [*set(l_epochs)]
    l_epochs.sort()
    epoch_found = None
    for running_epoch in l_epochs:
        if running_epoch >= epoch:
            epoch_found = running_epoch
            break
    if epoch_found is None:
        epoch_found = l_epochs[-1]
    print('Nearest epoch found is '+str(epoch_found))
    ## Extract samples from this epoch
    l_samples = []
    for file in l:
        match = re.search(pattern, file)
        if match:
            file_epoch = int(match.group(1))
            if file_epoch == epoch_found:
                l_samples.append(file)
    ## Collect all the samples from this epoch
    all_img = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples]
    
    ## Plot the first max_to_plot samples from l_samples, using plot_and_save_line
    if len(all_img) > max_to_plot:
        all_img = all_img[:max_to_plot]
    label_samp = diffuser.config["model_id"]
    title = 'Epoch '+str(epoch_found)+' for '+label_samp
    plot_and_save_line(all_img, amin = amin, amax = amax ,save_file = save_file,elementary_figsize = (10, 10), rgb = False, title = title)

def check_training_samples(diffuser, max_to_plot = 8, amin = -6 ,amax = 6, save_file = None, elementary_figsize = (10, 10), rgb = False):
    ''' Get the list of all epochs then plot the results at the first epoch sampled, 10%, 50% and 100% of the training '''
    pattern = r'epoch_(\d+)_'
    l = os.listdir(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"]))
    l_epochs = []
    for file in l:
        match = re.search(pattern, file)
        if match:
            l_epochs.append(int(match.group(1)))
    l_epochs = [*set(l_epochs)]
    l_epochs.sort()
    epochs = diffuser.config["epochs"]
    ## Extract samples from the corresponding epochs and "results_*.npy" for 100% of the training
    l_samples_0 = []
    l_samples_10 = []
    l_samples_50 = []
    l_samples_100 = []
    for file in l:
        match = re.search(pattern, file)
        if match:
            file_epoch = int(match.group(1))
            if file_epoch == 0:
                l_samples_0.append(file)
            elif file_epoch == epochs//10:
                l_samples_10.append(file)
            elif file_epoch == epochs//2:
                l_samples_50.append(file)
    pattern_results = r'results_(\d+)'
    for file in l:
        match = re.search(pattern_results, file)
        if match:
            l_samples_100.append(file)
    l_samples_0 = l_samples_0[:max_to_plot]
    l_samples_10 = l_samples_10[:max_to_plot]
    l_samples_50 = l_samples_50[:max_to_plot]
    l_samples_100 = l_samples_100[:max_to_plot]
    ## Collect all the samples from these epochs
    all_img_0 = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples_0]
    all_img_10 = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples_10]
    all_img_50 = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples_50]
    all_img_100 = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples_100]
    ## Plot the first max_to_plot samples from l_samples, using plot_and_save_lines

    label_samp = diffuser.config["model_id"]
    title = 'Samples as training progresses for '+label_samp
    plot_and_save_lines([all_img_0,all_img_10,all_img_50,all_img_100], amin = amin, amax = amax ,save_file = save_file,elementary_figsize = elementary_figsize, rgb = rgb, label_list = ['0%','10%','50%','100%'], title = title)


def plot_and_save_line(samples, amin = -6, amax = 6 ,save_file = None,elementary_figsize = (48, 20), rgb = False, title = None):
    n_samples = len(samples)
    assert n_samples<=8 ,'Too many samples'
    W, H = elementary_figsize
    figsize = (W*n_samples,H)
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    axs = fig.subplots(nrows=1, ncols=n_samples)
    ndim = len(samples[0].shape)
    if ndim == 2:
        for col, ax in enumerate(axs):
            im=ax.imshow(samples[col],vmin=amin,vmax=amax)
            ax.axis('off')
        fig.suptitle(title,fontsize=60)
        if not(save_file is None):
            plt.savefig(save_file)
    elif ndim == 3:
        if not rgb:
            for col, ax in enumerate(axs):
                im=ax.imshow(samples[col].sum(axis=0),vmin=amin,vmax=amax)
                ax.axis('off')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
        else:
            for col, ax in enumerate(axs):
                im=ax.imshow(np.moveaxis(samples[col],0,-1),vmin=amin,vmax=amax)
                ax.axis('off')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
    elif ndim >= 3:
        print('Summing over the first dimensions')
        if not rgb:
            for col, ax in enumerate(axs):
                im=ax.imshow(samples[col].sum(axis=tuple(range(0,ndim-2))),vmin=amin,vmax=amax)
                ax.axis('off')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
        else:
            for col, ax in enumerate(axs):
                im=ax.imshow(np.moveaxis(samples[col].sum(axis=tuple(range(0,ndim-3))),0,-1),vmin=amin,vmax=amax)
                ax.axis('off')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
    plt.show()

def plot_and_save_lines(samples_list, amin = -6, amax = 6 ,save_file = None,elementary_figsize = (10, 10), rgb = False, label_list = None, title = None):
    '''Does the same as plot_and_save_line but for a list of list (or array) of samples.
    Only plots the max number of samples so that the result is rectangular'''
    list_n = [len(samples) for samples in samples_list]
    n_list = len(samples_list)
    n_samples = min(list_n)
    assert n_samples<=8 ,'Too many samples'
    W, H = elementary_figsize
    figsize = (W*n_samples+2,H*n_list+2)
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    subfigs = fig.subfigures(nrows=n_list, ncols=1)
    ndim = len(samples_list[0][0].shape)
    if ndim == 2:
        for row, samples in enumerate(samples_list):
            axs = subfigs[row].subplots(nrows=1, ncols=n_samples)
            for col, ax in enumerate(axs):
                im=ax.imshow(samples[col],vmin=amin,vmax=amax)
                ax.axis('off')
            if label_list is not None:
                try:
                    subfigs[row].suptitle(label_list[row], fontsize = 40)
                except:
                    print('Failure to assign label')
        fig.suptitle(title,fontsize=60)
        if not(save_file is None):
            plt.savefig(save_file)
    elif ndim == 3:
        if not rgb:
            for row, samples in enumerate(samples_list):
                axs = subfigs[row].subplots(nrows=1, ncols=n_samples)
                for col, ax in enumerate(axs):
                    im=ax.imshow(samples[col].sum(axis=0),vmin=amin,vmax=amax)
                    ax.axis('off')
                if label_list is not None:
                    try:
                        subfigs[row].suptitle(label_list[row], fontsize = 40)
                    except:
                        print('Failure to assign label')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
        else:
            for row, samples in enumerate(samples_list):
                axs = subfigs[row].subplots(nrows=1, ncols=n_samples)
                for col, ax in enumerate(axs):
                    im=ax.imshow(np.moveaxis(samples[col],0,-1),vmin=amin,vmax=amax)
                    ax.axis('off')
                if label_list is not None:
                    try:
                        subfigs[row].suptitle(label_list[row], fontsize = 40)
                    except:
                        print('Failure to assign label')

            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
    elif ndim >= 3:
        print('Summing over the first dimensions')
        if not rgb:
            for row, samples in enumerate(samples_list):
                axs = subfigs[row].subplots(nrows=1, ncols=n_samples)
                for col, ax in enumerate(axs):
                    im=ax.imshow(samples[col].sum(axis=tuple(range(0,ndim-2))),vmin=amin,vmax=amax)
                    ax.axis('off')
                if label_list is not None:
                    try:
                        subfigs[row].suptitle(label_list[row], fontsize = 40, ha='right', va='center', x=0.0, y=0.5, rotation=90)
                    except:
                        print('Failure to assign label')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
        else:
            for row, samples in enumerate(samples_list):
                axs = subfigs[row].subplots(nrows=1, ncols=n_samples)
                for col, ax in enumerate(axs):
                    im=ax.imshow(np.moveaxis(samples[col].sum(axis=tuple(range(0,ndim-3))),0,-1),vmin=amin,vmax=amax)
                    ax.axis('off')
                if label_list is not None:
                    try:
                        subfigs[row].suptitle(label_list[row], fontsize = 40,)
                    except:
                        print('Failure to assign label')
            fig.suptitle(title,fontsize=60)
            if not(save_file is None):
                plt.savefig(save_file)
    plt.show()

        
def plot_losses(diffuser, save_file = None, burnin = 0, linewidth = 0.5, running_mean = 0, figsize = (10,10)):
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    title = 'Losses for '+diffuser.config["model_id"]
    fig.suptitle(title,fontsize=15)
    axs = fig.subplots(nrows=1, ncols=1)
    len_ratio = int(len(diffuser.losses)/len(diffuser.test_losses))
    avg_losses = np.mean(np.array(diffuser.losses).reshape(-1, len_ratio), axis=1)
    def compute_running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    if running_mean ==0:
        axs.plot(avg_losses[burnin:],label='Train loss', linewidth = linewidth)
        axs.plot(diffuser.test_losses[burnin:],label='Test loss', linewidth = linewidth)
        axs.legend()
    else:
        axs.plot(compute_running_mean(avg_losses[burnin:],running_mean),label='Train loss', linewidth = linewidth)
        axs.plot(compute_running_mean(diffuser.test_losses[burnin:],running_mean),label='Test loss', linewidth = linewidth)
        axs.legend()
    
    if not(save_file is None):
        plt.savefig(save_file)
    plt.show()


def plot_comparaison(diffuser, max_to_plot = 8, save_file = None, amin = -6, amax = 6, elementary_figsize = (10,10), rgb = False, transforms = None):
    ## Get max_to_plot samples from results_*
    pattern = r'results_(\d+)'
    l = os.listdir(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"]))
    l_samples = []
    for file in l:
        match = re.search(pattern, file)
        if match:
            l_samples.append(file)
    if len(l_samples) > max_to_plot:
        l_samples = l_samples[:max_to_plot]
    ## Collect all the samples from results_*
    all_img = [np.load(os.path.join(diffuser.config["sample_dir"],diffuser.config["model_id"],sample),allow_pickle=True) for sample in l_samples]
    if len(all_img[0].shape) == 2:
        all_img = np.concatenate([img[np.newaxis,np.newaxis,:,:] for img in all_img], axis = 0)
    elif len(all_img[0].shape) == 3:
        all_img = np.concatenate([img[np.newaxis,:,:,:] for img in all_img], axis = 0)
    elif len(all_img[0].shape) >= 4:
        all_img = np.concatenate([img for img in all_img], axis = 0)
    ## Get max_to_plot samples from the dataset
    dataloader = diffuser.train_dataloader
    batch = next(iter(dataloader))
    if len(batch) < max_to_plot:
        max_to_plot = len(batch)
    dataset = batch[:max_to_plot]
    if len(dataset.shape)< len(all_img[0].shape)+1:
        dataset = dataset.unsqueeze(1)
    if transforms is not None:
        dataset = transforms(dataset.numpy())
        all_img = transforms(all_img)
    ## Plot the images using plot_and_save_lines
    label_samp = diffuser.config["model_id"]
    try:
        label_dataset = diffuser.config["dataloaders"]["dataset"]["name"]
    except:
        label_dataset = "Dataset"
    plot_and_save_lines([dataset,all_img], amin = amin, amax = amax ,save_file = save_file,elementary_figsize = elementary_figsize, rgb = rgb, label_list = [label_dataset,label_samp])
        



def plot_comparaison_1channel(dataset, start, samples, amin, amax, save_file = None):
    fig = plt.figure(constrained_layout=True,figsize=(48,20))
    fig.suptitle('Data points vs generated examples comparison',fontsize=80)
    # create 2x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)

    assert len(dataset[0].shape)==2 , 'If there is a channel dim in the output of the dataset you should use 2 channel with the argument reduce (it sums over this dim)'## H and W no channel dim 

    subfigs[0].suptitle('Data points',fontsize=60)
    axs1 = subfigs[0].subplots(nrows=1, ncols=5)
    for col, ax in enumerate(axs1):
        ax.imshow(dataset[start+col],vmin=amin,vmax=amax)

    subfigs[1].suptitle('Generated examples',fontsize=60)
    axs2 = subfigs[1].subplots(nrows=1, ncols=5)
    for col, ax in enumerate(axs2):
        im=ax.imshow(samples[start+col],vmin=amin,vmax=amax)

    cb_ax = fig.add_axes([1, 0.25, 0.01, 0.5])
    cbar = fig.colorbar(im, cax=cb_ax)
    if not(save_file is None):
        plt.savefig(save_file)
    return fig

def plot_comparaison_multi_channel(dataset, start, samples, reduced=False, save_file = None, amin = amin, amax = amax):
    fig = plt.figure(constrained_layout=True,figsize=(48,20))
    fig.suptitle('Data points vs generated examples comparison',fontsize=80)
    # create 2x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)

    if not(reduced):
        subfigs[0].suptitle('Data points',fontsize=60)
        axs1 = subfigs[0].subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs1):
            ax.imshow(torch.cat(((dataset[start+col]+6)/12,torch.zeros(1,256,256))).permute(1,2,0),norm='linear')

        subfigs[1].suptitle('Generated examples',fontsize=60)
        axs2 = subfigs[1].subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs2):
            im=ax.imshow(np.moveaxis(np.concatenate([(samples[start+col]+6)/12,np.zeros((1,256,256))]),0,-1),norm='linear')
        if not(save_file is None):
            plt.savefig(save_file)
    else:
        subfigs[0].suptitle('Data points',fontsize=60)
        axs1 = subfigs[0].subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs1):
            ax.imshow(torch.sum(dataset[start+col], dim = 0),vmin=amin,vmax=amax)

        subfigs[1].suptitle('Generated examples',fontsize=60)
        axs2 = subfigs[1].subplots(nrows=1, ncols=5)
        for col, ax in enumerate(axs2):
            im=ax.imshow(np.sum(samples[start+col], axis = 0),vmin=amin,vmax=amax)

        cb_ax = fig.add_axes([1, 0.25, 0.01, 0.5])
        cbar = fig.colorbar(im, cax=cb_ax)
        if not(save_file is None):
            plt.savefig(save_file)
    return fig