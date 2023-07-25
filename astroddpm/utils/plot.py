import numpy as np
import torch
import matplotlib.pyplot as plt
import os

amin,amax=(-6.232629, 7.390278)
def plot_epoch():
    return 

def check_epoch_nearest_sample(epoch_number, SAMPLE_STEP_EPOCH, SAMPLE_FOLDER, MODEL_ID, l_samples, amin, amax, num_samples_per_sampling_epoch = 8, num_plot = 8):
    epoch_to_check=int(epoch_number/SAMPLE_STEP_EPOCH)
    nsamp = num_samples_per_sampling_epoch
    try:
        print('Here are the results for epoch '+str(epoch_to_check*SAMPLE_STEP_EPOCH))
        fig, ax =plt.subplots(1,num_plot,figsize=(20, 8))

        for i in range(num_plot):
            ax[i].imshow(np.sum(np.load(os.path.join(SAMPLE_FOLDER,MODEL_ID,l_samples[epoch_to_check*nsamp+i]),allow_pickle=True),axis=0),vmin=amin,vmax=amax) ##axis sum because of shape 1,
    except:
        print('Error when fetching the samples from the nearest epoch in the sample list. Your epoch is probably closest from the end so you should check results')


def plot_and_save_line(samples, amin, amax ,save_file = None,figsize = (48, 20)):
    
    n_samples = len(samples)
    assert n_samples<6 ,'Too many samples'
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    axs = fig.subplots(nrows=1, ncols=n_samples)

    for col, ax in enumerate(axs):
        im=ax.imshow(samples[col],vmin=amin,vmax=amax)
        ax.axis('off')
    if not(save_file is None):
        plt.savefig(save_file)


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