from quantimpy import morphology as mp
from quantimpy import minkowski as mk
import numpy as np
import matplotlib.pyplot as plt

##########################################################################################
## Minkowski functionals: Original code by Mudur, Nayantara and Finkbeiner, Douglas P 
## Authors of "Can denoising diffusion probabilistic models generate realistic astrophysical fields?""
##########################################################################################

def plot_mink_functionals(samplist, gs_vals, names, cols, savefig_dict={}):
    """
    Plot the Minkowski functionals for a list of samples
    Args:
        samplist: list of array of samples, each sample is a numpy array of shape (N, N, 1)
        gs_vals: list of g values
        names: list of names for the samples
        cols: list of colors for the samples
        savefig_dict: dict, contains the keys save_path and dpi
    Returns:
        sampwise_minkmean: list of numpy arrays of shape (len(gs_vals), 1), mean of the Minkowski functionals for each sample
        sampwise_minkstd: list of numpy arrays of shape (len(gs_vals), 1), standard deviation of the Minkowski functionals for each sample
    """
    ## TODO check the docstring
    sampwise_minkmean  = []
    sampwise_minkstd = []
    for samp in samplist:
        samp_minks = []
        for isa in range(len(samp)):#each image
            image = samp[isa]
            gs_masks = [image>=gs_vals[ig] for ig in range(len(gs_vals))]
            minkowski = []
            for i in range(len(gs_masks)):
                minkowski.append(mk.functionals(gs_masks[i], norm=True))
            minkowski = np.vstack(minkowski) #N_alphax3
            samp_minks.append(minkowski)
        samp_minks = np.stack(samp_minks) #NsampxN_alphax3
        sampwise_minkmean.append(samp_minks.mean(0))
        sampwise_minkstd.append(np.std(samp_minks, axis=0, ddof=1))
    
    fig, ax = plt.subplots(figsize=(10, 15), nrows=3)
    for iax in range(3):
        for isa in range(len(samplist)):
            style='solid' if isa==0 else 'dashed'
            ax[iax].plot(gs_vals, sampwise_minkmean[isa][:, iax], cols[isa], label=names[isa], linestyle=style)
            ax[iax].fill_between(gs_vals, sampwise_minkmean[isa][:, iax]-sampwise_minkstd[isa][:, iax], 
                    sampwise_minkmean[isa][:, iax]+sampwise_minkstd[isa][:, iax], color=cols[isa], alpha=0.2)
        ax[iax].set_xlabel('g')
        if iax==0:
            ax[iax].set_ylabel(r'$\mathcal{M}_{0}(g)$', fontsize=10)
        elif iax==1:
            ax[iax].set_ylabel(r'$\mathcal{M}_{1}(g)$', fontsize=10)
        else:
            ax[iax].set_ylabel(r'$\mathcal{M}_{2}(g)$', fontsize=10)
        if iax==0:
            ax[iax].legend(prop={'size': 20})
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_minkmean, sampwise_minkstd