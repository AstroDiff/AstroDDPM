import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from astroddpm.runners import Diffuser
from astroddpm.runners import get_samples

## Plotting
#####################################################


def plot_ps2d(bins, ps_list, labels=None, show=True, save_name=None,title=None, elementary_figsize = (5, 5), linewidth = 1, ): ## TODO implement linewidth, selective labeling and selective width
    bins_centers = (bins[:-1] + bins[1:])/2

    shape = ps_list[0].shape
    W, H = elementary_figsize
    if len(shape) == 1:
        fig, ax = plt.subplots(1, 1, figsize = elementary_figsize)
        for idx, ps in enumerate(ps_list):
            ax.plot(bins_centers.cpu(), ps[:-1].cpu(), label=labels[idx] if labels is not None else None)
        if labels is not None:
            ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        if not(title is None):
            plt.title(title)
        if save_name is not None:
            fig.savefig(save_name, facecolor='white', transparent=False)
        if show:
            fig.show(warn=False)
        else:
            plt.close(fig)
        return fig, ax
    elif len(shape) == 2:
        raise ValueError('power spectra should be 1D or 3D: 1D if only one channel, 3D if multiple channels because of the C x C cross spectrum')
    elif len(shape) == 3:
        fig, ax = plt.subplots(shape[1], shape[1], figsize=(shape[1]*W, shape[1]*H), layout='constrained',sharex=True)
        for idx, ps in enumerate(ps_list):
            for i in range(shape[1]):
                for j in range(shape[1]):
                    if i < j:
                        ax[i,j].axis('off')
                    elif i == j:
                        ax[i][j].plot(bins_centers.cpu(), ps[i,j,:-1].cpu(), label=labels[idx] if labels is not None else None)
                        ax[i][j].set_xscale('log')
                        ax[i][j].set_yscale('log')
                    else:
                        ps_to_plot = ps[i,j,:-1]/torch.sqrt(ps[i,i,:-1]*ps[j,j,:-1])
                        ax[i][j].plot(bins_centers.cpu(), ps_to_plot.cpu(), label=labels[idx] if labels is not None else None)
                        ax[i][j].set_xscale('log')

                    if labels is not None:
                        ax[i][j].legend()
        if not(title is None):
            fig.suptitle(title)
        if save_name is not None:
            fig.savefig(save_name, facecolor='white', transparent=False)
        if show:
            fig.show(warn=False)
        else:
            plt.close(fig)
        return fig, ax

def plot_set_power_spectrum_iso2d(data, bins = torch.linspace(0, np.pi, 100), title = None, show = True, save_name = None, elementary_figsize = (5, 5), use_gpu = True):
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    mean_, std_ , bins = set_power_spectrum_iso2d(data.to(temp_device), bins, use_gpu = use_gpu)
    bins_centers = (bins[:-1] + bins[1:])/2
    bins_centers = bins_centers.cpu()
    W, H = elementary_figsize
    if mean_.ndim == 1:
        fig, ax = plt.subplots(1, 1)
        mean = mean_[:-1].cpu()
        std = std_[:-1].cpu()
        ax.plot(bins_centers,mean,'-k')
        ax.fill_between(bins_centers,mean+std,mean-std)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if save_name is not None:
            fig.savefig(save_name)
    elif mean_.ndim == 3:
        shape = mean_.shape
        fig, ax = plt.subplots(shape[1], shape[1], figsize=(shape[1]*W, shape[1]*H), layout='constrained',sharex=True)
        for i in range(shape[1]):
            for j in range(shape[1]):
                if i < j:
                    ax[i,j].axis('off')
                elif i == j:
                    mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                    ax[i][j].plot(bins_centers, mean, '-k')
                    ax[i][j].fill_between(bins_centers, mean+std, mean-std)
                    ax[i][j].set_xscale('log')
                    ax[i][j].set_yscale('log')
                else:
                    normalization = torch.sqrt(mean_[i,i,:-1]*mean_[j,j,:-1]).cpu()
                    mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                    ax[i][j].plot(bins_centers, mean/normalization, '-k')
                    ax[i][j].fill_between(bins_centers, (mean+std)/normalization, (mean-std)/normalization)
                    ax[i][j].set_xscale('log')         

    else:
        raise ValueError('power spectra should be 1D or 3D: 1D if only one channel, 3D if multiple channels because of the C x C cross spectrum')
    
    if not(title is None):
        fig.suptitle(title)
    if show:
        fig.show(warn=False)
    else:
        plt.close(fig)
    return mean_, std_, bins


def plot_sets_power_spectrum_iso2d(data_list, bins = torch.linspace(0, np.pi, 100), max_width = 3, labels = None, elementary_figsize = (5, 5), save_name = None, show = True, title= None, use_gpu = True):
    ## Computations
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    bins_centers = (bins[:-1] + bins[1:])/2
    bins_centers = bins_centers.cpu()
    mean_list = []
    std_list = []
    for data in data_list:
        mean_, std_ , bins = set_power_spectrum_iso2d(data.to(temp_device), bins, use_gpu = use_gpu)
        mean_list.append(mean_)
        std_list.append(std_)
    
    ndim = mean_list[0].ndim
    ## Figure
    n = len(data_list)
    h , w = (n+max_width-1)//max_width, min(n, max_width)

    if n <= max_width:
        if n == 0:
            raise ValueError('No power spectrum to plot')
        elif n == 1:
            title = labels if labels is not None else None
            fig, mean_, std, bins = plot_set_power_spectrum_iso2d(data_list[0], bins = bins, label = title, show = True, elementary_figsize = elementary_figsize)

        else:
            fig = plt.figure(constrained_layout=True, figsize = (w*elementary_figsize[0] , h*elementary_figsize[1]))
            subfigs = fig.subfigures(nrows = h, ncols = w)
            for idx, mean_, std_ in (zip(range(n), mean_list, std_list)):
                if ndim == 1:
                    ax = subfigs[idx].subplots(1, 1)
                    mean = mean_[:-1].cpu()
                    std = std_[:-1].cpu()
                    ax.plot(bins_centers,mean,'-k')
                    ax.fill_between(bins_centers,mean+std,mean-std)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                elif ndim == 3:
                    axs = subfigs[idx].subplots(mean_.shape[1], mean_.shape[1], sharex = True)
                    shape = mean_.shape
                    for i in range(shape[1]):
                        for j in range(shape[1]):
                            if i < j:
                                axs[i,j].axis('off')
                            elif i == j:
                                mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                                axs[i][j].plot(bins_centers, mean, '-k')
                                axs[i][j].fill_between(bins_centers, mean+std, mean-std)
                                axs[i][j].set_xscale('log')
                                axs[i][j].set_yscale('log')
                            else:
                                normalization = torch.sqrt(mean_[i,i,:-1]*mean_[j,j,:-1]).cpu()
                                mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                                axs[i][j].plot(bins_centers, mean/normalization, '-k')
                                axs[i][j].fill_between(bins_centers, (mean+std)/normalization, (mean-std)/normalization)
                                axs[i][j].set_xscale('log')
                else:
                    raise ValueError('power spectra should be 1D or 3D: 1D if only one channel, 3D if multiple channels because of the C x C cross spectrum')
            for i in range(n, h*w):
                subfigs[i].axis('off')
    else:
        fig = plt.figure(constrained_layout=True, figsize = (w*elementary_figsize[0] , h*elementary_figsize[1]))
        subfigs = fig.subfigures(nrows = h, ncols = w)
        fig.supylabel('Isotropic Power')
        fig.supxlabel('Wavenumber')
        for idx, mean_, std_ in (zip(range(n), mean_list, std_list)):
            if ndim == 1:
                ax = subfigs[idx//h][idx%h].subplots(1, 1)
                mean = mean_[:-1].cpu()
                std = std_[:-1].cpu()
                ax.plot(bins_centers,mean,'-k')
                ax.fill_between(bins_centers,mean+std,mean-std)
                ax.set_xscale('log')
                ax.set_yscale('log')
            elif ndim == 3:
                axs = subfigs[idx//h][idx%h].subplots(mean_.shape[1], mean_.shape[1],sharex=True)
                shape = mean_.shape
                for i in range(shape[1]):
                    for j in range(shape[1]):
                        if i < j:
                            axs[i,j].axis('off')
                        elif i == j:
                            mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                            axs[i][j].plot(bins_centers, mean, '-k')
                            axs[i][j].fill_between(bins_centers, mean+std, mean-std)
                            axs[i][j].set_xscale('log')
                            axs[i][j].set_yscale('log')
                        else:
                            normalization = torch.sqrt(mean_[i,i,:-1]*mean_[j,j,:-1]).cpu()
                            mean, std = mean_[i,j,:-1].cpu(), std_[i,j,:-1].cpu()
                            axs[i][j].plot(bins_centers, mean/normalization, '-k')
                            axs[i][j].fill_between(bins_centers, (mean+std)/normalization, (mean-std)/normalization)
                            axs[i][j].set_xscale('log')
            else:
                raise ValueError('power spectra should be 1D or 3D: 1D if only one channel, 3D if multiple channels because of the C x C cross spectrum')
    if not(title is None):
        fig.suptitle(title)
    if save_name is not None:
        fig.savefig(save_name, facecolor='white', transparent=False)
    if show:
        fig.show(warn=False)
    else:
        plt.close(fig)

    _, _ = plot_ps2d(bins, mean_list, labels=labels, show=True)
 
    return mean_list, std_list, bins_centers, labels

def plot_ps_samples_dataset(diffuser, samples = None, title = None, max_num_samples = 128, save_name = None, bins = torch.linspace(0, np.pi, 100), use_gpu = True):
    temp_device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if samples is None:
        ## Get results from the sample_dir corresponding to the diffuser
        samples = torch.from_numpy(get_samples(diffuser))
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(1).to(temp_device)
    label_samp = diffuser.config["model_id"]
    try:
        label_dataset = diffuser.config["dataloaders"]["dataset"]["name"]
    except:
        label_dataset = "Dataset"
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    shape = dataset[0].shape
    if len(shape) == 3:
        datapoints = torch.cat([dataset[i].reshape(1,-1,shape[1], shape[2]) for i in range(min(len(dataset), max_num_samples))]).to(temp_device)
    elif len(shape) == 2:
        datapoints = torch.cat([dataset[i].reshape(1,1,shape[0], shape[1]) for i in range(min(len(dataset), max_num_samples))]).to(temp_device)
    else:
        datapoints = torch.cat([dataset[i] for i in range(min(len(dataset), max_num_samples))]).to(temp_device)
    
    ## Compute power spectrum
    if not(isinstance(bins, torch.Tensor)):
        bins = torch.linspace(0, np.pi, 100)
    bins = bins.to(temp_device)

    return plot_sets_power_spectrum_iso2d([samples, datapoints], bins = bins, labels = [label_samp, label_dataset], use_gpu=use_gpu, title = title, save_name=save_name)


def compare_separation_power_spectrum_iso(baseline, samples, noisy, bins = torch.linspace(0, np.pi, 100), title = None, only_trajectories= True, max_width = 1, relative_error = True, elementary_figsize = (5, 5), save_name = None, show = True, use_gpu = True):
    """ All tensor should be given as torch tensor of shape bs, ch, h, w, even for singular images""" ##TODO simplify with plot_ps i think. 
    temp_device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    #raise NotImplementedError("This function is not working properly")
    _, b0, _ = power_spectrum_iso2d(baseline,bins=bins)
    _, b3, _ = power_spectrum_iso2d(noisy[:1], bins=bins)
    b0 = b0.reshape(-1)
    b3 = b3.reshape(-1)
    num_samples = len(samples)
    power_spectra, mean_, std_, _ = set_power_spectrum_iso2d(samples.to(temp_device), bins = bins.to(temp_device), only_stat=False, use_gpu=use_gpu)
    bins_centers = (bins[:-1] + bins[1:])/2
    rel_err = torch.abs((power_spectra - b0)/b0)

    bias = torch.abs((mean_ - b0)/b0)

    noisy_rel_err = torch.abs((b3 - b0)/b0)

    w, h = elementary_figsize

    if not(relative_error):
        # fig, ax = plt.subplots(1, 1,figsize=(10,10), layout='constrained')

        # fig.supylabel('Isotropic Power')
        # fig.supxlabel('Wavenumber')
        # ax.plot(bins_centers,b3[:99],'g', label = 'Noisy')
        # ax.plot(bins_centers,mean,'-k',label = 'Mean-denoised')
        # ax.plot(bins_centers,b0[:99],'r',label = 'Truth')
        # idx = np.random.randint(len(power_spectra))
        # for i, ps in enumerate(power_spectra):
        #     if i==idx:
        #         linewidth = max_width 
        #         ax.plot(bins_centers, ps[:99],'b',linewidth=linewidth)
        #     else:
        #         linewidth = 1/np.sqrt(num_samples)
        #         ax.plot(bins_centers, ps[:99],linewidth=linewidth)

        # ax.legend()
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        # if not(title is None):
        #     ax.title.set_text(title)
        ndim = mean_.ndim
        shape = mean_.shape
        
        if ndim == 1:
            fig, ax = plt.subplots(1, 1, figsize = elementary_figsize)
            ax.plot(bins_centers.cpu(), mean_[:-1].cpu(), '-k', label = 'Mean-denoised')
            ax.plot(bins_centers.cpu(), b0[:-1].cpu(), 'r', label = 'Truth')
            ax.plot(bins_centers.cpu(), b3[:-1].cpu(), 'g', label = 'Noisy')
            idx = np.random.randint(len(power_spectra))
            for i, ps in enumerate(power_spectra):
                if i==idx:
                    linewidth = max_width 
                    ax.plot(bins_centers.cpu(), ps[:-1].cpu(), 'b', linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    ax.plot(bins_centers.cpu(), ps[:-1].cpu(), linewidth=linewidth)
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            if not(title is None):
                plt.title(title)
            if save_name is not None:
                fig.savefig(save_name, facecolor='white', transparent=False)
            if show:
                fig.show(warn=False)
            else:
                plt.close(fig)
        if ndim == 3:
            fig, axs = plt.subplots(shape[1], shape[1], figsize=(shape[1]*w, shape[1]*h), layout='constrained',sharex=True)
            idx = np.random.randint(len(power_spectra))
            for i in range(shape[1]):
                for j in range(shape[1]):
                    if i < j:
                        axs[i,j].axis('off')
                    elif i == j:
                        axs.plot(bins_centers.cpu(), mean_[i,j,:-1].cpu(), '-k', label = 'Mean-denoised')
                        axs.plot(bins_centers.cpu(), b0[i,j,:-1].cpu(), 'r', label = 'Truth')
                        axs.plot(bins_centers.cpu(), b3[i,j,:-1].cpu(), 'g', label = 'Noisy')
                        for i, ps in enumerate(power_spectra):
                            if i==idx:
                                linewidth = max_width 
                                axs.plot(bins_centers.cpu(), ps[i,j,:-1].cpu(), 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                axs.plot(bins_centers.cpu(), ps[i,j,:-1].cpu(), linewidth=linewidth)
                        axs.set_xscale('log')
                        axs.set_yscale('log')
                        axs.legend()
                    else:
                        normalization = torch.sqrt(mean_[i,i,:-1]*mean_[j,j,:-1]).cpu()
                        axs.plot(bins_centers.cpu(), mean_[i,j,:-1].cpu()/normalization, '-k', label = 'Mean-denoised')
                        axs.plot(bins_centers.cpu(), b0[i,j,:-1].cpu()/normalization, 'r', label = 'Truth')
                        axs.plot(bins_centers.cpu(), b3[i,j,:-1].cpu()/normalization, 'g', label = 'Noisy')
                        for i, ps in enumerate(power_spectra):
                            if i==idx:
                                linewidth = max_width 
                                axs.plot(bins_centers.cpu(), ps[i,j,:-1].cpu()/normalization, 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                axs.plot(bins_centers.cpu(), ps[i,j,:-1].cpu()/normalization, linewidth=linewidth)
            if not(title is None):
                axs.title.set_text(title)
            if save_name is not None:
                fig.savefig(save_name, facecolor='white', transparent=False)
            if show:
                fig.show(warn=False)
            
    else:
        # fig, ax = plt.subplots(2, 1, figsize=(10,10),height_ratios=[2,1],sharex=True)
        # fig.subplots_adjust(hspace=0)
        # fig.supxlabel('Wavenumber')
        # ax[0].plot(bins_centers,b3[:99],'g', label = 'Noisy')
        # ax[0].plot(bins_centers,mean,'-k',label = 'Mean-denoised')
        # ax[0].plot(bins_centers,b0[:99],'r',label = 'Truth')
        # idx = np.random.randint(len(power_spectra))
        # for i, ps in enumerate(power_spectra):
        #     if i==idx:
        #         linewidth = max_width 
        #         ax[0].plot(bins_centers, ps[:99],'b',linewidth=linewidth)
        #     else:
        #         linewidth = 1/np.sqrt(num_samples)
        #         ax[0].plot(bins_centers, ps[:99],linewidth=linewidth)

        # ax[0].legend()
        # ax[0].set_xscale('log')
        # ax[0].set_yscale('log')
        # ax[0].set_ylabel('Isotropic Power')

        # ax[1].plot(bins_centers,bias[:99],'-k', label = 'PS estimator relative bias') ##Relative error of the mean of PS
        # ax[1].plot(bins_centers,noisy_rel_err[:99],'-g', label = 'Noisy relative error')
        # for i, rel in enumerate(rel_err):
        #     if i==idx:
        #         linewidth = max_width 
        #         ax[1].plot(bins_centers, rel[:99],'b',linewidth=linewidth)
        #     else:
        #         linewidth = 1/np.sqrt(num_samples)
        #         ax[1].plot(bins_centers, rel[:99],linewidth=linewidth)
        # ax[1].legend()
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log') 

        # ax[1].set_ylabel('Rel. Error')
        ndim = mean_.ndim
        shape = mean_.shape
        if ndim == 1:
            fig, axs = plt.subplots(2, 1, figsize = elementary_figsize, height_ratios=[2,1],sharex=True)
            fig.subplots_adjust(hspace=0)
            fig.supxlabel('Wavenumber')
            axs[0].plot(bins_centers.cpu(), mean_[:-1].cpu(), '-k', label = 'Mean-denoised')
            axs[0].plot(bins_centers.cpu(), b0[:-1].cpu(), 'r', label = 'Truth')
            axs[0].plot(bins_centers.cpu(), b3[:-1].cpu(), 'g', label = 'Noisy')
            idx = np.random.randint(len(power_spectra))
            for i, ps in enumerate(power_spectra):
                if i==idx:
                    linewidth = max_width 
                    axs[0].plot(bins_centers.cpu(), ps[:-1].cpu(), 'b', linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    axs[0].plot(bins_centers.cpu(), ps[:-1].cpu(), linewidth=linewidth)
            axs[0].legend()
            axs[0].set_xscale('log')
            axs[0].set_yscale('log')
            axs[0].set_ylabel('Isotropic Power')

            axs[1].plot(bins_centers.cpu(), bias[:-1].cpu(), '-k', label = 'PS estimator relative bias') ##Relative error of the mean of PS
            axs[1].plot(bins_centers.cpu(), noisy_rel_err[:-1].cpu(), '-g', label = 'Noisy relative error')
            for i, rel in enumerate(rel_err):
                if i==idx:
                    linewidth = max_width 
                    axs[1].plot(bins_centers.cpu(), rel[:-1].cpu(), 'b', linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    axs[1].plot(bins_centers.cpu(), rel[:-1].cpu(), linewidth=linewidth)
            axs[1].legend()
            axs[1].set_xscale('log')
            axs[1].set_yscale('log')
            axs[1].set_ylabel('Rel. Error')

            if not(title is None):
                axs[0].title.set_text(title)
            if save_name is not None:
                fig.savefig(save_name, facecolor='white', transparent=False)
            if show:
                fig.show(warn=False)
            else:
                plt.close(fig)
        if ndim == 3:
            fig, axs = plt.subplots(2, 1, figsize=(shape[1]*w, shape[1]*h), height_ratios=[2,1], sharex=True)
            fig.subplots_adjust(hspace=0)
            fig.supxlabel('Wavenumber')

            subaxs = axs[0].subplots(shape[1], shape[1], sharex=True)
            idx = np.random.randint(len(power_spectra))
            for i in range(shape[1]):
                for j in range(shape[1]):
                    if i < j:
                        subaxs[i,j].axis('off')
                    elif i == j:
                        subaxs[i,j].plot(bins_centers.cpu(), mean_[i,j,:-1].cpu(), '-k', label = 'Mean-denoised')
                        subaxs[i,j].plot(bins_centers.cpu(), b0[i,j,:-1].cpu(), 'r', label = 'Truth')
                        subaxs[i,j].plot(bins_centers.cpu(), b3[i,j,:-1].cpu(), 'g', label = 'Noisy')
                        for ind, ps in enumerate(power_spectra):
                            if ind==idx:
                                linewidth = max_width 
                                subaxs[i,j].plot(bins_centers.cpu(), ps[i,j,:-1].cpu(), 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                subaxs[i,j].plot(bins_centers.cpu(), ps[i,j,:-1].cpu(), linewidth=linewidth)
                        subaxs[i,j].set_xscale('log')
                        subaxs[i,j].set_yscale('log')
                        subaxs[i,j].legend()
                    else:
                        normalization = torch.sqrt(mean_[i,i,:-1]*mean_[j,j,:-1]).cpu()
                        subaxs[i,j].plot(bins_centers.cpu(), mean_[i,j,:-1].cpu()/normalization, '-k', label = 'Mean-denoised')
                        subaxs[i,j].plot(bins_centers.cpu(), b0[i,j,:-1].cpu()/normalization, 'r', label = 'Truth')
                        subaxs[i,j].plot(bins_centers.cpu(), b3[i,j,:-1].cpu()/normalization, 'g', label = 'Noisy')
                        for ind, ps in enumerate(power_spectra):
                            if ind==idx:
                                linewidth = max_width 
                                subaxs[i,j].plot(bins_centers.cpu(), ps[i,j,:-1].cpu()/normalization, 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                subaxs[i,j].plot(bins_centers.cpu(), ps[i,j,:-1].cpu()/normalization, linewidth=linewidth)
                        subaxs[i,j].set_xscale('log')
                        subaxs[i,j].legend()
            axs[0].set_ylabel('Isotropic Power')

            subaxs = axs[1].subplots(shape[1], shape[1], sharex=True)
            for i in range(shape[1]):
                for j in range(shape[1]):
                    if i < j:
                        subaxs[i,j].axis('off')
                    elif i == j:
                        subaxs[i,j].plot(bins_centers.cpu(), bias[i,j,:-1].cpu(), '-k', label = 'PS estimator relative bias')
                        subaxs[i,j].plot(bins_centers.cpu(), noisy_rel_err[i,j,:-1].cpu(), '-g', label = 'Noisy relative error')
                        for ind, rel in enumerate(rel_err):
                            if ind==idx:
                                linewidth = max_width 
                                subaxs[i,j].plot(bins_centers.cpu(), rel[i,j,:-1].cpu(), 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                subaxs[i,j].plot(bins_centers.cpu(), rel[i,j,:-1].cpu(), linewidth=linewidth)
                        subaxs[i,j].set_xscale('log')
                        subaxs[i,j].set_yscale('log')
                        subaxs[i,j].legend()
                    else:
                        subaxs[i,j].plot(bins_centers.cpu(), bias[i,j,:-1].cpu(), '-k', label = 'PS estimator relative bias')
                        subaxs[i,j].plot(bins_centers.cpu(), noisy_rel_err[i,j,:-1].cpu(), '-g', label = 'Noisy relative error')
                        for ind, rel in enumerate(rel_err):
                            if ind==idx:
                                linewidth = max_width 
                                subaxs[i,j].plot(bins_centers.cpu(), rel[i,j,:-1].cpu(), 'b', linewidth=linewidth)
                            else:
                                linewidth = 1/np.sqrt(num_samples)
                                subaxs[i,j].plot(bins_centers.cpu(), rel[i,j,:-1].cpu(), linewidth=linewidth)

                        subaxs[i,j].set_xscale('log')
                        subaxs[i,j].legend()
            axs[1].set_ylabel('Rel. Error')

            if not(title is None):
                axs[0].title.set_text(title)
            if save_name is not None:
                fig.savefig(save_name, facecolor='white', transparent=False)
            if show:
                fig.show(warn=False)
            else:
                plt.close(fig)
    return b0, b3, mean_, std_, bins


#####################################################
## DIFUSER VERSION ( & torchified)
#####################################################

def fourriercov2d(data, data2=None, norm=None, use_gpu = True):
    """ Compute the power spectrum of the input data or the cross spectrum if data2 is not None. Returns a torch tensor on the device."""
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(temp_device)
        else:
            data = data.to(temp_device)
        if data2 is not None:
            if isinstance(data2, np.ndarray):
                data2 = torch.from_numpy(data2).to(temp_device)
            else:
                data2 = data2.to(temp_device)
        if data2 is None:
            result=torch.absolute(torch.fft.fft2(data, norm=norm))**2
        else:
            result=torch.conj(torch.fft.fft2(data, norm=norm))*torch.fft.fft2(data2, norm=norm)
        return result

def power_spectrum_2d(data, norm=None, use_gpu = True):
    '''Data is supposed to have shape B x C x H x W.
    If C = 1, then we compute the power spectrum of each image
    If C >=1 we compute the power spectrum of each channel & the cross spectrum for every pair of channels.
    This will lead to a tensor of shape (B x C X C x H x W).
    One should keep in mind that for i != j the cross spectrum is not symetric but it is conjugate transpose with complex values.
    '''
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if data.shape[1]==1:
        ## We only have to compute the power spectrum of each image
        return fourriercov2d(data.to(temp_device), norm=norm, use_gpu=use_gpu)
    else:
        B, C, H, W = data.shape
        dims_pair = [(i,j) for i in range(C) for j in range(i+1)]
        input_tensors1 = []
        input_tensors2 = []
        for pair in dims_pair:
            i , j = pair
            input_tensors1.append(data[:,i])
            input_tensors2.append(data[:,j])
        input_tensors1 = torch.cat(input_tensors1, dim=0).to(temp_device)
        input_tensors2 = torch.cat(input_tensors2, dim=0).to(temp_device)
        stacked = fourriercov2d(input_tensors1, data2=input_tensors2, norm=norm, use_gpu=use_gpu)
        res = torch.zeros((B, C, C, H, W), dtype=torch.complex64, device=temp_device)
        for idx, pair in enumerate(dims_pair):
            i , j = pair
            res[:,i,j] = stacked[idx*B:(idx+1)*B]
            if i != j:
                res[:,j,i] = torch.conj(stacked[idx*B:(idx+1)*B])
        return res


def _spectral_iso2d(data_sp, bins=None, sampling=1.0, return_counts=False, use_gpu = True):
    '''Given a B x C x H x W or a B x C x C x H x W tensor, compute the isotropic power spectrum of each image (only over the B dimension)'''
    input_dim = data_sp.ndim
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    if input_dim == 4:
        B, C, _, N = data_sp.shape
        assert C==1, "There was an issue with power_spectrum2d that didn't take into account the channel dimension"
        ## We only have a power spectrum for each image
        n_dim = 2
        wn = (2 * np.pi * torch.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (n_dim - 1)).to(temp_device)
        wn_iso = torch.zeros(data_sp.shape).to(temp_device)
        for i in range(n_dim):
            wn_iso += torch.moveaxis(wn, 0, i) ** 2
        wn_iso = torch.sqrt(wn_iso)
        wn_iso = wn_iso.reshape(B, C, -1) ## C = 1
        data_sp = data_sp.reshape(B, C, -1)

        if bins is None:
            bins = torch.sort(torch.unique(wn_iso))[0]
        BINS = len(bins)
        index = torch.bucketize(wn_iso, bins)
        index_mask = F.one_hot(index, BINS+1).to(temp_device) ## we will discard the first bin

        counts = torch.sum(index_mask, dim=[1, 2])
        ps_mean = torch.sum(index_mask * data_sp.unsqueeze(-1), dim=[1, 2]) / counts

        ps_std = torch.sqrt(torch.sum(index_mask * (data_sp.unsqueeze(-1) - ps_mean.reshape(B, 1, 1, BINS +1)) ** 2, dim=[1, 2]) / counts)

        ps_mean, ps_std = ps_mean[:,1:], ps_std[:,1:] ## discard the first bin
        
        if return_counts:
            return bins, ps_mean, ps_std, torch.tensor(counts)
        else:
            return bins, ps_mean, ps_std ## Bins, Shape B x len(bins) , idem
    elif input_dim == 5:
        ## We have a power spectrum for each pairs of channels
        B, C,_, _, N = data_sp.shape
        assert data_sp.shape == (B, C, C, N, N) 
        data_sp = data_sp.real ## Because we integrate over a circle on the fourrier plane, only real part is relevant (imag part cancels out, up to numerical errors)
        n_dim = 2
        wn = (2 * np.pi * torch.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (n_dim - 1)).to(temp_device)
        wn_iso = torch.zeros(data_sp.shape).to(temp_device)
        for i in range(n_dim):
            wn_iso += torch.moveaxis(wn, 0, i) ** 2
        wn_iso = torch.sqrt(wn_iso)
        wn_iso = wn_iso.reshape(B, C, C, -1)
        data_sp = data_sp.reshape(B, C, C, -1)

        if bins is None:
            bins = torch.sort(torch.unique(wn_iso))[0]
        BINS = len(bins)
        index = torch.bucketize(wn_iso, bins)
        index_mask = F.one_hot(index, BINS+1).to(temp_device) ## we will discard the first bin
        counts = torch.sum(index_mask, dim=3)
        ps_mean = torch.sum(index_mask * data_sp.unsqueeze(-1), dim=3) / counts
        temp = (data_sp.unsqueeze(-1) - ps_mean.unsqueeze(3)) ** 2
        ps_std = torch.sqrt(torch.sum(index_mask * temp, dim=3) / counts)

        ps_mean, ps_std = ps_mean[:,:,:,1:], ps_std[:,:,:,1:] ## discard the first bin

        if return_counts:
            return bins, ps_mean, ps_std, torch.tensor(counts)
        else:
            return bins, ps_mean, ps_std
    else:
        raise ValueError("The input tensor should have 4 or 5 dimensions")


def power_spectrum_iso2d(data, bins=None, sampling=1.0, norm=None, return_counts=False, use_gpu=True):
    '''Different behavior depending on the shape of data'''
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    return _spectral_iso2d(power_spectrum_2d(data.to(temp_device), norm=norm, use_gpu = use_gpu), bins=bins.to(temp_device), sampling=sampling, return_counts=return_counts, use_gpu = use_gpu) 
    ## Bins , Shape B x 1 x len(bins) or B x C x C x len(bins), idem
    
def set_power_spectrum_iso2d(data, bins = torch.linspace(0, np.pi, 100), only_stat = True, use_gpu = True):
    '''Data should have shape (N, C, H, W)'''
    temp_device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    _, power_spectra, _ = power_spectrum_iso2d(data.to(temp_device), bins=bins.to(temp_device), use_gpu=use_gpu)
    
    if power_spectra.ndim == 2:
        mean, std = torch.mean(power_spectra,axis=0).reshape(-1), torch.std(power_spectra,axis=0).reshape(-1) ## Average over dim 0 (B dimension)
    elif power_spectra.ndim == 3:
        raise ValueError('power spectra should be 1D or 3D: 1D if only one channel, 3D if multiple channels because of the C x C cross spectrum')
    elif power_spectra.ndim == 4:
        mean, std = torch.mean(power_spectra,axis=0), torch.std(power_spectra,axis=0)
    if only_stat:
        return mean, std, bins
    else:
        return power_spectra, mean, std, bins
