import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from astroddpm.runners import Diffuser
from astroddpm.analysis.validationMetrics.basics import get_samples
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Plotting
#####################################################


def plot_ps(bins, ps_list, labels=None, show=False, save_name=None,title=None):
    bins_centers = (bins[:-1] + bins[1:])/2

    fig, ax = plt.subplots(1, 1)
    for idx, ps in enumerate(ps_list):
        ax.plot(bins_centers, ps[:-1], label=labels[idx] if labels is not None else None)
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


def plot_set_power_spectrum_iso2d(data_list, bins = torch.linspace(0, np.pi, 100), max_width = 3, labels = None):
    n = len(data_list)
    h , w = (n+max_width-1)//max_width, n%max_width
    print(h, w)
    fig, ax = plt.subplots( h, w, figsize = (w*7 , h*7), layout='constrained')
    bins_centers = (bins[:-1] + bins[1:])/2
    bins_centers = bins_centers.cpu().numpy()
    mean_list = []
    if n <= max_width:
        if n == 0:
            raise ValueError('No power spectrum to plot')
        elif n == 1:
            mean, std, _ = set_power_spectrum_iso2d(data_list[0].to(device), bins)
            mean = mean[:99].cpu().numpy()
            std = std[:99].cpu().numpy()
            mean_list.append(mean)
            ax.plot(bins_centers,mean,'-k')
            ax.fill_between(bins_centers,mean+std,mean-std)

            ax.set_xscale('log')
            ax.set_yscale('log')
            if not(labels is None) and len(labels) > idx:
                ax.title.set_text(labels)

        else:
            for idx, data in enumerate(data_list):
                mean, std, _ = set_power_spectrum_iso2d(data.to(device), bins)
                mean = mean[:99].cpu().numpy()
                std = std[:99].cpu().numpy()
                mean_list.append(mean)
                ax[idx].plot(bins_centers,mean,'-k')
                ax[idx].fill_between(bins_centers,mean+std,mean-std)

                ax[idx].set_xscale('log')
                ax[idx].set_yscale('log')
                if not(labels is None) and len(labels) > idx:
                    ax[idx].title.set_text(labels[idx])
    else:
        fig.supylabel('Isotropic Power')
        fig.supxlabel('Wavenumber')
        for idx, data in enumerate(data_list):
            mean, std, _ = set_power_spectrum_iso2d(data.to(device), bins)
            mean = mean[:99].cpu().numpy()
            std = std[:99].cpu().numpy()
            mean_list.append(mean)
            ax[idx//max_width][idx%max_width].plot(bins_centers,mean,'-k')
            ax[idx//max_width][idx%max_width].fill_between(bins_centers,mean+std,mean-std)

            ax[idx//max_width][idx%max_width].set_xscale('log')
            ax[idx//max_width][idx%max_width].set_yscale('log')
            if not(labels is None) and len(labels) > idx:
                ax[idx//max_width][idx%max_width].title.set_text(labels[idx])
    plt.show()
    fig2, ax2 = plt.subplots(1, 1)
    for i, mean in enumerate(mean_list):
        if labels is None:
            ax2.plot(bins_centers,mean)
        else:
            ax2.plot(bins_centers,mean,label=labels[i])
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.legend()

        fig2.supylabel('Isotropic Power')
        fig2.supxlabel('Wavenumber')
    plt.show()
    return mean_list, bins_centers, labels

def plot_ps_samples_dataset(diffuser, samples = None, title = None, legend = True, max_num_samples = 128, savefig = None, bins = torch.linspace(0, np.pi, 100)):
    if samples is None:
        ## Get results from the sample_dir corresponding to the diffuser
        samples = torch.from_numpy(get_samples(diffuser))
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(1).to(device)
    label_samp = diffuser.config["model_id"]
    try:
        label_dataset = diffuser.config["dataloaders"]["dataset"]["name"]
    except:
        label_dataset = "Dataset"
    ## Get elements of the dataset
    dataset = diffuser.train_dataloader.dataset
    shape = dataset[0].shape
    if len(shape) == 3:
        datapoints = torch.cat([dataset[i].reshape(1,-1,shape[1], shape[2]) for i in range(min(len(dataset), max_num_samples))]).to(device)
    elif len(shape) == 2:
        datapoints = torch.cat([dataset[i].reshape(1,1,shape[0], shape[1]) for i in range(min(len(dataset), max_num_samples))]).to(device)
    else:
        datapoints = torch.cat([dataset[i] for i in range(min(len(dataset), max_num_samples))]).to(device)
    
    ## Compute power spectrum
    if not(isinstance(bins, torch.Tensor)):
        bins = torch.linspace(0, np.pi, 100)
    bins = bins.to(device)

    return plot_set_power_spectrum_iso2d([samples, datapoints], bins = bins, labels = [label_samp, label_dataset])


def compare_separation_power_spectrum_iso(baseline, samples, noisy, bins = np.linspace(0, np.pi, 100), title = None, only_trajectories= True, max_width = 2, relative_error = True):
    """ All tensor should be given as torch tensor of shape bs, ch, h, w, even for singular images"""
    raise NotImplementedError("This function is not working properly")
    _, b0, _ = power_spectrum_iso(baseline[0][0].cpu(),bins=bins)
    _, b2, _ = power_spectrum_iso(samples[0][0].detach().cpu(),bins=bins)
    _, b3, _ = power_spectrum_iso(noisy[0][0].detach().cpu(), bins=bins)
    num_samples = len(samples)
    power_spectra, mean_, std_, _ = set_power_spectrum_iso(samples[:,0].detach().cpu(), bins = bins, only_stat=False)
    bins_centers = (bins[:-1] + bins[1:])/2
    mean, std = mean_[:99], std_[:99]
    rel_err = np.abs((power_spectra - b0)/b0)
    sign_rel_err = (power_spectra - b0)/b0

    bias = np.abs((mean_ - b0)/b0)
    signed_bias = (mean_ - b0)/b0

    noisy_rel_err = np.abs((b3 - b0)/b0)

    if not(only_trajectories):
        if not(relative_error):
            fig, ax = plt.subplots(1, 1,figsize=(10,10), layout='constrained')

            fig.supylabel('Isotropic Power')
            fig.supxlabel('Wavenumber')
            
            ax.plot(bins_centers,b3[:99],'g', label = 'Noisy')
            ax.plot(bins_centers,mean,'-k',label = 'Mean-denoised')
            ax.plot(bins_centers,b0[:99],'r',label = 'Truth')
            ax.fill_between(bins_centers,mean+std,mean-std,label = 'Std-denoised')
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')

            if not(title is None):
                ax.title.set_text(title)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(15,10),height_ratios=[2,1], layout='constrained',sharex=True)
            fig.subplots_adjust(hspace=0)
            fig.supxlabel('Wavenumber')
            ax[0].plot(bins_centers,b3[:99],'g', label = 'Noisy')
            ax[0].plot(bins_centers,mean,'-k',label = 'Mean-denoised')
            ax[0].plot(bins_centers,b0[:99],'r',label = 'Truth')
            ax[0].fill_between(bins_centers,mean+std,mean-std,label = 'Std-denoised')
            ax[0].legend()
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_ylabel('Isotropic Power')

            ax[1].plot(bins_centers,bias[:99],'-k', label = 'PS estimator relative bias') ##Relative error of the mean of PS
            ax[1].plot(bins_centers,noisy_rel_err[:99],'-g', label = 'Noisy relative error')
            ax[1].plot(bins_centers,np.mean(rel_err[:99], axis = 0),'-b', label = 'Mean-denoised relative error') ## Mean of the relative error of PS
            ax[1].legend()
            ax[1].set_xscale('log')

            ax[1].set_ylabel('Rel. Error')

    else:
        if not(relative_error):
            fig, ax = plt.subplots(1, 1,figsize=(10,10), layout='constrained')

            fig.supylabel('Isotropic Power')
            fig.supxlabel('Wavenumber')
            ax.plot(bins_centers,b3[:99],'g', label = 'Noisy')
            ax.plot(bins_centers,mean,'-k',label = 'Mean-denoised')
            ax.plot(bins_centers,b0[:99],'r',label = 'Truth')
            idx = np.random.randint(len(power_spectra))
            for i, ps in enumerate(power_spectra):
                if i==idx:
                    linewidth = max_width 
                    ax.plot(bins_centers, ps[:99],'b',linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    ax.plot(bins_centers, ps[:99],linewidth=linewidth)

            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')

            if not(title is None):
                ax.title.set_text(title)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(10,10),height_ratios=[2,1],sharex=True)
            fig.subplots_adjust(hspace=0)
            fig.supxlabel('Wavenumber')
            ax[0].plot(bins_centers,b3[:99],'g', label = 'Noisy')
            ax[0].plot(bins_centers,mean,'-k',label = 'Mean-denoised')
            ax[0].plot(bins_centers,b0[:99],'r',label = 'Truth')
            idx = np.random.randint(len(power_spectra))
            for i, ps in enumerate(power_spectra):
                if i==idx:
                    linewidth = max_width 
                    ax[0].plot(bins_centers, ps[:99],'b',linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    ax[0].plot(bins_centers, ps[:99],linewidth=linewidth)

            ax[0].legend()
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_ylabel('Isotropic Power')

            ax[1].plot(bins_centers,bias[:99],'-k', label = 'PS estimator relative bias') ##Relative error of the mean of PS
            ax[1].plot(bins_centers,noisy_rel_err[:99],'-g', label = 'Noisy relative error')
            for i, rel in enumerate(rel_err):
                if i==idx:
                    linewidth = max_width 
                    ax[1].plot(bins_centers, rel[:99],'b',linewidth=linewidth)
                else:
                    linewidth = 1/np.sqrt(num_samples)
                    ax[1].plot(bins_centers, rel[:99],linewidth=linewidth)
            ax[1].legend()
            ax[1].set_xscale('log')
            ax[1].set_yscale('log') 

            ax[1].set_ylabel('Rel. Error')

    plt.show()
    if False and relative_error:
        fig, ax = plt.subplots(1, 1,figsize=(10,10), layout='constrained')

        fig.supylabel('Relative error on the Isotropic Power Spectrum')
        fig.supxlabel('Wavenumber')

        ax.plot(bins_centers,bias[:99],'-k', label = 'PS estimator relative bias')

        idx = np.random.randint(len(rel_err))
        for i, rel in enumerate(rel_err):
            if i==idx:
                linewidth = max_width 
                ax.plot(bins_centers, rel[:99],'b',linewidth=linewidth)
            else:
                linewidth = 1/np.sqrt(num_samples)
                ax.plot(bins_centers, rel[:99],linewidth=linewidth)

        ax.legend()
        ax.set_xscale('log')
    plt.show()
        
    return b0, b3, mean_, std_, bins


#####################################################
## DIFUSER VERSION ( & torchified)
#####################################################

def fourriercov2d(data, data2=None, norm=None):
    """ Compute the power spectrum of the input data or the cross spectrum if data2 is not None. Returns a torch tensor on the device."""
    with torch.no_grad():
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(device)
        else:
            data = data.to(device)
        if data2 is not None:
            if isinstance(data2, np.ndarray):
                data2 = torch.from_numpy(data2).to(device)
            else:
                data2 = data2.to(device)
        if data2 is None:
            result=torch.absolute(torch.fft.fft2(data, norm=norm))**2
        else:
            result=torch.conj(torch.fft.fft2(data, norm=norm))*torch.fft.fft2(data2, norm=norm)
        return result

def power_spectrum_2d(data, bins=None, sampling=1.0, norm=None, return_counts=False):
    '''Data is supposed to have shape B x C x H x W.
    If C = 1, then we compute the power spectrum of each image
    If C >=1 we compute the power spectrum of each channel & the cross spectrum for every pair of channels.
    This will lead to a tensor of shape (B x C X C x H x W).
    One should keep in mind that for i != j the cross spectrum is not symetric but it is conjugate transpose with complex values.
    '''
    if data.shape[1]==1:
        ## We only have to compute the power spectrum of each image
        return fourriercov2d(data, norm=norm)
    else:
        B, C, H, W = data.shape
        dims_pair = [(i,j) for i in range(C) for j in range(i+1)]
        input_tensors1 = []
        input_tensors2 = []
        for pair in dims_pair:
            i , j = pair
            input_tensors1.append(data[:,i])
            input_tensors2.append(data[:,j])
        input_tensors1 = torch.cat(input_tensors1, dim=0).to(device)
        input_tensors2 = torch.cat(input_tensors2, dim=0).to(device)
        res = torch.zeros((B, C, C, H, W), dtype=torch.complex64, device=device)
        stacked = fourriercov2d(input_tensors1, data2=input_tensors2, norm=norm).cpu()
        for idx, pair in enumerate(dims_pair):
            i , j = pair
            res[:,i,j] = stacked[idx*B:(idx+1)*B]
            if i != j:
                res[:,j,i] = torch.conj(stacked[idx*B:(idx+1)*B])
        return res


def _spectral_iso2d(data_sp, bins=None, sampling=1.0, return_counts=False):
    '''Given a B x C x H x W or a B x C x C x H x W tensor, compute the isotropic power spectrum of each image (only over the B dimension)'''
    input_dim = data_sp.ndim

    if input_dim == 4:
        B, C, _, N = data_sp.shape
        assert C==1, "There was an issue with power_spectrum2d that didn't take into account the channel dimension"
        ## We only have a power spectrum for each image
        n_dim = 2
        wn = (2 * np.pi * torch.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (n_dim - 1)).to(device)
        wn_iso = torch.zeros(data_sp.shape).to(device)
        for i in range(n_dim):
            wn_iso += torch.moveaxis(wn, 0, i) ** 2
        wn_iso = torch.sqrt(wn_iso)
        wn_iso = wn_iso.reshape(B, C, -1) ## C = 1
        data_sp = data_sp.reshape(B, C, -1)

        if bins is None:
            bins = torch.sort(torch.unique(wn_iso))[0]
        BINS = len(bins)
        index = torch.bucketize(wn_iso, bins)
        index_mask = F.one_hot(index, BINS+1).to(device) ## we will discard the first bin

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
        n_dim = 2
        wn = (2 * np.pi * torch.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (n_dim - 1))
        wn_iso = torch.zeros(data_sp.shape)
        for i in range(n_dim):
            wn_iso += torch.moveaxis(wn, 0, i) ** 2
        wn_iso = torch.sqrt(wn_iso)
        wn_iso = wn_iso.reshape(B, C, C, -1)
        data_sp = data_sp.reshape(B, C, C, -1)

        if bins is None:
            bins = torch.sort(torch.unique(wn_iso))[0]
        BINS = len(bins)
        index = torch.bucketize(wn_iso, bins)
        index_mask = F.one_hot(index, BINS+1).to(device) ## we will discard the first bin
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


def power_spectrum_iso2d(data, bins=None, sampling=1.0, norm=None, return_counts=False):
    '''Different behavior depending on the shape of data'''
    data_sp = power_spectrum_2d(data, norm=norm)
    return _spectral_iso2d(data_sp, bins=bins, sampling=sampling, return_counts=return_counts) ## Bins , Shape B x 1 x len(bins) or B x C x C x len(bins), idem

    
def set_power_spectrum_iso2d(data, bins = np.linspace(0, np.pi, 100), only_stat = True):
    '''Data should have shape (N, C, H, W)'''

    _, power_spectra, _ = power_spectrum_iso2d(data, bins=bins)
    mean, std = torch.mean(power_spectra,axis=0).reshape(-1), torch.std(power_spectra,axis=0).reshape(-1) ## Average over dim 0 (B dimension)
    if only_stat:
        return mean, std, bins
    else:
        return power_spectra, mean, std, bins
