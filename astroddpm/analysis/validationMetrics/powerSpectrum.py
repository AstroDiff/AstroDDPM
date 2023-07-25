import numpy as np
import matplotlib.pyplot as plt



def _spectral_iso(data_sp, bins=None, sampling=1.0, return_counts=False):
    """
    Internal function.
    Check power_spectrum_iso for documentation details.
    Parameters
    ----------
    data_sp : TYPE
        DESCRIPTION.
    bins : TYPE, optional
        DESCRIPTION. The default is None.
    sampling : TYPE, optional
        DESCRIPTION. The default is 1.0.
    return_coutns : bool, optional
        Return counts per bin.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        Return counts per bin return_counts=True.
    """
    N = data_sp.shape[0]
    ndim = data_sp.ndim
    # Build an array of isotropic wavenumbers making use of numpy broadcasting
    wn = (2 * np.pi * np.fft.fftfreq(N, d=sampling)).reshape((N,) + (1,) * (ndim - 1))
    wn_iso = np.zeros(data_sp.shape)
    for i in range(ndim):
        wn_iso += np.moveaxis(wn, 0, i) ** 2
    wn_iso = np.sqrt(wn_iso)
    # We do not need ND-arrays anymore
    wn_iso = wn_iso.ravel()
    data_sp = data_sp.ravel()
    # We compute associations between index and bins
    if bins is None:
        bins = np.sort(np.unique(wn_iso)) # Default binning
    index = np.digitize(wn_iso, bins) - 1
    # Stacking
    stacks = np.empty(len(bins), dtype=object)
    for i in range(len(bins)):
        stacks[i] = []
    for i in range(len(index)):
        if index[i] >= 0:
            stacks[index[i]].append(data_sp[i])
    counts = []
    # Computation for each bin of the mean power spectrum and standard deviations of the mean
    ps_mean = np.zeros(len(bins), dtype=data_sp.dtype) # Allow complex values (for cross-spectrum)
    ps_std = np.zeros(len(bins)) # If complex values, note that std first take the modulus
    for i in range(len(bins)):
        ps_mean[i] = np.mean(stacks[i])
        count = len(stacks[i])
        ps_std[i] = np.std(stacks[i]) / np.sqrt(count)
        counts.append(count)
    if return_counts:
        return bins, ps_mean, ps_std, np.array(counts)
    else:
        return bins, ps_mean, ps_std
    

def power_spectrum(data, data2=None, norm=None):
    """
    Compute the full power spectrum of input data.
    Parameters
    ----------
    data : array
        Input data.
    norm : str
        FFT normalization. Can be None or 'ortho'. The default is None.
    Returns
    -------
    None.
    """
    if data2 is None:
        result=np.absolute(np.fft.fftn(data, norm=norm))**2
    else:
        result=np.real(np.conjugate(np.fft.fftn(data, norm=norm))*np.fft.fftn(data2, norm=norm))
    return result

def power_spectrum_iso(data, data2=None, bins=None, sampling=1.0, norm=None, return_counts=False):
    """
    Compute the isotropic power spectrum of input data.
    bins parameter should be a list of bin edges defining:
    bins[0] <= bin 0 values < bins[1]
    bins[1] <= bin 1 values < bins[2]
                ...
    bins[N-2] <= bin N-2 values < bins[N-1]
    bins[N-1] <= bin N-1 values
    Note that the last bin has no superior limit.
    Parameters
    ----------
    data : array
        Input data.
    bins : array, optional
        Array of bins. If None, we use a default binning which corresponds to a full isotropic power spectrum.
        The default is None.
    sampling : float, optional
        Grid size. The default is 1.0.
    norm : TYPE, optional
        FFT normalization. Can be None or 'ortho'. The default is None.
    return_counts: bool, optional
        Return counts per bin. The default is None
    Raises
    ------
    Exception
        DESCRIPTION.
    Returns
    -------
    bins : TYPE
        DESCRIPTION.
    ps_mean : TYPE
        DESCRIPTION.
    ps_std : TYPE
        DESCRIPTION.
    counts : array, optional
        If return_counts=True, counts per bin.
    """
    # Check data shape
    for i in range(data.ndim):
        if data.shape[i] != data.shape[0]:
            raise Exception("Input data must be of shape (N, ..., N).")
    # Compute the full power spectrum of input data
    if data2 is None:
        data_ps = power_spectrum(data, norm=norm)
    else:
        data_ps = power_spectrum(data, data2=data2, norm=norm)
    return _spectral_iso(data_ps, bins=bins, sampling=sampling, return_counts=return_counts)

## Computing Statistics accross a set
#####################################################

def set_power_spectrum_iso(dataset, bins = np.linspace(0, np.pi, 100), only_stat = True):
    n = len(dataset)
    power_spectra=np.concatenate([power_spectrum_iso(dataset[i],bins=bins)[1].reshape(1,100) for i in range(n)],axis=0)
    mean, std = np.mean(power_spectra,axis=0), np.std(power_spectra,axis=0)
    if only_stat:
        return mean, std, bins
    else:
        return power_spectra, mean, std, bins


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


def plot_set_power_spectrum_iso(dataset_list, bins, max_width = 3, labels = None):
    n = len(dataset_list)
    h , w = (n+max_width-1)//max_width, n%max_width
    print(h, w)
    fig, ax = plt.subplots( h, w, figsize = (w*7 , h*7), layout='constrained')
    bins_centers = (bins[:-1] + bins[1:])/2
    mean_list = []
    if n <= max_width:
        if n == 0:
            raise ValueError('No power spectrum to plot')
        elif n == 1:
            mean, std, _ = set_power_spectrum_iso(dataset[0], bins)
            mean = mean[:99]
            std = std[:99]
            mean_list.append(mean)
            ax.plot(bins_centers,mean,'-k')
            ax.fill_between(bins_centers,mean+std,mean-std)

            ax.set_xscale('log')
            ax.set_yscale('log')
            if not(labels is None) and len(labels) > idx:
                ax.title.set_text(labels)

        else:
            for idx, dataset in enumerate(dataset_list):
                mean, std, _ = set_power_spectrum_iso(dataset, bins)
                mean = mean[:99]
                std = std[:99]
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
        for idx, dataset in enumerate(dataset_list):
            mean, std, _ = set_power_spectrum_iso(dataset, bins)
            mean = mean[:99]
            std = std[:99]
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

def compare_separation_power_spectrum_iso(baseline, samples, noisy, bins = np.linspace(0, np.pi, 100), title = None, only_trajectories= True, max_width = 2, relative_error = True):
    """ All tensor should be given as torch tensor of shape bs, ch, h, w, even for singular images"""
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
