import numpy as np
import scipy.stats as stats
import scipy.fftpack


def fftind(size):
    """ Returns a np array of shifted Fourier coordinates k_x k_y.     
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, np array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
        """
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )


def compute_autocovariance(data):
    """
    Compute the autocovariance matrix of the input data.
    
    Works for any dimension.

    Parameters
    ----------
    data : array
        Input data.

    Returns
    -------
    array
        Autocovariance matrix.

    """
    return np.real(np.fft.ifftn(np.absolute(np.fft.fftn(data - data.mean())) ** 2) / np.prod(data.shape))



def gaussian_random_field(alpha = 3.0,
                          size = 256, 
                          flag_normalize = True):
    """ Returns a np array of shifted Fourier coordinates k_x k_y. 
        Input args:
            alpha (double, default = 3.0): 
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0
        Returns:
            gfield (np array of shape (size, size)):
                The random gaussian random field      
        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = np.random.normal(size = (size, size)) \
        + 1j * np.random.normal(size = (size, size))
    
        # To real space
    gfield = np.fft.ifft2(noise * amplitude).imag
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield


def generate_grf(cov, mean=0.0):
    """
    Generate a realization of a 2D real-valued GRF with periodic boundary conditions
    for a given mean and autocovariance matrix.

    Parameters
    ----------
    cov : array
        2D autocovariance matrix.
    mean : float, optional
        Mean value of the GRF. The default is 0.0.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    array
        Realization of the corresponding GRF.

    """
    if cov.ndim != 2:
        raise Exception("This function expects cov to be be a 2D array.")
    M, N = cov.shape
    gamma = np.fft.fft2(cov)
    Z = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    X = np.fft.fft2(np.multiply(np.sqrt(gamma), Z / np.sqrt(M * N))).real
    return X + mean