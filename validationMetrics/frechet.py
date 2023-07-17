import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import os

from .vae import VAE, CKPT_FOLDER, MODEL_ID


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=VAE(image_channels=1).to(device)
ckpt = torch.load(os.path.join(CKPT_FOLDER, MODEL_ID, 'ckpt.pt'))
model.load_state_dict(ckpt['vae'])

encoder=model.encoder


DIM = 2048

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Run 50 itrs of newton-schulz to get the matrix sqrt of
    # sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50)
    if torch.any(torch.isnan(covmean)):
        return float('nan')
    covmean = covmean.squeeze()
    out = (diff.dot(diff) +
            torch.trace(sigma1) +
            torch.trace(sigma2) -
            2 * torch.trace(covmean)).cpu().item()
    return out


def get_statistics(images, num_images=None, batch_size=50, 
                   verbose=False, parallel=False):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None:
        try:
            num_images = len(images)
        except:
            raise ValueError(
                "when `images` is not a list like object (e.g. generator), "
                "`num_images` should be given")



    fid_acts = torch.empty((num_images, 512)).to(device)

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = encoder(batch_images)
            fid_acts[start: end] = pred[0].view(-1, 512)
        start = end

    m1 = torch.mean(fid_acts, axis=0)
    s1 = torch_cov(fid_acts, rowvar=False)
 
    return m1, s1


def get_fid_score(images, num_images=None, batch_size=50,
                   verbose=False, parallel=False,stats_cache=None,reference_images=None):
    if (reference_images is None)and (stats_cache is None):
        raise ValueError('No reference provided, fill out either stats_cache or reference_images')
    m1, s1 = get_statistics(
        images, num_images, batch_size, verbose, parallel)
    if not(stats_cache is None):
        f = np.load(stats_cache)
        m2, s2 = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        m2,s2= get_statistics(reference_images,None,batch_size,verbose,parallel)
    m2 = torch.tensor(m2).to(m1.dtype)
    s2 = torch.tensor(s2).to(s1.dtype)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    
    return fid_value