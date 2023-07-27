# GEMOA: GEnerative MOdeling for Astrophysical fields

This repository contains all the code related to a projet to use diffusions and score based models to generate realistic astrophysical fields with matching statistics. Ultimately, such models could be used as efficient priors or, using the computed score, conditionnal generators.

<!-- Source Separations with simulation informed priors SSSIP ?-->
---

## TODO 

- [ ] decide how datasets are managed for the wider audience (@fi & outside) (and ckpts too)
- [ ] develop fine-tune
- [ ] solidify resume training
- [X] ~translate analysis tools & older separations tools~
- [ ] add a verbose = reduced option for training
- [X] ~verify how models are saved to the local MODELS.json~ My bad
- [ ] Add continuous time models? Variance exploding? -> probably modify first the DiscreteSBM loss method.
- [ ] Add likelihood computations (forward time and reverse time ODE plus log like computing)
- [ ] Multi power spectrum score based models.
- [X] ~Integrate BM3D to allow an easy comparison~ In the demo separation notebook
- [ ] Add a DDIM & an EMA option (DDRM, PIGDM...later)
- [X] ~Complete requirements.txt~ with version
- [ ] Add the possibility of a validation set in addition of train/test set?
- [X] ~Manage location dependent links in setup.py~ Don't think we need to.
- [ ] Option for parallel training?
- [ ] Logging on WandB or Tensorboard


## How to use it

# Training a diffusion model to generate a realisitc physical field

We have a custom dataset object in PyTorch that takes a directory as argument and separates it into training and test set then loads both (and logs the test set elements).
Feel free to change the dataloading part.

# Generating realistic physical fields

Given a checkpoint folder and a model id, you can parse the flagfile associated with the arguments used for training this model. It can serve two purposes : 
- fine-tuning the model/training again a similar architecture, as you have access to every useful hyperparameter we used
- loading the model and then its weight from a checkpoint

Then use either the function generate_image or the method generate_image. As you can imagine their input signature differ but overall it requires:
- the loaded model
- the number of samples you want
- the number of channel
- the size of the images (height and width are equal)
- ...two other inputs that are only used for denoising

If you use the function, it returns a list of samples and the corresponding samples halfway through the reverse diffusion process. If you use the method, it returns a torch tensor with only the samples.

# Denoising images under various levels of the perturbation kernel used.

We propose a new method for denoising images under a gaussian random field noise, using a diffusion model trained under this noise as a prior. 

To denoise images like that, you need to load the model and its weight (see the previous subsection) and then you can use the last two arguments of the generate_image function/method. Given a noise level t and a batch to denoise:
- UPCOMING README

You can also use functions in the separation.py file or use the source_separation_interactive notebook:
- check_separation1score is used to check the method given some model. 
  - It takes the following arguments:
    - MODEL_ID1
    - CKPT_FOLDER1
    - noise_step (default 500) the timestep corresponding to the noise level you want
    - num_to_denoise (default 1), how many images from the test set do you want to denoise
    - NUM_SAMPLES (default 16), how many diffusion trajectories do you want to compute (use 1 in case of Tweedie's formula)
    - tweedie (default False)
  - it takes the first num_to_denoise images from the test set, add noise to them (equivalent up to rescaling by a time dependent constant to following the forward SDE process) and then denoise them in either one step or noise_step (multistep denoising)
  - it returns three lists of torch tensors, all of length num_to_denoise:
    - truth_list a list of tensors of size (1, channel, size, size)
    - noisy_list a list of tensors of size (NUM_SAMPLES, channel, size, size) the noisy_list is given in an additive noise convention (the model takes as inputs these but rescaled to preserve the variance see Yang Song's paper)  
    - denoised_list a list of tensors of size (NUM_SAMPLES, channel, size, size) the results
- separation1score fullfills the same role but given an observation as input. The observation is supposed to be noisy and you need to know the noise level to which it was noised. You can input observations in both the rescaled (variance preserving) and the additive noise conventions
  - It takes the following arguments:
    - MODEL_ID1
    - CKPT_FOLDER1
    - observation
    - noise_step (default 500) the timestep corresponding to the noise level you want
    - NUM_SAMPLES (default 16), how many diffusion trajectories do you want to compute (use 1 in case of Tweedie's formula)
    - tweedie (default False)
    - rescale_observation (default True) whether or not the model should rescale the observation
  
## Repository Structure

- **Root**
  - **[ddpm](./ddpm)**: Scripts and Notebooks for experimentation.
    - **[model.py](./ddpm/model.py)** (Python): .
    - **[diffusion](./ddpm/diffusion.py)** (Python): .
  - **[dataHandler](./dataHandler)**: The production-ready solution(s) composed of libraries, services, and jobs.
    - **[dataset.py](./dataHandler/dataset.py)** (Python): Utility functions that are distilled from the research phase and used across multiple scripts. Should only contain refactored and tested Python scripts/modules. Installable via pip.
  - **[validationMetrics](./validationMetrics)**: The production-ready solution(s) composed of libraries, services, and jobs.
    - **[frechet.py](./validationMetrics/frechet.py)** (Python): Script to compute Frechet Distance between two gaussians/ work in progress, code borrowed from this [pytorch implementation of Jonathan Ho's DDPM](https://github.com/w86763777/pytorch-ddpm).
    - **[minkowskiFunctional.py](./validationMetrics/frechet.py)** (Python): Script to compute the $\mathcal{M_0}$, $\mathcal{M_1}$ and $\mathcal{M_2}$ Minkowski functionals. Trimmed down version of [this file](https://github.com/nmudur/diffusion-models-astrophysical-fields-mlps/blob/15869027b4c57788129cb0985c20090e80418369/annotated/evaluate.py) from the repository linked to [Can denoising diffusion probabilistic models generate realistic astrophysical fields?](https://arxiv.org/abs/2211.12444).
    - **[powerSpectrum.py](./validationMetrics/powerSpectrum.py)** (Python): Script to compute power spectra, original code by [Bruno Regaldo Saint-Blancard](https://users.flatironinstitute.org/~bregaldosaintblancard/).
    - **[vae.py](./validationMetrics/vae.py)**/**[vae.ipynb](./validationMetrics/vae.ipynb)**/**[vae_copy.ipynb](./validationMetrics/vae_copy.ipynb)** (Python/Jupyter): Work in progress to develop networks whose activation maps could be used for an equivalent of the Frechet Inception Distance (FID).
  - **[train](./train.py)** (Python): Folder for logs outputs (sample and checkpoints on a different cluster memory partition).
  - **[train_interactive](./train_interactive.ipynb)** (Jupyter): Folder for logs outputs (sample and checkpoints on a different cluster memory partition).
  - **[data_processing](./data_processing.ipynb)**: Jupyter Notebook to process data from raw 3D FITS cube and also create the toy dataset. For internal FI cluster use, using provided data directories is prefered.
  - **[check](./check.ipynb)**: Jupyter Notebook to visually check samples.
  - **[checkSummaryStat](./checkSummaryStat.ipynb)**: Jupyter Notebook to check summary statistics for a single model against the baseline.
  - **[checkTraining](./checkTraining.ipynb)**: Jupyter Notebook to check how training went (loss visualization, overfitting checks, reconstruction checks).
  - **[compareSummaryStat](./compareSummaryStat.ipynb)**: Jupyter notebook to compare summary statistics between samples for two models and the baseline (data set).
  - **[ddpm_exp.sh](./ddpm_exp.sh)** (Bash): Example of a bash script to launch an experiment.
  - **[ddpm1.17](./ddpm1.17.sh)** (Bash): Example of single model training using a bash script.

---

## Requirements

- Numpy, Matplotlib...
- Torch, cuda...
- astropy
- absl
- tqdm

(Upcoming MoMo/MuP)

For use on the FI cluster, use provided module &, in addition to previous packages/library, use the following modules:

module load cuda
module load cudnn
module load nvhpc
module load nvtop
