### Defines different SDE solver that will be incorporated into the generative process of the models
### Currently, only Euler-Maruyama method is implemented de facto for the discrete time model.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_schedule(schedule_type, **kwargs):
    """
    Returns a schedule of time steps for a differential equation solver (SDE or ODE)
    Args:
        schedule_type: str, type of schedule
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
    """
    if schedule_type == 'linear':
        return linear_schedule(**kwargs)
    elif schedule_type == 'power_law':
        return power_law_schedule(**kwargs)
    else:
        raise NotImplementedError

def linear_schedule(**kwargs):
    """
    Returns a linear schedule of time steps for a differential equation solver (SDE or ODE)
    **kwargs should contain the following keys:
        t_min: float, minimum time step
        t_max: float, maximum time step
        n_iter: int, number of time steps
    Args:
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
        """
    ## TODO parrallelize this
    if 't_min' not in kwargs:
        kwargs['t_min'] = 1e-4
    if 't_max' not in kwargs:
        kwargs['t_max'] = 1
    if 'n_iter' not in kwargs:
        kwargs['n_iter'] = 1000
    return torch.linspace(kwargs['t_min'], kwargs['t_max'], kwargs['n_iter'])

def power_law_schedule(**kwargs):
    """
    Returns a power law schedule of time steps for a differential equation solver (SDE or ODE)
    **kwargs should contain the following keys:
        t_min: float, minimum time step
        t_max: float, maximum time step
        n_iter: int, number of time steps
        power: float, power law exponent
    Args:
        **kwargs: additional arguments for the schedule
    Returns:
        schedule: torch.Tensor, schedule of time steps
    """
    ## TODO parrallelize this
    if 't_min' not in kwargs:
        kwargs['t_min'] = 1e-4
    if 't_max' not in kwargs:
        kwargs['t_max'] = 1
    if 'n_iter' not in kwargs:
        kwargs['n_iter'] = 1000
    if 'power' not in kwargs:
        kwargs['power'] = 1.1
    n_iter = kwargs['n_iter']
    return torch.linspace(0,1,n_iter)**kwargs['power'] * (kwargs['t_max'] - kwargs['t_min']) + kwargs['t_min']

class EulerMaruyama():
    """
    Euler-Maruyama method for solving stochastic differential equations
    """
    def __init__(self, schedule):
        self.schedule = schedule

    def forward(self, x_init, f, gdW, reverse_time = True, verbose = False):
        """
        Solves the SDE dx = f(x, t) dt + g(x, t) dW_t
        Args:
            x_init: torch.Tensor, initial value of x
            f: function, drift term
            gdW: function, diffusion term (should also contain the random part of the diffusion term, will only be multiplied by sqrt(dt) in the solver)
            reverse_time: bool, whether to reverse the time schedule (used for backward SDEs)
            verbose: bool, whether to print progress bar
        Returns:
            x: torch.Tensor, a sample from the SDE at the final time step
        """
        if reverse_time:
            times = self.schedule
            times = times.flip(0)
        else:
            times = self.schedule
        x = x_init

        progress_bar = tqdm.tqdm(total = len(times)-1, disable=not verbose)
        for i in range(len(times)-1):
            dt = times[i+1] - times[i]
            timesteps = times[i]*torch.ones(x.shape[0],1).to(x.device)
            x = x + f(x, timesteps) * dt + gdW(x, timesteps) * torch.sqrt(torch.abs(dt))
            if x.isnan().any():
                print("Nan encountered at time step {}".format(i))
                break
            progress_bar.update(1)
        if reverse_time:
            times = x.flip(0) ## flip back to the original time order
        return x

