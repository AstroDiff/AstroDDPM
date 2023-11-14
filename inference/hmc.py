""" Adapted from https://github.com/modichirag/VBS/blob/main/src/pyhmc.py """

import numpy as np
import torch
from tqdm import tqdm


class DualAveragingStepSize():
    """ Dual averaging step size adaptation. """

    def __init__(self, initial_step_size,
                 target_accept=0.65,
                 gamma=0.05,
                 t0=10.0,
                 kappa=0.75,
                 nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = torch.log(initial_step_size) #torch.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma * 2 #parameter to tune
        self.t = t0
        self.kappa = kappa
        self.error_sum = torch.zeros_like(self.initial_step_size).to(initial_step_size.device) #0
        self.log_averaged_step = torch.zeros_like(self.initial_step_size).to(initial_step_size.device) #0
        self.nadapt = nadapt
        
    def update(self, p_accept):
        p_accept[p_accept > 1] = 1.
        p_accept[torch.isnan(p_accept)] = 0.
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return torch.exp(log_step), torch.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n" % step_size)
        else:
            step_size = torch.exp(self.log_averaged_step)
        return step_size
    


class HMC():
    """ Hamiltonian Monte Carlo Sampler"""

    def __init__(self, log_prob,
                 grad_log_prob=None,
                 log_prob_and_grad=None,
                 invmetric_diag=None,
                 precision=torch.float32):
        """Constructor

        Args:
            log_prob (_type_): Function that returns the log probability of the target distribution.
            grad_log_prob (_type_, optional): Function that returns the gradient of the log probability of the target distribution. Defaults to None.
            log_prob_and_grad (_type_, optional): Function that returns both the log probability of the target distribution and its gradient. Defaults to None.
            invmetric_diag (_type_, optional): Inverse vector of masses. Defaults to None.
            precision (_type_, optional): Float precision. Defaults to torch.float32.
        """

        self.precision = precision
        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        
        assert not((self.grad_log_prob is None) and (self.log_prob_and_grad is None))

        # Convert to prescribed precision
        if self.log_prob is not None:
            self.log_prob = lambda x: log_prob(x).to(self.precision)
        if self.grad_log_prob is not None:
            self.grad_log_prob = lambda x: grad_log_prob(x).to(self.precision)
        if self.log_prob_and_grad is not None:
            self.log_prob_and_grad = lambda x: tuple([y.to(self.precision) for y in log_prob_and_grad(x)])

        # Inverse mass matrix
        if invmetric_diag is None:
            self.invmetric_diag = 1.
        else:
            self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        # Define the potential energy, kinetic energy and its gradient
        self.V = lambda x: -self.log_prob(x)
        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum(-1) #Sum across rows
        self.KE_g = lambda p: p * self.invmetric_diag

        # Collision function
        self.collision_fn = None

        # Counters
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0
    
    def reset_counters(self):
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0

    def V_g(self, q):
        """Returns the gradient of the potential energy.

        Args:
            q (torch.Tensor): Position vector.

        Returns:
            torch.Tensor: Gradient of the potential energy at q.
        """
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(q)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(q)
        return -v_g.detach()

    def H(self, q, p, Vq=None):
        """Returns the Hamiltonian.

        Args:
            q (torch.Tensor): Position vector.
            p (torch.Tensor): Momentum vector.
            Vq (torch.Tensor, optional): Potential energy at q. Defaults to None.

        Returns:
            torch.Tensor: Hamiltonian at (q, p).
        """
        if Vq is None:
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)

    def set_collision_fn(self, collision_fn):
        """Sets custom collision management function for leapfrog updates.

        A collision management function takes as input the position and momentum vectors and returns
        the updated position and momentum vectors.

        Args:
            collision_fn (_type_): Collision function.
        """
        self.collision_fn = collision_fn

    def leapfrog(self, q, p, N, step_size):
        """Leapfrog integrator.

        Args:
            q (torch.Tensor): Position vector.
            p (torch.Tensor): Momentum vector.
            N (int): Number of leapfrog steps.
            step_size (torch.Tensor): Step sizes per chain.

        Returns:
            (torch.Tensor, torch.Tensor): Updated (q, p).
        """
        self.leapcount += 1
        s = step_size.unsqueeze(-1)

        p = p - 0.5 * s * self.V_g(q)
        for i in range(N - 1):
            q = q + s * self.KE_g(p)
            if self.collision_fn is not None:
                q, p = self.collision_fn(q, p)
            p = p - s * self.V_g(q)
        q = q + s * self.KE_g(p)
        if self.collision_fn is not None:
            q, p = self.collision_fn(q, p)
        p = p - 0.5 * s * self.V_g(q)
        return q, p
        
    def metropolis(self, q0, p0, q1, p1, V_q0=None, V_q1=None):
        """Metropolis-Hastings acceptance step.

        Args:
            q0 (_type_): Position vector at step n.
            p0 (_type_): Momentum vector at step n.
            q1 (_type_): Position vector at step n+1.
            p1 (_type_): Momentum vector at step n+1.
            V_q0 (_type_, optional): Potential energy at q0. Defaults to None.
            V_q1 (_type_, optional): Potential energy at q1. Defaults to None.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor): Position, momentum, acceptance rate,
            Hamiltonians at (q0, p0) and (q1, p1) after the acceptance step.
        """
        
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = torch.exp(H0 - H1)

        u = torch.rand(prob.shape[0], device=prob.device)

        qq = q1.clone()
        pp = p1.clone()
        acc = torch.ones_like(prob)

        cond1 = torch.logical_or(torch.isnan(prob), torch.isinf(prob))
        cond1 = torch.logical_or(cond1, torch.sum(q0 - q1, dim=-1) == 0)

        qq[cond1] = q0[cond1]
        pp[cond1] = p0[cond1]
        acc[cond1] = -1.0

        cond2 = torch.logical_and(u > torch.min(torch.ones_like(u), prob), ~cond1)
        qq[cond2] = q0[cond2]
        pp[cond2] = p0[cond2]
        acc[cond2] = 0.0
        
        return qq, pp, acc, torch.stack([H0, H1], dim=-1)

    def step(self, q, nleap, step_size):
        """Performs a single HMC step.

        Args:
            q (torch.Tensor): Position vector.
            nleap (torch.Tensor): Number of leapfrog steps.
            step_size (torch.Tensor): Step sizes per chain.

        Returns:
            (torch.Tensor,)*5: Position, momentum, acceptance rate,
            Hamiltonians at (q0, p0) and (q1, p1), misc counts after HMC step.
        """
        p = torch.randn(q.shape, device=q.device, dtype=self.precision) * self.metricstd
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis(q, p, q1, p1)
        return q, p, accepted, Hs, torch.tensor([self.Hcount, self.Vgcount, self.leapcount])

    def adapt_stepsize(self, q, step_size, epsadapt, nleap):
        """ Dual averaging step size adaptation.

        Args:
            q (torch.Tensor): Position vector
            step_size (torch.Tensor): Intial step size.
            epsadapt (int): Number of iterations for step size adaptation.
            nleap (int): Number of leapfrog steps per HMC step.

        Returns:
            (torch.Tensor, torch.Tensor): Updated position vector, step size.
        """
        print("Adapting step size using %d iterations" % epsadapt)
        epsadapt_kernel = DualAveragingStepSize(step_size)

        for i in tqdm(range((epsadapt + 1))):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            q = q.detach()
            prob = torch.exp(Hs[...,0] - Hs[...,1])

            if i < epsadapt:
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
        return q, step_size
    
    def sample(self, q,
               step_size = 0.01,
               nsamples=20,
               burnin=10,
               nleap=30,
               skipburn=True,
               epsadapt=0,
               verbose=False,
               ret_side_quantities=False):
        """Performs HMC sampling.

        Args:
            q (torch.Tensor): Position vector. Possible shapes are (nchains, ndim) or (ndim).
            step_size (float, optional): Step size. If no adaptive step size Defaults to 0.01.
            nsamples (int, optional): Number of samples. Defaults to 20.
            burnin (int, optional): Number of burn-in steps. Defaults to 10.
            nleap (int, optional): Number of leapfrog steps per HMC step. Defaults to 30.
            skipburn (bool, optional): Should we save burning samples? Defaults to True.
            epsadapt (int, optional): Epsilon adapt parameter. Defaults to 0.
            verbose (bool, optional): Verbose mode. Defaults to False.
            ret_side_quantities (bool, optional): Should we return side quantities? Defaults to False.

        Returns:
            Sampler: Sampler object.
        """
        if q.ndim == 1: q = q.unsqueeze(0) # We add a chain dimension if there is none.
        assert q.ndim == 2, "q must be 2D" # Shape of q is (nchains, ndim)

        self.reset_counters()

        q = q.to(self.precision)
        
        step_size = step_size * torch.ones((q.shape[0]), device=q.device, dtype=self.precision)

        # We store the samples, acceptance rates, Hamiltonian values, and misc counts
        samples_list = []
        accepts_list = []
        Hs_list = []
        counts_list = []

        if epsadapt > 0:
            q, step_size = self.adapt_stepsize(q, step_size, epsadapt, nleap)
            self.step_size = step_size
        for i in tqdm(range(nsamples + burnin), disable=not verbose):
            q, p, acc, Hs, count = self.step(q, nleap, step_size)
            accepts_list.append(acc)
            if (skipburn and (i >= burnin)) or not skipburn:
                samples_list.append(q)
                Hs_list.append(Hs)
                counts_list.append(count)

        # To torch tensors
        samples_list = torch.stack(samples_list, dim=-2)
        accepts_list = torch.stack(accepts_list, dim=-1)
        Hs_list = torch.stack(Hs_list, dim=-2)
        counts_list = torch.stack(counts_list, dim=-2)
        
        if ret_side_quantities:
            return samples_list, accepts_list, Hs_list, counts_list
        else:
            return samples_list
