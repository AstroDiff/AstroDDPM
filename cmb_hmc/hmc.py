""" Adapted from https://github.com/modichirag/VBS/blob/main/src/pyhmc.py """

import numpy as np
import torch

class Sampler():

    def __init__(self):
        self.samples = []
        self.accepts = []
        self.Hs = []
        self.counts = []
        self.i = 0

    def to_array(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == list:
                self.__dict__[key] = np.array(self.__dict__[key])            

    def to_list(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == np.ndarray:
                self.__dict__[key] = list(self.__dict__[key])

    def appends(self, q, acc, Hs, count):
        self.i += 1
        self.accepts.append(acc)
        self.samples.append(q)
        self.Hs.append(Hs)
        self.counts.append(count)
        
    def save(self, path):
        pass



class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt
        
    def update(self, p_accept):

        if np.isnan(p_accept) : p_accept = 0.
        if p_accept > 1: p_accept = 1. 
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
        return np.exp(log_step), np.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n"%step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size


    
class HMC():

    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        assert not((self.grad_log_prob == None) & (self.log_prob_and_grad == None))
        #
        self.V = lambda x : self.log_prob(x)*-1.
        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
        #
        self.leapcount = 0 
        self.Vgcount = 0 
        self.Hcount = 0
 
   
    def V_g(self, x):
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(x)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(x)
        return v_g *-1.

        
    def V_vandg(self, x):
        if self.log_prob_and_grad is not None:
            self.Vgcount += 1
            v, v_g = self.log_prob_and_grad(x)
            return v*-1., v_g*-1
        else:
            raise NotImplementedError
        

    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()


    def unit_norm_KE_g(self, p):
        return p


    def H(self, q, p, Vq=None):
        if Vq is None: 
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)


    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p

        except Exception as e:
            print("exception : ", e)
            return q0, p0


    def leapfrog_Vgq(self, q, p, N, step_size, V_q=None, V_gq=None):
        self.leapcount += 1 
        q0, p0, V_q0, V_gq0 = q, p, V_q, V_gq
        try:
            if V_gq is None:
                p = p - 0.5*step_size * self.V_g(q) 
            else:
                p = p - 0.5*step_size * V_gq
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 

            q = q + step_size * self.KE_g(p)
            if self.log_prob_and_grad is not None:
                V_q1, V_gq1 = self.V_vandg(q) 
            else:
                V_q1, V_gq1 = None, self.V_g(q) 
            p = p - 0.5*step_size * V_gq1
            return q, p, V_q1, V_gq1

        except Exception as e:
            print("exception : ", e)
            return q0, p0, V_q0, V_gq0


    def metropolis(self, qp0, qp1, V_q0=None, V_q1=None, u=None):
        
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = np.exp(H0 - H1)
        #prob = min(1., np.exp(H0 - H1))
        
        if u is None: u =  np.random.uniform(0., 1., size=1)
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, -1, [H0, H1]
        elif  u > min(1., prob):
            return q0, p0, 0., [H0, H1]
        else: return q1, p1, 1., [H0, H1]


    def step(self, q, nleap, step_size, **kwargs):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def _parse_kwargs_sample(self, **kwargs):

        self.nsamples = kwargs['nsamples']
        self.burnin = kwargs['burnin']
        self.step_size = kwargs['step_size']
        self.nleap = kwargs['nleap']


    def adapt_stepsize(self, q, epsadapt, **kwargs):
        print("Adapting step size for %d iterations"%epsadapt)
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size)
        self._parse_kwargs_sample(**kwargs)
        
        for i in range(epsadapt+1):
            q, p, acc, Hs, count = self.step(q, self.nleap, step_size)
            prob = np.exp(Hs[0] - Hs[1])
            if i < epsadapt:
                if np.isnan(prob): prob = 0.
                if prob > 1: prob = 1.
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
                self.step_size = step_size
        return q

    
    def sample(self, q, p=None, callback=None, skipburn=True, epsadapt=0, **kwargs):

        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()

        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, **kwargs)
            
        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count = self.step(q, self.nleap, self.step_size)
            state.i += 1
            state.accepts.append(acc)
            if skipburn & (i > self.burnin):
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)
            
        state.to_array()
        return state
