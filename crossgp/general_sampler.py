import numpy as np
import GPy
from GPy.core.parameterization.priors import Uniform
import emcee
import matplotlib.pyplot as plt
import corner
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util import diag
from GPy.util.linalg import pdinv, dpotrs, tdot, dpotri, jitchol
log_2_pi = np.log(2*np.pi)


class GPHyperparameterSampler:
    def __init__(self, freqs, Y_all, kernel, noise_variance, param_names, prior_bounds):
        self.freqs = freqs
        self.Y_all = Y_all
        self.kernel = kernel.copy()
        self.param_names = param_names
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names)
        self.noise_variance = noise_variance
        self.model = GPy.models.GPRegression(self.freqs, Y_all, kernel=self.kernel)
        self.model.Gaussian_noise.variance = noise_variance
        self.model.Gaussian_noise.fix()

    def _set_params(self, theta):
        for name, val in zip(self.param_names, theta):
            parts = name.split('.')
            attr = self.kernel
            for p in parts[:-1]:
                attr = getattr(attr, p)
            setattr(attr, parts[-1], val)
        self.model.kern = self.kernel

    def log_likelihood(self, theta):
        self._set_params(theta)
        return self.model.log_likelihood()

    def log_prior(self, theta):
        for val, (low, high) in zip(theta, self.prior_bounds):
            if not (low < val < high):
                return -np.inf
        return 0.0

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_sampler(self, initial_position, nwalkers=50, nsteps=50, discard=10):
        p0 = [initial_position + 0.1 * np.random.randn(self.ndim) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(p0, nsteps, progress=True)
        self.posterior_samples = sampler.get_chain(discard=discard, flat=True)
        self.update_model_with_posterior_mean()
        self.print_posterior_means()
        self.plot_corner()
        return self.model

    def update_model_with_posterior_mean(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        self._set_params(mean_vals)

    def print_posterior_means(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        for name, val in zip(self.param_names, mean_vals):
            print(f"Posterior mean {name}: {val}")

    def plot_corner(self):
        corner.corner(self.posterior_samples, labels=self.param_names)



class DiagSampler:
    def __init__(self, freqs, Y_all, kerns, noise_var, param_names, prior_bounds):
        self.freqs = freqs
        self.Y_all = Y_all
        self.kerns = [kerns[0].copy(),kerns[1].copy()]
        self.param_names = param_names
        self.param_names_flat = [item for sublist in self.param_names for item in sublist]
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names[0])+len(param_names[1])
        self.noise_var = noise_var
        self.N_theta1 = len(param_names[0])
        self.models = None
        
    def K_comb(self, X):
        N = len(X)
        kern1, kern2 = self.kerns
        K1 = kern1.K(X)
        K2 = kern2.K(X)
        K = np.zeros((2*N,2*N))
        K[:N,:N] = K1
        K[N:,N:] = K2
        return K

    def lml(self, kerns, X, Y):
        N = len(X)
        kern1, kern2 = kerns
        m = 0
        YYT_factor = Y-m
        K = self.K_comb(X)
        Ky = K.copy()
        diag.add(Ky, self.noise_var+1e-8)
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
        return log_marginal
        
    def set_params(self, thetas, kerns):
        thetas = [thetas[:self.N_theta1],thetas[self.N_theta1:]]
        if self.N_theta1 == 1:
            for i in range(2):
                for j in range(len(self.param_names[i])):
                    setattr(kerns[i], self.param_names[i][j], thetas[i][j])
        else:
            for i in range(2):
                for j in range(len(self.param_names[i])):
                    parts = self.param_names[i][j].split('.')
                    attr = getattr(kerns[i],parts[0])
                    setattr(attr, parts[1], thetas[i][j])
        return kerns

    def log_likelihood(self, thetas):
        self.kerns = self.set_params(thetas, self.kerns)
        return self.lml(self.kerns, self.freqs, self.Y_all)

    def log_prior(self, thetas):
        for val, (low, high) in zip(thetas, self.prior_bounds):
            if not (low < val < high):
                return -np.inf
        return 0.0

    def log_posterior(self, thetas):
        lp = self.log_prior(thetas)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(thetas)

    def run_sampler(self, initial_position, nwalkers=50, nsteps=50, discard=10):
        p0 = [initial_position + 0.1 * np.random.randn(self.ndim) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(p0, nsteps, progress=True)
        self.posterior_samples = sampler.get_chain(discard=discard, flat=True)
        self.update_model_with_posterior_mean()
        self.print_posterior_means()
        self.plot_corner()
        self.models = [GPy.models.GPRegression(self.freqs, self.Y_all[:len(self.freqs),:], kernel=self.kerns[0]),GPy.models.GPRegression(self.freqs, self.Y_all[len(self.freqs):,:], kernel=self.kerns[1])]
    
    def update_model_with_posterior_mean(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        self.set_params(mean_vals, self.kerns)

    def print_posterior_means(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        for name, val in zip(self.param_names_flat, mean_vals):
            print(f"Posterior mean {name}: {val}")

    def plot_corner(self):
        corner.corner(self.posterior_samples, labels=self.param_names_flat)
