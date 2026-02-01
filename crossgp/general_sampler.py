import numpy as np
import GPy
from GPy.core.parameterization.priors import Uniform
import emcee
import matplotlib.pyplot as plt
import corner
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util import diag
from GPy.util.linalg import pdinv, dpotrs, tdot, dpotri, jitchol
from GPy.core.parameterization.priors import Uniform
log_2_pi = np.log(2*np.pi)

from .funcs import *

class SharedSampler:
    """
    Perform GPR on two data sets with linked hyperparameters with cross-covariances.

    Arguments:
    freqs (list): List of frequencies (can be any domain)
    Y_all (array): Multiple realizations of input signal are stacked horizontally forming each data set. Two such data sets are vertically stacked to form Y_all.
    kerns (list): List of two GPy.kern objects. First is coherent and second is incoherent.
    noise_var (float): The noise variance
    param_names (list): List of two lists, containing hyperparameter names that are optimized. First list is for coherent and second for incoherent kernel.
    prior_bounds (list): List of GPy.prior objects, one for each hyperparameter that is optimized, in order: coherent first, incoherent next.
    diag (bool): If True, cross covariances are not used.
    """
    def __init__(self, freqs, Y_all, kerns, noise_var, param_names, prior_bounds, diag=False):
        self.freqs = freqs
        self.Y_all = Y_all
        self.kerns = [kerns[0].copy(),kerns[1].copy()]
        self.param_names = param_names
        self.param_names_flat = [item for sublist in self.param_names for item in sublist]
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names[0])+len(param_names[1])
        self.N_theta1 = len(param_names[0])
        self.noise_var = noise_var
        self.models = None
        self.diag = diag
        
    def K_comb(self, X):
        N = len(X)
        kern1, kern2 = self.kerns
        diag = kern1+kern2
        offdiag = kern1
        K1 = diag.K(X)
        K2 = offdiag.K(X)
        K = np.zeros((2*N,2*N))
        K[:N,:N] = K1
        K[N:,N:] = K1
        if not self.diag:
            K[:N,N:] = K2
            K[N:,:N] = K2
        return K

    def lml(self, X, Y):
        N = len(X)
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
        for i in range(2):
            if any('.' in s for s in self.param_names[i]):
                for j in range(len(self.param_names[i])):
                    parts = self.param_names[i][j].split('.')
                    attr = getattr(kerns[i],parts[0])
                    setattr(attr, parts[1], thetas[i][j])
            else:
                for j in range(len(self.param_names[i])):
                    setattr(kerns[i], self.param_names[i][j], thetas[i][j])
        return kerns

    def log_likelihood(self, thetas):
        self.kerns = self.set_params(thetas, self.kerns)
        return self.lml(self.freqs, self.Y_all)

    def log_prior(self, thetas):
        logp = 0.0
        for val, prior in zip(thetas, self.prior_bounds):
            if prior is None:
                continue
            lp = prior.lnpdf(val)
            if type(prior) == GPy.core.parameterization.priors.Uniform:
                lp = np.log(lp)
            logp += lp
        return logp

    def log_posterior(self, thetas):
        lp = self.log_prior(thetas)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(thetas)

    def run_sampler(self, initial_position, nwalkers=50, nsteps=50, discard=10):
        """
        Run MCMC.

        Arguments:
        initial_position (list): List of initial values of hyperparameters that are optimized.
        nwalkers (int): Number of walkers
        nsteps (int): Number of MCMC steps
        discard (int): Number of steps to discard from the end.
        """
        p0 = [initial_position + 0.001 * np.random.randn(self.ndim) for _ in range(nwalkers)]
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
        
    def predict_coh(self):
        """
        Predict the coherent component.
        """
        K_p = self.kerns[0].K(self.freqs)
        K = self.K_comb(self.freqs)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all, lower=1)
        y_mean = np.vstack((K_p,K_p)).T.dot(alpha)
        v, _ = dpotrs(LW, np.vstack((K_p,K_p)), lower=1)
        y_cov = K_p - np.vstack((K_p,K_p)).T.dot(v)
        return y_mean, y_cov

    def predict(self, kern_pred):
        """
        Predict an incoherent component.

        Argument:
        kern_pred: GPy.kern object for the incoherent component to predict.
        """
        K_p = kern_pred.K(self.freqs)
        K = (self.kerns[0]+self.kerns[1]).K(self.freqs)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha1, _ = dpotrs(LW, self.Y_all[:len(self.freqs),:], lower=1)
        y_mean1 = K_p.T.dot(alpha1)
        alpha2, _ = dpotrs(LW, self.Y_all[len(self.freqs):,:], lower=1)
        y_mean2 = K_p.T.dot(alpha2)
        K_full = self.K_comb(self.freqs)
        diag.add(K_full, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K_full)
        v, _ = dpotrs(LW, np.vstack((K_p,K_p)), lower=1)
        y_cov = K_p - np.vstack((K_p,K_p)).T.dot(v)
        return y_mean1, y_mean2, y_cov


class SharedDiagSampler:
    """
    Perform GPR on two data sets with linked hyperparameters

    Arguments:
    freqs (list): List of frequencies (can be any domain)
    Y_all (array): Multiple realizations of input signal are stacked horizontally forming each data set. Two such data sets are vertically stacked to form Y_all.
    kerns (list): List of two GPy.kern objects, for the two datasets
    noise_var (float): The noise variance
    param_names (list): List of strings containing the linked hyperparameters that are optimized
    prior_bounds (list): List of tuples containing the linked uniform prior bounds    
    """
    def __init__(self, freqs, Y_all, kerns, noise_var, param_names, prior_bounds):
        self.freqs = freqs
        self.Y_all = Y_all
        self.kerns = [kerns[0].copy(),kerns[1].copy()]
        self.param_names = param_names
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names)
        self.noise_var = noise_var
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

    def lml(self, X, Y):
        N = len(X)
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
        if len(thetas) == 1:
            for i in range(2):
                setattr(kerns[i], self.param_names[0], thetas[0])
        else:
            for i in range(2):
                for j in range(len(self.param_names)):
                    parts = self.param_names[j].split('.')
                    attr = getattr(kerns[i],parts[0])
                    setattr(attr, parts[1], thetas[j])
        return kerns

    def log_likelihood(self, thetas):
        self.kerns = self.set_params(thetas, self.kerns)
        return self.lml(self.freqs, self.Y_all)

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
        """
        Run MCMC.

        Arguments:
        initial_position (list): List of initial values of hyperparameters that are optimized.
        nwalkers (int): Number of walkers
        nsteps (int): Number of MCMC steps
        discard (int): Number of steps to discard from the end.
        """
        p0 = [initial_position + 0.1 * np.random.randn(self.ndim) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior, moves=emcee.moves.KDEMove())
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
        for name, val in zip(self.param_names, mean_vals):
            print(f"Posterior mean {name}: {val}")

    def plot_corner(self):
        corner.corner(self.posterior_samples, labels=self.param_names)
        
    def predict(self, kern_pred):
        """
        Predict any component.

        Argument:
        kern_pred: GPy.kern object for the component to predict.
        """
        K_p = kern_pred.K(self.freqs)
        K = self.kerns[0].K(self.freqs)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha1, _ = dpotrs(LW, self.Y_all[:len(self.freqs),:], lower=1)
        y_mean1 = K_p.T.dot(alpha1)
        alpha2, _ = dpotrs(LW, self.Y_all[len(self.freqs):,:], lower=1)
        y_mean2 = K_p.T.dot(alpha2)
        K_full = self.K_comb(self.freqs)
        diag.add(K_full, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K_full)
        v, _ = dpotrs(LW, np.vstack((K_p,K_p)), lower=1)
        y_cov = K_p - np.vstack((K_p,K_p)).T.dot(v)
        return y_mean1, y_mean2, y_cov


class DiagSampler:
    """
    Perform GPR on two data sets with independent hyperparameters

    Arguments:
    freqs (list): List of frequencies (can be any domain)
    Y_all (array): Multiple realizations of input signal are stacked horizontally forming each data set. Two such data sets are vertically stacked to form Y_all.
    kerns (list): List of two GPy.kern objects, for the two datasets
    noise_var (float): The noise variance
    param_names (list): List of two lists, for first and second data set. Each list containes strings with names of the hyperparameters that are optimized
    prior_bounds (list): List of tuples containing uniform prior bounds    
    """
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
        """
        Run MCMC.

        Arguments:
        initial_position (list): List of initial values of hyperparameters that are optimized.
        nwalkers (int): Number of walkers
        nsteps (int): Number of MCMC steps
        discard (int): Number of steps to discard from the end.
        """
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


class GPHyperparameterSampler:
    """
    Perform GPR on a single data set

    Arguments:
    freqs (list): List of frequencies (can be any domain)
    Y_all (array): Multiple realizations of input signal, stacked horizontally
    kernel (GPy.kern): The GP kernel
    noise_variance (float): The noise variance
    param_names (list): List of strings containing the hyperparameters that are optimized
    prior_bounds (list): List of tuples containing uniform prior bounds    
    """
    def __init__(self, freqs, Y_all, kernel, noise_variance, param_names, prior_bounds):
        self.freqs = freqs
        self.Y_all = Y_all
        self.kernel = kernel.copy()
        self.param_names = param_names
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names)
        self.noise_variance = noise_variance
        self.model = GPy.models.GPRegression(freqs, Y_all, kernel=self.kernel)
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
        logp = 0.0
        for val, prior in zip(theta, self.prior_bounds):
            if prior is None:
                continue
            lp = prior.lnpdf(val)
            if type(prior) == GPy.core.parameterization.priors.Uniform:
                lp = np.log(lp)
            logp += lp
        return logp

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_sampler(self, initial_position, nwalkers=50, nsteps=50, discard=10):
        """
        Run MCMC.

        Arguments:
        initial_position (list): List of initial values of hyperparameters that are optimized.
        nwalkers (int): Number of walkers
        nsteps (int): Number of MCMC steps
        discard (int): Number of steps to discard from the end.
        """
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

    def predict(self, kern_pred):
        K_p = kern_pred.K(self.freqs)
        K = self.kernel.K(self.freqs)
        diag.add(K, self.model.Gaussian_noise.variance)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all[:len(self.freqs),:], lower=1)
        y_mean = K_p.T.dot(alpha)
        Wi, LW, LWi, W_logdet = pdinv(K)
        v, _ = dpotrs(LW, K_p, lower=1)
        y_cov = K_p - K_p.T.dot(v)
        return y_mean, y_cov