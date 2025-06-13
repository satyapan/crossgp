import torch
from ps_eor import datacube, pspec, psutil, simu, fitutil, fgfit, flagger, ml_gpr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import GPy
from GPy.core.parameterization.priors import Uniform
import emcee
import corner
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util import diag
from GPy.util.linalg import pdinv, dpotrs, tdot, dpotri, jitchol
log_2_pi = np.log(2*np.pi)


class Log10Uniform(GPy.core.parameterization.priors.Prior):
    ''' Log10 prior '''

    domain = GPy.priors._POSITIVE

    def __new__(cls, *args):
        return object.__new__(cls)

    def __init__(self, l, u):
        self.lower = l
        self.upper = u

    def __str__(self):
        return "Log10[{:.2g}, {:.2g}]".format(self.lower, self.upper)

    def lnpdf(self, x):
        region = (x >= 10 ** self.lower) * (x <= 10 ** self.upper)
        return np.log(region)

    def lnpdf_grad(self, x):
        return np.zeros(x.shape)

    def rvs(self, n):
        return 10 ** np.random.uniform(self.lower, self.upper, size=n)



class SharedVisSampler:
    def __init__(self, data_nights, kerns, noise_var, param_names, prior_bounds):
        self.data1 = data_nights[0]
        self.data2 = data_nights[1]
        self.freqs = self.data1.freqs.reshape(-1,1)*1e-6
        data_stack = np.vstack((self.data1.data, self.data2.data))
        self.Y_all = np.hstack((data_stack.real, data_stack.imag))
        self.kerns = [kerns[0].copy(),kerns[1].copy()]
        self.param_names = param_names
        self.param_names_flat = [item for sublist in self.param_names for item in sublist]
        self.prior_bounds = prior_bounds
        self.ndim = len(param_names[0])+len(param_names[1])
        self.N_theta1 = len(param_names[0])
        self.noise_var = noise_var
        self.result = None
        
    def K_comb(self, X, kerns):
        N = len(X)
        kern1, kern2 = kerns
        diag = kern1+kern2
        offdiag = kern1
        K1 = diag.K(X)
        K2 = offdiag.K(X)
        K = np.zeros((2*N,2*N))
        K[:N,:N] = K1
        K[N:,N:] = K1
        K[:N,N:] = K2
        K[N:,:N] = K2
        return K

    def lml(self, X, Y):
        N = len(X)
        m = 0
        YYT_factor = Y-m
        K = self.K_comb(X, self.kerns)
        Ky = K.copy()
        diag.add(Ky, self.noise_var+1e-8)
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
        return log_marginal

    def set_params(self, thetas, kerns):
        kerns = [kerns[0].copy(), kerns[1].copy()]
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

    def run_sampler(self, nwalkers=50, nsteps=200, discard=100):
        p0 = [np.array([prior.rvs(1)[0] for prior in self.prior_bounds]) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior)
        sampler.run_mcmc(p0, nsteps, progress=True)
        self.result =  sampler
        self.posterior_samples = sampler.get_chain(discard=discard, flat=True)
        self.update_model_with_posterior_mean()
        self.print_posterior_means()
        self.plot_corner()
        
    def update_model_with_posterior_mean(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        self.kerns = self.set_params(mean_vals, self.kerns)

    def print_posterior_means(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        for name, val in zip(self.param_names_flat, mean_vals):
            print(f"Posterior mean {name}: {val}")

    def plot_corner(self):
        corner.corner(self.posterior_samples, labels=self.param_names_flat)

    def posterior_mean(self, kern_pred, coh=True):
        N = len(self.freqs)
        K = self.K_comb(self.freqs, self.kerns)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all, lower=1)
        if coh:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            K_p[:N,N:] = kern_pred.K(self.freqs)
            K_p[N:,:N] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)[:N,:]
            y_mean_complex = y_mean[:,:y_mean.shape[1]//2] + 1j*y_mean[:,y_mean.shape[1]//2:]
            data_pred = self.data1.copy()
            data_pred.data = y_mean_complex
            return data_pred
        else:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)
            y_mean1 = y_mean[:N,:]
            y_mean2 = y_mean[N:,:]
            y_mean1_complex = y_mean1[:,:y_mean1.shape[1]//2] + 1j*y_mean1[:,y_mean1.shape[1]//2:]
            y_mean2_complex = y_mean2[:,:y_mean2.shape[1]//2] + 1j*y_mean2[:,y_mean2.shape[1]//2:]
            data_pred1 = self.data1.copy()
            data_pred2 = self.data2.copy()
            data_pred1.data = y_mean1_complex
            data_pred2.data = y_mean2_complex
            return data_pred1, data_pred2
    
    def predict(self, kern_pred, kern_full, coh=True):
        N = len(self.freqs)
        K = self.K_comb(self.freqs, kern_full)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all, lower=1)
        if coh:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            K_p[:N,N:] = kern_pred.K(self.freqs)
            K_p[N:,:N] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)[:N,:]
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = (K_p - K_p.T.dot(v))[:N,:N]
            y_mean = np.array([np.random.multivariate_normal(y_mean[:,i], y_cov) for i in range(y_mean.shape[1])]).T
            y_mean_complex = y_mean[:,:y_mean.shape[1]//2] + 1j*y_mean[:,y_mean.shape[1]//2:]
            data_pred = self.data1.copy()
            data_pred.data = y_mean_complex
            return data_pred
        else:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = K_p - K_p.T.dot(v)
            y_mean = np.array([np.random.multivariate_normal(y_mean[:,i], y_cov) for i in range(y_mean.shape[1])]).T
            y_mean1 = y_mean[:N,:]
            y_mean2 = y_mean[N:,:]
            y_mean1_complex = y_mean1[:,:y_mean1.shape[1]//2] + 1j*y_mean1[:,y_mean1.shape[1]//2:]
            y_mean2_complex = y_mean2[:,:y_mean2.shape[1]//2] + 1j*y_mean2[:,y_mean2.shape[1]//2:]
            data_pred1 = self.data1.copy()
            data_pred2 = self.data2.copy()
            data_pred1.data = y_mean1_complex
            data_pred2.data = y_mean2_complex
            return data_pred1, data_pred2

    def predict_dist(self, pred_name, kern_full, coh=True, n_pick=10, subtract_from=None):
        N = len(self.freqs)
        K = self.K_comb(self.freqs, kern_full)
        diag.add(K, self.noise_var)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all, lower=1)
        kern_pred = self.kern_from_name(pred_name, kern_full, coh=coh)
        if coh:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            K_p[:N,N:] = kern_pred.K(self.freqs)
            K_p[N:,:N] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)[:N,:]
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = (K_p - K_p.T.dot(v))[:N,:N]
            y_mean_gen = np.array([np.random.multivariate_normal(y_mean[:,i], y_cov, size=n_pick) for i in range(y_mean.shape[1])])
            y_mean_gen = list(y_mean_gen.transpose((1,2,0)))
            y_mean_complex = [y[:,:y.shape[1]//2] + 1j*y[:,y.shape[1]//2:] for y in y_mean_gen]
            data_pred = [self.data1.copy() for i in range(n_pick)]
            for i in range(n_pick):
                data_pred[i].data = y_mean_complex[i]
            if subtract_from is not None:
                for i in range(n_pick):
                    data_pred[i].data = subtract_from.data - data_pred[i].data
            return data_pred
        else:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = K_p - K_p.T.dot(v)
            y_mean_gen = np.array([np.random.multivariate_normal(y_mean[:,i], y_cov, size=n_pick) for i in range(y_mean.shape[1])])
            y_mean_gen = list(y_mean_gen.transpose((1,2,0)))
            y_mean1 = [y[:N,:] for y in y_mean_gen]
            y_mean2 = [y[N:,:] for y in y_mean_gen]
            y_mean1_complex = [y[:,:y.shape[1]//2] + 1j*y[:,y.shape[1]//2:] for y in y_mean1]
            y_mean2_complex = [y[:,:y.shape[1]//2] + 1j*y[:,y.shape[1]//2:] for y in y_mean2]
            data_pred = [(self.data1.copy(),self.data2.copy()) for i in range(n_pick)]
            for i in range(n_pick):
                data_pred[i][0].data = y_mean1_complex[i]
                data_pred[i][1].data = y_mean2_complex[i]
            if subtract_from is not None:
                for i in range(n_pick):
                    data_pred[i][0].data = subtract_from[0].data - data_pred[i][0].data
                    data_pred[i][1].data = subtract_from[1].data - data_pred[i][1].data
            return data_pred

    def kern_from_name(self, pred_name, kern_full, coh=True):
        if any('.' in s for s in self.param_names[(not coh)*1]):
            if type(pred_name) == str:
                kern_pred = getattr(kern_full[(not coh)*1], pred_name)
            else:
                pred_list = []
                for pred in pred_name:
                    pred_list.append(getattr(kern_full[(not coh)*1], pred))
                kern_pred = GPy.kern.Add(pred_list)
        else:
            kern_pred = kern_full[(not coh)*1]
        return kern_pred
    
    def sample_cubes(self, pred_name, coh=True, discard=100, n_pick=10, subtract_from=None):
        samples_left = self.result.get_chain(discard=discard)
        flat_samples = samples_left.reshape(-1, samples_left.shape[-1])
        idx = np.random.choice(flat_samples.shape[0], size=n_pick, replace=False)
        picked_samples = flat_samples[idx]
        cubes = []
        bar = tqdm(total=n_pick)
        for i in range(n_pick):
            theta = picked_samples[i,:]
            kerns_theta = self.set_params(theta, self.kerns)
            kern_pred = self.kern_from_name(pred_name, kerns_theta, coh=coh)
            cube = self.predict(kern_pred, kerns_theta, coh=coh)
            if subtract_from is not None:
                if coh:
                    cube.data = subtract_from.data - cube.data
                else:
                    cube[0].data = subtract_from[0].data - cube[0].data
                    cube[1].data = subtract_from[1].data - cube[1].data
            cubes.append(cube)
            bar.update(1)
        return cubes
    
    def get_ps3d(self, ps_gen, kbins, pred_name, coh=True, kind='dist', discard=100, n_pick=10, subtract_from=None):
        if kind == 'dist':
            print('Sampling from GP posterior')
            cubes = self.predict_dist(pred_name, self.kerns, coh=coh, n_pick=n_pick, subtract_from=subtract_from)
        else:
            print('Sampling from hyperparameter distribution')
            cubes = self.sample_cubes(pred_name, coh=coh, discard=discard, n_pick=n_pick, subtract_from=subtract_from)
        if coh:
            ps = []
            for i in range(n_pick):
                ps.append(ps_gen.get_ps3d(kbins, cubes[i]))
            ps = pspec.SphericalPowerSpectraMC(ps)
            return ps
        else:
            ps1 = []
            ps2 = []
            for i in range(n_pick):
                cube = cubes[i]
                ps1.append(ps_gen.get_ps3d(kbins, cube[0]))
                ps2.append(ps_gen.get_ps3d(kbins, cube[1]))
            ps1 = pspec.SphericalPowerSpectraMC(ps1)
            ps2 = pspec.SphericalPowerSpectraMC(ps2)
            return ps1, ps2
    
    def get_ps2d(self, ps_gen, pred_name, coh=True, kind='dist', discard=100, n_pick=10, subtract_from=None):
        if kind == 'dist':
            cubes = self.predict_dist(pred_name, self.kerns, coh=coh, n_pick=n_pick, subtract_from=subtract_from)
        else:
            cubes = self.sample_cubes(pred_name, coh=coh, discard=discard, n_pick=n_pick, subtract_from=subtract_from)
        if coh:
            ps = []
            for i in range(n_pick):
                ps.append(ps_gen.get_ps2d(cubes[i]))
            ps = pspec.CylindricalPowerSpectraMC(ps)
            return ps
        else:
            ps1 = []
            ps2 = []
            for i in range(n_pick):
                cube = cubes[i]
                ps1.append(ps_gen.get_ps2d(cube[0]))
                ps2.append(ps_gen.get_ps2d(cube[1]))
            ps1 = pspec.CylindricalPowerSpectraMC(ps1)
            ps2 = pspec.CylindricalPowerSpectraMC(ps2)
            return ps1, ps2