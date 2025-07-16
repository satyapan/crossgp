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


def nearest_postive_definite(A, maxtries=10):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
        if k > maxtries:
            break

    return A3

def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = GPy.util.linalg.jitchol(B, 0)
        return True
    except np.linalg.LinAlgError:
        return False



class SharedVisSampler:
    def __init__(self, data_nights, kerns, noise_nights, param_names, prior_bounds):
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
        self.noise1_var = noise_nights[0].data.real.var()
        self.noise2_var = noise_nights[1].data.real.var()
        self.result = None
        self.discard = None
        self.posterior_samples = None
        
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
        Ky[:N,:N] += self.noise1_var*np.eye(N)
        Ky[N:,N:] += self.noise2_var*np.eye(N)
        Wi, LW, LWi, W_logdet = pdinv(Ky)
        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))
        return log_marginal

    def set_params(self, thetas, kerns):
        kerns = [kerns[0], kerns[1]]
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

    def run_sampler(self, nwalkers=50, nsteps=200, discard=100, emcee_moves='stretch'):
        if emcee_moves == 'stretch':
            moves = emcee.moves.StretchMove()
        elif emcee_moves == 'kde':
            moves = emcee.moves.KDEMove()
        p0 = [np.array([prior.rvs(1)[0] for prior in self.prior_bounds]) for _ in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_posterior, moves=moves)
        sampler.run_mcmc(p0, nsteps, progress=True)
        self.result =  sampler
        self.plot_samples()
        self.discard = discard
        self.posterior_samples, _ = self.clip_outliers(discard)
        self.update_model_with_posterior_mean()
        self.print_posterior_means()
        self.plot_corner()

    def clip_outliers(self, discard, clip_nsigma=6, discard_walkers_nsigma=10):
        samples = self.result.get_chain(discard=discard)
        log_prob = self.result.get_log_prob(discard=discard)
        max_log_prob = log_prob.max(axis=0)
        mask = max_log_prob > np.median(max_log_prob) - discard_walkers_nsigma * np.median(log_prob.std(axis=0))
        if (~mask).sum() > 0:
            print(f'Discarding {(~mask).sum()} walkers')
        samples = samples[:, mask, :].reshape(-1, samples.shape[-1])        
        log_prob = log_prob[:, mask].flatten()
        samples_outliers = np.zeros_like(samples)
        for i in range(samples.shape[1]):
            m = np.median(samples[:, i])
            s = psutil.robust_std(samples[:, i])
            samples_outliers[abs(samples[:, i] - m) > clip_nsigma * s, i] = 1
        mask = (samples_outliers.sum(axis=1) == 0)
        return samples[mask], log_prob[mask]
        
    def update_model_with_posterior_mean(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        self.kerns = self.set_params(mean_vals, self.kerns)

    def print_posterior_means(self):
        mean_vals = np.mean(self.posterior_samples, axis=0)
        for name, val in zip(self.param_names_flat, mean_vals):
            print(f"Posterior mean {name}: {val}")

    def plot_samples(self):
        chain = self.result.chain
        shape = chain.shape
        ncols=4
        nrows = int(np.ceil((self.ndim + 1) / ncols))
        fig,ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 1 + 2.2 * nrows), sharex=True)
        for i in range(shape[2]):
            if 'variance' in self.param_names_flat[i]:
                ax[i//ncols,i%ncols].set_yscale('log')
            ax[i//ncols,i%ncols].text(0.01, 0.95, self.param_names_flat[i]+'=%.4f'%(np.median(chain[:,:,i])), transform=ax[i//ncols,i%ncols].transAxes, fontsize=9, va='top', ha='left')
            for j in range(shape[0]):
                ax[i//ncols,i%ncols].plot(chain[j,:,i], color='tab:orange', alpha=0.6)
        log_prob = self.result.get_log_prob()
        for j in range(shape[0]):
            ax[self.ndim//ncols, self.ndim%ncols].plot(log_prob[:,j], color='tab:orange', alpha=0.6)
        ax[self.ndim//ncols, self.ndim%ncols].text(0.01, 0.95, 'likelihood=%.4f'%(np.median(log_prob)), transform=ax[self.ndim//ncols, self.ndim%ncols].transAxes, fontsize=9, va='top', ha='left')
    

    def plot_corner(self):
        corner.corner(self.posterior_samples, labels=self.param_names_flat, smooth=1)

    def posterior_mean(self, kern_pred, coh=True):
        N = len(self.freqs)
        K = self.K_comb(self.freqs, self.kerns)
        K[:N,:N] += self.noise1_var*np.eye(N)
        K[N:,N:] += self.noise2_var*np.eye(N)
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

    def unpack_name(self, pred_name, kern_full, coh=True):
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

    def kern_from_name(self, pred_name, kern_full, coh=True):
        if coh==True or coh==False:
            kern_pred = self.unpack_name(pred_name, kern_full, coh=coh)
        else:
            pred_coh = pred_name[0]
            pred_inc = pred_name[1]
            kern_pred_coh = self.unpack_name(pred_coh, kern_full, coh=True)
            kern_pred_inc = self.unpack_name(pred_inc, kern_full, coh=False)
            kern_pred = (kern_pred_coh, kern_pred_inc)
        return kern_pred
    
    def combine_cubes(self, cube1, cube2):
        if cube1.weights == None or cube2.weights == None:
            cube_comb = cube1.copy()
            cube_comb.data = (cube1.data+cube2.data)/2
        else:
            weights = cube1.weights.data+cube2.weights.data
            data = (cube1.data*cube1.weights.data+cube2.data*cube2.weights.data)/weights
            cube_comb = cube1.copy()
            cube_comb.data = data
            cube_comb.weights.data = weights
        return cube_comb
            
    def predict_dist(self, pred_name, kern_full, coh=True, n_pick=100, subtract_from=None, combine=False):
        N = len(self.freqs)
        K = self.K_comb(self.freqs, kern_full)
        K[:N,:N] += self.noise1_var*np.eye(N)
        K[N:,N:] += self.noise2_var*np.eye(N)
        Wi, LW, LWi, W_logdet = pdinv(K)
        alpha, _ = dpotrs(LW, self.Y_all, lower=1)
        kern_pred = self.kern_from_name(pred_name, kern_full, coh=coh)
        if coh==True:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            K_p[:N,N:] = kern_pred.K(self.freqs)
            K_p[N:,:N] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)[:N,:]
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = (K_p - K_p.T.dot(v))[:N,:N]
            if not is_positive_definite(y_cov):
                y_cov = nearest_postive_definite(y_cov)
            y_cov_samples = np.transpose(np.random.multivariate_normal(np.zeros(N), y_cov, size=(y_mean.shape[1],n_pick)), axes=(1,2,0))
            y_mean_gen = list(y_mean[None,:,:]+y_cov_samples)
            y_mean_complex = [y[:,:y.shape[1]//2] + 1j*y[:,y.shape[1]//2:] for y in y_mean_gen]
            data_pred = [self.data1.copy() for i in range(n_pick)]
            for i in range(n_pick):
                data_pred[i].data = y_mean_complex[i]
            if subtract_from is not None:
                if type(subtract_from) == list:
                    for i in range(n_pick):
                        data_pred[i] = (data_pred[i].copy(),data_pred[i].copy())
                        data_pred[i][0].data = subtract_from[0].data - data_pred[i][0].data
                        data_pred[i][1].data = subtract_from[1].data - data_pred[i][1].data
                    if combine == True:
                        for i in range(n_pick):
                            data_pred[i] = self.combine_cubes(data_pred[i][0], data_pred[i][1])
                else:
                    for i in range(n_pick):
                        data_pred[i].data = subtract_from.data - data_pred[i].data
        elif coh==False:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred.K(self.freqs)
            K_p[N:,N:] = kern_pred.K(self.freqs)
            y_mean = K_p.T.dot(alpha)
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = K_p - K_p.T.dot(v)
            if not is_positive_definite(y_cov):
                y_cov = nearest_postive_definite(y_cov)
            y_cov_samples = np.transpose(np.random.multivariate_normal(np.zeros(2*N), y_cov, size=(y_mean.shape[1],n_pick)), axes=(1,2,0))
            y_mean_gen = list(y_mean[None,:,:]+y_cov_samples)
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
            if combine == True:
                for i in range(n_pick):
                    data_pred[i] = self.combine_cubes(data_pred[i][0], data_pred[i][1])
        else:
            K_p = np.zeros_like(K)
            K_p[:N,:N] = kern_pred[0].K(self.freqs)
            K_p[N:,N:] = kern_pred[0].K(self.freqs)
            K_p[:N,N:] = kern_pred[0].K(self.freqs)
            K_p[N:,:N] = kern_pred[0].K(self.freqs)
            K_p[:N,:N] += kern_pred[1].K(self.freqs)
            K_p[N:,N:] += kern_pred[1].K(self.freqs)
            y_mean = K_p.T.dot(alpha)
            v, _ = dpotrs(LW, K_p, lower=1)
            y_cov = K_p - K_p.T.dot(v)
            if not is_positive_definite(y_cov):
                y_cov = nearest_postive_definite(y_cov)
            y_cov_samples = np.transpose(np.random.multivariate_normal(np.zeros(2*N), y_cov, size=(y_mean.shape[1],n_pick)), axes=(1,2,0))
            y_mean_gen = list(y_mean[None,:,:]+y_cov_samples)
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
            if combine == True:
                for i in range(n_pick):
                    data_pred[i] = self.combine_cubes(data_pred[i][0], data_pred[i][1])
        return data_pred

    def sample_cubes(self, pred_name, coh=True, discard=None, n_pick=100, subtract_from=None, combine=False):
        if discard is None:
            discard=self.discard
        flat_samples, _ = self.clip_outliers(discard)
        idx = np.random.choice(flat_samples.shape[0], size=n_pick, replace=False)
        picked_samples = flat_samples[idx]
        cubes = []
        bar = tqdm(total=n_pick)
        for i in range(n_pick):
            theta = picked_samples[i,:]
            kerns_theta = self.set_params(theta, self.kerns)
            cube = self.predict_dist(pred_name, kerns_theta, coh=coh, n_pick=1, subtract_from=subtract_from, combine=combine)[0]
            cubes.append(cube)
            bar.update(1)
        return cubes
    
    def get_ps3d(self, ps_gen, kbins, pred_name, coh=True, kind='dist', discard=None, n_pick=100, subtract_from=None, combine=False):
        if discard is None:
            discard=self.discard
        if kind == 'dist':
            print('Sampling from GP posterior')
            cubes = self.predict_dist(pred_name, self.kerns, coh=coh, n_pick=n_pick, subtract_from=subtract_from, combine=combine)
        else:
            print('Sampling from hyperparameter distribution')
            cubes = self.sample_cubes(pred_name, coh=coh, discard=discard, n_pick=n_pick, subtract_from=subtract_from, combine=combine)
        if (coh==True and type(subtract_from) != list) or combine==True:
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
    
    def get_ps2d(self, ps_gen, pred_name, coh=True, kind='dist', discard=None, n_pick=100, subtract_from=None, combine=False):
        if discard is None:
            discard=self.discard
        if kind == 'dist':
            print('Sampling from GP posterior')
            cubes = self.predict_dist(pred_name, self.kerns, coh=coh, n_pick=n_pick, subtract_from=subtract_from, combine=combine)
        else:
            print('Sampling from hyperparameter distribution')
            cubes = self.sample_cubes(pred_name, coh=coh, discard=discard, n_pick=n_pick, subtract_from=subtract_from, combine=combine)
        if (coh==True and type(subtract_from) != list) or combine==True:
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