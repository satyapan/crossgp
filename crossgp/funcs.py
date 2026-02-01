import torch
from ps_eor import datacube, pspec, psutil
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
import multiprocessing as mp
import warnings
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

def get_uv_bins(umin, umax, du):
    '''Return uv bins from umin to umax with a bin width of du'''
    return psutil.pairwise(np.arange(umin, umax + du, du))