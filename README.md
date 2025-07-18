# CrossGP: Coherent and incoherent component separation with Gaussian Processes
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16084622.svg)](https://doi.org/10.5281/zenodo.16084622)

crossgp is a Python-based tool for performing Gaussian process regression (GPR) to separate out coherent and incoherent components across multiple data sets. The tool works with general 1D signals as well as gridded visibility cubes in the form obtained using [ps_eor](https://gitlab.com/flomertens/ps_eor). The algorithm used by the tool is described by Munshi et al. (in prep).

# Dependencies
crossgp requires the following python libraries:
- ps_eor
- GPy
- emcee
- corner
- tqdm
- numpy
- matplotlib

# Installation
crossgp can be installed via pip:
```
pip install crossgp
```

# Documentation
A step-by-step guide is presented in the wiki page.
