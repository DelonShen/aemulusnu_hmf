from .gaussian_process import *
from .massfunction import *
from .utils import *

from functools import cache
from pkg_resources import resource_filename

import numpy as np
import scipy

key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev']
mass_function_params = ['d0', 'd1',
                        'e0', 'e1',
                        'f0', 'f1',
                        'g0', 'g1']

gp_parameters_fname = {'mean_module': 
                       {'base_means': 
                        [{'weights': 'data/mean_module.base_means.0.weights.txt',
                          'bias': 'data/mean_module.base_means.0.bias.txt'},
                         {'weights': 'data/mean_module.base_means.1.weights.txt',
                          'bias': 'data/mean_module.base_means.1.bias.txt'},
                         {'weights': 'data/mean_module.base_means.2.weights.txt',
                          'bias': 'data/mean_module.base_means.2.bias.txt'},
                         {'weights': 'data/mean_module.base_means.3.weights.txt',
                          'bias': 'data/mean_module.base_means.3.bias.txt'},
                         {'weights': 'data/mean_module.base_means.4.weights.txt',
                          'bias': 'data/mean_module.base_means.4.bias.txt'},
                         {'weights': 'data/mean_module.base_means.5.weights.txt',
                          'bias': 'data/mean_module.base_means.5.bias.txt'},
                         {'weights': 'data/mean_module.base_means.6.weights.txt',
                          'bias': 'data/mean_module.base_means.6.bias.txt'},
                         {'weights': 'data/mean_module.base_means.7.weights.txt',
                          'bias': 'data/mean_module.base_means.7.bias.txt'}]},
                       'covar_module': 
                       {'data_covar_module': 
                        {'matern': {
                            'nu': 'data/covar_module.data_covar_module.matern.nu.txt',
                            'lengthscale': 'data/covar_module.data_covar_module.matern.lengthscale.txt'},
                         'scale': 'data/covar_module.data_covar_module.scale.txt',
                         'constant': 'data/covar_module.data_covar_module.constant.txt'},
                        'task_covar_module': 
                        {'covar_factor': 'data/covar_module.task_covar_module.covar_factor.txt',
                         'var': 'data/covar_module.task_covar_module.var.txt'}
                        }
                       }


def load_data(fname):
    return np.loadtxt(resource_filename('aemulusnu_hmf', fname))

train_x = load_data('data/train_x.txt')
train_y = load_data('data/train_y.txt')
raw_x = load_data('data/raw_x.txt')

in_scaler = Normalizer()
in_scaler.min_val = load_data('data/in_normalizer_min_vals.txt')
in_scaler.max_val = load_data('data/in_normalizer_max_vals.txt')


### MEAN MODULE
base_means = []
for i in range(len(train_y[0])):
    curr_weights = load_data(gp_parameters_fname['mean_module']['base_means'][i]['weights'])
    curr_bias    = load_data(gp_parameters_fname['mean_module']['base_means'][i]['bias'])
    base_means += [LinearMean(curr_weights, curr_bias)]
mean_module = MultitaskMean(base_means)


### COVAR MODULE

##### f^\Omega
nu = load_data(gp_parameters_fname['covar_module']['data_covar_module']['matern']['nu'])
lengthscale = load_data(gp_parameters_fname['covar_module']['data_covar_module']['matern']['lengthscale'])
f_matern = MaternKernel(nu, lengthscale)

scale = load_data(gp_parameters_fname['covar_module']['data_covar_module']['scale'])
f_scale = ScaleKernel(f_matern, scale)

constant = load_data(gp_parameters_fname['covar_module']['data_covar_module']['constant'])
f_constant = ConstantKernel(constant)

data_covar_module = AdditiveKernel(f_scale, f_constant)


##### f^\theta
covar_factor = load_data(gp_parameters_fname['covar_module']['task_covar_module']['covar_factor'])
var          = load_data(gp_parameters_fname['covar_module']['task_covar_module']['var'])
task_covar_module = IndexKernel(covar_factor, var)

##### full covariance
covar_module = MultitaskKernel(task_covar_module = task_covar_module,
                               data_covar_module = data_covar_module)



def _predict_with_GP(test_x):
    #this basically implements the computation of the mean
    #from Rasmussen and Williams (2006) Eq. (2.19)
    KXSX = covar_module(test_x, train_x)
    KXX = covar_module(train_x, train_x)

    #use cholesky decomposition to invert KXX
    L = np.linalg.cholesky(KXX)
    
    _y = train_y - mean_module(train_x)
    _y = _y.flatten()
    
    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, _y, lower=True))
    
    test_y  = np.reshape(np.einsum('ij,j->i', KXSX, alpha), (len(test_x), len(train_y[0])))
    test_y += mean_module(test_x)

    return test_y


@cache
def _predict_params_full(cosmo_vals_tuple):
    test_x = in_scaler.transform(np.array([cosmo_vals_tuple]))
    return _predict_with_GP(test_x = test_x)[0]

def predict_params_full(cosmo_dict):
    cosmo_vals = [cosmo_dict[curr_key] for curr_key in key_ordering]
    #we convert to tuple to make it hashable
    cosmo_vals_tuple = tuple(cosmo_vals)

    predicted_params = _predict_params_full(cosmo_vals_tuple)

    return predicted_params

def predict_params(cosmo_dict, a):
    curr_params = predict_params_full(cosmo_dict)
    paired_params = list(zip(curr_params, curr_params[1:]))[::2]

    param_at_z = {'d':-1, 'e':-1, 'f':-1, 'g':-1}
    for (p0,p1), key in zip(paired_params, param_at_z):
        param_at_z[key] = p(p0, p1, a)
            
    return param_at_z

@cache
def _create_cosmology(cosmo_vals_tuple):
    """
    tuple of parameters made from dict of parameters 
    outputs `aemulusnu_hmf.massfunction.cosmology` object
    """
    print('initializing cosmology. this will only happen once per cosmology')
    curr_cosmo_dict = dict(zip(key_ordering, cosmo_vals_tuple))
    return cosmology(curr_cosmo_dict)

def get_cosmology(cosmo_dict):
    cosmo_vals = [cosmo_dict[curr_key] for curr_key in key_ordering]
    cosmo_vals_tuple = tuple(cosmo_vals)

    return _create_cosmology(cosmo_vals_tuple)

def dn_dM(cosmo_dict, M, a):
    """ 
    cosmo_dict is a dictionary with entries
        - 10^9 As: As * 10^9
        - ns: Spectral index
        - H0: Hubble parameter in [km/s/Mpc]
        - w0: Dark Energy Equation fo State
        - ombh2: Ω_b h^2
        - omch2: Ω_m h^2
        - nu_mass_ev: Neutrino mass sum in [eV]
    M is the halo mass in units of Msol / h 
    a is the scale factor
    
    returns the mass function dn/dM in units h^4 / (Mpc^3  Msun)
    """
    z = scaleToRedshift(a)

    cosmo_vals = [cosmo_dict[curr_key] for curr_key in key_ordering]
    #we convert to tuple to make it hashable
    cosmo_vals_tuple = tuple(cosmo_vals)

    curr_cosmology = _create_cosmology(cosmo_vals_tuple)
    curr_parameters = predict_params(cosmo_dict, a)

    #cosmology dependent quantities
    sigma_cb = curr_cosmology.sigma_cb(M, z)
    d_ln_sigma_cb_dM = curr_cosmology.dln_sigma_cb_dM(M, z)
    rho_cb = curr_cosmology.f_rho_cb(0.0)

    #multiplicity function
    f = f_G(sigma_cb, **curr_parameters)

    return f * rho_cb/M * (-d_ln_sigma_cb_dM)


def multiplicity_function(cosmo_dict, sigma_cb, a):
    """ 
    cosmo_dict is a dictionary with entries
        - 10^9 As: As * 10^9
        - ns: Spectral index
        - H0: Hubble parameter in [km/s/Mpc]
        - w0: Dark Energy Equation fo State
        - ombh2: Ω_b h^2
        - omch2: Ω_m h^2
        - nu_mass_ev: Neutrino mass sum in [eV]
    sigma_cb is the variance of the smoothed CDM + baryon density field
    a is the scale factor
    
    returns the multiplicity function
    """
    z = scaleToRedshift(a)

    cosmo_vals = [cosmo_dict[curr_key] for curr_key in key_ordering]
    #we convert to tuple to make it hashable
    cosmo_vals_tuple = tuple(cosmo_vals)

    curr_cosmology = _create_cosmology(cosmo_vals_tuple)
    curr_parameters = predict_params(cosmo_dict, a)

    return f_G(sigma_cb, **curr_parameters)
