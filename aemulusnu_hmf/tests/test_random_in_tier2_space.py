from aemulusnu_hmf.emulator import predict_params_full, load_data
import numpy as np


random_x = load_data('data/random_tier2_input.txt')
random_y = load_data('data/random_tier2_output.txt')
key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev']

def _test_individual(i, cosmo_vals):
    cosmo_params = dict(zip(key_ordering, cosmo_vals))
    y = predict_params_full(cosmo_params)

    assert(np.allclose(y, random_y[i], rtol=1e-5, atol = 1e-4))


def test_at_random():
    for i,cosmo in enumerate(random_x):
        _test_individual(i, cosmo)




