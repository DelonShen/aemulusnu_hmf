from aemulusnu_hmf.emulator import dn_dM, load_data
import numpy as np

random_x = load_data('data/random_tier2_dn_dM_input.txt')
random_y = load_data('data/random_tier2_dn_dM_output.txt')
key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev']

Ms = np.logspace(12, 16, 1000)
a_s = np.linspace(0.33, 1, 100)

def _test_individual(i, cosmo_vals):
    cosmo_params = dict(zip(key_ordering, cosmo_vals))
    curr_vals = []
    for a in a_s:
        curr_vals += [dn_dM(cosmo_params, Ms, a)]
    curr_vals = np.array(curr_vals)
    y = np.reshape(curr_vals, (1000 * 100))
    assert(np.allclose(y, random_y[i], rtol=1e-8))


def test_at_random():
    for i,cosmo in enumerate(random_x):
        _test_individual(i, cosmo)
