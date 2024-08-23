from aemulusnu_hmf.emulator import load_data
from aemulusnu_hmf.utils import Normalizer


import numpy as np


train_x = load_data('data/train_x.txt')
train_y = load_data('data/train_y.txt')
raw_x = load_data('data/raw_x.txt')

in_scaler = Normalizer()
in_scaler.min_val = load_data('data/in_normalizer_min_vals.txt')
in_scaler.max_val = load_data('data/in_normalizer_max_vals.txt')


def test_inscaler():
    np.allclose(in_scaler.transform(raw_x), train_x,
                atol = 0, rtol = 1e-8)
