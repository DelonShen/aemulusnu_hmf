# Aemulus ν: Precision halo mass functions in wνCDM cosmologies
This package contains the halo mass function emulator described in [ [24XX.XXXXX]](TODO) for cluster mass scales ($\gtrsim 10^{13}M_\odot  / h$) up to redshift $z \eqsim 2$ with support for the whole space of $w\nu {\rm CDM}$ cosmologies allowed by current data. This emulator is built from measurements of halo abundances in the Aemulus $\nu$ suite of simulations described in [ [2303.09762]](https://arxiv.org/abs/2303.09762).

The repository with all the numerical studies and code to build this emulator can be found [here](https://github.com/DelonShen/aemulusnu_hmf_development)

---

*If you have any questions at all about the code, I'm always open to chat, just let me know at [delon@stanford.edu](mailto:delon@stanford.edu)*

## Dependencies
- `numpy`
- `scipy`
- `classy`
- `pytest`

## Installation 
Download this repository and in the base directory run
```
python setup.py install
```
To validate your installation, you can run
```
pytest
```

# Getting Started
For a thorough introduction, see `Tutorial.ipynb`. For something simple:
```python
from aemulusnu_hmf.emulator import dn_dM
cosmology = {'ns': 0.97,
 'H0': 67.0,
 'w0': -1.0,
 'ombh2': 0.0223,
 'omch2': 0.12,
 'nu_mass_ev': 0.07,
 '10^9 As': 2.1}

Ms = np.logspace(12, 16, 100) #in units of Msol / h
mass_function = dn_dM(cosmology, Ms, a = 1.0) #in units of h^4 / (Mpc^3  Msol)
```

# Citation
If you use the Aemulus $\nu$ halo mass function emulator, please cite [ [24XX.XXXXX]](TODO) and [ [2303.09762]](https://arxiv.org/abs/2303.09762)

```
@article{Shen:2024XXX, 
TODO inspire entry}
```

```
@article{DeRose:2023dmk,
    author = "DeRose, Joseph and Kokron, Nickolas and Banerjee, Arka and Chen, Shi-Fan and White, Martin and Wechsler, Risa and Storey-Fisher, Kate and Tinker, Jeremy and Zhai, Zhongxu",
    title = "{Aemulus \ensuremath{\nu}: precise predictions for matter and biased tracer power spectra in the presence of neutrinos}",
    eprint = "2303.09762",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2023/07/054",
    journal = "JCAP",
    volume = "07",
    pages = "054",
    year = "2023"
}
```
