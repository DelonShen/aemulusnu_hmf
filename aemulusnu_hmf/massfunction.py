from scipy.special import gamma
from scipy.interpolate import interp1d, UnivariateSpline
from classy import Class
from .utils import scaleToRedshift, redshiftToScale

import numpy as np


_G_ = 6.67428e-11  # Newton constant in m^3/Kg/s^2 (CLASS precision)
_c_ = 2.99792458e8 # speed of light in m/s         (CLASS precision)
G_over_c2 = 4.78538e-20 #Mpc / Msol

class cosmology:
    def __init__(self, cosmology):
        self.cosmology = cosmology
        self.h = self.cosmology['H0']/100


        h = self.cosmology['H0']/100
        cosmo_dict = {
            'h': h,
            'Omega_b': self.cosmology['ombh2'] / h**2,
            'Omega_cdm': self.cosmology['omch2'] / h**2,
            'N_ur': 0.00641,
            'N_ncdm': 1,
            'output': 'mPk mTk',
            'z_pk': '0.0,99',
            'P_k_max_h/Mpc': 20.,
            'm_ncdm': self.cosmology['nu_mass_ev']/3,
            'deg_ncdm': 3,
            'ncdm_quadrature_strategy': 2,
            'T_cmb': 2.7255,
            'A_s': self.cosmology['10^9 As'] * 10**-9,
            'n_s': self.cosmology['ns'],
            'Omega_Lambda': 0.0,
            'w0_fld': self.cosmology['w0'],
            'wa_fld': 0.0,
            'cs2_fld': 1.0,
            'fluid_equation_of_state': "CLP"
        }


        self.pkclass = Class()
        self.pkclass.set(cosmo_dict)
        self.pkclass.compute()

        self.background = self.pkclass.get_background()
        self.z_bg = self.background['z']

        # conversion factor converts rho from CLASS background.c
        # which is in natural units to h^2 Msol / Mpc^3
        conversion_factor = 3 / (8 * np.pi * G_over_c2) / h / h 

        self.rho_c_natural = self.background['(.)rho_cdm']
        self.rho_b_natural = self.background['(.)rho_b']
        self.rho_crit_natural = self.background['(.)rho_crit']


        # rho_cb in units of h^2 Msol / Mpc^3
        self.rho_cb = (self.rho_c_natural + self.rho_b_natural) * conversion_factor
        self.f_rho_cb = interp1d(x = self.z_bg, y = self.rho_cb)

        # rho_m,0
        self.Omega_m = self.pkclass.get_current_derived_parameters(['Omega_m'])['Omega_m']
        self.rho_m_0 = self.rho_crit_natural[-1] * self.Omega_m * conversion_factor

        # empty dictionary to store splines
        self.f_dln_sigma_cb = {}
        self.f_sigmas_cb = {}

        self.f_dln_sigma_m = {}
        self.f_sigmas_m = {}



    def Pm(self, k, z):
        """
        k in units of h/Mpc
        z is reddshift


        returns linear matter power spectrum in units of Mpc^3 / h^3
        """
        return np.array([self.pkclass.pk_lin(k_curr * self.h, np.array([z])) for k_curr in k]) * self.h ** 3

    def Pcb(self, k, z):
        """
        k in units of h/Mpc
        z is reddshift


        returns linear matter power spectrum in units of Mpc^3 / h^3
        """
        return np.array([self.pkclass.pk_cb_lin(k_curr * self.h, np.array([z])) for k_curr in k]) * self.h ** 3

    def _sigma_cb(self, M, z):
        """
        M in units Msol / h
            gets converted to Lagrangian scale of halo R with
            R = (3 M / 4 pi rho_cb,0) ** (1/3) # Mpc / h
        z is redshift

        returns sigma_cb 
        """
        M_use = np.atleast_1d(M)
        R = (3 * M_use / (4  * np.pi * self.f_rho_cb(0.0))) ** (1/3) # Mpc / h
        return np.array([self.pkclass.sigma_cb(R = R_curr, z = z, h_units = True) for R_curr in R])

    def f_sigma_cb(self, z):
        """
        z is redshift

        returns sigma interpolation using `scipy.interpolate.UnivariateSpline`
        """
        if(z in self.f_sigmas_cb):
            return self.f_sigmas_cb[z]

        Msample = np.logspace(12, 17, 1000)
        sigma_cb_samples = self._sigma_cb(M = Msample, z = z)

        f_log_sigma_cb = UnivariateSpline(x = np.log(Msample), y = np.log(sigma_cb_samples))

        self.f_sigmas_cb[z] = f_log_sigma_cb
        return self.f_sigmas_cb[z]


    def sigma_cb(self, M, z):
        """
        M in units Msol / h
            gets converted to Lagrangian scale of halo R with
            R = (3 M / 4 pi rho_cb,0) ** (1/3) # Mpc / h
        z is redshift

        returns sigma_cb using spline
        """
        return np.exp((self.f_sigma_cb(z))(np.log(M)))

 
    def f_dln_sigma_cb_dM(self, z):
        """
        z is redshift

        returns dln(sigma_cb)/dM spline using `scipy.interpolate.UnivariateSpline`
        """
        if(z in self.f_dln_sigma_cb):
            return self.f_dln_sigma_cb[z]


        f_log_sigma_cb = self.f_sigma_cb(z)
        f_dlog_sigma_cb_dlog_M = f_log_sigma_cb.derivative()

        #d log sig / d log M = M * d log sig / dM 
        self.f_dln_sigma_cb[z] = lambda M: 1/M * f_dlog_sigma_cb_dlog_M(np.log(M))

        return self.f_dln_sigma_cb[z]

    def dln_sigma_cb_dM(self, M, z):
        """
        M in units Msol / h
        z is redshift

        returns dln(sigma_cb)/dM for this mass and redshift in units h / Msol
        """
        return (self.f_dln_sigma_cb_dM(z))(M)

    def _sigma_m(self, M, z):
        """
        M in units Msol / h
            gets converted to Lagrangian scale of halo R with
            R = (3 M / 4 pi rho_m,0) ** (1/3) # Mpc / h
        z is redshift

        returns sigma_m 
        """
        M_use = np.atleast_1d(M)
        R = (3 * M_use / (4  * np.pi * self.rho_m_0)) ** (1/3) # Mpc / h
        return np.array([self.pkclass.sigma(R = R_curr, z = z, h_units = True) for R_curr in R])

    def f_sigma_m(self, z):
        """
        z is redshift

        returns sigma interpolation using `scipy.interpolate.UnivariateSpline`
        """
        if(z in self.f_sigmas_m):
            return self.f_sigmas_m[z]

        Msample = np.logspace(12, 17, 1000)
        sigma_m_samples = self._sigma_m(M = Msample, z = z)

        f_log_sigma_m = UnivariateSpline(x = np.log(Msample), y = np.log(sigma_m_samples))

        self.f_sigmas_m[z] = f_log_sigma_m
        return self.f_sigmas_m[z]


    def sigma_m(self, M, z):
        """
        M in units Msol / h
            gets converted to Lagrangian scale of halo R with
            R = (3 M / 4 pi rho_m,0) ** (1/3) # Mpc / h
        z is redshift

        returns sigma_m using spline
        """
        return np.exp((self.f_sigma_m(z))(np.log(M)))

 
    def f_dln_sigma_m_dM(self, z):
        """
        z is redshift

        returns dln(sigma_m)/dM spline using `scipy.interpolate.UnivariateSpline`
        """
        if(z in self.f_dln_sigma_m):
            return self.f_dln_sigma_m[z]


        f_log_sigma_m = self.f_sigma_m(z)
        f_dlog_sigma_m_dlog_M = f_log_sigma_m.derivative()

        #d log sig / d log M = M * d log sig / dM 
        self.f_dln_sigma_m[z] = lambda M: 1/M * f_dlog_sigma_m_dlog_M(np.log(M))

        return self.f_dln_sigma_m[z]

    def dln_sigma_m_dM(self, M, z):
        """
        M in units Msol / h
        z is redshift

        returns dln(sigma_m)/dM for this mass and redshift in units h / Msol
        """
        return (self.f_dln_sigma_m_dM(z))(M)



def B(d, e, f, g):
    oup = e**(d)*g**(-d/2)*gamma(d/2)
    oup += g**(-f/2)*gamma(f/2)
    return 2/oup


def f_G(σM, d, e, f, g):
    oup = B(d, e, f, g)
    oup *= ((σM/e)**(-d)+σM**(-f))
    oup *= np.exp(-g/σM**2)
    return oup

def p(p0, p1, a):
    return p0 + (a-0.5)*p1
