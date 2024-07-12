from pyccl import ccllib as lib
from pyccl.pk2d import *

from pyccl import check
import numpy as np

def custom_get_camb_pk_lin(cosmo, *, nonlin=False):
    """Run CAMB and return the linear CDM+Baryon spectrum.
    Used to implement Costanzi et al. 2013 (JCAP12(2013)012) 
    perscription to evaluate HMF in nuCDM cosmology in the
    Aemulus-nu HMF emulator

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological
            parameters. The cosmological parameters with
            which to run CAMB.
        nonlin (:obj:`bool`): Whether to compute and return the
            non-linear power spectrum as well.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: Power spectrum object. The linear power \
            spectrum. If ``nonlin=True``, returns a tuple \
            ``(pk_lin, pk_nonlin)``.
    """
    import camb
    import camb.model

    # Get extra CAMB parameters that were specified
    extra_camb_params = {}
    try:
        extra_camb_params = cosmo["extra_parameters"]["camb"]
    except (KeyError, TypeError):
        pass

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status, cosmo=cosmo)
    a_arr = np.sort(a_arr)
    zs = 1.0 / a_arr - 1
    zs = np.clip(zs, 0, np.inf)

    # deal with normalization
    if np.isfinite(cosmo["A_s"]):
        A_s_fid = cosmo["A_s"]
    elif np.isfinite(cosmo["sigma8"]):
        A_s_fid = 2.1e-9
        sigma8_target = cosmo["sigma8"]
    else:
        raise CCLError(
            "Could not normalize the linear power spectrum. "
            "A_s = %f, sigma8 = %f" % (
                cosmo['A_s'], cosmo['sigma8']))

    # init camb params
    cp = camb.model.CAMBparams()

    # turn some stuff off
    cp.WantCls = False
    cp.DoLensing = False
    cp.Want_CMB = False
    cp.Want_CMB_lensing = False
    cp.Want_cl_2D_array = False
    cp.WantTransfer = True

    # basic background stuff
    h2 = cosmo['h']**2
    cp.H0 = cosmo['h'] * 100
    cp.ombh2 = cosmo['Omega_b'] * h2
    cp.omch2 = cosmo['Omega_c'] * h2
    cp.omk = cosmo['Omega_k']

    # "constants"
    cp.TCMB = cosmo['T_CMB']

    # neutrinos
    # We maually setup the CAMB neutrinos to match the adjustments CLASS
    # makes to their temperatures.
    cp.share_delta_neff = False
    cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
    cp.num_nu_massless = cosmo['N_nu_rel']
    cp.num_nu_massive = int(cosmo['N_nu_mass'])
    cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])

    delta_neff = cosmo['Neff'] - 3.044  # used for BBN YHe comps

    # CAMB defines a neutrino degeneracy factor as T_i = g^(1/4)*T_nu
    # where T_nu is the standard neutrino temperature from first order
    # computations
    # CLASS defines the temperature of each neutrino species to be
    # T_i_eff = T_ncdm * T_cmb where T_ncdm is a fudge factor to get the
    # total mass in terms of eV to match second-order computations of the
    # relationship between m_nu and Omega_nu.
    # We are trying to get both codes to use the same neutrino temperature.
    # thus we set T_i_eff = T_i = g^(1/4) * T_nu and solve for the right
    # value of g for CAMB. We get g = (T_ncdm / (11/4)^(-1/3))^4
    g = np.power(
        cosmo["T_ncdm"] / np.power(11.0/4.0, -1.0/3.0),
        4.0)

    if cosmo['N_nu_mass'] > 0:
        nu_mass_fracs = cosmo['m_nu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)

        cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
        cp.nu_mass_fractions = nu_mass_fracs
        cp.nu_mass_degeneracies = np.ones(int(cosmo['N_nu_mass'])) * g
    else:
        cp.nu_mass_numbers = []
        cp.nu_mass_fractions = []
        cp.nu_mass_degeneracies = []

    # get YHe from BBN
    cp.bbn_predictor = camb.bbn.get_predictor()
    cp.YHe = cp.bbn_predictor.Y_He(
        cp.ombh2 * (camb.constants.COBE_CMBTemp / cp.TCMB) ** 3,
        delta_neff)

    camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
    camb_de_model = extra_camb_params.get('dark_energy_model', 'fluid')
    if camb_de_model not in camb_de_models:
        raise ValueError("The only dark energy models CCL supports with "
                         "CAMB are fluid and ppf.")
    cp.set_classes(
        dark_energy_model=camb_de_model
    )

    if camb_de_model not in camb_de_models[:2] and cosmo['wa'] and \
            (cosmo['w0'] < -1 - 1e-6 or
                1 + cosmo['w0'] + cosmo['wa'] < - 1e-6):
        raise ValueError("If you want to use w crossing -1,"
                         " then please set the dark_energy_model to ppf.")
    cp.DarkEnergy.set_params(
        w=cosmo['w0'],
        wa=cosmo['wa']
    )

    cp.set_for_lmax(extra_camb_params.get("lmax", 5000))
    cp.InitPower.set_params(
        As=A_s_fid,
        ns=cosmo['n_s'])

    cp.set_matter_power(
        redshifts=[_z for _z in zs],
        kmax=extra_camb_params.get("kmax", 10.0))

    camb_res = camb.get_transfer_functions(cp)

    def construct_Pk2D(camb_res, nonlin=False):
        k, z, pk = camb_res.get_linear_matter_power_spectrum(
            var1='delta_nonu', var2='delta_nonu',
            hubble_units=True, nonlinear=nonlin)

        # convert to non-h inverse units
        k *= cosmo['h']
        pk /= (h2 * cosmo['h'])

        # now build interpolant
        nk = k.shape[0]
        lk_arr = np.log(k)
        a_arr = 1.0 / (1.0 + z)
        na = a_arr.shape[0]
        sinds = np.argsort(a_arr)
        a_arr = a_arr[sinds]
        ln_p_k_and_z = np.zeros((na, nk), dtype=np.float64)
        for i, sind in enumerate(sinds):
            ln_p_k_and_z[i, :] = np.log(pk[sind, :])

        pk = Pk2D(
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=ln_p_k_and_z,
            is_logp=True,
            extrap_order_lok=1,
            extrap_order_hik=2)

        return pk

    if np.isfinite(cosmo["sigma8"]):
        camb_res.calc_power_spectra()
        pk = construct_Pk2D(camb_res, nonlin=False)
        sigma8_tmp = sigma8(cosmo, p_of_k_a=pk)
        camb_res.Params.InitPower.As *= sigma8_target**2 / sigma8_tmp**2

    if nonlin:
        camb_res.Params.NonLinear = camb.model.NonLinear_pk
        camb_res.Params.NonLinearModel = camb.nonlinear.Halofit()
        halofit_version = extra_camb_params.get("halofit_version", "mead")
        options = {k: extra_camb_params[k] for k in
                   ["HMCode_A_baryon",
                    "HMCode_eta_baryon",
                    "HMCode_logT_AGN"] if k in extra_camb_params}
        camb_res.Params.NonLinearModel.set_params(
            halofit_version=halofit_version,
            **options)
    else:
        assert camb_res.Params.NonLinear == camb.model.NonLinear_none

    camb_res.calc_power_spectra()
    pk_lin = construct_Pk2D(camb_res, nonlin=False)

    if not nonlin:
        return pk_lin
    else:
        pk_nonlin = construct_Pk2D(camb_res, nonlin=True)

        return pk_lin, pk_nonlin


def custom_compute_linear_power(self):
    """Return the linear power spectrum.
    Customized so that it returns the P_{CDM+Baryons}
    instead of P_m

    This is done to implement the Costanzi et al. 2013 (JCAP12(2013)012) 
    perscription to evaluate HMF in nuCDM cosmology
    """
    self.compute_growth()

    # Populate power spectrum splines
    trf = self.transfer_function_type
    pk = None
    rescale_s8 = True
    rescale_mg = True
    if trf == "boltzmann_camb":
        rescale_s8 = False
        # For MG, the input sigma8 includes the effects of MG, while the
        # sigma8 that CAMB uses is the GR definition. So we need to rescale
        # sigma8 afterwards.
        if self.mg_parametrization.mu_0 != 0:
            rescale_s8 = True

    elif trf in ['bbks', 'eisenstein_hu', 'eisenstein_hu_nowiggles', 'boltzmann_isitgr', 'emulator', 'boltzmann_class']:
        raise NotImplementedError("Using Aemulus Nu HMF with `%s`\nis currently not implemented\nPlease use `boltzmann_camb`"%trf)

    # Compute the CAMB nonlin power spectrum if needed,
    # to avoid repeating the code in `compute_nonlin_power`.
    # Because CAMB power spectra come in pairs with pkl always computed,
    # we set the nonlin power spectrum first, but keep the linear via a
    # status variable to use it later if the transfer function is CAMB too.
    pkl = None
    if self.matter_power_spectrum_type == "camb":
        rescale_mg = False
        if self.mg_parametrization.mu_0 != 0:
            raise ValueError("Can't rescale non-linear power spectrum "
                             "from CAMB for mu-Sigma MG.")
        name = "delta_matter:delta_matter"
        pkl, self._pk_nl[name] = custom_get_camb_pk_lin(self, nonlin=True) #TODO

    if trf == "boltzmann_camb":
        pk = pkl if pkl is not None else custom_get_camb_pk_lin(self) #TODO

    # Rescale by sigma8/mu-sigma if needed
    if pk:
        status = 0
        status = lib.rescale_linpower(self.cosmo, pk.psp,
                                      int(rescale_mg),
                                      int(rescale_s8),
                                      status)
        check(status, self)

    return pk



