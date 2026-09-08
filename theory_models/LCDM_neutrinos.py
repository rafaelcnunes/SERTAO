"""
Flat LCDM + massive neutrinos + optional free N_eff.
"""

import numpy as np
from SERTAO.neutrinos import create_neutrinos

# ================================================================
# 1. Datasets
# ================================================================

DATASETS = ['CC', 'BAO_DESI', 'BBN', 'CMB θ*']

# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN']
# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['Pantheon+SHOES', 'BAO_DESI', 'BBN']
# DATASETS = ['Union3',    'BAO_DESI', 'BBN']
# DATASETS = ['RSD',       'BAO_DESI', 'BBN']

ANALYSIS_NAME = "LCDM_neutrinos"

# ================================================================
# 2. Neutrino configuration
# ================================================================

SAMPLE_NU_MASS = True       # True → sample Σmν; False → fix it
SUM_M_FIXED    = 0.06       # eV  (used when SAMPLE_NU_MASS = False)

SAMPLE_NEFF    = False      # True → sample N_eff; False → fix it
NEFF_FIXED     = 3.044      # SM prediction

NU_CONFIG      = 'standard' # 'standard' | '1ncdm' | 'hierarchy_normal'
                             # | 'hierarchy_inverted' | (m1, m2, m3)

# ================================================================
# 3. Priors
# ================================================================

PRIORS = {
    'H0':        (40.0, 90.0),
    'Omega_cdm': (0.10, 0.50),
    'Omega_b':   (0.02, 0.06),
}

if SAMPLE_NU_MASS:
    PRIORS['sum_m'] = (0.06, 0.60)

if SAMPLE_NEFF:
    PRIORS['Neff'] = (1.0, 5.0)

use_pp = any(d in DATASETS for d in ('Pantheon+', 'Pantheon+SHOES'))
use_u3 = 'Union3' in DATASETS

if use_pp and use_u3:
    raise ValueError("Pantheon+ and Union3 cannot be used simultaneously.")
if use_pp:
    PRIORS['M_B']  = (-21.0, -18.0)
if use_u3:
    PRIORS['Mcal'] = (-20.0, -17.0)
if any(d in DATASETS for d in ('RSD', 'f')):
    PRIORS['sigma8'] = (0.5, 1.0)

# ================================================================
# H(z) model
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # photon density today


class _NeutrinoHModel:
    """
    Pickle-safe H(z) callable for LCDM + massive neutrinos.
    """

    def __init__(
        self,
        sample_nu_mass, sum_m_fixed,
        sample_neff,    neff_fixed,
        nu_config,
    ):
        self.sample_nu_mass = sample_nu_mass
        self.sum_m_fixed    = sum_m_fixed
        self.sample_neff    = sample_neff
        self.neff_fixed     = neff_fixed
        self.nu_config      = nu_config

        # Pre-compute fixed neutrino object when both are fixed —
        # captured once here, reused at every call with zero overhead.
        self._nu_fixed        = None
        self._Omega_nu0_fixed = None

        if not sample_nu_mass and not sample_neff:
            self._nu_fixed = create_neutrinos(
                config     = tuple(nu_config) if isinstance(nu_config, list) else nu_config,
                total_mass = sum_m_fixed if sum_m_fixed > 1e-6 else None,
                N_eff      = neff_fixed,
                h          = 0.6736,
            )
            self._Omega_nu0_fixed = self._nu_fixed.Omega_nu0

    def __call__(self, z, p):
        H0          = p['H0']
        h           = H0 / 100.0
        Omega_m     = p['Omega_cdm'] + p['Omega_b']
        Omega_gamma = _OMEGA_GAMMA_H2 / h**2

        neff  = p.get('Neff',  self.neff_fixed)
        sum_m = p.get('sum_m', self.sum_m_fixed)

        if neff <= 0.0 or sum_m < 0.0:
            return np.nan

        if self.sample_nu_mass or self.sample_neff:
            nu = create_neutrinos(
                config     = tuple(self.nu_config) if isinstance(self.nu_config, list) else self.nu_config,
                total_mass = sum_m if sum_m > 1e-6 else None,
                N_eff      = round(neff, 5),
                h          = round(h,    5),
            )
            Omega_nu0 = nu.Omega_nu0
            # density_parameter() returns scalar; vectorise over z if needed
            if np.ndim(z) == 0:
                Omega_nu_z = nu.contribution_to_H2(float(z))
            else:
                Omega_nu_z = np.array([nu.contribution_to_H2(float(zi)) for zi in z])
        else:
            Omega_nu0 = self._Omega_nu0_fixed
            if np.ndim(z) == 0:
                Omega_nu_z = self._nu_fixed.contribution_to_H2(float(z))
            else:
                Omega_nu_z = np.array([self._nu_fixed.contribution_to_H2(float(zi)) for zi in z])

        Omega_L = 1.0 - Omega_m - Omega_gamma - Omega_nu0
        if Omega_L < -0.1:
            return np.nan

        H2 = (
            Omega_m     * (1.0 + z)**3
            + Omega_gamma * (1.0 + z)**4
            + Omega_nu_z
            + Omega_L
        )

        if np.ndim(H2) == 0:
            return np.nan if H2 <= 0.0 else H0 * np.sqrt(H2)
        return H0 * np.sqrt(np.where(H2 > 0.0, H2, np.nan))


# Instantiate at module level — pickle can locate this object
# because it references the class _NeutrinoHModel defined above,
# which is importable from theory_models.LCDM_neutrinos
H_model = _NeutrinoHModel(
    sample_nu_mass = SAMPLE_NU_MASS,
    sum_m_fixed    = SUM_M_FIXED,
    sample_neff    = SAMPLE_NEFF,
    neff_fixed     = NEFF_FIXED,
    nu_config      = NU_CONFIG,
)
