"""
theory_models/LCDM_neutrinos.py
--------------------------------
Flat ΛCDM + massive neutrinos + optional free N_eff.

Neutrino density follows the WMAP-7 Fermi-Dirac formalism
(Komatsu et al. 2011, Appendix C), tracking the full
relativistic→non-relativistic transition per species.

Configuration
-------------
Set the four flags below and run via run.py:

    SAMPLE_NU_MASS : bool   — sample Σmν (True) or fix it (False)
    SUM_M_FIXED    : float  — eV, used when SAMPLE_NU_MASS = False
    SAMPLE_NEFF    : bool   — sample N_eff (True) or fix it (False)
    NEFF_FIXED     : float  — used when SAMPLE_NEFF = False (SM = 3.044)

Usage via run.py
----------------
    python run.py --model LCDM_neutrinos
    mpirun -np 4 python run.py --model LCDM_neutrinos
"""

import numpy as np
from SERTAO.neutrinos import create_neutrinos

# ================================================================
# 1. Datasets
# ================================================================

DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN']

# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['Pantheon+SHOES', 'BAO_DESI', 'BBN']
# DATASETS = ['Union3',    'BAO_DESI', 'BBN']
# DATASETS = ['CC',        'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['RSD',       'BAO_DESI', 'BBN']

ANALYSIS_NAME = "LCDM_neutrinos"

# ================================================================
# 2. Neutrino configuration
# ================================================================

SAMPLE_NU_MASS = False      # True → sample Σmν; False → fix it
SUM_M_FIXED    = 0.06       # eV  (0.0 = massless)

SAMPLE_NEFF    = False      # True → sample N_eff; False → fix it
NEFF_FIXED     = 3.044      # SM prediction (Planck 2018 + QED corrections)

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
    PRIORS['sum_m'] = (0.06, 0.60)   # eV

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
# 4. H(z) model
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # Fixsen 2009

def _build_H_model():
    """
    Returns H_model(z, p) with the chosen neutrino treatment.

    When both sum_m and N_eff are fixed, Omega_nu0 is pre-computed
    once here and captured in the closure — zero overhead per call.
    When either is sampled, create_neutrinos() hits the lru_cache
    for repeated proposals.
    """
    _nu_fixed        = None
    _Omega_nu0_fixed = None

    if not SAMPLE_NU_MASS and not SAMPLE_NEFF:
        _nu_fixed = create_neutrinos(
            config     = tuple(NU_CONFIG) if isinstance(NU_CONFIG, list) else NU_CONFIG,
            total_mass = SUM_M_FIXED if SUM_M_FIXED > 1e-6 else None,
            N_eff      = NEFF_FIXED,
            h          = 0.6736,
        )
        _Omega_nu0_fixed = _nu_fixed.Omega_nu0

    def H_model(z, p):
        H0          = p['H0']
        h           = H0 / 100.0
        Omega_m     = p['Omega_cdm'] + p['Omega_b']
        Omega_gamma = _OMEGA_GAMMA_H2 / h**2

        neff  = p.get('Neff',  NEFF_FIXED)
        sum_m = p.get('sum_m', SUM_M_FIXED)

        if neff <= 0.0 or sum_m < 0.0:
            return np.nan

        if SAMPLE_NU_MASS or SAMPLE_NEFF:
            nu = create_neutrinos(
                config     = tuple(NU_CONFIG) if isinstance(NU_CONFIG, list) else NU_CONFIG,
                total_mass = sum_m if sum_m > 1e-6 else None,
                N_eff      = round(neff, 5),
                h          = round(h,    5),
            )
            Omega_nu_z = nu.contribution_to_H2(z)
            Omega_nu0  = nu.Omega_nu0
        else:
            Omega_nu_z = _nu_fixed.contribution_to_H2(z)
            Omega_nu0  = _Omega_nu0_fixed

        Omega_L = 1.0 - Omega_m - Omega_gamma - Omega_nu0
        if Omega_L < -0.1:
            return np.nan

        H2 = (Omega_m     * (1+z)**3
            + Omega_gamma * (1+z)**4
            + Omega_nu_z
            + Omega_L)

        if np.ndim(H2) == 0:
            return np.nan if H2 <= 0.0 else H0 * np.sqrt(H2)
        return H0 * np.sqrt(np.where(H2 > 0.0, H2, np.nan))

    return H_model


H_model = _build_H_model()
