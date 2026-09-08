"""
Flat w0waCDM: LCDM with a dynamical dark energy equation of state
following the Chevallier–Polarski–Linder (CPL) parametrisation.

Usage via run.py
----------------
    python run.py --model w0waCDM
    mpirun -np 4 python run.py --model w0waCDM
"""

import numpy as np

# ================================================================
# Datasets
# ================================================================

DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN']

# Other common combinations:
# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['Pantheon+SHOES', 'BAO_DESI', 'BBN']
# DATASETS = ['Union3',    'BAO_DESI', 'BBN']
# DATASETS = ['CC',        'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['RSD',       'BAO_DESI', 'BBN']

ANALYSIS_NAME = "w0waCDM"

# ================================================================
# Priors
# ================================================================

PRIORS = {
    'H0':        (40.0, 90.0),
    'Omega_cdm': (0.10, 0.50),
    'Omega_b':   (0.02, 0.06),
    'w0':        (-3.0,  1.0),   # present-day DE equation of state
    'wa':        (-3.0,  2.0),   # DE equation of state time derivative
}

use_pp = any(d in DATASETS for d in ('Pantheon+', 'Pantheon+SHOES'))
use_u3 = 'Union3' in DATASETS

if use_pp and use_u3:
    raise ValueError("Pantheon+ and Union3 cannot be used simultaneously.")
if use_pp:
    PRIORS['M_B']    = (-21.0, -18.0)
if use_u3:
    PRIORS['Mcal']   = (-20.0, -17.0)
if any(d in DATASETS for d in ('RSD', 'f')):
    PRIORS['sigma8'] = (0.5, 1.0)

# ================================================================
# H(z) model — CPL parametrisation
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # photon density today (Fixsen 2009)

def H_model(z, p):
    """
    Hubble rate for flat w0waCDM (CPL)
    Limits:
        z → 0 :  f_DE → 1                      (by construction)
        wa = 0 :  f_DE → (1+z)^{3(1+w0)}       (wCDM)
        w0=-1, wa=0 : f_DE → 1                  (ΛCDM)
        z → ∞ :  f_DE → (1+z)^{3(1+w0+wa)}    (early-time behaviour)
    """
    H0      = p['H0']
    h       = H0 / 100.0
    Omega_m = p['Omega_cdm'] + p['Omega_b']
    Omega_r = _OMEGA_GAMMA_H2 / h**2
    w0      = p['w0']
    wa      = p['wa']

    # Flat universe: Omega_DE absorbs the remainder at z=0
    Omega_DE0 = 1.0 - Omega_m - Omega_r

    # Guard: unphysical energy budget
    if Omega_DE0 <= 0.0:
        return np.nan

    # CPL dark energy density factor (exact closed form)
    #   (1+z)^{3(1+w0+wa)} · exp(−3·wa·z/(1+z))
    exponent_power = 3.0 * (1.0 + w0 + wa)
    exponent_exp   = -3.0 * wa * z / (1.0 + z)

    f_DE = (1.0 + z)**exponent_power * np.exp(exponent_exp)

    H2 = (
        Omega_m     * (1.0 + z)**3
        + Omega_r   * (1.0 + z)**4
        + Omega_DE0 * f_DE
    )

    if np.ndim(H2) == 0:
        return np.nan if H2 <= 0.0 else H0 * np.sqrt(H2)
    return H0 * np.sqrt(np.where(H2 > 0.0, H2, np.nan))
