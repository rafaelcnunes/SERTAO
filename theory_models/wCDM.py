"""
Flat wCDM: LCDM with a constant dark energy equation of state w ≠ -1.
Usage via run.py
----------------
    python run.py --model wCDM
    mpirun -np 4 python run.py --model wCDM
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
# DATASETS = ['DES_Dovekie', 'BAO_DESI', 'BBN']  # DES-Dovekie (M marginalized)
# DATASETS = ['CC',        'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['RSD',       'BAO_DESI', 'BBN']

ANALYSIS_NAME = "wCDM"

# ================================================================
# Priors
# ================================================================

PRIORS = {
    'H0':        (40.0, 90.0),
    'Omega_cdm': (0.10, 0.50),
    'Omega_b':   (0.02, 0.06),
    'w0':        (-3.0,  1.0),  
}

use_pp = any(d in DATASETS for d in ('Pantheon+', 'Pantheon+SHOES'))
use_u3 = 'Union3' in DATASETS

if sum([use_pp, use_u3, 'DES_Dovekie' in DATASETS]) > 1:
    raise ValueError("Pantheon+/SHOES, Union3 and DES_Dovekie cannot be used simultaneously.")
if use_pp:
    PRIORS['M_B']    = (-21.0, -18.0)
if use_u3:
    PRIORS['Mcal']   = (-20.0, -17.0)
if any(d in DATASETS for d in ('RSD', 'f', 'FS_DESI')):
    PRIORS['sigma8'] = (0.5, 1.0)

# ================================================================
# H(z) model
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # photon density today (Fixsen 2009)

def H_model(z, p):
    """
    Hubble rate for flat wCDM + photon radiation.

    H²(z) = H0² [Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_DE0·(1+z)^{3(1+w0)}]

    The DE density factor (1+z)^{3(1+w0)} is exact for constant w.
    No numerical integration needed.
    """
    H0      = p['H0']
    h       = H0 / 100.0
    Omega_m = p['Omega_cdm'] + p['Omega_b']
    Omega_r = _OMEGA_GAMMA_H2 / h**2
    w0      = p['w0']

    # Flat universe: Omega_DE absorbs the remainder at z=0
    Omega_DE0 = 1.0 - Omega_m - Omega_r

    # Guard: unphysical energy budget
    if Omega_DE0 <= 0.0:
        return np.nan

    # DE density factor: (1+z)^{3(1+w0)}
    # For w0 = -1 this reduces to 1 (cosmological constant).
    f_DE = (1.0 + z) ** (3.0 * (1.0 + w0))

    H2 = (
        Omega_m   * (1.0 + z)**3
        + Omega_r * (1.0 + z)**4
        + Omega_DE0 * f_DE
    )

    if np.ndim(H2) == 0:
        return np.nan if H2 <= 0.0 else H0 * np.sqrt(H2)
    return H0 * np.sqrt(np.where(H2 > 0.0, H2, np.nan))
