"""
Flat LCDM with photon radiation.
Usage via run.py
----------------
    python run.py --model LCDM
    mpirun -np 4 python run.py --model LCDM
"""

import numpy as np

# ================================================================
# Datasets
# ================================================================

DATASETS = ['BAO_DESI', 'BBN']

# Other common combinations:
# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN']
# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['Pantheon+SHOES', 'BAO_DESI', 'BBN']
# DATASETS = ['Union3',    'BAO_DESI', 'BBN']
# DATASETS = ['CC',        'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['RSD',       'BAO_DESI', 'BBN']

ANALYSIS_NAME = "LCDM"

# ================================================================
# Priors
# ================================================================

PRIORS = {
    'H0':        (40.0, 90.0),
    'Omega_cdm': (0.10, 0.50),
    'Omega_b':   (0.02, 0.06),
}

use_pp = any(d in DATASETS for d in ('Pantheon+', 'Pantheon+SHOES'))
use_u3 = 'Union3' in DATASETS

if use_pp and use_u3:
    raise ValueError("Pantheon+ and Union3 cannot be used simultaneously.")
if use_pp:
    PRIORS['M_B'] = (-21.0, -18.0)
if use_u3:
    PRIORS['Mcal'] = (-20.0, -17.0)
if any(d in DATASETS for d in ('RSD', 'f')):
    PRIORS['sigma8'] = (0.5, 1.0)

# ================================================================
# H(z) model
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # Fixsen 2009

def H_model(z, p):
    H0      = p['H0']
    h       = H0 / 100.0
    Omega_m = p['Omega_cdm'] + p['Omega_b']
    Omega_r = _OMEGA_GAMMA_H2 / h**2
    Omega_L = 1.0 - Omega_m - Omega_r
    H2 = Omega_m*(1+z)**3 + Omega_r*(1+z)**4 + Omega_L
    if np.ndim(H2) == 0:
        return np.nan if H2 <= 0.0 else H0 * np.sqrt(H2)
    return H0 * np.sqrt(np.where(H2 > 0.0, H2, np.nan))
