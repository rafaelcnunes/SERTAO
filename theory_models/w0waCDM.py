"""
Flat w0waCDM: ΛCDM with a dynamical dark energy equation of state
following the Chevallier–Polarski–Linder (CPL) parametrisation.

Dark energy equation of state (CPL)
------------------------------------
    w(a) = w0 + wa · (1 − a) = w0 + wa · z/(1+z)

    w0 : present-day value  (recovered at a=1, z=0)
    wa : rate of change      (w → w0+wa at a→0, z→∞)

    ΛCDM limit: w0 = -1, wa = 0

Dark energy density evolution
------------------------------
Integrating the continuity equation exactly:

    ρ_DE(z) / ρ_DE(0) = (1+z)^{3(1+w0+wa)} · exp(−3·wa·z/(1+z))

This closed-form result avoids numerical integration of ρ_DE and
is numerically stable for all z ≥ 0 and all (w0, wa) values.

Derivation:
    d ln ρ_DE / d ln(1+z) = 3[1 + w(z)]
    ∫₀ᶻ [1 + w0 + wa·z'/(1+z')] / (1+z') dz'
        = (1+w0+wa)·ln(1+z) − wa·z/(1+z)

Flatness constraint:
    Ω_DE0 = 1 − Ω_m − Ω_r   (derived, not sampled)

Free parameters
---------------
H0, Omega_cdm, Omega_b, w0, wa
+ nuisance: M_B (Pantheon+), Mcal (Union3), sigma8 (RSD/f)

Physical priors on (w0, wa)
----------------------------
The standard ranges used in DESI DR2 (arXiv:2503.14738) and
Sabogal & Nunes 2025 (arXiv:2505.24465) are:
    w0 ∈ (−3, 1)
    wa ∈ (−3, 2)

Note: No additional flatness cut on (w0, wa) is applied here;
the engine rejects unphysical H²<0 via the np.nan guard.

Usage via run.py
----------------
    python run.py --model w0waCDM
    mpirun -np 4 python run.py --model w0waCDM
"""

import numpy as np

# ================================================================
# 1. Datasets
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
# 2. Priors
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
# 3. H(z) model — CPL parametrisation
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # photon density today (Fixsen 2009)

def H_model(z, p):
    """
    Hubble rate for flat w0waCDM (CPL) + photon radiation.

    H²(z) = H0² [Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_DE0·f_DE(z)]

    where the CPL dark energy density factor is:

        f_DE(z) = (1+z)^{3(1+w0+wa)} · exp(−3·wa·z/(1+z))

    This is exact — no numerical integration required.

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
