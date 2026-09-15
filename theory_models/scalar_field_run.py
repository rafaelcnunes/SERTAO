"""
theory_models/scalar_field_run.py
----------------------------------
Quintessence scalar field — power-law potential V(φ) ∝ φⁿ.

Free parameters
---------------
H0, Omega_cdm, Omega_b   — background cosmology
lambda_phi               — initial slope λ = −Mpl V,φ/V
n_pot                    — potential exponent n

Usage
-----
    python run.py --model scalar_field_run
    mpirun -np 4 python run.py --model scalar_field_run
"""

import numpy as np
from functools import lru_cache

from SERTAO.scalar_field import QuintessenceDynamics, PowerLawPotential

# ================================================================
# Datasets
# ================================================================

DATASETS = ['BAO_DESI', 'BBN']

# Other combinations:
# DATASETS = ['Pantheon+', 'BAO_DESI', 'BBN', 'CMB θ*']
# DATASETS = ['CC', 'BAO_DESI', 'BBN']

ANALYSIS_NAME = "Quintessence_PowerLaw"

# ================================================================
# Priors
# ================================================================

PRIORS = {
    'H0':         (40.0, 90.0),
    'Omega_cdm':  (0.10, 0.50),
    'Omega_b':    (0.02, 0.06),
    'lambda_phi': (0.01, 2.0),   # initial slope λ
    'n_pot':      (1.1,  6.0),   # potential exponent n > 1
}

use_pp = any(d in DATASETS for d in ('Pantheon+', 'Pantheon+SHOES'))
use_u3 = 'Union3' in DATASETS

if use_pp and use_u3:
    raise ValueError("Pantheon+ and Union3 cannot be used simultaneously.")
if use_pp:
    PRIORS['M_B']  = (-21.0, -18.0)
if use_u3:
    PRIORS['Mcal'] = (-20.0, -17.0)
if any(d in DATASETS for d in ('RSD', 'f', 'FS_DESI')):
    PRIORS['sigma8'] = (0.5, 1.0)

# ================================================================
# Cached ODE solver
#
# The quintessence ODE takes ~7 ms per call.  Since the sampler
# proposes many points close to previously evaluated ones, a cache
# on quantized (λ, n) gives a large speedup.
#
# Quantization:
#   λ → 0.005 (grid of 400 points over [0.01, 2.0])
#   n → 0.05  (grid of ~100 points over [1.1, 6.0])
# This gives ~40,000 possible keys — well above maxsize=8000,
# so the LRU will retain the most-visited region of parameter space.
# ================================================================

_Q_LAM = 0.005
_Q_N   = 0.05

def _quantize(x, dx):
    return dx * round(x / dx)

@lru_cache(maxsize=8_000)
def _solve_quintessence(lam_q, n_q):
    """
    Solve the quintessence ODE for quantized (λ, n) and return
    an interpolator Omega_phi(z).

    The result is cached so repeated nearby proposals reuse the
    same solution without re-integrating.
    """
    potential = PowerLawPotential(n=n_q)
    dyn       = QuintessenceDynamics(potential)
    dyn.solve(
        N_ini=-7.0,
        N_fin=0.0,
        x_ini=1e-8,
        y_ini=1e-5,
        lam_ini=lam_q,
        npts=500,
    )
    # Return the interpolator directly so the caller can evaluate
    # Omega_phi at any z without re-interpolating.
    return dyn._Om_interp


# ================================================================
# Physical pre-check (fast, no ODE)
# ================================================================

def _pre_check(p):
    """
    Reject cosmologically unphysical points before running the ODE.
    Returning False causes the engine to assign logL = -inf immediately.
    """
    lam = p['lambda_phi']
    n   = p['n_pot']

    # Exponential attractor condition: λ² < 3(1+w_bg) ≈ 3
    # Beyond this the field dominates and w_φ → non-accelerating
    if lam**2 >= 3.0:
        return False

    # Avoid pathological slopes
    if lam < 1e-3:
        return False

    # Power-law requires n > 1 for a physical potential minimum
    if n <= 1.0:
        return False

    return True


# ================================================================
# H(z) model
# ================================================================

_OMEGA_GAMMA_H2 = 2.469e-5   # photon density today (Fixsen 2009)

def H_model(z, p):
    """
    Hubble rate for flat quintessence + matter + radiation.
    """
    H0      = p['H0']
    h       = H0 / 100.0
    Omega_m = p['Omega_cdm'] + p['Omega_b']
    Omega_r = _OMEGA_GAMMA_H2 / h**2
    lam     = p['lambda_phi']
    n       = p['n_pot']

    if Omega_m <= 0.0 or Omega_m >= 1.0:
        return np.nan

    # ΛCDM limit (field frozen)
    if lam < 1e-6:
        Omega_L = 1.0 - Omega_m - Omega_r
        H2 = Omega_m*(1+z)**3 + Omega_r*(1+z)**4 + Omega_L
        return np.nan if H2 <= 0 else H0*np.sqrt(H2)

    # Quantize and fetch cached interpolator
    lam_q = _quantize(lam, _Q_LAM)
    n_q   = _quantize(n,   _Q_N)

    try:
        Om_interp = _solve_quintessence(lam_q, n_q)
    except Exception:
        return np.nan

    # Evaluate Omega_phi at requested z
    z_scalar = np.ndim(z) == 0
    z_arr    = np.atleast_1d(np.asarray(z, dtype=float))
    Omega_phi = Om_interp(z_arr)

    # Guard: unphysical values
    bad = ~np.isfinite(Omega_phi) | (Omega_phi < 0.0)
    if np.any(bad):
        return np.nan

    H2 = Omega_m*(1+z_arr)**3 + Omega_r*(1+z_arr)**4 + Omega_phi
    H  = np.where(H2 > 0, H0*np.sqrt(H2), np.nan)

    return float(H[0]) if z_scalar else H


# ================================================================
# Pre-check wrapper for the engine
# ================================================================
pre_check = _pre_check
