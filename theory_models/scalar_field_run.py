import numpy as np
from functools import lru_cache

from SERTAO.engine import LikelihoodEngine
from SERTAO.MCMC import run_mcmc
from SERTAO.scalar_field import (
    QuintessenceDynamics,
    PowerLawPotential,
)

# ============================================================
# Helpers
# ============================================================
def quantize(x, dx):
    return dx * round(x / dx)

# ============================================================
# Fast physical priors (NO dynamics here)
# ============================================================

def fast_physical_priors(p):
    lam = p["lambda_phi"]
    n   = p["n_pot"]

    if lam**2 > 3.0:
        return False
    if lam < 1e-3:
        return False
    if n <= 1.0:
        return False

    return True


# ============================================================
# Quintessence solver (GLOBAL cached)
# ============================================================

@lru_cache(maxsize=2000)
def solve_quintessence_cached(lam_ini, n_pot):
    lam_ini = quantize(lam_ini, 0.01)
    n_pot   = quantize(n_pot, 0.05)

    potential = PowerLawPotential(n=n_pot)
    model = QuintessenceDynamics(potential)

    model.solve(
        N_ini=-7.0,
        N_fin=0.0,
        x_ini=1e-8,
        y_ini=1e-5,
        lam_ini=lam_ini,
        npts=600,
    )

    z = model.z()
    Omega_phi = model.Omega_phi()

    return z, Omega_phi


# ============================================================
# H(z) model
# ============================================================

def H_model(z, p):
    H0 = p["H0"]
    Omega_m = p["Omega_cdm"] + p["Omega_b"]

    if Omega_m <= 0.0 or Omega_m >= 1.0:
        return np.nan

    lam = p["lambda_phi"]

    # LambdaCDM limit
    if lam < 1e-6:
        return H0 * np.sqrt(
            Omega_m * (1.0 + z)**3 + (1.0 - Omega_m)
        )

    z_grid, Ophi_grid = solve_quintessence_cached(
        p["lambda_phi"],
        p["n_pot"]
    )

    Omega_phi = np.interp(
        z,
        z_grid[::-1],
        Ophi_grid[::-1],
        left=np.nan,
        right=np.nan,
    )

    if not np.isfinite(Omega_phi) or Omega_phi < 0.0:
        return np.nan

    return H0 * np.sqrt(
        Omega_m * (1.0 + z)**3 + Omega_phi
    )


# ============================================================
# DATASETS
# ============================================================
datasets = ['Pantheon+', 'BAO_DESI', 'BBN']

# ============================================================
# PRIORS
# ============================================================
priors = {
    "H0": (40.0, 90.0),
    "Omega_cdm": (0.10, 0.50),
    "Omega_b": (0.02, 0.06),

    # Quintessence
    "lambda_phi": (0.01, 2.0),
    "n_pot": (1.1, 6.0),
}

if "Union3" in datasets:
    priors["Mcal"] = (-20.0, -17.0)


# ============================================================
# LIKELIHOOD ENGINE
# ============================================================

engine = LikelihoodEngine(
    datasets=datasets,
    use_cache=True
)


def log_likelihood(theta):
    p = dict(zip(priors.keys(), theta))

    # Fast physical rejection
    if not fast_physical_priors(p):
        return -np.inf

    return engine.log_likelihood(
        theta=theta,
        priors=priors,
        H_model=H_model
    )


# ============================================================
# RUN
# ============================================================

run_mcmc(
    log_likelihood=log_likelihood,
    priors=priors,
    analysis_name="Quintessence_PowerLaw",
    datasets=datasets,
    nlive=300,
    dlogz=1.0,
    resume=True,
    output_dir="chains"
)











