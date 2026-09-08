"""
Example usage of SERTAO's HybridCosmology.

This example demonstrates how to combine:

    - a custom background expansion H(z),
    - SERTAO background quantities,
    - SERTAO linear growth,
    - CLASS linear matter power spectrum,
    - CLASS nonlinear matter power spectrum.

The same sigma8 normalization is used by both SERTAO and CLASS.

Units
-----
H0:
    km/s/Mpc

Distances:
    Mpc

k passed to CLASS:
    1/Mpc

P(k):
    Mpc^3
"""

import numpy as np
import matplotlib.pyplot as plt

from SERTAO.hybrid_cosmology import HybridCosmology


# ============================================================
# Cosmological parameters
# ============================================================

H0 = 70.0
Omega_m = 0.30
Omega_b = 0.05
Omega_cdm = Omega_m - Omega_b
sigma8 = 0.80
w = -0.90

# ============================================================
# Custom background expansion
# ============================================================

def H_wCDM(z):
    """
    Hubble parameter for a flat constant-w CDM model.

    Parameters
    ----------
    z : float or ndarray
        Redshift.

    Returns
    -------
    float or ndarray
        H(z) in km/s/Mpc.
    """

    return H0 * np.sqrt(
        Omega_m * (1.0 + z)**3
        + (1.0 - Omega_m)
        * (1.0 + z)**(3.0 * (1.0 + w))
    )


# ============================================================
# CLASS configuration
# ============================================================

class_params = {
    "n_s": 0.965,
    "Omega_k": 0.0,
    "output": "mPk",
    "non_linear": "hmcode",
    "P_k_max_h/Mpc": 10.0,
}


# ============================================================
# Initialize the hybrid cosmology
# ============================================================

cosmo = HybridCosmology(
    H_of_z=H_wCDM,
    H0=H0,
    Omega_b=Omega_b,
    Omega_cdm=Omega_cdm,
    sigma8=sigma8,
    class_params=class_params,
)


# ============================================================
# Background and growth
# ============================================================

z = np.linspace(
    0.0,
    2.0,
    100,
)

H = cosmo.Hubble(z)
DM = cosmo.DM(z)
DH = cosmo.DH(z)
fsigma8 = cosmo.fsigma8(z)


# ============================================================
# Matter power spectrum
# ============================================================

# CLASS expects k in 1/Mpc.
k = np.logspace(
    -3,
    0,
    150,
)

Pk_linear = cosmo.Pk_linear(
    k,
    z=0.0,
)

Pk_nonlinear = cosmo.Pk(
    k,
    z=0.0,
)


# ============================================================
# Print basic information
# ============================================================

print("\n" + "=" * 60)
print("SERTAO + CLASS HybridCosmology")
print("=" * 60)

print(f"H0       = {cosmo.H0:.2f} km/s/Mpc")
print(f"h        = {cosmo.h:.3f}")
print(f"Omega_m  = {cosmo.Omega_m:.3f}")
print(f"Omega_b  = {cosmo.Omega_b:.3f}")
print(f"Omega_cdm = {cosmo.Omega_cdm:.3f}")
print(f"sigma8   = {cosmo.sigma8_input:.3f}")

print(
    f"sigma8(CLASS) = "
    f"{cosmo.sigma8_CLASS:.4f}"
)

print("=" * 60)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)

# ------------------------------------------------------------
# Hubble parameter
# ------------------------------------------------------------

ax = axes[0, 0]

ax.plot(
    z,
    H,
    lw=2,
)

ax.set_xlabel(
    r"$z$",
)

ax.set_ylabel(
    r"$H(z)\,[{\rm km\,s^{-1}\,Mpc^{-1}}]$",
)

ax.set_title(
    "Hubble Parameter",
)

ax.grid(
    True,
    alpha=0.3,
)


# ------------------------------------------------------------
# Comoving distances
# ------------------------------------------------------------

ax = axes[0, 1]

ax.plot(
    z,
    DM,
    lw=2,
    label=r"$D_M(z)$",
)

ax.plot(
    z,
    DH,
    lw=2,
    label=r"$D_H(z)$",
)

ax.set_xlabel(
    r"$z$",
)

ax.set_ylabel(
    r"Distance [Mpc]",
)

ax.set_title(
    "Cosmological Distances",
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()


# ------------------------------------------------------------
# Growth
# ------------------------------------------------------------

ax = axes[1, 0]

ax.plot(
    z,
    fsigma8,
    lw=2,
)

ax.set_xlabel(
    r"$z$",
)

ax.set_ylabel(
    r"$f(z)\sigma_8(z)$",
)

ax.set_title(
    "Linear Growth",
)

ax.grid(
    True,
    alpha=0.3,
)


# ------------------------------------------------------------
# Matter power spectrum
# ------------------------------------------------------------

ax = axes[1, 1]

ax.loglog(
    k,
    Pk_linear,
    lw=2,
    label="Linear",
)

ax.loglog(
    k,
    Pk_nonlinear,
    lw=2,
    label="Nonlinear",
)

ax.set_xlabel(
    r"$k\,[{\rm Mpc}^{-1}]$",
)

ax.set_ylabel(
    r"$P(k)\,[{\rm Mpc}^3]$",
)

ax.set_title(
    "Matter Power Spectrum",
)

ax.grid(
    True,
    which="both",
    alpha=0.3,
)

ax.legend()


plt.tight_layout()
plt.show()


# ============================================================
# Release CLASS
# ============================================================

cosmo.close_class()










    
