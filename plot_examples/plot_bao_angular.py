import os
import numpy as np
import matplotlib.pyplot as plt

from SERTAO.cosmology import GenericCosmology

# ============================================================
# Cosmological model
# ============================================================
H0 = 67.4
Omega_m = 0.315
Omega_b = 0.049

def H_LCDM(z):
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + (1.0 - Omega_m))


cosmo = GenericCosmology(
    H_of_z=H_LCDM,
    H0=H0,
    Omega_m=Omega_m,
    Omega_b=Omega_b
)

# ============================================================
# Load BAO angular data
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

data_file = os.path.join(DATA_DIR, "bao_t_on_data.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

data = np.loadtxt(data_file)

(
    z_eff, theta_bao, sigma_theta,
    z_ON, theta_ON, sigma_ON
) = data.T

# Remove zero entries
mask_bao = z_eff != 0.0
mask_on  = z_ON  != 0.0

z_eff = z_eff[mask_bao]
theta_bao = theta_bao[mask_bao]
sigma_theta = sigma_theta[mask_bao]

z_ON = z_ON[mask_on]
theta_ON = theta_ON[mask_on]
sigma_ON = sigma_ON[mask_on]

# ============================================================
# Theoretical prediction
# ============================================================

conv = 180.0 / np.pi
rd = cosmo.rd_sound_horizon()

# Continuous redshift grid
z_min = 0.09
z_max = max(z_eff.max(), z_ON.max()) * 1.05
z_grid = np.linspace(z_min, z_max, 400)

theta_LCDM = np.array([
    conv * rd / ((1.0 + z) * cosmo.DA(z))
    for z in z_grid
])

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

# LCDM prediction (single smooth curve)
plt.plot(
    z_grid, theta_LCDM,
    color="black", lw=2,
    label=r"$\Lambda$CDM"
)

# Standard angular BAO data
plt.errorbar(
    z_eff, theta_bao, yerr=sigma_theta,
    fmt="o", capsize=3,
    label="BAO angular"
)

# BAO-ON data
plt.errorbar(
    z_ON, theta_ON, yerr=sigma_ON,
    fmt="s", capsize=3,
    label="BAO-ON"
)

plt.xlabel(r"$z$")
plt.ylabel(r"$\theta_{\rm BAO}$ [graus]")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()



