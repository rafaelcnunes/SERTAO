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
sigma8 = 0.78  

def H_LCDM(z):
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + (1.0 - Omega_m))

cosmo = GenericCosmology(
    H_of_z=H_LCDM,
    H0=H0,
    Omega_m=Omega_m,
    Omega_b=Omega_b,
    sigma8_0=sigma8
)

# ============================================================
# Load RSD data
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

data_file = os.path.join(DATA_DIR, "RSD.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

data = np.loadtxt(data_file, comments="#")

z_rsd   = data[:, 0]
fs8_obs = data[:, 1]
fs8_err = data[:, 2]

# ============================================================
# Theoretical prediction
# ============================================================
z_min = 0.0
z_max = z_rsd.max() * 1.05
z_grid = np.linspace(z_min, z_max, 500)

fs8_LCDM = np.array([
    cosmo.fsigma8(z) for z in z_grid
])

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

# ΛCDM prediction
plt.plot(
    z_grid, fs8_LCDM,
    color="black", lw=2,
    label=r"$\Lambda$CDM"
)

# RSD data
plt.errorbar(
    z_rsd, fs8_obs, yerr=fs8_err,
    fmt="o", capsize=3,
    label=r"RSD $f\sigma_8$"
)

plt.xlabel(r"$z$")
plt.ylabel(r"$f\sigma_8(z)$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


