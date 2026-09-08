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
# Load Cosmic Chronometers data
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

data_file = os.path.join(DATA_DIR, "HzTable_MM_BC03.txt")
cov_file  = os.path.join(DATA_DIR, "cov_cc_MM_BC03.txt")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found: {data_file}")

if not os.path.exists(cov_file):
    raise FileNotFoundError(f"Covariance file not found: {cov_file}")

# Load data
data = np.loadtxt(data_file, comments="#")

z_cc = data[:, 0]
H_obs = data[:, 1]

# Load covariance (for error bars)
cov = np.loadtxt(cov_file)
sigma_H = np.sqrt(np.diag(cov))

# ============================================================
# Theoretical prediction 
# ============================================================

z_min = 0.0
z_max = z_cc.max() * 1.05
z_grid = np.linspace(z_min, z_max, 500)

H_LCDM_grid = H_LCDM(z_grid)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

# LCDM prediction
plt.plot(
    z_grid, H_LCDM_grid,
    color="black", lw=2,
    label=r"$\Lambda$CDM"
)

# CC data
plt.errorbar(
    z_cc, H_obs, yerr=sigma_H,
    fmt="o", capsize=3,
    label="Cosmic Chronometers"
)

plt.xlabel(r"$z$")
plt.ylabel(r"$H(z)\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


