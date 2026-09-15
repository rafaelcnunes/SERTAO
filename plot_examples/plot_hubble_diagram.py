import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── SERTAO path ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from SERTAO.cosmology import GenericCosmology

# ============================================================
# Cosmological model  (edit here to change the theory curve)
# ============================================================

H0      = 67.36
Omega_m = 0.315
Omega_b = 0.049

def H_LCDM(z):
    Or = 2.469e-5 / (H0 / 100)**2
    OL = 1.0 - Omega_m - Or
    return H0 * np.sqrt(Omega_m*(1+z)**3 + Or*(1+z)**4 + OL)

cosmo = GenericCosmology(
    H_of_z=H_LCDM,
    H0=H0,
    Omega_m=Omega_m,
    Omega_b=Omega_b,
)

# ============================================================
# Load DES-Dovekie data
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

data_file = os.path.join(DATA_DIR, "DES-Dovekie_HD.csv")
cov_file  = os.path.join(DATA_DIR, "DES-Dovekie_STAT_SYS.npz")
# Try alternative filename with + sign
if not os.path.exists(cov_file):
    cov_file = os.path.join(DATA_DIR, "DES-Dovekie_STAT+SYS.npz")

if not os.path.exists(data_file):
    raise FileNotFoundError(
        f"Data file not found: {data_file}\n"
        f"Place DES-Dovekie_HD.csv in the data/ directory."
    )

# Load Hubble diagram
data = pd.read_csv(data_file, sep=r'\s+', comment='#',
                   usecols=['zHD', 'zHEL', 'MU', 'MUERR'])

zHD  = data['zHD'].to_numpy()
zHEL = data['zHEL'].to_numpy()
MU   = data['MU'].to_numpy()
MUERR = data['MUERR'].to_numpy()   # statistical-only error bars

# Survey flag: DES z > 0.1, low-z z < 0.1
mask_des  = zHD >= 0.1
mask_lowz = zHD <  0.1

print(f"Loaded {len(zHD)} SNe Ia  ({mask_des.sum()} DES,  {mask_lowz.sum()} low-z)")

# ============================================================
# Theory distance modulus
# mu_theory(z) = 5 * log10( (1+zHEL) * DM(zHD) ) + 25
# ============================================================

z_grid   = np.logspace(np.log10(0.01), np.log10(1.3), 500)
DM_grid  = cosmo.DM(z_grid)
mu_grid  = 5.0 * np.log10((1 + z_grid) * DM_grid) + 25.0

# Evaluate theory at data redshifts (using zHEL for kinematic correction)
DM_data     = cosmo.DM(zHD)
mu_theory   = 5.0 * np.log10((1 + zHEL) * DM_data) + 25.0

# Best-fit M: weighted mean offset (equivalent to Goliath best-fit M)
weights     = 1.0 / MUERR**2
M_bestfit   = np.average(MU - mu_theory, weights=weights)
mu_grid_shifted = mu_grid + M_bestfit
mu_theory_shifted = mu_theory + M_bestfit
residuals   = MU - mu_theory_shifted

print(f"Best-fit M offset = {M_bestfit:.4f} mag")
print(f"Residual RMS      = {np.std(residuals):.4f} mag")

# ============================================================
# Binning for cleaner plot
# ============================================================

def bin_data(z, mu, mu_err, res, n_bins=20, z_min=None, z_max=None):
    """Bin SN data into redshift bins for plotting."""
    z_min = z_min or z.min()
    z_max = z_max or z.max()
    edges = np.linspace(z_min, z_max, n_bins + 1)
    z_bin, mu_bin, err_bin, res_bin = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (z >= lo) & (z < hi)
        if mask.sum() == 0:
            continue
        w = 1.0 / mu_err[mask]**2
        z_bin.append(np.average(z[mask], weights=w))
        mu_bin.append(np.average(mu[mask], weights=w))
        err_bin.append(1.0 / np.sqrt(w.sum()))
        res_bin.append(np.average(res[mask], weights=w))
    return (np.array(z_bin), np.array(mu_bin),
            np.array(err_bin), np.array(res_bin))

# Bin DES and low-z separately
z_bin_des,  mu_bin_des,  err_bin_des,  res_bin_des  = bin_data(
    zHD[mask_des],  MU[mask_des],  MUERR[mask_des],  residuals[mask_des],
    n_bins=20)

z_bin_lowz, mu_bin_lowz, err_bin_lowz, res_bin_lowz = bin_data(
    zHD[mask_lowz], MU[mask_lowz], MUERR[mask_lowz], residuals[mask_lowz],
    n_bins=5)

# ============================================================
# Plot
# ============================================================

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9, 7),
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
    sharex=True,
)

# ── Top panel: Hubble diagram ─────────────────────────────────

# Individual SNe (faint, for context)
ax_top.scatter(zHD[mask_des],  MU[mask_des]  - M_bestfit,
               c='steelblue', s=1.5, alpha=0.25, rasterized=True)
ax_top.scatter(zHD[mask_lowz], MU[mask_lowz] - M_bestfit,
               c='darkorange', s=1.5, alpha=0.25, rasterized=True)

# Binned points
ax_top.errorbar(z_bin_des,  mu_bin_des  - M_bestfit, yerr=err_bin_des,
                fmt='o', ms=5, color='steelblue',  capsize=3,
                label=f'DES ({mask_des.sum()} SNe Ia)',  zorder=3)
ax_top.errorbar(z_bin_lowz, mu_bin_lowz - M_bestfit, yerr=err_bin_lowz,
                fmt='s', ms=5, color='darkorange', capsize=3,
                label=f'Low-z ({mask_lowz.sum()} SNe Ia)', zorder=3)

# Theory curve
ax_top.plot(z_grid, mu_grid, color='black', lw=1.8,
            label=r'$\Lambda$CDM')

ax_top.set_ylabel(r'Distance modulus  $\mu$', fontsize=12)
ax_top.legend(fontsize=10, loc='upper left')
ax_top.grid(alpha=0.3)
ax_top.set_ylim(33, 47)

# ── Bottom panel: residuals ───────────────────────────────────

ax_bot.axhline(0, color='black', lw=1.2, ls='--')

ax_bot.errorbar(z_bin_des,  res_bin_des,  yerr=err_bin_des,
                fmt='o', ms=5, color='steelblue',  capsize=3, zorder=3)
ax_bot.errorbar(z_bin_lowz, res_bin_lowz, yerr=err_bin_lowz,
                fmt='s', ms=5, color='darkorange', capsize=3, zorder=3)

# Individual residuals (faint)
ax_bot.scatter(zHD[mask_des],  residuals[mask_des],
               c='steelblue', s=1.5, alpha=0.2, rasterized=True)
ax_bot.scatter(zHD[mask_lowz], residuals[mask_lowz],
               c='darkorange', s=1.5, alpha=0.2, rasterized=True)

ax_bot.set_xlabel(r'Redshift  $z$', fontsize=12)
ax_bot.set_ylabel(r'$\mu - \mu_{\Lambda\rm CDM}$', fontsize=11)
ax_bot.set_ylim(-0.55, 0.55)
ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_bot.grid(alpha=0.3)

# ── Log x-axis ───────────────────────────────────────────────

ax_bot.set_xscale('log')
ax_bot.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f'{x:.2g}'.rstrip('0').rstrip('.')))

plt.tight_layout()

# Save
out_path = os.path.join(BASE_DIR, 'hubble_diagram_dovekie.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"\nSaved: {out_path}")

plt.show()

# ============================================================
# Print summary statistics
# ============================================================

print()
print("=" * 50)
print(f"  DES-Dovekie  —  Hubble diagram summary")
print("=" * 50)
print(f"  N_SNe (DES)   = {mask_des.sum()}")
print(f"  N_SNe (low-z) = {mask_lowz.sum()}")
print(f"  z range       = [{zHD.min():.3f}, {zHD.max():.3f}]")
print(f"  Residual RMS  = {np.std(residuals):.4f} mag")
print(f"  Weighted RMS  = {np.sqrt(np.average(residuals**2, weights=1/MUERR**2)):.4f} mag")
print("=" * 50)
