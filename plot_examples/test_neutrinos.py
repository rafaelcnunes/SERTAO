"""
Diagnostic plots for the SERTAO neutrinos module.
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(4):   # search up to 4 levels up
    if os.path.isdir(os.path.join(_ROOT, 'SERTAO')) and        os.path.isfile(os.path.join(_ROOT, 'SERTAO', '__init__.py')):
        break
    _ROOT = os.path.dirname(_ROOT)

sys.path.insert(0, _ROOT)   # local package takes priority over site-packages

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SERTAO.neutrinos import (
    Neutrinos, create_neutrinos,
    fermi_dirac_f, T_nu0_eV, T_nu0_K,
)

plt.rcParams.update({
    "figure.dpi":        130,
    "font.size":         11,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "lines.linewidth":   1.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
SAVE = dict(bbox_inches="tight", dpi=150)


# ================================================================
# Configurations to test
# ================================================================

CONFIGS = [
    ("Standard  Σm=0.06 eV", create_neutrinos(config="standard",          total_mass=0.06,  h=0.6736)),
    ("Standard  Σm=0.15 eV", create_neutrinos(config="standard",          total_mass=0.15,  h=0.6736)),
    ("Standard  Σm=0.30 eV", create_neutrinos(config="standard",          total_mass=0.30,  h=0.6736)),
    ("1ncdm     m=0.06 eV",  create_neutrinos(config="1ncdm",             m_ncdm=0.06,      h=0.6736)),
    ("NH        Σm=0.06 eV", create_neutrinos(config="hierarchy_normal",  total_mass=0.06,  h=0.6736)),
    ("IH        Σm=0.10 eV", create_neutrinos(config="hierarchy_inverted",total_mass=0.10,  h=0.6736)),
    ("Massless  Σm=0 eV",    create_neutrinos(config="standard",          total_mass=0.0,   h=0.6736)),
]

COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(CONFIGS)))

z = np.logspace(-2, 3.5, 400)


# ================================================================
# Terminal output
# ================================================================

print("=" * 65)
print("  SERTAO — neutrinos module diagnostic")
print("=" * 65)

for name, nu in CONFIGS:
    print(f"\n{name}:")
    print(f"  masses   = {nu.masses} eV")
    print(f"  N_eff    = {nu.N_eff:.4f}")
    print(f"  Omega_nu0 = {nu.Omega_nu0:.4e}")
    print(f"  omega_nu0 = {nu.omega_nu0:.4e}")
    if nu.total_mass_eV > 0:
        print(f"  z_trans  = {np.round(nu.z_trans, 1)}")
        # Cross-check with standard formula
        om_std = nu.total_mass_eV / 93.14
        diff   = abs(nu.omega_nu0 / om_std - 1) * 100
        print(f"  Σm/93.14 = {om_std:.4e}  (diff = {diff:.2f}%)")


# ================================================================
# Figure 1 — Fermi-Dirac f(y) fitting formula
# ================================================================

fig, ax = plt.subplots(figsize=(7, 4))

y = np.logspace(-2, 3, 300)
f = fermi_dirac_f(y)

ax.loglog(y, f,     color="#1D9E75", lw=2.2, label=r"$f(y) = [1+(Ay)^p]^{1/p}$")
ax.loglog(y, np.ones_like(y), "k:", lw=1, alpha=0.4, label=r"$y\to 0$: $f\to 1$")
ax.loglog(y, 0.3173*y,        "k--", lw=1, alpha=0.4, label=r"$y\to\infty$: $f\to Ay$")

ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel(r"$y = m_\nu\,/\,T_\nu(z)$")
ax.set_ylabel(r"$f(y)$")
ax.set_title("Fermi-Dirac energy integral  (WMAP-7 fitting formula)")
ax.legend()
ax.set_xlim(y[0], y[-1])

plt.tight_layout()
plt.savefig("nu_fig1_fermi_dirac.png", **SAVE)
plt.close()
print("\nSaved nu_fig1_fermi_dirac.png")


# ================================================================
# Figure 2 — Omega_nu(z) for all configurations
# ================================================================

fig, ax = plt.subplots(figsize=(9, 5))

for (name, nu), col in zip(CONFIGS, COLORS):
    dens = np.array([nu.density_parameter(float(zi)) for zi in z])
    ax.loglog(z, dens, color=col, label=name)

    # Mark z_trans for massive species
    for zt in nu.z_trans:
        if 0.01 < zt < 1100:
            ax.axvline(zt, color=col, ls=":", lw=0.8, alpha=0.5)

# Reference lines
Omega_gamma0 = 2.4728e-5 / 0.6736**2
ax.loglog(z, Omega_gamma0*(1+z)**4, "k--", lw=1, alpha=0.3, label=r"$\propto(1+z)^4$ (rel.)")
ax.loglog(z, CONFIGS[0][1].Omega_nu0*(1+z)**3, "k:",  lw=1, alpha=0.3, label=r"$\propto(1+z)^3$ (NR)")

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$\Omega_\nu(z)$")
ax.set_title(r"Neutrino density parameter  $\Omega_\nu(z)$")
ax.legend(fontsize=9, ncol=2)
ax.set_xlim(z[0], z[-1])

plt.tight_layout()
plt.savefig("nu_fig2_density.png", **SAVE)
plt.close()
print("Saved nu_fig2_density.png")


# ================================================================
# Figure 3 — Equation of state w_nu(z)
# ================================================================

fig, ax = plt.subplots(figsize=(9, 4))

for (name, nu), col in zip(CONFIGS, COLORS):
    w = np.array([nu.equation_of_state(float(zi)) for zi in z])
    ax.semilogx(z, w, color=col, label=name)

    for zt in nu.z_trans:
        if 0.01 < zt < 1100:
            ax.axvline(zt, color=col, ls=":", lw=0.8, alpha=0.5)

ax.axhline(1/3, color="k", ls="--", lw=1, alpha=0.4, label=r"$w=1/3$ (relativistic)")
ax.axhline(0,   color="k", ls=":",  lw=1, alpha=0.4, label=r"$w=0$  (non-relativistic)")

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$w_\nu(z)$")
ax.set_title("Neutrino effective equation of state")
ax.set_ylim(-0.05, 0.40)
ax.legend(fontsize=9, ncol=2)
ax.set_xlim(z[0], z[-1])

plt.tight_layout()
plt.savefig("nu_fig3_eos.png", **SAVE)
plt.close()
print("Saved nu_fig3_eos.png")


# ================================================================
# Figure 4 — Omega_nu / Omega_m ratio  (neutrino fraction)
# ================================================================

fig, ax = plt.subplots(figsize=(9, 4))
Omega_m0 = 0.315

for (name, nu), col in zip(CONFIGS[:-1], COLORS[:-1]):  # skip massless
    dens  = np.array([nu.density_parameter(float(zi)) for zi in z])
    rho_m = Omega_m0 * (1+z)**3
    ax.semilogx(z, dens / rho_m * 100, color=col, label=name)

ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$\Omega_\nu(z)\,/\,\Omega_m(z)$  [%]")
ax.set_title(r"Neutrino fraction relative to matter")
ax.legend(fontsize=9, ncol=2)
ax.set_xlim(z[0], z[-1])
ax.set_ylim(0, None)

plt.tight_layout()
plt.savefig("nu_fig4_fraction.png", **SAVE)
plt.close()
print("Saved nu_fig4_fraction.png")


# ================================================================
# Figure 5 — N_eff sensitivity
# ================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

z_neff = np.logspace(-2, 3.5, 300)
neff_vals = [2.5, 3.044, 3.5, 4.0]
colors_neff = plt.cm.plasma(np.linspace(0.1, 0.9, len(neff_vals)))

for Neff, col in zip(neff_vals, colors_neff):
    nu = create_neutrinos(config="standard", total_mass=0.06, N_eff=Neff, h=0.6736)
    dens = np.array([nu.density_parameter(float(zi)) for zi in z_neff])
    axes[0].loglog(z_neff, dens, color=col, label=f"$N_{{\\rm eff}}={Neff}$")
    # Ratio to N_eff=3.044
    nu_ref = create_neutrinos(config="standard", total_mass=0.06, N_eff=3.044, h=0.6736)
    dens_ref = np.array([nu_ref.density_parameter(float(zi)) for zi in z_neff])
    axes[1].semilogx(z_neff, dens/dens_ref, color=col, label=f"$N_{{\\rm eff}}={Neff}$")

axes[0].set_xlabel(r"$z$"); axes[0].set_ylabel(r"$\Omega_\nu(z)$")
axes[0].set_title(r"$N_{\rm eff}$ effect on $\Omega_\nu(z)$")
axes[0].legend()
axes[1].axhline(1, color="k", ls="--", lw=1, alpha=0.4)
axes[1].set_xlabel(r"$z$"); axes[1].set_ylabel(r"ratio to $N_{\rm eff}=3.044$")
axes[1].set_title(r"$\Omega_\nu(z)\,/\,\Omega_\nu^{\rm SM}(z)$")
axes[1].legend()
for ax in axes: ax.set_xlim(z_neff[0], z_neff[-1])

plt.tight_layout()
plt.savefig("nu_fig5_neff.png", **SAVE)
plt.close()
print("Saved nu_fig5_neff.png")


# ================================================================
# Terminal validation
# ================================================================

print("\n" + "=" * 65)
print("  Validation tests")
print("=" * 65)

nu_std = create_neutrinos(config="standard", total_mass=0.06, h=0.6736)

# 1. Massless limit: Omega_nu ∝ (1+z)^4
nu_ml = create_neutrinos(config="standard", total_mass=0.0, h=0.6736)
Om0   = nu_ml.density_parameter(0.01)
for z_val, label in [(1, "z=1"), (10, "z=10"), (100, "z=100")]:
    ratio    = nu_ml.density_parameter(z_val) / Om0
    expected = (1 + z_val)**4 / (1 + 0.01)**4
    err      = abs(ratio / expected - 1) * 100
    ok       = "✔" if err < 0.5 else "✗"
    print(f"  Massless (1+z)^4 at {label}: ratio={ratio:.4f}  expected={expected:.4f}  "
          f"err={err:.3f}%  {ok}")

# 2. NR limit: w(z→0) ≈ 0
w0 = nu_std.equation_of_state(0.0)
ok = "✔" if w0 < 0.01 else "✗"
print(f"\n  NR limit: w(z=0) = {w0:.6f}  (expected ~0)  {ok}")

# 3. Relativistic limit: w(z=1000) ≈ 1/3
w_hi = nu_std.equation_of_state(1000.0)
ok   = "✔" if abs(w_hi - 1/3) < 0.005 else "✗"
print(f"  Rel limit: w(z=1000) = {w_hi:.6f}  (expected 1/3={1/3:.6f})  {ok}")

# 4. Transition redshift: z_trans ~ m/T_nu0 - 1
z_trans_expected = 0.02 / T_nu0_eV - 1
z_trans_actual   = nu_std.z_trans[0]
err = abs(z_trans_actual / z_trans_expected - 1) * 100
ok  = "✔" if err < 0.1 else "✗"
print(f"  z_trans = {z_trans_actual:.2f}  (expected {z_trans_expected:.2f})  {ok}")

# 5. Standard formula cross-check: omega_nu ≈ Σm / 93.14
om_wmap = nu_std.omega_nu0
om_std  = nu_std.total_mass_eV / 93.14
err     = abs(om_wmap / om_std - 1) * 100
ok      = "✔" if err < 1.0 else "✗"
print(f"  omega_nu0 (WMAP) = {om_wmap:.4e}  (Σm/93.14 = {om_std:.4e},  diff={err:.2f}%)  {ok}")

# 6. contribution_to_H2 == density_parameter
for z_val in [0.0, 1.0, 100.0]:
    d1 = nu_std.density_parameter(z_val)
    d2 = nu_std.contribution_to_H2(z_val)
    ok = "✔" if abs(d1 - d2) < 1e-12 else "✗"
    print(f"  density_parameter == contribution_to_H2 at z={z_val}: {ok}")

print("\n" + "=" * 65)
print("  Figures: nu_fig1..5 saved.")
print("=" * 65)






