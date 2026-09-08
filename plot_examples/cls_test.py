"""
Angular power spectra for CMB lensing × galaxy clustering using the Limber approximation.

Run this file directly to produce diagnostic plots:
    python cls.py
"""

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator

C_LIGHT  = 299792.458   # km/s
Z_STAR   = 1089.0       # CMB last-scattering redshift

# ================================================================
# Numerical integration
# ================================================================
def integrate(y, x):
    """Integrate y(x) using Simpson's rule."""
    return simpson(y=np.asarray(y, dtype=float),
                   x=np.asarray(x, dtype=float))

# ================================================================
# Source distribution
# ================================================================

def phi_LSST(z, z0=0.81):
    """
    LSST-like redshift distribution  z^2 exp(-z/z0) / (2 z0^3).
    Normalisation is applied inside compute_kernels().
    """
    z = np.asarray(z, dtype=float)
    return z**2 * np.exp(-z / z0) / (2.0 * z0**3)


def phi_gaussian(z, z_mean, sigma_z):
    """
    Gaussian photometric bin centred on z_mean with width sigma_z.
    Normalised so that integral over z = 1.
    Used for the DESI LRG bins z2/z3/z4.
    """
    z   = np.asarray(z, dtype=float)
    phi = np.exp(-0.5 * ((z - z_mean) / sigma_z)**2)
    return phi / integrate(phi, z)


# ================================================================
# Galaxy bias
# ================================================================

def galaxy_bias(z, b1=1.0, b2=0.84):
    """
    Linear bias  b(z) = b1 + b2 * z.
    """
    return b1 + b2 * np.asarray(z, dtype=float)


# ================================================================
# Background
# ================================================================

def compute_background(cosmo, z, z_star=Z_STAR):
    """
    Compute comoving distances chi(z), H(z), and chi_star.
    """
    z = np.asarray(z, dtype=float)
    chi      = np.array([cosmo.DM(zi)     for zi in z], dtype=float)
    H        = np.array([cosmo.Hubble(zi) for zi in z], dtype=float)
    chi_star = float(cosmo.DM(z_star))
    return chi, H, chi_star


# ================================================================
# Magnification bias kernel  W^mu  (eq. 2.15)
# ================================================================

def magnification_bias_kernel(z, chi, H, chi_star, phi, Omega_m, H0, s_mu):
    """
    Magnification bias correction to the galaxy window function.

    W^mu(z) = (5 s_mu - 2) * prefactor * (1+z) * H(z)
              * int_z^z_star dz' chi(z)[chi(z')-chi(z)]/chi(z') phi(z')

    where prefactor = 3 Omega_m H0^2 / (2 c^2).

    Returns zero array when (5 s_mu - 2) = 0.
    """
    amp = 5.0 * s_mu - 2.0
    if np.isclose(amp, 0.0):
        return np.zeros_like(z, dtype=float)

    z   = np.asarray(z,   dtype=float)
    chi = np.asarray(chi, dtype=float)
    H   = np.asarray(H,   dtype=float)
    phi = np.asarray(phi, dtype=float)

    prefactor = amp * 3.0 * Omega_m * H0**2 / (2.0 * C_LIGHT**2)

    # Inner integral over z' > z for each z_i
    inner = np.zeros(len(z), dtype=float)
    for i in range(len(z)):
        mask = z >= z[i]
        if mask.sum() < 2:
            continue
        z_t, chi_t, phi_t = z[mask], chi[mask], phi[mask]
        integrand    = chi[i] * (chi_t - chi[i]) / np.where(chi_t > 0, chi_t, 1.0) * phi_t
        integrand[0] = 0.0
        inner[i]     = integrate(integrand, z_t)

    return prefactor * (1.0 + z) * H * inner


# ================================================================
# Projection kernels
# ================================================================

def compute_kernels(
    z, chi, H, chi_star,
    Omega_m, H0,
    b1=1.0, b2=0.84,
    phi=None,
    s_mu=0.0,
):
    """
    Compute lensing kernel, galaxy distribution, galaxy bias,
    and the geometric Limber integration weights.

    Parameters
    ----------
    z, chi, H   : arrays on the redshift grid
    chi_star    : float, comoving distance to CMB
    Omega_m, H0 : cosmological parameters
    b1, b2      : galaxy bias  b(z) = b1 + b2*z
    phi         : optional normalised redshift distribution.
                  If None, phi_LSST is used and normalised here.
    s_mu        : magnitude slope for magnification bias (eq. 2.15).
                  Default 0 disables magnification.

    Returns
    -------
    W_kappa, phi, bias, geom_kk, geom_kg, geom_gg
    """
    z   = np.asarray(z,   dtype=float)
    chi = np.asarray(chi, dtype=float)
    H   = np.asarray(H,   dtype=float)

    # ── Source distribution ───────────────────────────────────
    if phi is None:
        phi = phi_LSST(z)
        phi = phi / integrate(phi, z)
    else:
        phi = np.asarray(phi, dtype=float)

    # ── Galaxy bias ───────────────────────────────────────────
    bias = galaxy_bias(z, b1=b1, b2=b2)

    # ── CMB lensing kernel  W^kappa  (eq. 2.10) ──────────────
    # W^kappa = (3 Omega_m H0^2)/(2c) * (1+z)/H(z) * chi*(1-chi/chi*)
    # The /H(z) factor is already separated from the Limber measure H/c,
    # so geom_kk = (H/c) * W_kappa^2 / chi^2 reproduces eq. 2.13.
    geom_factor = (chi_star - chi) / chi_star
    geom_factor = np.where(chi < chi_star, geom_factor, 0.0)

    W_kappa = (
        3.0 * Omega_m * H0**2 / (2.0 * C_LIGHT)
        * (1.0 + z) / H
        * chi * geom_factor
    )

    # ── Galaxy kernel with magnification bias (eq. 2.14) ─────
    W_mu      = magnification_bias_kernel(z, chi, H, chi_star,
                                          phi, Omega_m, H0, s_mu)
    W_g_total = phi * bias + W_mu

    # ── Limber geometric factors ──────────────────────────────
    # C_ell^{XY} = int dz (H/c) * W^X * W^Y / chi^2 * P_{XY}
    measure = H / C_LIGHT / chi**2

    geom_kk = measure * W_kappa**2
    geom_kg = measure * W_kappa * W_g_total
    geom_gg = measure * W_g_total**2

    return W_kappa, phi, bias, geom_kk, geom_kg, geom_gg


# ================================================================
# Effective redshift  (eq. 2.20)
# ================================================================

def effective_redshift(z, weight):
    """
    Kernel-weighted effective redshift:
        z_eff = int dz z*weight / int dz weight
    """
    return integrate(z * weight, z) / integrate(weight, z)


# ================================================================
# Limber wavenumber
# ================================================================

def limber_k(ells, chi):
    """
    k_Limber = (ell + 1/2) / chi.
    Returns array of shape (N_ell, N_z).
    """
    return (np.asarray(ells, dtype=float)[:, None] + 0.5) \
           / np.asarray(chi, dtype=float)[None, :]


# ================================================================
# P(k,z) interpolator
# ================================================================

def build_power_interpolator(cosmo, z, k_min=1e-4, k_max=30.0, n_k=250):
    """
    Build a 2D log-log interpolator for P(k, z).

    Interpolation grid: (z, log k) -> log P.
    Log-space is considerably more stable than linear in k.

    Parameters
    ----------
    cosmo       : HybridCosmology (SERTAO + CLASS)
    z           : array, redshift grid
    k_min/k_max : float, 1/Mpc   (set k_max >= max Limber k expected)
    n_k         : int,  number of k points (log-spaced)

    Returns
    -------
    RegularGridInterpolator mapping (z, log k) -> log P
    """
    z      = np.asarray(z, dtype=float)
    k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    log_k  = np.log(k_grid)

    log_P = np.empty((len(z), n_k), dtype=float)
    for iz, zi in enumerate(z):
        pk_row = np.array([cosmo.Pk(float(ki), float(zi)) for ki in k_grid])
        pk_row = np.where(pk_row > 0.0, pk_row, 1e-40)
        log_P[iz] = np.log(pk_row)

    return RegularGridInterpolator(
        (z, log_k), log_P,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


# ================================================================
# Evaluate P on Limber grid
# ================================================================

def evaluate_power(P_interp, z, k):
    """
    Evaluate P(k,z) on the Limber grid  k[i_ell, i_z].

    Parameters
    ----------
    P_interp : RegularGridInterpolator from build_power_interpolator
    z        : array, shape (N_z,)
    k        : array, shape (N_ell, N_z)

    Returns
    -------
    P : array, shape (N_ell, N_z), Mpc^3
    """
    z = np.asarray(z, dtype=float)
    k = np.asarray(k, dtype=float)
    N_ell, N_z = k.shape

    points = np.column_stack([
        np.broadcast_to(z[None, :], (N_ell, N_z)).ravel(),
        np.log(k.ravel()),
    ])
    return np.exp(P_interp(points).reshape(N_ell, N_z))


# ================================================================
# Project into angular power spectra
# ================================================================

def project_cls(P, z, geom_kk, geom_kg, geom_gg):
    """
    Integrate P(k_Limber, z) against the geometric factors.

    C_ell^{XY} = int dz  geom_{XY}(z) * P(k_Limber(ell,z), z)
    """
    Cl_kk = np.array([integrate(P[i] * geom_kk, z) for i in range(len(P))])
    Cl_kg = np.array([integrate(P[i] * geom_kg, z) for i in range(len(P))])
    Cl_gg = np.array([integrate(P[i] * geom_gg, z) for i in range(len(P))])
    return Cl_kk, Cl_kg, Cl_gg


# ================================================================
# Main entry point
# ================================================================

def compute_all_cls(
    cosmo,
    z,
    ells,
    Omega_m,
    H0,
    b1=1.0,
    b2=0.84,
    phi=None,
    s_mu=0.0,
    k_min=1e-4,
    k_max=30.0,
    n_k=250,
):
    """
    Compute C_ell^{kk}, C_ell^{kg}, C_ell^{gg}.

    Parameters
    ----------
    cosmo       : HybridCosmology  (SERTAO + CLASS)
    z           : array, redshift integration grid
    ells        : array, multipoles
    Omega_m     : float
    H0          : float, km/s/Mpc
    b1, b2      : galaxy bias parameters  b(z) = b1 + b2*z
    phi         : optional normalised redshift distribution.
                  If None, phi_LSST(z) is used.
    s_mu        : float, magnitude slope for magnification bias.
                  Use values from Sailer+2024 Table 2 per bin.
                  Default 0 disables magnification.
    k_min/k_max : float, 1/Mpc  — range of P(k,z) interpolation grid
    n_k         : int  — number of k grid points

    Returns
    -------
    Cl_kk, Cl_kg, Cl_gg : arrays, shape (N_ell,)
    z_eff : tuple  (z_eff_kk, z_eff_kg, z_eff_gg)
    results : dict  — intermediate arrays for diagnostics / plotting
    """
    z    = np.asarray(z,    dtype=float)
    ells = np.asarray(ells, dtype=float)

    chi, H, chi_star = compute_background(cosmo, z)

    W_kappa, phi_out, bias, geom_kk, geom_kg, geom_gg = compute_kernels(
        z=z, chi=chi, H=H, chi_star=chi_star,
        Omega_m=Omega_m, H0=H0,
        b1=b1, b2=b2,
        phi=phi, s_mu=s_mu,
    )

    z_eff = (
        effective_redshift(z, geom_kk),
        effective_redshift(z, geom_kg),
        effective_redshift(z, geom_gg),
    )

    k = limber_k(ells, chi)

    P_interp = build_power_interpolator(cosmo, z, k_min=k_min,
                                        k_max=k_max, n_k=n_k)
    P = evaluate_power(P_interp, z, k)

    Cl_kk, Cl_kg, Cl_gg = project_cls(P=P, z=z,
                                        geom_kk=geom_kk,
                                        geom_kg=geom_kg,
                                        geom_gg=geom_gg)

    results = dict(
        z=z, ells=ells, chi=chi, H=H, chi_star=chi_star,
        W_kappa=W_kappa, phi=phi_out, bias=bias,
        geom_kk=geom_kk, geom_kg=geom_kg, geom_gg=geom_gg,
        k_limber=k, P_limber=P, z_eff=z_eff,
    )

    return Cl_kk, Cl_kg, Cl_gg, z_eff, results


# ================================================================
# Diagnostic plots  (run with:  python cls.py)
# ================================================================

if __name__ == "__main__":

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("cls.py — diagnostic plots (mock LCDM, no CLASS needed)")
    print("=" * 58)

    # ── Mock flat LCDM cosmology ──────────────────────────────
    class _MockCosmo:
        """
        Analytic flat LCDM for offline testing.
        H(z)  = H0 * sqrt(Om*(1+z)^3 + OL)
        chi(z) = c * int_0^z dz'/H(z')
        Pk     = A * k^ns * T_EH^2(k) * D^2(z)
        """
        def __init__(self, H0=67.36, Omega_m=0.315):
            self.H0      = H0
            self.Omega_m = Omega_m
            self._OL     = 1.0 - Omega_m
            self._h      = H0 / 100.0
            # precompute chi
            self._zg = np.linspace(0.0, 1200.0, 80_000)
            Hg       = self._Hfunc(self._zg)
            dc       = C_LIGHT / Hg
            self._cg = np.zeros_like(self._zg)
            self._cg[1:] = np.cumsum(
                0.5*(dc[:-1]+dc[1:])*np.diff(self._zg))

        def _Hfunc(self, z):
            return self.H0*np.sqrt(self.Omega_m*(1+z)**3 + self._OL)

        def Hubble(self, z): return float(self._Hfunc(z))

        def DM(self, z):
            return float(np.interp(z, self._zg, self._cg))

        def _T(self, k):
            om = self.Omega_m * self._h**2
            ob = 0.022 * self._h**2
            T  = 2.728/2.7
            keq= 7.46e-2*om/T**2
            s  = 44.5*np.log(9.83/om)/np.sqrt(1+10*ob**0.75)
            q  = k/(13.41*keq)
            return np.log(1+2.34*q)/(2.34*q)*(
                1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**(-0.25)

        def _D(self, z):
            def f(zp): return (1+zp)/self._Hfunc(zp)**3
            za = np.linspace(z, 1000.0, 500)
            z0 = np.linspace(0.0, 1000.0, 500)
            return self._Hfunc(z)*integrate(f(za), za)/(
                   self._Hfunc(0)*integrate(f(z0), z0))

        def Pk(self, k, z):
            return 5e4 * k**0.965 * self._T(k)**2 * self._D(z)**2

    cosmo   = _MockCosmo()
    Omega_m = cosmo.Omega_m
    H0      = cosmo.H0

    # ── Redshift grid ─────────────────────────────────────────
    z = np.linspace(0.01, 4.0, 250)

    # ── DESI LRG bins (paper Table 2) ────────────────────────
    BINS = {
        "z2": dict(z_eff=0.625, s_mu=1.044, bias=1.8, color="#378ADD"),
        "z3": dict(z_eff=0.785, s_mu=0.974, bias=2.0, color="#1D9E75"),
        "z4": dict(z_eff=0.914, s_mu=0.988, bias=2.2, color="#D85A30"),
    }

    ells = np.unique(
        np.round(np.geomspace(20, 300, 50)).astype(int)
    ).astype(float)

    # ── Compute for each bin ──────────────────────────────────
    print("Computing C_ell for each bin...")
    bin_res = {}
    for bname, bp in BINS.items():
        phi_bin = phi_gaussian(z, z_mean=bp["z_eff"], sigma_z=0.10)
        Cl_kk, Cl_kg, Cl_gg, z_eff, info = compute_all_cls(
            cosmo=cosmo, z=z, ells=ells,
            Omega_m=Omega_m, H0=H0,
            b1=bp["bias"], b2=0.0,
            phi=phi_bin,
            s_mu=bp["s_mu"],
        )
        bin_res[bname] = dict(
            Cl_kk=Cl_kk, Cl_kg=Cl_kg, Cl_gg=Cl_gg,
            z_eff_tuple=z_eff, info=info,
            z_eff_bin=bp["z_eff"],
            s_mu=bp["s_mu"], bias=bp["bias"], color=bp["color"],
        )
        print(f"  {bname}: z_eff(gg)={z_eff[2]:.3f}, "
              f"z_eff(kg)={z_eff[1]:.3f}")
    print("Done.\n")

    # ── Style ─────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 11,
        "axes.labelsize": 12, "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    SAVE = dict(bbox_inches="tight", dpi=150)
    lw   = 1.8

    # =========================================================
    # Figure 1 — Projection kernels
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Projection kernels — DESI LRG bins (z2/z3/z4)", y=1.02)

    ax0, ax1, ax2 = axes

    # W_kappa (same for all bins)
    ref  = bin_res["z2"]["info"]
    Wk   = ref["W_kappa"]
    ax0.plot(z, Wk / Wk.max(), color="k", lw=lw,
             label=r"$W^\kappa(z)$ (normalised)")
    ax0.set_xlabel(r"$z$");  ax0.set_xlim(0, 3.5)
    ax0.set_ylabel("Normalised amplitude")
    ax0.set_title(r"CMB lensing kernel $W^\kappa$")
    ax0.legend()

    for bname, br in bin_res.items():
        c   = br["color"]
        phi = br["info"]["phi"]
        ax1.plot(z, phi / phi.max(), color=c,
                 label=f"{bname} ($z_{{eff}}={br['z_eff_tuple'][2]:.2f}$)")
        # W_mu contribution (dashed)
        W_mu = magnification_bias_kernel(
            z, ref["chi"], ref["H"], ref["chi_star"],
            phi, Omega_m, H0, br["s_mu"])
        if W_mu.max() > 0:
            ax1.plot(z, W_mu / phi.max(), color=c, ls="--", alpha=0.55)

    ax1.set_xlabel(r"$z$");  ax1.set_xlim(0, 2.5)
    ax1.set_title(r"$\phi(z)$ [solid]  +  $W^\mu$ [dashed]")
    ax1.legend()

    for bname, br in bin_res.items():
        info   = br["info"]
        # reconstruct W_g_total from geom_kg
        # geom_kg = H/c * W_kappa * W_g_total / chi^2
        H_arr  = info["H"];  chi_arr = info["chi"]
        W_g_tot = info["geom_kg"] * C_LIGHT * chi_arr**2 / (H_arr * Wk)
        ax2.plot(z, W_g_tot / np.abs(W_g_tot).max(), color=br["color"],
                 label=bname)

    ax2.set_xlabel(r"$z$");  ax2.set_xlim(0, 2.5)
    ax2.set_title(r"$W^g_\mathrm{total} = \phi\,b + W^\mu$ (normalised)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("cls_fig1_kernels.pdf", **SAVE)
    plt.savefig("cls_fig1_kernels.png", **SAVE)
    plt.close();  print("Saved cls_fig1_kernels")

    # =========================================================
    # Figure 2 — Limber k(z) for representative multipoles
    # =========================================================
    chi_arr  = bin_res["z2"]["info"]["chi"]
    k_limber = limber_k(ells, chi_arr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ell_plot = [20, 50, 100, 178, 243]
    colors   = plt.cm.plasma(np.linspace(0.1, 0.85, len(ell_plot)))
    for ev, col in zip(ell_plot, colors):
        idx  = np.argmin(np.abs(ells - ev))
        mask = (z > 0.1) & (z < 3.0)
        ax.semilogy(z[mask], k_limber[idx][mask], color=col,
                    label=rf"$\ell={ev}$")

    ax.axhspan(0.1, 10, alpha=0.07, color="gray",
               label="Nonlinear  k > 0.1")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$k_\mathrm{Limber}$ [Mpc$^{-1}$]")
    ax.set_title(r"Limber wavenumber  $k = (\ell+\frac{1}{2})/\chi(z)$")
    ax.legend(ncol=2);  ax.set_xlim(0.1, 3.0)
    plt.tight_layout()
    plt.savefig("cls_fig2_limber_k.pdf", **SAVE)
    plt.savefig("cls_fig2_limber_k.png", **SAVE)
    plt.close();  print("Saved cls_fig2_limber_k")

    # =========================================================
    # Figure 3 — C_ell for all bins
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Angular power spectra — mock flat LCDM", y=1.02)

    specs = [
        ("Cl_kk", r"$C_\ell^{\kappa\kappa}$"),
        ("Cl_kg", r"$C_\ell^{\kappa g}$"),
        ("Cl_gg", r"$C_\ell^{gg}$"),
    ]
    for ax, (key, title) in zip(axes, specs):
        for bname, br in bin_res.items():
            Cl = br[key]
            ax.loglog(ells,
                      ells*(ells+1)/(2*np.pi) * np.abs(Cl),
                      color=br["color"],
                      label=f"{bname} ($z_{{eff}}={br['z_eff_tuple'][2]:.2f}$)")
        ax.axvline(20,  ls=":", color="k", alpha=0.35, lw=1)
        ax.axvline(243, ls=":", color="k", alpha=0.35, lw=1)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$\ell(\ell+1)C_\ell/2\pi$")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig("cls_fig3_cls.pdf", **SAVE)
    plt.savefig("cls_fig3_cls.png", **SAVE)
    plt.close();  print("Saved cls_fig3_cls")

    # =========================================================
    # Figure 4 — Integrand dC_ell/dz at ell=100
    # =========================================================
    ell_fixed = 100.0
    i_ell     = int(np.argmin(np.abs(ells - ell_fixed)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Limber integrand dCl/dz at ell={int(ell_fixed)}",
                 y=1.02)

    geom_keys = [("geom_kk", r"$\kappa\kappa$"),
                 ("geom_kg", r"$\kappa g$"),
                 ("geom_gg", r"$gg$")]

    for ax, (gk, title) in zip(axes, geom_keys):
        for bname, br in bin_res.items():
            info = br["info"]
            P_row = info["P_limber"][i_ell]
            intgd = P_row * info[gk]
            norm  = np.abs(intgd).max()
            ax.plot(z, intgd / norm, color=br["color"], label=bname)
            # mark z_eff
            ze_idx = {"geom_kk": 0, "geom_kg": 1, "geom_gg": 2}[gk]
            ax.axvline(br["z_eff_tuple"][ze_idx], color=br["color"],
                       ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel(r"$z$")
        ax.set_ylabel("Normalised integrand")
        ax.set_title(f"Integrand: {title}")
        ax.set_xlim(0, 3.0);  ax.legend()

    plt.tight_layout()
    plt.savefig("cls_fig4_integrand.pdf", **SAVE)
    plt.savefig("cls_fig4_integrand.png", **SAVE)
    plt.close();  print("Saved cls_fig4_integrand")

    # =========================================================
    # Figure 5 — Effective redshifts
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x      = np.arange(3)
    bnames = list(bin_res.keys())
    width  = 0.22
    spec_kw = [
        ("kk", r"$z_\mathrm{eff}^{\kappa\kappa}$", "#534AB7", -width),
        ("kg", r"$z_\mathrm{eff}^{\kappa g}$",     "#1D9E75",  0.0  ),
        ("gg", r"$z_\mathrm{eff}^{gg}$",            "#D85A30", +width),
    ]

    for key, label, sc, off in spec_kw:
        ze_idx = {"kk": 0, "kg": 1, "gg": 2}[key]
        vals   = [bin_res[b]["z_eff_tuple"][ze_idx] for b in bnames]
        bars   = ax.bar(x + off, vals, width=width,
                        label=label, color=sc, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Paper Table 2 z_eff values (dashed lines)
    for i, bname in enumerate(bnames):
        ze = BINS[bname]["z_eff"]
        ax.hlines(ze, i - 0.40, i + 0.40,
                  colors="k", linestyles="--", lw=1.2, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bnames)
    ax.set_ylabel(r"$z_\mathrm{eff}$")
    ax.set_title("Effective redshifts per bin\n"
                 "(dashed = paper Table 2)")
    ax.legend(ncol=3);  ax.set_ylim(0, 2.2)

    plt.tight_layout()
    plt.savefig("cls_fig5_zeff.pdf", **SAVE)
    plt.savefig("cls_fig5_zeff.png", **SAVE)
    plt.close();  print("Saved cls_fig5_zeff")

    # =========================================================
    # Figure 6 — Magnification bias impact on C_ell^kg
    # =========================================================
    print("Computing C_ell without magnification bias...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(r"Magnification bias: $C_\ell^{\kappa g}$ ratio "
                 r"(with / without $W^\mu$)", y=1.02)

    for ax, bname in zip(axes, bin_res.keys()):
        br  = bin_res[bname]
        phi_b = br["info"]["phi"]

        Cl_kk0, Cl_kg0, Cl_gg0, _, _ = compute_all_cls(
            cosmo=cosmo, z=z, ells=ells,
            Omega_m=Omega_m, H0=H0,
            b1=br["bias"], b2=0.0,
            phi=phi_b,
            s_mu=0.0,        # magnification OFF
        )

        ratio = br["Cl_kg"] / np.where(np.abs(Cl_kg0) > 0, Cl_kg0, np.nan)

        ax.semilogx(ells, ratio, color=br["color"], lw=lw)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"ratio")
        ax.set_title(f"{bname}  ($s_\\mu = {br['s_mu']:.3f}$)")
        ax.set_xlim(ells[0], ells[-1])

    plt.tight_layout()
    plt.savefig("cls_fig6_magnification.pdf", **SAVE)
    plt.savefig("cls_fig6_magnification.png", **SAVE)
    plt.close();  print("Saved cls_fig6_magnification")

    print("\nAll figures saved (cls_fig1 … cls_fig6).")
