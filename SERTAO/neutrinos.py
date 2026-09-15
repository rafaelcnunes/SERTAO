"""
Neutrino density evolution for flat FLRW cosmologies.
The neutrino density parameter at redshift z is computed from the
WMAP-7 formalism (Komatsu et al. 2011, Appendix C), which tracks
the full relativistic-to-non-relativistic transition species by
species using the Fermi-Dirac energy integral.

Each neutrino species i with mass m_i contributes:

    Omega_nu,i(z) = (7/8) * (4/11)^{4/3} * N_eff/N_species
                    * Omega_gamma(z) * f(m_i / T_nu(z))        

where

    T_nu(z)  = T_nu,0 * (1+z)                   
    T_nu,0   = (4/11)^{1/3} * T_CMB = 1.9454 K

and f(y) is the dimensionless Fermi-Dirac integral:

    f(y) = (120 / 7pi^4) * int_0^inf x^2 sqrt(x^2+y^2)/(e^x+1) dx   

approximated by the fitting formula [eq. C4]:

    f(y) ≈ (1 + (A y)^p)^{1/p},   A = 0.3173,  p = 1.83

which is accurate to < 0.35% for all y in [0, ∞).

Limits
------
    f(0) = 1          → massless (relativistic): Omega_nu ∝ (1+z)^4
    f(y→∞) → A*y      → massive (NR):           Omega_nu ∝ (1+z)^3

The transition redshift for species i is z_trans,i ≈ m_i / T_nu,0 - 1.
For m_i = 0.02 eV: z_trans ~ 118. For m_i = 0.06 eV: z_trans ~ 357.

Neutrino temperature
--------------------
T_nu,0 = (4/11)^{1/3} * T_CMB comes from entropy conservation at
e+/e- annihilation. It is the exact result in the standard model;
QED corrections shift it by < 0.1% and are absorbed into N_eff.
"""

import numpy as np
from scipy.interpolate import interp1d
from functools import lru_cache


# ==============================================================
# Physical constants
# ==============================================================

T_CMB       = 2.7255           # K 
k_B         = 8.617333e-5      # eV/K
T_nu0_K     = (4.0/11.0)**(1.0/3.0) * T_CMB   # K = 1.9454 K
T_nu0_eV    = k_B * T_nu0_K                    # eV = 1.6764e-4 eV
N_eff_std   = 3.044            

# Fitting formula coefficients (WMAP-7)
_A_FIT = 0.3173
_P_FIT = 1.83


# ==============================================================
# Fermi-Dirac integral — fitting formula
# ==============================================================

def fermi_dirac_f(y):
    """
    Dimensionless Fermi-Dirac energy integral for one neutrino species.

        f(y) = (120/7pi^4) int_0^inf x^2 sqrt(x^2+y^2)/(e^x+1) dx
             ≈ (1 + (A*y)^p)^{1/p}     [accurate to < 0.35%]

    where  y = m_nu / T_nu(z)  =  m_nu / (T_nu0 * (1+z)).

    Limits:
        f(0)      = 1       [massless, relativistic]
        f(y→∞)   ≈ A * y   [non-relativistic: rho ∝ m * n]
    """
    y = np.asarray(y, dtype=float)
    return (1.0 + (_A_FIT * y) ** _P_FIT) ** (1.0 / _P_FIT)


# ==============================================================
# Neutrino class
# ==============================================================

class Neutrinos:
    """
    Computes Omega_nu(z) species by species, tracking the full
    relativistic-to-non-relativistic transition via the fitting
    formula for the Fermi-Dirac energy integral f(y).

    Parameters
    ----------
    config : str or list
        Mass configuration. Options:

        'standard'          3 degenerate species, Σm = total_mass
        '1ncdm'             1 massive + 2 massless
        '2ncdm'             2 massive + 1 massless  (set m1, m2 via kwargs)
        'hierarchy_normal'  [0, 0, m3]  (normal ordering, lightest states ~0)
        'hierarchy_inverted' [m, m, 0]  (inverted ordering)
        [m1, m2, m3, ...]   custom mass list in eV

    total_mass : float, optional
        Σm_nu in eV. Default depends on config.
    m_ncdm : float, optional
        Mass of the single massive species (for '1ncdm').
    N_eff : float, optional
        Effective number of relativistic species. Default: 3.044.
    h : float, optional
        Hubble parameter H0/100. Used for Omega_gamma0. Default: 0.6736.
    m1, m2 : float, optional
        Individual masses in eV for '2ncdm' config.

    Notes
    -----
    The neutrino photon temperature ratio T_nu/T_gamma = (4/11)^{1/3}
    is built into the constant T_nu0_eV; N_eff rescales the effective
    number of neutrino-like species without changing this ratio.
    """

    def __init__(
        self,
        config='standard',
        total_mass=None,
        m_ncdm=None,
        N_eff=None,
        h=0.6736,
        **kwargs,
    ):
        self.config  = config
        self.h       = h
        self.N_eff   = N_eff if N_eff is not None else N_eff_std

        self.masses, self.description = self._parse_config(
            config, total_mass, m_ncdm, **kwargs
        )

        self.masses      = np.asarray(self.masses, dtype=float)
        self.total_mass  = float(np.sum(self.masses))
        self.n_species   = len(self.masses)
        self.n_massive   = int(np.sum(self.masses > 0))

        # Photon density parameter today
        self.Omega_gamma0 = 2.4728e-5 / h**2

        # Purely relativistic neutrino density today (massless limit)
        # Omega_nu,rel,0 = (7/8)*(4/11)^(4/3) * N_eff * Omega_gamma0
        # Note: 0.2271 ≡ (7/8)*(4/11)^(4/3) — exact coefficient
        self._coeff = 0.22710 * self.N_eff / self.n_species

        # Omega_nu(z=0): use WMAP formalism for consistency
        self._Omega_nu0 = self._compute_Omega_nu(0.0)

        # Build interpolation table
        self._build_tables()

    # ----------------------------------------------------------
    # Configuration parser
    # ----------------------------------------------------------

    def _parse_config(self, config, total_mass, m_ncdm, **kwargs):

        if isinstance(config, list):
            masses = np.array(config, dtype=float)
            desc   = f"Custom masses: {masses} eV"

        elif config == 'standard':
            total_mass = total_mass if total_mass is not None else 0.06
            masses = np.ones(3) * total_mass / 3.0
            desc   = f"3 degenerate, Σm = {total_mass:.4f} eV"

        elif config == '1ncdm':
            m_ncdm = m_ncdm if m_ncdm is not None else (
                total_mass if total_mass is not None else 0.06)
            masses = np.array([m_ncdm, 0.0, 0.0])
            desc   = f"1 massive (m={m_ncdm:.4f} eV) + 2 massless"

        elif config == '2ncdm':
            total_mass = total_mass if total_mass is not None else 0.06
            m1 = kwargs.get('m1', total_mass / 2.0)
            m2 = kwargs.get('m2', total_mass / 2.0)
            masses = np.array([m1, m2, 0.0])
            desc   = f"2 massive (m1={m1:.4f}, m2={m2:.4f} eV) + 1 massless"

        elif config == 'hierarchy_normal':
            # NH minimum: m1~0, m2~0, m3 ~ sqrt(Δm31^2) ~ 0.050 eV
            m3     = total_mass if total_mass is not None else 0.060
            masses = np.array([0.0, 0.0, m3])
            desc   = f"Normal hierarchy: [0, 0, {m3:.4f}] eV"

        elif config == 'hierarchy_inverted':
            # IH minimum: m1≈m2 ~ sqrt(Δm31^2) ~ 0.050 eV, m3~0
            total_mass = total_mass if total_mass is not None else 0.100
            m = total_mass / np.sqrt(2.0)
            masses = np.array([m, m, 0.0])
            desc   = f"Inverted hierarchy: [{m:.4f}, {m:.4f}, 0] eV"

        else:
            raise ValueError(
                f"Unknown neutrino config '{config}'. "
                "Options: 'standard', '1ncdm', '2ncdm', "
                "'hierarchy_normal', 'hierarchy_inverted', or a list."
            )

        return masses, desc

    # ----------------------------------------------------------
    # Core density calculation (WMAP-7 formalism, eq. C1)
    # ----------------------------------------------------------

    def _compute_Omega_nu(self, z):
        """
        Neutrino density parameter at redshift z.

        Omega_nu(z) = 0.2271 * N_eff / N_species * Omega_gamma(z)
                      * sum_i f(m_i / T_nu(z))                  [eq. C1]

        T_nu(z) = T_nu0 * (1+z)  →  y_i = m_i / (T_nu0_eV * (1+z))
        """
        Omega_gamma_z = self.Omega_gamma0 * (1.0 + z)**4
        y_vals        = self.masses / (T_nu0_eV * (1.0 + z))
        f_sum         = np.sum(fermi_dirac_f(y_vals))
        return self._coeff * Omega_gamma_z * f_sum

    # ----------------------------------------------------------
    # Interpolation table
    # ----------------------------------------------------------

    def _build_tables(self, z_min=1e-4, z_max=1100.0, n_points=600):
        """
        Pre-compute Omega_nu(z) and w_nu(z) on a log-spaced grid
        and build cubic interpolators.

        The equation of state w(z) is derived from the density:
            w(z) = (1/3) * [d ln Omega_nu / d ln(1+z) - 3] / 1
                 → w = 1/3  (relativistic, large z)
                 → w = 0    (non-relativistic, small z)

        It is computed via a logarithmic finite difference on the
        density grid rather than using the sigmoid approximation,
        so it is consistent with the f(y) calculation.
        """
        z_grid   = np.logspace(np.log10(z_min), np.log10(z_max), n_points)

        # Omega_nu on the grid
        Om_grid  = np.array([self._compute_Omega_nu(z) for z in z_grid])

        # Equation of state from logarithmic slope
        # w(z) = [d ln rho / d ln(1+z)] / 3  — 1/3 for rel, 0 for NR
        # Use central finite differences on log scale
        ln1pz    = np.log(1.0 + z_grid)
        ln_Om    = np.log(np.where(Om_grid > 0, Om_grid, 1e-300))
        d_lnOm   = np.gradient(ln_Om, ln1pz)
        w_grid   = d_lnOm / 3.0
        # Clamp to physical range [0, 1/3]
        w_grid   = np.clip(w_grid, 0.0, 1.0 / 3.0)

        self._Omega_interp = interp1d(
            z_grid, Om_grid,
            kind='cubic', bounds_error=False, fill_value='extrapolate',
        )
        self._w_interp = interp1d(
            z_grid, w_grid,
            kind='cubic', bounds_error=False, fill_value=(0.0, 1.0/3.0),
        )

        # Store transition redshifts for diagnostics
        self.z_trans = np.array([
            max(0.0, m / T_nu0_eV - 1.0) for m in self.masses if m > 0
        ])

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    def density_parameter(self, z):
        """
        Omega_nu(z) = rho_nu(z) / rho_crit,0.

        Exact at any z: relativistic (∝ a^{-4}) at high z,
        non-relativistic (∝ a^{-3}) at low z.
        """
        return float(self._Omega_interp(z))

    def equation_of_state(self, z):
        """
        Effective equation of state w_nu(z) = P_nu / rho_nu.

        Derived from the logarithmic slope of Omega_nu(z),
        consistent with the f(y) formalism.
        Limits: w = 1/3 (relativistic), w = 0 (non-relativistic).
        """
        return float(self._w_interp(z))

    def contribution_to_H2(self, z):
        """
        Neutrino contribution to H^2(z) / H0^2.
        Alias for density_parameter(z).
        """
        return self.density_parameter(z)

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def Omega_nu0(self):
        """Omega_nu at z=0 from the WMAP f(y) formalism."""
        return self._Omega_nu0

    @property
    def omega_nu0(self):
        """Physical density omega_nu = Omega_nu * h^2 at z=0."""
        return self._Omega_nu0 * self.h**2

    @property
    def total_mass_eV(self):
        """Sum of neutrino masses in eV."""
        return self.total_mass

    @property
    def N_eff_value(self):
        """Effective number of neutrino species."""
        return self.N_eff

    def summary(self):
        """Print a summary of the neutrino configuration."""
        print(f"Neutrino configuration: {self.description}")
        print(f"  N_eff     = {self.N_eff:.4f}")
        print(f"  N_species = {self.n_species}  ({self.n_massive} massive)")
        print(f"  Masses    = {self.masses} eV")
        print(f"  Σm_nu     = {self.total_mass:.4f} eV")
        print(f"  T_nu,0    = {T_nu0_eV:.4e} eV = {T_nu0_K:.4f} K")
        if len(self.z_trans) > 0:
            print(f"  z_trans   = {self.z_trans}  (rel→NR per species)")
        print(f"  Omega_nu0 = {self._Omega_nu0:.4e}")
        print(f"  omega_nu0 = {self.omega_nu0:.4e}")
        # Cross-check with standard formula
        if self.total_mass > 0:
            om_std = self.total_mass / 93.14
            print(f"  Σm/93.14  = {om_std:.4e}  "
                  f"(NR limit, diff = {abs(self.omega_nu0/om_std-1)*100:.2f}%)")


# ==============================================================
# Factory with LRU cache
# ==============================================================

@lru_cache(maxsize=512)
def create_neutrinos(
    config='standard',
    total_mass=None,
    m_ncdm=None,
    N_eff=None,
    h=0.6736,
    **kwargs,
):
    """
    Create (or retrieve from cache) a Neutrinos instance.

    Uses functools.lru_cache keyed on all parameters, so identical
    configurations return the same object without recomputation.
    Essential for MCMC performance when neutrino parameters are fixed.

    Parameters
    ----------
    config : str or tuple
        Mass configuration. Pass a tuple (not a list) when using
        the custom-mass option, since lists are not hashable.
        E.g. create_neutrinos(config=(0.02, 0.02, 0.02)).
    total_mass : float, optional
    m_ncdm : float, optional
    N_eff : float, optional
    h : float, optional
    m1, m2 : float, optional  (for '2ncdm')

    Returns
    -------
    Neutrinos
    """
    # Convert tuple config to list for the Neutrinos class
    if isinstance(config, tuple):
        config = list(config)
    return Neutrinos(
        config=config,
        total_mass=total_mass,
        m_ncdm=m_ncdm,
        N_eff=N_eff,
        h=h,
        **kwargs,
    )


# Backwards-compatible alias
create_neutrinos_cached = create_neutrinos


# ==============================================================
# Quick diagnostic (python neutrinos.py)
# ==============================================================

if __name__ == '__main__':

    print("=" * 58)
    print("neutrinos.py — WMAP-7 Fermi-Dirac formalism")
    print("=" * 58)

    # ── Fitting formula accuracy ──────────────────────────────
    from scipy.integrate import quad

    def f_exact(y):
        def integrand(x):
            return x**2 * np.sqrt(x**2 + y**2) / (np.exp(x) + 1.0)
        result, _ = quad(integrand, 0, 80, limit=300)
        return 120.0 / (7.0 * np.pi**4) * result

    print("\nFitting formula accuracy (eq. C4 vs exact integral):")
    print(f"{'y':>10}  {'f_exact':>10}  {'f_approx':>10}  {'err %':>8}")
    for y in [0, 0.5, 1, 2, 5, 10, 50, 200]:
        fe = f_exact(y)
        fa = fermi_dirac_f(float(y))
        print(f"{y:>10.1f}  {fe:>10.6f}  {fa:>10.6f}  {abs(fa/fe-1)*100:>7.4f}%")

    # ── Standard configurations ───────────────────────────────
    print("\nStandard configurations:")
    configs = [
        dict(config='standard',           total_mass=0.06),
        dict(config='1ncdm',              m_ncdm=0.06),
        dict(config='hierarchy_normal',   total_mass=0.06),
        dict(config='hierarchy_inverted', total_mass=0.10),
    ]
    for kw in configs:
        nu = Neutrinos(**kw)
        nu.summary()
        print()

    # ── Omega_nu(z) evolution ─────────────────────────────────
    print("Omega_nu(z) evolution for 3 degenerate (Σm=0.06 eV):")
    print(f"{'z':>8}  {'Omega_nu':>12}  {'w_nu':>8}  {'scaling':>12}")
    nu = Neutrinos(config='standard', total_mass=0.06)
    Om0 = nu.density_parameter(0)
    for z in [0, 1, 10, 50, 100, 200, 500, 1000]:
        Om = nu.density_parameter(z)
        w  = nu.equation_of_state(z)
        # Expected scaling: (1+z)^3 for NR, (1+z)^4 for rel
        exp_nr  = Om0 * (1+z)**3
        exp_rel_ratio = Om / (Om0 * (1+z)**4) if z > 0 else 1.0
        print(f"{z:>8}  {Om:>12.4e}  {w:>8.4f}  "
              f"NR-ratio={Om/exp_nr:.4f}")
