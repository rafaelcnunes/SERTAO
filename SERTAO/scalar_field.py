"""
SERTAO/scalar_field.py
----------------------
Quintessence scalar field dynamics using autonomous variables.

The canonical quintessence action gives a scalar field φ with
kinetic energy K = φ̇²/2 and potential V(φ). In a flat FLRW
background the field obeys:

    φ̈ + 3Hφ̇ + V,φ = 0

Using the dimensionless phase-space variables (Copeland et al. 1998):

    x ≡ φ̇ / (√6 H Mpl)       [kinetic fraction  x² = ρ_K/ρ_crit]
    y ≡ √V / (√3 H Mpl)        [potential fraction y² = ρ_V/ρ_crit]
    λ ≡ -Mpl V,φ / V            [potential slope]

and the independent variable N = ln a, the equations of motion become
the autonomous system implemented in QuintessenceDynamics.equations().

The dark energy density parameter is:
    Ω_φ(N) = x² + y²
and the equation of state is:
    w_φ(N) = (x² - y²) / (x² + y²)

Supported potentials
--------------------
ExponentialPotential  V ∝ exp(-λφ)     → Γ = 1          (tracker)
PowerLawPotential     V ∝ φⁿ           → Γ = (n-1)/n    (thawing)
PNGBPotential         V ∝ 1+cos(φ/f)  → Γ ≈ 1 - 1/λ²  (pseudo-NGB)

Adding a new potential: subclass GenericPotential and implement Gamma(λ).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────
# Potentials
# ──────────────────────────────────────────────────────────────

class GenericPotential:
    """
    Base class for quintessence potentials.

    Subclasses must implement Gamma(lam), which encodes the shape:

        Γ(λ) = V · V,φφ / (V,φ)²

    The evolution of λ along the field trajectory is:

        dλ/dN = -√6 λ² (Γ - 1) x

    Limits:
        Γ = 1        → λ is constant (exponential potential)
        Γ > 1        → λ grows with time (freezing models)
        Γ < 1        → λ decays with time (thawing models)
    """

    def Gamma(self, lam):
        raise NotImplementedError


class ExponentialPotential(GenericPotential):
    """
    V(φ) = V₀ exp(−λφ/Mpl)

    Γ = 1 exactly → λ is constant → autonomous system reduces to 2D.
    Admits attractor (tracker) solutions for λ² < 3(1+w_bg).
    """

    def Gamma(self, lam):
        return 1.0


class PowerLawPotential(GenericPotential):
    """
    V(φ) = V₀ (φ/Mpl)ⁿ

    Γ = (n−1)/n  (constant, independent of λ).
    Thawing model: field is frozen at high z, rolls at late times.
    """

    def __init__(self, n=2.0):
        if n <= 1.0:
            raise ValueError(f"PowerLawPotential requires n > 1, got n={n}")
        self.n = float(n)

    def Gamma(self, lam):
        return (self.n - 1.0) / self.n


class PNGBPotential(GenericPotential):
    """
    V(φ) = V₀ [1 + cos(φ/f)]   (pseudo-Nambu-Goldstone boson)

    Γ ≈ 1 − 1/λ²  (standard approximation in the literature).

    Warning: Γ diverges as λ → 0. A guard λ_min = 0.1 is
    applied automatically to avoid numerical blow-up.
    """

    _LAM_MIN = 0.1   # below this Gamma becomes very negative → unstable

    def Gamma(self, lam):
        if abs(lam) < self._LAM_MIN:
            # Use L'Hôpital-style limit: Gamma → 0 as λ→0 for this approximation
            lam = np.sign(lam) * self._LAM_MIN if lam != 0 else self._LAM_MIN
        return 1.0 - 1.0 / lam**2


# ──────────────────────────────────────────────────────────────
# Quintessence dynamics
# ──────────────────────────────────────────────────────────────

class QuintessenceDynamics:
    """
    Solves the quintessence autonomous system (x, y, λ) as a
    function of e-folds N = ln a.

    Variables
    ---------
    x = φ̇ / (√6 H Mpl)     kinetic energy fraction  (x² = Ω_K)
    y = √V  / (√3 H Mpl)    potential energy fraction (y² = Ω_V)
    λ = −Mpl V,φ / V         field slope

    The flat-universe constraint is automatically satisfied by
    the dynamical system: Ω_φ = x²+y² ≤ 1, and the matter/radiation
    contributions fill the remainder (tracked by the background).

    Equations (Copeland, Liddle & Wands 1998, eqs 17-19)
    -----------------------------------------------------
    dx/dN = −3x + λ√(3/2) y² + (3/2)x(1 + x² − y²)
    dy/dN = −λ√(3/2) xy   + (3/2)y(1 + x² − y²)
    dλ/dN = −√6 λ²(Γ−1) x

    Parameters
    ----------
    potential : GenericPotential
        The quintessence potential (provides Gamma).
    Mpl : float
        Reduced Planck mass (default 1 in units where 8πG=1).
    """

    def __init__(self, potential, Mpl=1.0):
        self.potential = potential
        self.Mpl       = Mpl
        self._solved   = False

    # ── ODE right-hand side ──────────────────────────────────

    def equations(self, N, Y):
        x, y, lam = Y

        Gamma = self.potential.Gamma(lam)

        # Guard: keep x, y physical
        # Ω_φ = x²+y² must not exceed 1 significantly
        Omega_phi = x**2 + y**2
        if Omega_phi > 1.0 + 1e-6:
            # Return strong damping to push back into physical region
            return [-10*x, -10*y, 0.0]

        # Standard equations (Copeland et al. 1998)
        sqrt6 = np.sqrt(6.0)
        sq3o2 = np.sqrt(1.5)

        factor = 1.5 * (1.0 + x**2 - y**2)

        dx   = -3.0*x + lam*sq3o2*y**2 + x*factor
        dy   = -lam*sq3o2*x*y          + y*factor
        dlam = -sqrt6 * lam**2 * (Gamma - 1.0) * x

        return [dx, dy, dlam]

    # ── Solve ────────────────────────────────────────────────

    def solve(
        self,
        N_ini=-7.0,
        N_fin=0.0,
        x_ini=1e-8,
        y_ini=1e-5,
        lam_ini=0.5,
        npts=500,
        rtol=1e-8,
        atol=1e-10,
    ):
        """
        Integrate the quintessence ODE from N_ini to N_fin.

        Default initial conditions place the field deep in the
        matter-dominated era (N_ini = −7 → z ≈ 1095) where the
        field is nearly frozen (x ≪ 1, y ≪ 1).

        Parameters
        ----------
        N_ini : float    Start e-fold (default -7, z ≈ 1095)
        N_fin : float    End e-fold   (default  0, z = 0)
        x_ini : float    Initial kinetic fraction
        y_ini : float    Initial potential fraction
        lam_ini : float  Initial slope parameter λ
        npts  : int      Number of output points (default 500)
        rtol, atol : float  ODE tolerances

        Returns
        -------
        sol : OdeResult  (scipy)
        """
        N_eval = np.linspace(N_ini, N_fin, npts)

        sol = solve_ivp(
            self.equations,
            (N_ini, N_fin),
            [x_ini, y_ini, lam_ini],
            t_eval=N_eval,
            method='DOP853',   # 8th-order Dormand-Prince, more accurate
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Quintessence ODE failed: {sol.message}\n"
                f"  lam_ini={lam_ini}, x_ini={x_ini}, y_ini={y_ini}"
            )

        self.N   = sol.t
        self.x   = sol.y[0]
        self.y   = sol.y[1]
        self.lam = sol.y[2]
        self._solved = True

        # Build fast interpolators in z (increasing order)
        z_grid = np.exp(-self.N) - 1.0    # decreasing: z_max → 0
        idx    = np.argsort(z_grid)       # sort ascending
        z_asc  = z_grid[idx]
        Om_asc = (self.x**2 + self.y**2)[idx]
        w_asc  = self._w_phi_raw()[idx]

        self._z_min = float(z_asc[0])
        self._z_max = float(z_asc[-1])

        self._Om_interp = interp1d(
            z_asc, Om_asc,
            kind='cubic', bounds_error=False,
            fill_value=(Om_asc[0], Om_asc[-1]),
        )
        self._w_interp = interp1d(
            z_asc, w_asc,
            kind='cubic', bounds_error=False,
            fill_value=(w_asc[0], w_asc[-1]),
        )

        return sol

    # ── Derived quantities ───────────────────────────────────

    def _w_phi_raw(self):
        """Equation of state on the ODE grid (handles x=y=0 safely)."""
        num  = self.x**2 - self.y**2
        denom = self.x**2 + self.y**2
        # Where Omega_phi ≈ 0 the field is not dynamical; use w = -1
        safe = np.where(denom > 1e-30, denom, 1.0)
        return np.where(denom > 1e-30, num / safe, -1.0)

    def Omega_phi(self, z=None):
        """
        Dark energy density parameter Ω_φ(z).

        Parameters
        ----------
        z : float, array, or None
            If None, return the full ODE grid array.
            If scalar/array, interpolate.
        """
        self._check_solved()
        if z is None:
            return self.x**2 + self.y**2
        return float(self._Om_interp(z)) if np.ndim(z)==0 \
               else self._Om_interp(np.asarray(z))

    def w_phi(self, z=None):
        """
        Equation of state w_φ(z).

        Parameters
        ----------
        z : float, array, or None
        """
        self._check_solved()
        if z is None:
            return self._w_phi_raw()
        return float(self._w_interp(z)) if np.ndim(z)==0 \
               else self._w_interp(np.asarray(z))

    def z(self):
        """Redshift grid (decreasing, from N_ini to N_fin)."""
        self._check_solved()
        return np.exp(-self.N) - 1.0

    def z_range(self):
        """(z_min, z_max) covered by the ODE solution."""
        self._check_solved()
        return self._z_min, self._z_max

    # ── Helpers ──────────────────────────────────────────────

    def _check_solved(self):
        if not self._solved:
            raise RuntimeError(
                "Call solve() before accessing derived quantities."
            )
