import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d

# Planck 2018 baryon density 
_OMEGA_B_PLANCK = 0.04930

class GenericCosmology:
    """
    Generic cosmological class.
    """

    c       = 299792.458   # km/s
    Z_PIVOT = 4.0          # segment boundary
    NZ_LOW  = 1_000        # points in [0, Z_PIVOT]
    NZ_HIGH = 5_000        # points in [Z_PIVOT, z_max] for CMB

    # ======================================================
    # Initialisation
    # ======================================================

    def __init__(
        self,
        H_of_z,
        H0,
        Omega_m,
        Omega_b=None,
        sigma8_0=0.8,
        Neff=3.044,
    ):
        if not callable(H_of_z):
            raise TypeError("H_of_z must be a callable H(z) in km/s/Mpc.")

        self.H       = H_of_z
        self.H0      = H0
        self.h       = H0 / 100.0

        self.Omega_m = Omega_m

        self.Omega_b = Omega_b if Omega_b is not None else _OMEGA_B_PLANCK

        self.sigma8_0 = sigma8_0
        self.Neff     = Neff

        self.omega_m = self.Omega_m * self.h**2
        self.omega_b = self.Omega_b * self.h**2

        # Internal caches — populated on first use
        self._distance_cache = None
        self._z_cache_max    = None

    # ======================================================
    # Distance cache — two-segment grid
    # ======================================================

    def _build_distance_cache(self, z_max):
        """
        Build (or extend) the distance interpolation grid.

        Uses a two-segment strategy: dense for z < Z_PIVOT,
        sparser for z in [Z_PIVOT, z_max] if needed.
        The cache is reused across calls until z_max exceeds
        the previously stored maximum.
        """
        if self._distance_cache is not None and z_max <= self._z_cache_max:
            return

        z_max = max(z_max, 1e-3)   # guard against z_max = 0

        # --------------------------------------------------
        # Segment 1: [0, min(z_max, Z_PIVOT)]
        # --------------------------------------------------
        z1_max = min(z_max, self.Z_PIVOT)
        z1     = np.linspace(0.0, z1_max, self.NZ_LOW)
        H1     = np.array([self.H(z) for z in z1], dtype=float)
        H1[~np.isfinite(H1)] = np.inf
        Dc1    = cumulative_trapezoid(self.c / H1, z1, initial=0.0)

        if z_max <= self.Z_PIVOT:
            z_grid  = z1
            H_grid  = H1
            Dc_grid = Dc1
        else:
            # --------------------------------------------------
            # Segment 2: [Z_PIVOT, z_max]  — sparser, smooth
            # --------------------------------------------------
            z2  = np.linspace(self.Z_PIVOT, z_max, self.NZ_HIGH)
            H2  = np.array([self.H(z) for z in z2], dtype=float)
            H2[~np.isfinite(H2)] = np.inf
            dDc = cumulative_trapezoid(self.c / H2, z2, initial=0.0)

            # Stitch: segment 2 starts exactly where segment 1 ended
            z_grid  = np.concatenate([z1,            z2[1:]])
            H_grid  = np.concatenate([H1,            H2[1:]])
            Dc_grid = np.concatenate([Dc1, Dc1[-1] + dDc[1:]])

        self._distance_cache = {
            "z":  z_grid,
            "Dc": Dc_grid,
            "H":  H_grid,
        }
        self._z_cache_max = z_max

    # ======================================================
    # Cosmological distances
    # ======================================================

    def comoving_distance(self, z):
        z     = np.atleast_1d(np.asarray(z, dtype=float))
        z_max = 1.05 * float(np.max(z))
        self._build_distance_cache(z_max)
        Dc = np.interp(z, self._distance_cache["z"], self._distance_cache["Dc"])
        return float(Dc[0]) if Dc.size == 1 else Dc

    def DL(self, z):
        z  = np.atleast_1d(np.asarray(z, dtype=float))
        Dc = self.comoving_distance(z)
        DL = (1.0 + z) * Dc
        return float(DL[0]) if DL.size == 1 else DL

    def DA(self, z):
        z  = np.atleast_1d(np.asarray(z, dtype=float))
        Dc = self.comoving_distance(z)
        DA = Dc / (1.0 + z)
        return float(DA[0]) if DA.size == 1 else DA

    def distance_modulus(self, z):
        return 5.0 * np.log10(self.DL(z)) + 25.0

    # ======================================================
    # BAO distances
    # ======================================================

    def DM(self, z):
        """Comoving angular diameter distance (= comoving distance, flat)."""
        return self.comoving_distance(z)

    def DH(self, z):
        """Hubble distance  c / H(z).  Uses the cached H grid."""
        z     = np.atleast_1d(np.asarray(z, dtype=float))
        z_max = 1.05 * float(np.max(z))
        self._build_distance_cache(z_max)
        H_val = np.interp(z, self._distance_cache["z"], self._distance_cache["H"])
        DH    = self.c / H_val
        return float(DH[0]) if DH.size == 1 else DH

    def DV(self, z):
        """Volume-averaged BAO distance."""
        z  = np.atleast_1d(np.asarray(z, dtype=float))
        DM = self.DM(z)
        DH = self.DH(z)
        DV = (DM**2 * z * DH) ** (1.0 / 3.0)
        return float(DV[0]) if DV.size == 1 else DV

    # ======================================================
    # Sound horizon (fitting formula)
    # ======================================================

    def rd_sound_horizon(self):
        """
        Comoving sound horizon at the drag epoch (Mpc). Fitting formula calibrated against CLASS/CAMB.
        """
        return (
            147.05
            * (self.omega_m / 0.1432) ** (-0.23)
            * (self.Neff    / 3.044)  ** (-0.10)
            * (self.omega_b / 0.02236) ** (-0.13)
        )

    # ======================================================
    # Linear growth
    # ======================================================

    def _growth_ode(self, a, y):
        """
        Linear growth ODE in scale factor a.

        D'' + [3/a + dlnH/dlna / a] D' = 3/2 * Omega_m(a) * D / a^2

        dlnH/dlna is computed analytically:
            dlnH/dlna = -3/2 * Omega_m(a)
        valid for GR with smooth dark energy (no anisotropic stress).
        This replaces the original central finite-difference stencil,
        which required three H(z) evaluations per ODE step.
        """
        D, dD_da = y
        z = 1.0 / a - 1.0
        H = self.H(z)

        Omega_m_a  = self.Omega_m * (1.0 + z)**3 * (self.H0 / H)**2
        dlnH_dlnA  = -1.5 * Omega_m_a          # exact for smooth DE in GR

        d2D_da2 = (
            -(3.0 / a + dlnH_dlnA / a) * dD_da
            + 1.5 * Omega_m_a * D / a**2
        )

        return [dD_da, d2D_da2]

    def _precompute_growth_grid(self, z_max=5.0, n_points=300):
        """
        Solve the growth ODE once and build interpolators for D(z) and f(z).

        Initial condition: deep matter domination at a_ini = 1/(1+50),
        where D ~ a to better than 10^{-5}. Using z_max as the lower
        boundary (the original choice) is less accurate when z_max < 50
        because the matter-dominated attractor is not yet established.
        """
        A_INI_Z = 50.0                         # start well in matter domination
        a_ini   = 1.0 / (1.0 + A_INI_Z)
        a_max   = 1.0

        a_grid  = np.logspace(np.log10(a_ini), 0.0, n_points)
        y0      = [a_ini, 1.0]                 # D ~ a, dD/da ~ 1 in MD

        sol = solve_ivp(
            self._growth_ode,
            [a_ini, a_max],
            y0,
            t_eval=a_grid,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
        )

        D      = sol.y[0]
        dD_da  = sol.y[1]

        # Normalise so D(z=0) = 1
        D0     = D[-1]
        D     /= D0
        dD_da /= D0

        # Growth rate f = d ln D / d ln a = a/D * dD/da
        f = a_grid * dD_da / D

        # z is decreasing as a increases: reverse so z is monotonically
        # increasing for interp1d.
        z_grid = 1.0 / a_grid - 1.0        # z_grid[0] = A_INI_Z, z_grid[-1] = 0

        z_asc = z_grid[::-1]               # 0 → z_max
        D_asc = D[::-1]
        f_asc = f[::-1]

        # fill_value: for z < 0 return D(0)=1; for z > z_max return D(z_max)
        # (the original tuple was inverted after the [::-1] reversal)
        self._D_interp = interp1d(
            z_asc, D_asc,
            kind="cubic",
            bounds_error=False,
            fill_value=(D_asc[0], D_asc[-1]),   # (left=z→0, right=z→large)
        )

        self._f_interp = interp1d(
            z_asc, f_asc,
            kind="cubic",
            bounds_error=False,
            fill_value=(f_asc[0], f_asc[-1]),
        )

    def growth_factor(self, z):
        """Linear growth factor D(z), normalised to D(z=0) = 1."""
        if not hasattr(self, "_D_interp"):
            self._precompute_growth_grid()
        return self._D_interp(z)

    def growth_rate(self, z):
        """Linear growth rate f(z) = d ln D / d ln a."""
        if not hasattr(self, "_f_interp"):
            self._precompute_growth_grid()
        return self._f_interp(z)

    def sigma8(self, z):
        """sigma8(z) = sigma8_0 * D(z)."""
        return self.sigma8_0 * self.growth_factor(z)

    def fsigma8(self, z):
        """f(z) * sigma8(z)."""
        return self.growth_rate(z) * self.sigma8(z)

    # ======================================================
    # Utility
    # ======================================================

    def Omega_m_z(self, z):
        """Matter density parameter at redshift z."""
        return self.Omega_m * (1.0 + z)**3 * (self.H0 / self.H(z))**2


# ======================================================================
# ScalarFieldCosmology
# ======================================================================

class ScalarFieldCosmology(GenericCosmology):
    """
    Cosmology driven by a quintessence scalar field.
    """

    def __init__(
        self,
        scalar_solver,
        H0,
        Omega_m,
        Omega_b=None,
        sigma8_0=0.8,
        Neff=3.044,
    ):
        self.scalar = scalar_solver

        z_grid        = self.scalar.z()
        Omega_phi_grid = self.scalar.Omega_phi()

        # Interpolator for Omega_phi(z); z_grid is decreasing from the solver
        self._Omega_phi_interp = lambda z: np.interp(
            z,
            z_grid[::-1],
            Omega_phi_grid[::-1],
        )

        def H_of_z(z):
            Omega_phi = self._Omega_phi_interp(z)
            Omega_phi = max(Omega_phi, 0.0)   # guard against tiny negatives
            return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_phi)

        super().__init__(
            H_of_z=H_of_z,
            H0=H0,
            Omega_m=Omega_m,
            Omega_b=Omega_b,
            sigma8_0=sigma8_0,
            Neff=Neff,
        )
