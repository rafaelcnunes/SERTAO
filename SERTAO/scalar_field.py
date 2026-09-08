import numpy as np
from scipy.integrate import solve_ivp

# ==========================================================
# Generic potential via Gamma
# ==========================================================

class GenericPotential:
    """
    Generic quintessence potential defined
    """

    def Gamma(self, lam):
        """
        Must be implemented by the user.
        """
        raise NotImplementedError


class ExponentialPotential(GenericPotential):
    """
    V(phi) = V0 exp(-lambda phi)  => Gamma = 1
    """

    def Gamma(self, lam):
        return 1.0


class PowerLawPotential(GenericPotential):
    """
    V(phi) = V0 * phi^n  => Gamma = (n - 1)/n
    """

    def __init__(self, n=2.0):
        self.n = n

    def Gamma(self, lam):
        return (self.n - 1.0) / self.n


class PNGBPotential(GenericPotential):
    """
    V(phi) = V0 [1 + cos(phi/f)]
    """

    def __init__(self, f=1.0):
        self.f = f

    def Gamma(self, lam):
        # Standard approximation used in the literature
        return 1.0 - (1.0 / lam**2)


# ==========================================================
# Quintessence dynamics
# ==========================================================

class QuintessenceDynamics:
    """
    Quintessence dynamical system in (x, y, lambda).
    """

    def __init__(self, potential, Mpl=1.0):
        self.potential = potential
        self.Mpl = Mpl

    def equations(self, N, Y):
        x, y, lam = Y

        Gamma = self.potential.Gamma(lam)

        dx = (
            -3.0 * x
            + lam * np.sqrt(3.0 / 2.0) * y**2
            + 1.5 * x * (1.0 + x**2 - y**2)
        )

        dy = (
            -lam * np.sqrt(3.0 / 2.0) * x * y
            + 1.5 * y * (1.0 + x**2 - y**2)
        )

        dlam = -np.sqrt(6.0) * lam**2 * (Gamma - 1.0) * x

        return [dx, dy, dlam]

    def solve(
        self,
        N_ini=-7.0,
        N_fin=0.0,
        x_ini=1e-8,
        y_ini=1e-5,
        lam_ini=0.5,
        npts=800,
    ):
        N_eval = np.linspace(N_ini, N_fin, npts)

        sol = solve_ivp(
            self.equations,
            (N_ini, N_fin),
            [x_ini, y_ini, lam_ini],
            t_eval=N_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        self.N = sol.t
        self.x, self.y, self.lam = sol.y

        return sol

    # -------------------------
    # Derived quantities
    # -------------------------

    def Omega_phi(self):
        return self.x**2 + self.y**2

    def w_phi(self):
        return (self.x**2 - self.y**2) / (self.x**2 + self.y**2)

    def z(self):
        return np.exp(-self.N) - 1.0







