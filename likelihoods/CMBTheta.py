# ============================================================
# Compressed CMB likelihood
# Based on arXiv:2502.07185
# ============================================================
#
# Observable vector: (R, l_a, omega_b h^2)
#
# R          = 100 sqrt(omega_m h^2) D_M(z*) / c   [shift parameter]
# l_a        = pi  D_M(z*) / r_s                   [acoustic scale]
# omega_b h^2                                       [physical baryon density]
#
# r_s = 144.43 Mpc is fixed — it is part of the data product from
# arXiv:2502.07185, derived from the full Planck 2018 CMB analysis.
# It must NOT be replaced by rd_sound_horizon() or any other formula,
# as that would change the definition of l_a and break consistency
# with the data vector.
# ============================================================

import numpy as np

class CMBThetaLikelihood:
    """
    Compressed CMB likelihood
    """

    # Sound horizon 
    RS_STAR = 144.43   # Mpc

    # Redshift of last scattering
    Z_STAR = 1090.0

    # Speed of light
    C_LIGHT = 299792.458  # km/s

    # Data vector [R, l_a, omega_b h^2] 
    DATA = np.array([1.7504, 301.77, 0.022371])

    # Full covariance matrix 
    COV = 1e-8 * np.array([
        [ 1559.83,  -1325.41,   -36.45],
        [-1325.41, 714691.80,   269.77],
        [  -36.45,    269.77,     2.10],
    ])

    def __init__(self):
        self.data    = self.DATA.copy()
        self.cov     = self.COV.copy()
        self.inv_cov = np.linalg.inv(self.cov)
        self._sigma  = np.sqrt(np.diag(self.cov))

    # ── Theory vector ─────────────────────────────────────────────

    def get_theory_vector(self, cosmo):
        """
        Compute (R, l_a, omega_b h^2) for a given cosmology.
        """
        DM         = cosmo.DM(self.Z_STAR)
        h          = cosmo.H0 / 100.0
        omega_m_h2 = cosmo.omega_m
        omega_b_h2 = cosmo.Omega_b * h**2

        R  = 100.0 * np.sqrt(omega_m_h2) * DM / self.C_LIGHT
        la = np.pi * DM / self.RS_STAR

        return np.array([R, la, omega_b_h2])

    # ── Log-likelihood ────────────────────────────────────────────

    def loglike(self, cosmo):
        """
        Compute the compressed CMB log-likelihood.

        Returns
        -------
        logL : float   -0.5 * chi^2
        chi2 : float
        """
        theory = self.get_theory_vector(cosmo)
        delta  = theory - self.data
        chi2   = float(delta @ self.inv_cov @ delta)
        return -0.5 * chi2, chi2

    # ── Diagnostics ───────────────────────────────────────────────

    def print_theory(self, cosmo):
        """Print theory vs data with pulls for debugging."""
        theory = self.get_theory_vector(cosmo)
        labels = ['R', 'l_a', 'omega_b h^2']
        print(f"CMBTheta (rs = {self.RS_STAR} Mpc fixed):")
        for lab, th, obs, sig in zip(labels, theory, self.data, self._sigma):
            pull = (th - obs) / sig
            print(f"  {lab:<12}: theory={th:.6f}  data={obs:.6f}  "
                  f"sigma={sig:.6f}  pull={pull:+.2f}")
