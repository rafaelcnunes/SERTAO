# ============================================================
# Compressed CMB likelihood
# Information from arXiv:2502.07185
# ============================================================

import numpy as np

class CMBThetaLikelihood:

    def __init__(self):

        # ----------------------------------
        # Settings
        # ----------------------------------
        self.z_star = 1090.0
        self.c = 299792.458  # km/s

        # ----------------------------------
        # Data vector
        # ----------------------------------
        self.data = np.array([
            1.7504,
            301.77,
            0.022371
        ])

        # ----------------------------------
        # Covariance matrix 
        # ----------------------------------
        self.cov = 1e-8 * np.array([
            [1559.83,   -1325.41,  -36.45],
            [-1325.41, 714691.80,  269.77],
            [-36.45,     269.77,    2.10],
        ])

        self.inv_cov = np.linalg.inv(self.cov)

    # ==========================================
    # Compute R, l_a, omega_b h2
    # ==========================================
    def get_theory_vector(self, cosmo):

        z = self.z_star

        DM = cosmo.DM(z)
        rs = 144.43
        # Future option:
        # rs = cosmo.rs_at_z(z)

        h = cosmo.H0 / 100.0

        # Physical densities
        omega_b_h2 = cosmo.Omega_b * h**2
        omega_m_total_h2 = cosmo.omega_m

        # Shift parameter
        R = 100.0 * np.sqrt(omega_m_total_h2) * DM / self.c

        # Acoustic scale
        la = np.pi * DM / rs

        return np.array([R, la, omega_b_h2])

    # ==========================================
    # Log-likelihood
    # ==========================================
    def loglike(self, cosmo):

        theory = self.get_theory_vector(cosmo)
        delta = theory - self.data

        chi2 = delta @ self.inv_cov @ delta
        logL = -0.5 * chi2

        return logL, chi2










