# ============================================================
# RSD likelihood
# Samples from arXiv:2201.07829
# ============================================================

import os
import numpy as np

class RSDFastLikelihood:
    """
    Likelihood para dados RSD (f*sigma8).
    """

    def __init__(self, data_dir="data"):
        # -------------------------------------------------
        # File path
        # -------------------------------------------------
        data_file = os.path.join(data_dir, "RSD.txt")

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------
        data = np.loadtxt(data_file, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self.z = data[:, 0].astype(np.float64)
        self.fs8_obs = data[:, 1].astype(np.float64)
        self.fs8_err = data[:, 2].astype(np.float64)

        self.n_data = len(self.z)

        # Inverse variance (independent points)
        self.inv_var = 1.0 / (self.fs8_err ** 2)

        # Optional sanity check
        if not np.all(np.isfinite(self.inv_var)):
            raise ValueError("RSD errors contain non-finite values")

    # -------------------------------------------------
    # Log-likelihood
    # -------------------------------------------------
    def __call__(self, cosmo):
        fs8_theo = cosmo.fsigma8(self.z)
        residuals = fs8_theo - self.fs8_obs
        chi2 = np.sum(residuals**2 * self.inv_var)
        return -0.5 * chi2

    # -------------------------------------------------
    # Chi-square only (optional)
    # -------------------------------------------------
    def chi2(self, cosmo):
        fs8_theo = cosmo.fsigma8(self.z)
        residuals = fs8_theo - self.fs8_obs
        return np.sum(residuals**2 * self.inv_var)

