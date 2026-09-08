# ============================================================
# CC likelihood
# data from arXiv:1601.01701 and https://gitlab.com/mmoresco/CCcovariance
# ============================================================

import os
import numpy as np

class CC:
    """
    Cosmic Chronometers likelihood with full covariance matrix.
    """

    def __init__(self, data_dir="data"):
        data_file = os.path.join(data_dir, "HzTable_MM_BC03.txt")
        cov_file  = os.path.join(data_dir, "cov_cc_MM_BC03.txt")

        # ── Load data ─────────────────────────────────────────
        data = np.loadtxt(data_file, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)

        self.z     = data[:, 0].astype(np.float64)
        self.H_obs = data[:, 1].astype(np.float64)
        self.n_data = len(self.z)

        # ── Load and invert covariance matrix ─────────────────
        self.cov = np.loadtxt(cov_file)

        if self.cov.shape != (self.n_data, self.n_data):
            raise ValueError(
                f"CC covariance matrix shape {self.cov.shape} "
                f"!= ({self.n_data}, {self.n_data})"
            )

        self.invcov = np.linalg.inv(self.cov)

    # ── Log-likelihood ─────────────────────────────────────────

    def __call__(self, cosmo):

        H_theo = np.array([cosmo.H(zi) for zi in self.z])

        delta = H_theo - self.H_obs
        chi2  = delta @ self.invcov @ delta

        return -0.5 * chi2
