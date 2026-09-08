# ============================================================
# 2D BAO angular likelihood
# arXiv:2510.16141
# ============================================================
import os
import numpy as np

class BAOAngular:

    def __init__(self, data_dir="data"):
        # -------------------------------------------------
        # File paths
        # -------------------------------------------------
        
        data_file = os.path.join(data_dir, "bao_t_on_data.txt")
        cov_file  = os.path.join(data_dir, "cov_matrix_BAOT.txt")

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------
        data = np.loadtxt(data_file)

        (
            z_eff, theta_bao, sigma_theta,
            z_ON, theta_ON, sigma_ON
        ) = data.T

        # Remove zero entries
        mask_bao = z_eff != 0.0
        mask_on  = z_ON  != 0.0

        self.z_eff = z_eff[mask_bao]
        self.theta_bao = theta_bao[mask_bao]
        self.sigma_theta = sigma_theta[mask_bao]

        self.z_ON = z_ON[mask_on]
        self.theta_ON = theta_ON[mask_on]
        self.sigma_ON = sigma_ON[mask_on]

        # -------------------------------------------------
        # Covariance matrix
        # -------------------------------------------------
        cov = np.loadtxt(cov_file)

        if cov.shape != (len(self.z_eff), len(self.z_eff)):
            raise ValueError(
                f"Covariance shape {cov.shape} "
                f"!= ({len(self.z_eff)}, {len(self.z_eff)})"
            )

        self.invcov = np.linalg.inv(cov)

        # Conversion: radians -> degrees
        self.conv = 180.0 / np.pi

    # -------------------------------------------------
    # Log-likelihood
    # -------------------------------------------------
    def __call__(self, cosmo):

        rd = cosmo.rd_sound_horizon()

        # ---------------------------------------------
        # Angular diameter distances
        # ---------------------------------------------
        DA_eff = np.array([cosmo.DA(z) for z in self.z_eff])
        DA_ON  = np.array([cosmo.DA(z) for z in self.z_ON])

        # ---------------------------------------------
        # Theoretical angular BAO scale
        # ---------------------------------------------
        theta_th = self.conv * rd / ((1.0 + self.z_eff) * DA_eff)
        theta_th_ON = self.conv * rd / ((1.0 + self.z_ON) * DA_ON)

        # ---------------------------------------------
        # Residuals
        # ---------------------------------------------
        delta = theta_th - self.theta_bao

        # ---------------------------------------------
        # Chi-square contributions
        # ---------------------------------------------
        chi2_bao = delta @ self.invcov @ delta

        chi2_on = np.sum(
            ((theta_th_ON - self.theta_ON) / self.sigma_ON) ** 2
        )

        return -0.5 * (chi2_bao + chi2_on)



