# ============================================================
# DES-Dovekie Supernova Ia likelihood
# Reference: arXiv:2511.07517 (Popovic et al. 2025)
# Based on: https://github.com/des-science/DES-SN5YR
#
# Sample: ~1820 SNe Ia (1623 DES + 197 low-z), 0.025 < z < 1.14
#
# Data files (in data/)
# ----------------------
# DES-Dovekie_HD.csv          — Hubble diagram (zHD, zHEL, MU)
# DES-Dovekie_STAT_SYS.npz   — inverse covariance (stat + sys)
# -----------------------------------------------------
# Δmu = MU - M - mu_theory     (M = M0 + 5*log10(c/H0), free param)
# chit2 = Δmu^T C^{-1} Δmu
# B     = sum( C^{-1} Δmu )
# C_sum = sum( C^{-1} )        (sum of ALL elements of inv_cov)
# chi2  = chit2 - B^2/C_sum + log(C_sum / 2π)
#
# The Goliath marginalization absorbs any residual global offset
# in M not captured by the free parameter. 
# ============================================================

import os
import numpy as np
import pandas as pd

try:
    import numexpr as ne
    _NE = True
except ImportError:
    _NE = False   # graceful fallback to numpy (no accuracy loss)


class DESDovekieLikelihood:
    """
    DES-Dovekie Type Ia Supernova likelihood (arXiv:2511.07517).
    """

    C_LIGHT = 299792.458   # km/s

    def __init__(self, data_dir='.', z_min=0.0):
        self.data_dir = data_dir
        self.z_min    = z_min
        self._load_and_preprocess()

    # ── Data loading ──────────────────────────────────────────────

    def _load_and_preprocess(self):
        data_file = os.path.join(self.data_dir, 'DES-Dovekie_HD.csv')
        # Try both filename variants (+ vs _ separator)
        for _fname in ('DES-Dovekie_STAT+SYS.npz', 'DES-Dovekie_STAT_SYS.npz'):
            _candidate = os.path.join(self.data_dir, _fname)
            if os.path.exists(_candidate):
                cov_file = _candidate
                break
        else:
            cov_file = os.path.join(self.data_dir, 'DES-Dovekie_STAT_SYS.npz')

        for path in [data_file, cov_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"DES-Dovekie data file not found: {path}\n"
                    f"Download from:\n"
                    f"  https://github.com/des-science/DES-SN5YR\n"
                    f"and place in {self.data_dir}/"
                )

        # Load Hubble diagram
        data = pd.read_csv(data_file, sep=r'\s+', comment='#',
                           usecols=['zHD', 'zHEL', 'MU'])
        zHD_all  = data['zHD'].to_numpy()
        zHEL_all = data['zHEL'].to_numpy()
        MU_all   = data['MU'].to_numpy()

        mask = zHD_all > self.z_min

        self.zHD  = zHD_all[mask]
        self.zHEL = zHEL_all[mask]
        self.MU   = MU_all[mask]
        self.n_sne = int(mask.sum())

        # Pre-compute (1+zHEL) factor used in every theory call
        self._one_plus_zHEL = 1.0 + self.zHEL

        # Load and apply mask to covariance
        self._load_covariance(mask)

    def _load_covariance(self, mask):
        """
        Load inverse covariance from NPZ file.

        NPZ format (DES-Dovekie):
          'nsn' : [n_sne]          total number of SNe
          'cov' : upper triangle of the INVERSE covariance matrix
                  (stored as 1D array, length = n*(n+1)/2)
        """
        npz_path = os.path.join(self.data_dir, 'DES-Dovekie_STAT_SYS.npz')
        npz      = np.load(npz_path)

        # Use named keys — do not rely on npz.files index ordering
        n_full   = int(npz['nsn'][0])
        inv_cov_full = np.zeros((n_full, n_full), dtype=np.float64)
        inv_cov_full[np.triu_indices(n_full)] = npz['cov'].astype(np.float64)

        # Symmetrize (upper → lower)
        i_lower = np.tril_indices(n_full, -1)
        inv_cov_full[i_lower] = inv_cov_full.T[i_lower]

        # Apply z_min mask
        self.inv_cov = np.ascontiguousarray(
            inv_cov_full[mask, :][:, mask]
        )

        # Pre-compute C_sum = sum of all elements of inv_cov
        # (used in Goliath marginalization every call)
        self.C_sum = float(self.inv_cov.sum())

    # ── Log-likelihood ────────────────────────────────────────────

    def __call__(self, cosmo, M):
        """
        Compute the DES-Dovekie log-likelihood.
        """
        return self.loglkl(cosmo, M)

    def loglkl(self, cosmo, M):
        """
        Core likelihood computation.

        Theory distance modulus (paper eq. 7):
            mu_th = 5 * log10( (1+zHEL) * (c/H0) * DM(zHD) ) + 25

        Note on units:
            DM is in Mpc. The factor c/H0 converts to Mpc.
            The factor (1+zHEL) is the kinematic Doppler correction
            for the heliocentric-to-CMB frame conversion.

        Chi2 (marginalized over global additive offset):
            Δmu    = MU - M - mu_theory
            chit2  = Δmu^T C^{-1} Δmu
            B      = Σ_i (C^{-1} Δmu)_i   [scalar]
            C_sum  = Σ_{ij} C^{-1}_{ij}    [scalar, pre-computed]
            chi2   = chit2 - B^2/C_sum + log(C_sum / 2π)
        """
        # Comoving distance at CMB-frame redshifts
        DM = cosmo.DM(self.zHD)   # shape (n_sne,)

        # Theory distance modulus
        # D_l = (1+zHEL) * DM  [using paper eq. 7: D_l = (1+z_obs)*chi(z_CMB)]
        c   = self.C_LIGHT
        H0  = cosmo.H0

        # D_l = (1+zHEL) * DM(zHD)   [Mpc]
        # mu  = 5 * log10(D_l / 10 pc) + 25  = 5 * log10(D_l) + 25
        # (cosmo.DM already returns Mpc — do NOT multiply by c/H0)
        mu_theory = 5.0 * np.log10(self._one_plus_zHEL * DM) + 25.0

        # Residuals: data − theory − M
        delta = self.MU - M - mu_theory

        # (eq. A9-A12 of Goliath et al. 2001, arXiv:astro-ph/0104009)
        inv_cov = self.inv_cov
        ic_delta = inv_cov @ delta            # C^{-1} Δmu

        chit2  = float(delta @ ic_delta)      # Δmu^T C^{-1} Δmu
        B      = float(ic_delta.sum())        # Σ (C^{-1} Δmu)_i
        C_sum  = self.C_sum

        chi2 = chit2 - (B**2 / C_sum) + np.log(C_sum / (2.0 * np.pi))

        return -0.5 * chi2, chit2

    # ── Alternative interface (params dict) ───────────────────────

    def loglkl_from_params(self, cosmo, params):
        """
        Alternative interface accepting a parameter dictionary.
        Compatible with SERTAO engine parameter convention.
        """
        if 'Mcal' not in params:
            raise KeyError("'Mcal' not found in params dict")
        return self.loglkl(cosmo, params['Mcal'])

    # ── Diagnostics ───────────────────────────────────────────────

    def get_data(self):
        """
        Return (zHD, MU, MU_err) for plotting the Hubble diagram.
        MU_err is the 1-sigma diagonal error from the covariance.
        """
        MU_err = np.sqrt(np.diag(np.linalg.inv(self.inv_cov)))
        return self.zHD.copy(), self.MU.copy(), MU_err
