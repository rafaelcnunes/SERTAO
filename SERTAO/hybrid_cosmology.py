"""
Hybrid SERTAO + CLASS cosmology.

This module combines the background and growth framework
implemented in SERTAO with the matter power spectrum and
transfer functions provided by CLASS.

Conventions
-----------
H0:
    km/s/Mpc

Distances:
    Mpc

CLASS P(k):
    k in 1/Mpc
    P(k) in Mpc^3

CLASS P_k_max_h/Mpc:
    k_max in h/Mpc

sigma8:
    Defined for R = 8 h^-1 Mpc.
"""

import numpy as np
from classy import Class

from SERTAO.cosmology import GenericCosmology


class HybridCosmology(GenericCosmology):
    """
    SERTAO + CLASS hybrid cosmology.

    The background expansion and growth are handled by
    GenericCosmology, while CLASS provides the matter power
    spectrum and transfer functions.

    Parameters
    ----------
    H_of_z : callable, optional
        Custom Hubble function H(z) in km/s/Mpc.
        If None, a flat LambdaCDM background is used.

    H0 : float
        Hubble constant in km/s/Mpc.

    Omega_b : float
        Baryon density parameter.

    Omega_cdm : float
        Cold dark matter density parameter.

    sigma8 : float
        Present-day matter fluctuation amplitude.

    class_params : dict, optional
        Additional CLASS parameters.
    """

    def __init__(
        self,
        H_of_z=None,
        H0=70.0,
        Omega_b=0.05,
        Omega_cdm=0.25,
        sigma8=0.8,
        class_params=None,
    ):

        self.H0 = H0
        self.Omega_b = Omega_b
        self.Omega_cdm = Omega_cdm
        self.Omega_m = Omega_b + Omega_cdm
        self.h = H0 / 100.0
        self.sigma8_input = sigma8

        if H_of_z is None:
            H_of_z = lambda z: H0 * np.sqrt(
                self.Omega_m * (1.0 + z)**3
                + (1.0 - self.Omega_m)
            )

        super().__init__(
            H_of_z=H_of_z,
            H0=H0,
            Omega_m=self.Omega_m,
            Omega_b=Omega_b,
            sigma8_0=sigma8,
        )

        self.class_cosmo = None
        self.class_params = None

        if class_params is not None:
            self.init_class(class_params)

    # ========================================================
    # CLASS
    # ========================================================

    def init_class(self, class_params=None):
        """
        Initialize CLASS using sigma8 as the power-spectrum
        normalization.
        """

        self.close_class()

        params = dict(class_params or {})

        params.setdefault("output", "mPk")
        params.setdefault("h", self.h)
        params.setdefault("Omega_b", self.Omega_b)
        params.setdefault("Omega_cdm", self.Omega_cdm)

        # sigma8 is the normalization used by SERTAO.
        #
        # CLASS internally determines A_s such that the
        # requested sigma8 is reproduced.
        params.pop("A_s", None)
        params.pop("ln10^{10}A_s", None)
        params["sigma8"] = self.sigma8_input

        self.class_params = params

        self.class_cosmo = Class()

        try:
            self.class_cosmo.set(params)
            self.class_cosmo.compute()

        except Exception:
            self.close_class()
            raise

    # ========================================================
    # Matter power spectrum
    # ========================================================

    def Pk(self, k, z=0.0):
        """
        Return the nonlinear matter power spectrum.

        Parameters
        ----------
        k : float or ndarray
            Wavenumber in 1/Mpc.

        z : float
            Redshift.

        Returns
        -------
        float or ndarray
            Matter power spectrum in Mpc^3.
        """

        if self.class_cosmo is None:
            raise RuntimeError(
                "CLASS has not been initialized."
            )

        k = np.asarray(k)

        if k.ndim == 0:
            return self.class_cosmo.pk(
                float(k),
                z,
            )

        return np.array([
            self.class_cosmo.pk(
                float(ki),
                z,
            )
            for ki in k.ravel()
        ]).reshape(k.shape)

    # ========================================================
    # Linear matter power spectrum
    # ========================================================

    def Pk_linear(self, k, z=0.0):
        """
        Return the linear matter power spectrum.

        Parameters
        ----------
        k : float or ndarray
            Wavenumber in 1/Mpc.

        z : float
            Redshift.

        Returns
        -------
        float or ndarray
            Linear matter power spectrum in Mpc^3.
        """

        if self.class_cosmo is None:
            raise RuntimeError(
                "CLASS has not been initialized."
            )

        k = np.asarray(k)

        if k.ndim == 0:
            return self.class_cosmo.pk_lin(
                float(k),
                z,
            )

        return np.array([
            self.class_cosmo.pk_lin(
                float(ki),
                z,
            )
            for ki in k.ravel()
        ]).reshape(k.shape)

    # ========================================================
    # Transfer function
    # ========================================================

    def T_k(self, k, z=0.0):
        """
        Return the CLASS transfer functions.

        Parameters
        ----------
        k : float
            Wavenumber in 1/Mpc.

        z : float
            Redshift.
        """

        if self.class_cosmo is None:
            raise RuntimeError(
                "CLASS has not been initialized."
            )

        return self.class_cosmo.get_transfer(
            float(k),
            z,
        )

    # ========================================================
    # sigma(R)
    # ========================================================

    def sigma_R(self, R, z=0.0):
        """
        Return sigma(R,z).

        Parameters
        ----------
        R : float
            Radius in Mpc.

        z : float
            Redshift.
        """

        if self.class_cosmo is None:
            raise RuntimeError(
                "CLASS has not been initialized."
            )

        return self.class_cosmo.sigmaR(
            R,
            z,
        )

    # ========================================================
    # sigma(M)
    # ========================================================

    def sigmaM(self, M, z=0.0, R=None):
        """
        Return sigma(M,z).

        Parameters
        ----------
        M : float
            Mass in solar masses.

        z : float
            Redshift.

        R : float, optional
            Radius in Mpc.

        If R is not provided, it is obtained from

            M = 4 pi / 3 rho_m R^3.
        """

        if R is None:
            rho_m0 = (
                2.77536627e11
                * self.Omega_m
                * self.h**2
            )

            R = (
                3.0 * M
                / (4.0 * np.pi * rho_m0)
            )**(1.0 / 3.0)

        return self.sigma_R(
            R,
            z,
        )

    # ========================================================
    # CLASS sigma8
    # ========================================================

    @property
    def sigma8_CLASS(self):
        """
        Return sigma8 computed by CLASS.
        """

        if self.class_cosmo is None:
            raise RuntimeError(
                "CLASS has not been initialized."
            )

        return float(
            self.class_cosmo.sigma8()
        )

    # ========================================================
    # SERTAO interface
    # ========================================================

    def Hubble(self, z):
        """
        Return H(z) in km/s/Mpc.
        """
        return self.H(z)

    def DM(self, z):
        """
        Return the comoving distance in Mpc.
        """
        return super().DM(z)

    def DH(self, z):
        """
        Return the Hubble distance in Mpc.
        """
        return super().DH(z)

    def DV(self, z):
        """
        Return the volume-averaged distance in Mpc.
        """
        return super().DV(z)

    def fsigma8(self, z):
        """
        Return f(z) sigma8(z).
        """
        return super().fsigma8(z)

    # ========================================================
    # Cleanup
    # ========================================================

    def close_class(self):
        """
        Release the CLASS instance.
        """

        if self.class_cosmo is not None:
            try:
                self.class_cosmo.empty()
            finally:
                self.class_cosmo = None

    def __del__(self):
        try:
            self.close_class()
        except Exception:
            pass
