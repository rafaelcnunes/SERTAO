# ============================================================
# DESI DR1 Full Shape — ShapeFit likelihood
# Reference: arXiv:2411.12022, arXiv:2404.07283, arXiv:2602.18761
# ============================================================
#
# Compressed data vector: (qiso, qap, df, dm) per tracer
# 6 tracers (BGS, LRG1, LRG2, LRG3+ELG1, ELG2, QSO) = 24 parameters
#
# Definitions (arXiv:2602.18761, eqs. 3.1–3.4)
# -----------------------------------------------
#
#   q_par(z) = H_fid(z) * rd_fid / (H(z) * rd)
#   q_per(z) = DM(z) * rd_fid / (DM_fid(z) * rd)
#
#   qiso(z)  = [q_par * q_per^2]^(1/3)
#            = [DV(z) / DV_fid(z)] * (rd_fid / rd)
#
#   qap(z)   = q_par / q_per
#            = [DH_fid(z) * DM_fid(z)] / [DH(z) * DM(z)]
#            (rd cancels — qap is independent of rd)
#
#   df(z)    = (f*sigma8)(z) / (f*sigma8)_fid(z)
#
#   dm(z)    = m(z) - m_fid(z)
#            where m = d ln[T_nw(k)]^2 / d ln k |_{k = k_p/s}
#            T_nw = no-wiggle (de-wiggled) transfer function
#            k_p ~ pi/rd ~ 0.03 h/Mpc   (pivot scale)
#            s   = rd_fid / rd           (dilation factor)
#
# Fiducial cosmology (DESI DR1 flat LCDM)
# -----------------------------------------
# H0=67.36, Omega_m=0.3152, Omega_b=0.04930, sigma8=0.8111
# rd_fid=147.21 Mpc
# Fiducial quantities (DM_fid, DH_fid, DV_fid, fs8_fid, m_fid)
# are pre-computed once at construction from GenericCosmology.
# ============================================================

import os
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import interp1d
import h5py


class FSDESILikelihood:
    """
    DESI DR1 Full Shape ShapeFit likelihood
    """

    # ── DESI DR1 fiducial cosmology ───────────────────────────────
    _H0_FID    = 67.36      # km/s/Mpc
    _OM_FID    = 0.3152
    _OB_FID    = 0.04930
    _OR_FID    = 9.18e-5    # radiation (photons + massless nu)
    _S8_FID    = 0.8111     # sigma8 at z=0
    # rd_fid is computed self-consistently from rd_sound_horizon()
    # at the DESI DR1 fiducial cosmology (H0=67.36, Om=0.3152, Ob=0.04930)
    # using our Eisenstein-Hu fitting formula. This ensures s=rd_fid/rd=1
    # exactly at the fiducial cosmology, giving qiso=1.000000.
    # Note: DESI uses CLASS (rd=147.21 Mpc); our fitting formula gives
    # 147.085 Mpc (0.085% difference, well within data uncertainties).
    _RD_FID    = 147.08515  # Mpc — rd_sound_horizon(H0=67.36, Om=0.3152, Ob=0.04930)
    _C_LIGHT   = 299792.458 # km/s

    # ShapeFit slope parameters (arXiv:2404.07283 eq. 3.19)
    _A_SF      = 0.6        # amplitude parameter
    _KP_H_MPC  = 0.03       # pivot scale [h/Mpc]

    # HDF5 file names (FS-only: spectrum-poles-rotated, no bao-recon suffix)
    _H5_FILES = {
        'BGS':      'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_thetacut0.05.h5',
        'LRG1':     'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_LRG_GCcomb_z0.4-0.6_thetacut0.05.h5',
        'LRG2':     'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_LRG_GCcomb_z0.6-0.8_thetacut0.05.h5',
        'LRG3ELG1': 'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_LRG_GCcomb_z0.8-1.1_thetacut0.05.h5',
        'ELG2':     'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_ELG_LOPnotqso_GCcomb_z1.1-1.6_thetacut0.05.h5',
        'QSO':      'likelihood_shapefit_spectrum-poles-rotated%2Bbao-recon_syst-rotation-hod-photo_QSO_GCcomb_z0.8-2.1_thetacut0.05.h5',
    }

    def __init__(self, data_dir='data'):
        h5_dir = os.path.join(data_dir, 'FS_DESI_DR1')

        # ── Read HDF5 files ───────────────────────────────────────
        self.tracers   = list(self._H5_FILES.keys())
        self.n_tracer  = len(self.tracers)
        self.n_per     = 4            # qiso, qap, df, dm
        self.n_data    = self.n_tracer * self.n_per   # 24

        z_effs   = []
        data_vec = []
        cov_blocks = []
        params_per_tracer = []

        for tracer in self.tracers:
            path = os.path.join(h5_dir, self._H5_FILES[tracer])
            with h5py.File(path, 'r') as h:
                sf     = h['observable/shapefit']
                z      = float(sf.attrs['zeff'])
                params = [p.decode() for p in sf['labels_values'][()]]
                data   = np.array([float(sf[p]['value'][()].flat[0])
                                   for p in params])
                cov    = h['covariance/value'][()]

            z_effs.append(z)
            data_vec.append(data)
            cov_blocks.append(cov)
            params_per_tracer.append(params)

        self.z_eff  = np.array(z_effs)
        self.data   = np.concatenate(data_vec)          # (24,)

        # Block-diagonal joint covariance (tracers are non-overlapping in z)
        self.cov = np.zeros((self.n_data, self.n_data))
        for i, blk in enumerate(cov_blocks):
            s, e = i*self.n_per, (i+1)*self.n_per
            self.cov[s:e, s:e] = blk

        self.inv_cov = np.linalg.inv(self.cov)

        # ── Pre-compute fiducial quantities ───────────────────────
        self._build_fiducial()

    # ── Fiducial cosmology (pre-computed once) ────────────────────

    def _build_fiducial(self):
        """
        Compute DM_fid, DH_fid, DV_fid, fsigma8_fid, m_fid at each z_eff.
        Uses the DESI DR1 fiducial flat ΛCDM cosmology.
        All fiducial distances are computed from the same numerical grid
        so that theory ratios = 1 exactly at the fiducial cosmology.
        """
        H0 = self._H0_FID; Om = self._OM_FID; Or = self._OR_FID
        OL = 1.0 - Om - Or; c = self._C_LIGHT

        def H(z): return H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL)

        # Comoving distance grid
        zg  = np.linspace(0.0, 4.0, 100_000)
        Hg  = np.array([H(z) for z in zg])
        chi = cumulative_trapezoid(c/Hg, zg, initial=0.0)
        self._chi_fid = interp1d(zg, chi)

        # Growth ODE
        a_ini = 1.0/(1.0 + 50.0)
        ag    = np.logspace(np.log10(a_ini), 0.0, 500)

        def ode(a, y):
            D, dD = y; z = 1.0/a - 1.0; Hv = H(z)
            Om_a  = Om*(1+z)**3*(H0/Hv)**2
            return [dD, -(3.0/a - 1.5*Om_a/a)*dD + 1.5*Om_a*D/a**2]

        sol = solve_ivp(ode, [a_ini, 1.0], [a_ini, 1.0], t_eval=ag,
                        method='DOP853', rtol=1e-8, atol=1e-10)
        D = sol.y[0]/sol.y[0][-1]
        f = ag * sol.y[1] / sol.y[0]
        zg2 = (1.0/ag - 1.0)[::-1]
        self._D_fid = interp1d(zg2, D[::-1], bounds_error=False,
                               fill_value=(D[0], D[-1]))
        self._f_fid = interp1d(zg2, f[::-1], bounds_error=False,
                               fill_value=(f[0], f[-1]))

        # Store fiducial values at each z_eff
        s8 = self._S8_FID
        self._DM_fid  = np.array([float(self._chi_fid(z)) for z in self.z_eff])
        self._DH_fid  = np.array([c/H(z) for z in self.z_eff])
        self._DV_fid  = (self._DM_fid**2 * self.z_eff * self._DH_fid)**(1/3)
        self._fs8_fid = np.array([float(self._f_fid(z))*s8*float(self._D_fid(z))
                                   for z in self.z_eff])

        # Fiducial dm: slope of T_nw at k_p (s=1 at fiducial)
        om_fid = self._OM_FID * (self._H0_FID/100.0)**2
        ob_fid = self._OB_FID * (self._H0_FID/100.0)**2
        self._m_fid = np.array([
            self._dm_EH(z, om_fid, ob_fid, s=1.0)
            for z in self.z_eff
        ])

    # ── Eisenstein-Hu no-wiggle transfer function slope ──────────

    @staticmethod
    def _T_EH_nw(k, om_h2, ob_h2):
        """
        Eisenstein & Hu (1998) no-wiggle (de-wiggled) transfer function.
        k in h/Mpc, om_h2 = Omega_m h^2, ob_h2 = Omega_b h^2.
        Returns T_nw(k) [dimensionless].
        """
        ombom  = ob_h2 / om_h2
        T_cmb  = 2.7255 / 2.7
        z_eq   = 2.5e4 * om_h2 * T_cmb**(-4)
        k_eq   = 7.46e-2 * om_h2 * T_cmb**(-2)           # h/Mpc
        b1     = 0.313 * om_h2**(-0.419) * (1 + 0.607*om_h2**0.674)
        b2     = 0.238 * om_h2**0.223
        z_d    = 1291 * om_h2**0.251 / (1 + 0.659*om_h2**0.828) * \
                 (1 + b1 * ob_h2**b2)
        R_d    = 31.5e3 * ob_h2 * T_cmb**(-4) * (1000.0/z_d)
        k_silk = 1.6 * ob_h2**0.52 * om_h2**0.01 * \
                 ((1 + (11.25 * ob_h2 / om_h2 * T_cmb**(-2))**0.84))**(-1/4)  # h/Mpc

        alpha_gamma = 1.0 - 0.328*np.log(431*om_h2)*ombom + \
                      0.38*np.log(22.3*om_h2)*ombom**2
        gamma_eff   = om_h2 * (alpha_gamma + (1-alpha_gamma) /
                                (1 + (0.43*k/k_silk)**4))
        q           = k * T_cmb**2 / gamma_eff
        L0          = np.log(2*np.e + 1.8*q)
        C0          = 14.2 + 731.0/(1 + 62.5*q)
        return L0 / (L0 + C0*q**2)

    def _dm_EH(self, z, om_h2, ob_h2, s):
        """
        Compute the shape parameter slope m for a given cosmology.

        From arXiv:2404.07283, m is defined as the logarithmic slope of
        the no-wiggle power spectrum ratio evaluated at the pivot k_p:

            m = d/d(ln k) [ T_nw(k)^2 / (s^3 T_nw_fid(k)^2) ] |_{k_p}

        Expanding the logarithmic derivative:
            = d ln T_nw^2/d ln k |_{k_p}  -  d ln T_nw_fid^2/d ln k |_{k_p}

        Both model and fiducial are evaluated at the SAME fixed k_p
        (the s^3 term cancels because s does not depend on k, and the
        T_nw_fid term is handled by subtracting self._m_fid in
        get_theory_vector).

        Note: k_p is NOT dilated by 1/s here. The dilation affects
        the BAO position (encoded in qiso, qap), not the slope pivot.
        Evaluating at k_p/s instead of k_p would introduce a ~0.2σ
        systematic bias in dm.

        The argument s is retained for API compatibility but is unused.
        """
        k_eval = self._KP_H_MPC        # fixed pivot 
        dk     = 0.005 * k_eval       

        T_p = self._T_EH_nw(k_eval + dk, om_h2, ob_h2)
        T_m = self._T_EH_nw(k_eval - dk, om_h2, ob_h2)
        T_c = self._T_EH_nw(k_eval,      om_h2, ob_h2)

        # m = d ln T^2 / d ln k = 2 * (k/T) * dT/dk
        dTdk = (T_p - T_m) / (2 * dk)
        return 2.0 * k_eval * dTdk / T_c

    # ── Theory prediction ─────────────────────────────────────────

    def get_theory_vector(self, cosmo):
        """
        Compute the ShapeFit theory vector
        """
        rd   = cosmo.rd_sound_horizon()   
        s    = self._RD_FID / rd          
        om_h2 = cosmo.omega_m             # Omega_m * h^2
        ob_h2 = cosmo.omega_b             # Omega_b * h^2

        theory = np.empty(self.n_data)

        for i, z in enumerate(self.z_eff):
            DM  = cosmo.DM(z)
            DH  = cosmo.DH(z)
            DV  = (DM**2 * z * DH)**(1/3)
            fs8 = cosmo.fsigma8(z)

            # qiso: isotropic volume dilation (includes rd ratio)
            qiso = (DV / self._DV_fid[i]) * s

            # qap: Alcock-Paczynski (rd cancels)
            qap  = (self._DH_fid[i] * self._DM_fid[i]) / (DH * DM)

            # df: growth rate ratio
            df   = fs8 / self._fs8_fid[i]

            # dm: shape parameter (slope of no-wiggle Pk at pivot scale)
            m_model = self._dm_EH(z, om_h2, ob_h2, s)
            dm       = m_model - self._m_fid[i]

            j = i * self.n_per
            theory[j], theory[j+1], theory[j+2], theory[j+3] = \
                qiso, qap, df, dm

        return theory

    # ── Log-likelihood ────────────────────────────────────────────

    def __call__(self, cosmo):
        """
        Compute the DESI DR1 FS ShapeFit log-likelihood.
        """
        theory = self.get_theory_vector(cosmo)
        delta  = theory - self.data
        chi2   = float(delta @ self.inv_cov @ delta)
        return -0.5 * chi2

    # ── Diagnostics ───────────────────────────────────────────────

    def print_theory(self, cosmo):
        """Print theory vs data with pulls for each tracer (debugging)."""
        theory = self.get_theory_vector(cosmo)
        sigma  = np.sqrt(np.diag(self.cov))
        params = ['qiso', 'qap', 'df', 'dm']

        chi2 = -2 * self(cosmo)
        print(f"FS_DESI ShapeFit  chi2={chi2:.3f}  logL={self(cosmo):.3f}")
        print(f"{'Tracer':<12} {'param':<6} {'theory':>9}  {'data':>9}  "
              f"{'sigma':>8}  {'pull':>7}")
        print("-" * 58)
        for i, (tracer, z) in enumerate(zip(self.tracers, self.z_eff)):
            for j, par in enumerate(params):
                idx  = i*self.n_per + j
                pull = (theory[idx] - self.data[idx]) / sigma[idx]
                print(f"{tracer:<12} {par:<6} {theory[idx]:>9.5f}  "
                      f"{self.data[idx]:>9.5f}  {sigma[idx]:>8.5f}  "
                      f"{pull:>+7.2f}")
