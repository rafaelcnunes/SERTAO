import os
import numpy as np
from collections import OrderedDict

from SERTAO.cosmology import GenericCosmology

# ---------------- Likelihoods ---------------- #
from likelihoods.BAO_desi import desi_likelihood
from likelihoods.BAO_2D import BAOAngular
from likelihoods.CC import CC
from likelihoods.rsd_likelihood import RSDFastLikelihood
from likelihoods.f_likelihood import fFastLikelihood

from likelihoods.PP import PantheonPlusLikelihood
from likelihoods.PP_SHOES import PantheonPlusSHOESLikelihood
from likelihoods.Union3 import Union3Likelihood
from likelihoods.DES_Dovekie import DESDovekieLikelihood
from likelihoods.BBN import BBNLikelihood
from likelihoods.CMBTheta import CMBThetaLikelihood
from likelihoods.FS_DESI import FSDESILikelihood


# ==============================================================
# Planck 2018 baseline values.
# ==============================================================
PLANCK_DEFAULTS = {
    "H0":        67.36,
    "Omega_b":   0.04930,
    "Omega_cdm": 0.26442,
    "sigma8":    0.8101,
    "Neff":      3.044,
    "M_B":       None,
    "Mcal":      None,
}


# ==============================================================
# Bounded LRU cache
# ==============================================================

class _BoundedLRUCache:
    __slots__ = ("_data", "_maxsize", "hits", "misses")

    def __init__(self, maxsize: int = 50_000):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key):
        try:
            value = self._data[key]
            self._data.move_to_end(key)
            self.hits += 1
            return value
        except KeyError:
            self.misses += 1
            return None

    def set(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
            self._data[key] = value

    def clear(self):
        self._data.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._data)

    @property
    def info(self):
        total = self.hits + self.misses
        rate  = self.hits / total if total else 0.0
        return {
            "size":     len(self._data),
            "maxsize":  self._maxsize,
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": f"{rate:.1%}",
        }


# ==============================================================
# LikelihoodEngine
# ==============================================================

class LikelihoodEngine:
    """
    Engine for combining cosmological likelihoods.

    Supported datasets
    ------------------
    "BBN"            Primordial nucleosynthesis prior on omega_b h^2
    "CMB θ*"         Compressed Planck CMB: R, l_a, omega_b h^2
    "BAO_DESI"       DESI DR1 BAO (7 tracers)
    "BAO_2D"         Angular BAO (2D)
    "CC"             Cosmic Chronometers H(z)
    "RSD"            Redshift-Space Distortions f*sigma8(z)
    "f"              Growth rate f(z)
    "FS_DESI"        DESI DR1 Full Shape — ShapeFit (6 tracers, 24 params)
                     Requires: sigma8 in priors; HDF5 files in data/FS_DESI_DR1/
    "Pantheon+"      SNe Ia Pantheon+ (1701 SNe). Requires: M_B in priors.
    "Pantheon+SHOES" Pantheon+ with SH0ES anchor. Requires: M_B in priors.
    "Union3"         SNe Ia Union 3.0. Requires: Mcal in priors.

    "DES_Dovekie"    DES-Dovekie SNe Ia (1820 SNe Ia, 0.025<z<1.14).
                     M is analytically marginalized — no nuisance parameter needed.
                     Cannot be combined with Pantheon+ or Union3.
    """

    _KNOWN_DATASETS = {
        "BBN", "CMB θ*",
        "BAO_DESI", "BAO_2D",
        "CC",
        "RSD", "f",
        "FS_DESI",
        "Pantheon+", "Pantheon+SHOES", "Union3", "DES_Dovekie",
    }

    def __init__(
        self,
        datasets,
        data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        use_cache=True,
        cache_maxsize=50_000,
        fixed_params=None,
    ):
        self.datasets  = list(datasets)
        self.data_dir  = data_dir
        self.use_cache = use_cache

        # ── Unknown dataset warning ───────────────────────────
        unknown = set(self.datasets) - self._KNOWN_DATASETS
        if unknown:
            import warnings
            warnings.warn(
                f"Unrecognised dataset(s): {unknown}. "
                "They will be silently ignored.",
                UserWarning, stacklevel=2,
            )

        # ── Fixed / default parameters ────────────────────────
        self._fixed = dict(PLANCK_DEFAULTS)
        if fixed_params:
            self._fixed.update(fixed_params)

        # ── LRU cache ─────────────────────────────────────────
        self._cache = _BoundedLRUCache(maxsize=cache_maxsize)

        # ── SNe Ia mutual exclusivity ─────────────────────────
        use_pp  = any(d in datasets for d in ("Pantheon+", "Pantheon+SHOES"))
        use_u3  = "Union3" in datasets
        use_dov = "DES_Dovekie" in datasets
        if sum([use_pp, use_u3, use_dov]) > 1:
            raise ValueError(
                "Pantheon+/SHOES, Union3 and DES_Dovekie cannot be used simultaneously."
            )

        # ── Instantiate likelihoods ───────────────────────────
        self.bbn      = BBNLikelihood()                       if "BBN"           in datasets else None
        self.cmb      = CMBThetaLikelihood()                  if "CMB θ*"        in datasets else None
        self.cc       = CC(data_dir)                          if "CC"            in datasets else None
        self.bao_2d   = BAOAngular(data_dir)                  if "BAO_2D"        in datasets else None
        self.rsd      = RSDFastLikelihood(data_dir)           if "RSD"           in datasets else None
        self.f        = fFastLikelihood(data_dir)             if "f"             in datasets else None
        self.fs_desi  = FSDESILikelihood(data_dir)            if "FS_DESI"       in datasets else None
        if "FS_DESI" in datasets and "sigma8" not in (fixed_params or {}):
            import warnings as _w
            _w.warn(
                "FS_DESI active but sigma8 not in fixed_params. "
                "Add sigma8 to PRIORS so it is sampled; otherwise "
                "it defaults to the Planck value (sigma8=0.8101).",
                UserWarning, stacklevel=2)
        self.pp       = PantheonPlusLikelihood(data_dir)      if "Pantheon+"     in datasets else None
        self.pps      = PantheonPlusSHOESLikelihood(data_dir) if "Pantheon+SHOES" in datasets else None
        self.u3       = Union3Likelihood(data_dir)            if "Union3"        in datasets else None
        self.dov      = DESDovekieLikelihood(data_dir)        if "DES_Dovekie"   in datasets else None
        self.use_bao_desi = "BAO_DESI" in datasets

        # ── Parameter index cache (set on first call) ─────────
        self._param_names = None
        self._idx_cdm     = None
        self._idx_b       = None

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _init_indices(self, priors):
        names = list(priors.keys())
        self._param_names = names
        self._idx_cdm = names.index("Omega_cdm") if "Omega_cdm" in names else None
        self._idx_b   = names.index("Omega_b")   if "Omega_b"   in names else None

    @staticmethod
    def _make_cache_key(theta):
        return np.round(theta, 10).tobytes()

    def _build_params(self, priors, theta):
        p = dict(self._fixed)
        p.update(zip(priors.keys(), theta))
        return p

    def _early_omega_m_guard(self, theta):
        if self._idx_cdm is None and self._idx_b is None:
            return False
        Omega_cdm = (float(theta[self._idx_cdm])
                     if self._idx_cdm is not None
                     else self._fixed.get("Omega_cdm", 0.0))
        Omega_b   = (float(theta[self._idx_b])
                     if self._idx_b is not None
                     else self._fixed.get("Omega_b", 0.0))
        return not (
            0.0 < Omega_cdm < 1.0
            and 0.0 < Omega_b   < 1.0
            and Omega_cdm + Omega_b < 1.0
        )

    # ==========================================================
    # Public API
    # ==========================================================

    @property
    def cache_info(self):
        return self._cache.info

    def clear_cache(self):
        self._cache.clear()

    # ----------------------------------------------------------

    def log_likelihood(
        self,
        theta,
        priors,
        H_model,
        pre_check=None,
        debug=False,
    ):
        """
        Compute the total log-likelihood for parameter vector ``theta``.

        Parameters
        ----------
        theta     : array-like
        priors    : dict  {param_name: (min, max)}
        H_model   : callable  H_model(z, p) -> H(z) in km/s/Mpc
        pre_check : callable, optional  pre_check(p) -> bool
        debug     : bool

        Returns
        -------
        float — total log-likelihood, or -inf if rejected.
        """
        theta = np.asarray(theta, dtype=np.float64)

        if self._param_names is None:
            self._init_indices(priors)

        # ── LRU cache ─────────────────────────────────────────
        key = None
        if self.use_cache:
            key = self._make_cache_key(theta)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # ── Early Omega_m guard ───────────────────────────────
        if self._early_omega_m_guard(theta):
            if debug:
                print("[engine] rejected: Omega_m outside (0, 1)")
            return -np.inf

        # ── Build parameter dict ──────────────────────────────
        p = self._build_params(priors, theta)
        p["Omega_m"] = p["Omega_cdm"] + p["Omega_b"]

        # ── Optional pre-check ────────────────────────────────
        if pre_check is not None and not pre_check(p):
            if debug:
                print("[engine] rejected by pre_check")
            return -np.inf

        # ── Build cosmology object ────────────────────────────
        cosmo = GenericCosmology(
            lambda z, _p=p: H_model(z, _p),
            H0=p["H0"],
            Omega_m=p["Omega_m"],
            Omega_b=p["Omega_b"],
            sigma8_0=p.get("sigma8", self._fixed["sigma8"]),
            Neff=p.get("Neff", self._fixed.get("Neff", 3.044)),
        )

        # Pre-warm distance cache to z=4 in one build, preventing
        # multiple cache rebuilds when BAO/CC/RSD call DM(z) individually.
        cosmo._build_distance_cache(4.0)

        logL = 0.0

        def _add(name, value):
            nonlocal logL
            if not np.isfinite(value):
                if debug:
                    print(f"[engine] rejected by {name}: logL={value}")
                return False
            logL += value
            return True

        # ================================================================
        # Likelihoods — cheap → expensive for early short-circuit
        # ================================================================

        # 1. BBN 
        if self.bbn:
            l, _ = self.bbn.loglkl(p["H0"], p["Omega_b"])
            if not _add("BBN", l):
                return -np.inf

        # 2. CMB θ* 
        if self.cmb:
            l, _ = self.cmb.loglike(cosmo)
            if not _add("CMB θ*", l):
                return -np.inf

        # 3. BAO DESI 
        if self.use_bao_desi:
            l, _ = desi_likelihood(cosmo)
            if not _add("BAO_DESI", l):
                return -np.inf

        # 4. BAO Angular 2D
        if self.bao_2d:
            l = self.bao_2d(cosmo)
            if not _add("BAO_2D", l):
                return -np.inf

        # 5. Cosmic Chronometers 
        if self.cc:
            l = self.cc(cosmo)
            if not _add("CC", l):
                return -np.inf

        # 6. RSD 
        if self.rsd:
            l = self.rsd(cosmo)
            if not _add("RSD", l):
                return -np.inf

        # 7. Growth rate f(z)
        if self.f:
            l = self.f(cosmo)
            if not _add("f(z)", l):
                return -np.inf

        # 8. Full Shape DESI — ShapeFit
        if self.fs_desi:
            l = self.fs_desi(cosmo)
            if not _add("FS_DESI", l):
                return -np.inf

        # 9. SNe Ia 
        if self.pp:
            M_B = p.get("M_B")
            if M_B is None:
                if debug:
                    print("[engine] rejected: M_B not provided for Pantheon+")
                return -np.inf
            l, _ = self.pp.loglkl(cosmo, M_B=M_B)
            if not _add("Pantheon+", l):
                return -np.inf

        if self.pps:
            M_B = p.get("M_B")
            if M_B is None:
                if debug:
                    print("[engine] rejected: M_B not provided for Pantheon+SHOES")
                return -np.inf
            l, _ = self.pps.loglkl(cosmo, M_B=M_B)
            if not _add("Pantheon+SHOES", l):
                return -np.inf

        if self.u3:
            Mcal = p.get("Mcal")
            if Mcal is None:
                if debug:
                    print("[engine] rejected: Mcal not provided for Union3")
                return -np.inf
            l, _ = self.u3.loglkl(cosmo, Mcal)
            if not _add("Union3", l):
                return -np.inf

        if self.dov:
            # M is analytically marginalized inside DES_Dovekie
            l, _ = self.dov(cosmo, M=0.0)
            if not _add("DES_Dovekie", l):
                return -np.inf

        # ── Store in cache ────────────────────────────────────
        if self.use_cache and key is not None:
            self._cache.set(key, logL)

        return logL
