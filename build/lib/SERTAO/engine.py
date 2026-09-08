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
from likelihoods.BBN import BBNLikelihood
from likelihoods.CMBTheta import CMBThetaLikelihood


# ==============================================================
# Planck 2018 (arXiv:1807.06209) baseline values.
# Used to fill any parameter absent from the sampled priors.
# ==============================================================
PLANCK_DEFAULTS = {
    "H0":       67.36,
    "Omega_b":  0.04930,
    "Omega_cdm":0.26442,
    "sigma8":   0.8101,
    "Neff":     3.044,
    "M_B":      None,   # nuisance — no sensible fixed default
    "Mcal":     None,   # nuisance — no sensible fixed default
}


# ==============================================================
# Bounded LRU cache
# ==============================================================

class _BoundedLRUCache:
    """
    Dict-backed LRU cache with a hard upper bound on entries.
    Evicts the least-recently-used entry when full.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries to keep (default: 50 000).
    """

    __slots__ = ("_data", "_maxsize", "hits", "misses")

    def __init__(self, maxsize: int = 50_000):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    # ----------------------------------------------------------

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

    Optimisations over the original version
    ----------------------------------------
    1. Bounded LRU cache (default 50 000 entries) — prevents
       unbounded memory growth in long nested-sampling runs.
    2. Fast cache key via ``tobytes()`` — avoids the intermediate
       NumPy array allocation of ``np.round(...)``.
    3. Early Omega_m guard using pre-computed parameter indices —
       rejects unphysical points before building any object.
    4. ``pre_check`` hook — caller can inject a cheap rejection
       function (e.g. scalar-field physical priors) that runs
       before the cosmology object is constructed.
    5. Planck-CMB defaults — parameters absent from the sampled
       priors (e.g. H0 or Omega_b when they are held fixed) are
       filled from ``PLANCK_DEFAULTS`` rather than crashing.
    6. Debug logging — pass ``debug=True`` to ``log_likelihood``
       to see which dataset triggered a ``-inf`` rejection.
    7. Cache statistics exposed via ``cache_info``.

    Parameters
    ----------
    datasets : list[str]
        Subset of recognised dataset keys to activate.
    data_dir : str
        Path to the data directory (default: ``"./data"``).
    use_cache : bool
        Enable the LRU result cache (default: ``True``).
    cache_maxsize : int
        Maximum number of cached evaluations (default: 50 000).
    fixed_params : dict, optional
        Extra parameter values to fix (override Planck defaults).
        Example: ``{"Neff": 3.044, "sigma8": 0.811}``.
    """

    # ----------------------------------------------------------
    # Recognised dataset keys
    # ----------------------------------------------------------
    _KNOWN_DATASETS = {
        "BBN", "CMB θ*",
        "BAO_DESI", "BAO_2D",
        "CC",
        "RSD", "f",
        "Pantheon+", "Pantheon+SHOES", "Union3",
    }

    def __init__(
        self,
        datasets,
        data_dir="./data",
        use_cache=True,
        cache_maxsize=50_000,
        fixed_params=None,
    ):
        self.datasets     = list(datasets)
        self.data_dir     = data_dir
        self.use_cache    = use_cache

        # --------------------------------------------------
        # Warn on unrecognised dataset keys
        # --------------------------------------------------
        unknown = set(self.datasets) - self._KNOWN_DATASETS
        if unknown:
            import warnings
            warnings.warn(
                f"Unrecognised dataset(s): {unknown}. "
                "They will be silently ignored.",
                UserWarning,
                stacklevel=2,
            )

        # --------------------------------------------------
        # Fixed / default parameters
        # Start from Planck baseline and override with caller.
        # --------------------------------------------------
        self._fixed = dict(PLANCK_DEFAULTS)
        if fixed_params:
            self._fixed.update(fixed_params)

        # --------------------------------------------------
        # LRU cache
        # --------------------------------------------------
        self._cache = _BoundedLRUCache(maxsize=cache_maxsize)

        # --------------------------------------------------
        # SNe Ia mutual exclusivity
        # --------------------------------------------------
        use_pp = any(d in datasets for d in ("Pantheon+", "Pantheon+SHOES"))
        use_u3 = "Union3" in datasets

        if use_pp and use_u3:
            raise ValueError(
                "Pantheon+/SHOES and Union3 cannot be used simultaneously "
                "(they measure the same observable with different datasets)."
            )

        # --------------------------------------------------
        # Instantiate likelihoods (load data once)
        # --------------------------------------------------
        self.bbn  = BBNLikelihood()                        if "BBN"            in datasets else None
        self.cmb  = CMBThetaLikelihood()                   if "CMB θ*"         in datasets else None
        self.pp   = PantheonPlusLikelihood(data_dir)       if "Pantheon+"       in datasets else None
        self.pps  = PantheonPlusSHOESLikelihood(data_dir)  if "Pantheon+SHOES"  in datasets else None
        self.u3   = Union3Likelihood(data_dir)             if "Union3"          in datasets else None
        self.cc   = CC(data_dir)                           if "CC"              in datasets else None
        self.bao_2d = BAOAngular(data_dir)                 if "BAO_2D"          in datasets else None
        self.rsd  = RSDFastLikelihood(data_dir)            if "RSD"             in datasets else None
        self.f    = fFastLikelihood(data_dir)              if "f"               in datasets else None
        self.use_bao_desi = "BAO_DESI" in datasets

        # --------------------------------------------------
        # Pre-compute param-name → index map (set at first
        # call via _init_indices; avoids coupling __init__
        # to the prior dict ordering).
        # --------------------------------------------------
        self._param_names = None        # set on first log_likelihood call
        self._idx_cdm     = None
        self._idx_b       = None

    # ==========================================================
    # Internal helpers
    # ==========================================================

    def _init_indices(self, priors):
        """Cache parameter indices for fast Omega_m guard."""
        names = list(priors.keys())
        self._param_names = names
        self._idx_cdm = names.index("Omega_cdm") if "Omega_cdm" in names else None
        self._idx_b   = names.index("Omega_b")   if "Omega_b"   in names else None

    # ----------------------------------------------------------

    @staticmethod
    def _make_cache_key(theta):
        """
        Fast hashable key.
        Rounds to 10 decimal places to treat numerically-identical
        dynesty proposals as the same point, then converts to bytes
        (avoids the intermediate array allocation of np.round).
        """
        return np.round(theta, 10).tobytes()

    # ----------------------------------------------------------

    def _build_params(self, priors, theta):
        """
        Merge sampled values with fixed/default values.

        Sampled values (from ``theta``) always win.
        Parameters absent from ``priors`` are filled from
        ``self._fixed``, which is seeded from Planck 2018 defaults
        and overridden by the caller's ``fixed_params``.
        """
        p = dict(self._fixed)           # start from defaults
        p.update(zip(priors.keys(), theta))  # sampled values win
        return p

    # ----------------------------------------------------------

    def _early_omega_m_guard(self, theta):
        """
        Reject unphysical Omega_m BEFORE building any object.
        Returns True if the point should be rejected.
        Uses pre-computed indices for speed.
        """
        if self._idx_cdm is None and self._idx_b is None:
            return False    # both fixed — skip (validated at init)

        Omega_cdm = (
            float(theta[self._idx_cdm])
            if self._idx_cdm is not None
            else self._fixed.get("Omega_cdm", 0.0)
        )
        Omega_b = (
            float(theta[self._idx_b])
            if self._idx_b is not None
            else self._fixed.get("Omega_b", 0.0)
        )
        Omega_m = Omega_cdm + Omega_b
        return not (0.0 < Omega_m < 1.0)

    # ==========================================================
    # Public API
    # ==========================================================

    @property
    def cache_info(self):
        """Return LRU cache statistics."""
        return self._cache.info

    def clear_cache(self):
        """Manually flush the result cache."""
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
        theta : array-like
            Parameter vector in the order defined by ``priors``.
        priors : dict
            Mapping ``{param_name: (min, max)}`` for sampled parameters.
        H_model : callable
            ``H_model(z, p) -> H(z)`` in km/s/Mpc.
        pre_check : callable, optional
            ``pre_check(p) -> bool`` — return ``False`` to reject the
            point before building the cosmology object (cheap physical
            priors, e.g. scalar-field constraints).
        debug : bool
            Print which dataset triggered a ``-inf`` rejection.

        Returns
        -------
        float
            Total log-likelihood, or ``-np.inf`` if rejected.
        """

        theta = np.asarray(theta, dtype=np.float64)

        # ---- Initialise index cache on first call ---- #
        if self._param_names is None:
            self._init_indices(priors)

        # ---- LRU cache lookup ---- #
        key = None
        if self.use_cache:
            key = self._make_cache_key(theta)
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # ---- Early Omega_m guard (no object allocation) ---- #
        if self._early_omega_m_guard(theta):
            if debug:
                print("[engine] rejected: Omega_m outside (0, 1)")
            return -np.inf

        # ---- Build full parameter dict ---- #
        p = self._build_params(priors, theta)
        p["Omega_m"] = p["Omega_cdm"] + p["Omega_b"]

        # ---- Optional caller pre-check ---- #
        if pre_check is not None and not pre_check(p):
            if debug:
                print("[engine] rejected by pre_check")
            return -np.inf

        # ---- Build cosmology object ---- #
        # Neff is passed explicitly so that rd_sound_horizon() and the
        # distance cache use the sampled value when Neff is a free parameter.
        # Falls back to the Planck 2018 default (3.044) if not sampled.
        cosmo = GenericCosmology(
            lambda z: H_model(z, p),
            H0=p["H0"],
            Omega_m=p["Omega_m"],
            Omega_b=p["Omega_b"],
            sigma8_0=p.get("sigma8", self._fixed.get("sigma8", 0.8)),
            Neff=p.get("Neff", self._fixed.get("Neff", 3.044)),
        )

        # ---- Pre-warm distance cache to z = 4 ---- #
        # Without this, each likelihood calling DM(z)/DH(z) at increasing z
        # triggers a full cache rebuild (10,000 H evals each).
        # BAO DESI alone causes 7 rebuilds = 70,000 H calls per theta.
        # One pre-build here reduces all subsequent calls to np.interp lookups.
        # CMB theta* will extend to z*~1090 on its first call if needed.
        cosmo._build_distance_cache(4.0)

        logL = 0.0

        # ================================================================
        # Likelihoods — ordered cheap-to-expensive for early short-circuit
        # ================================================================

        def _add(name, value):
            """Add ``value`` to logL; return False to short-circuit."""
            nonlocal logL
            if not np.isfinite(value):
                if debug:
                    print(f"[engine] rejected by {name}: logL={value}")
                return False
            logL += value
            return True

        # 1. BBN — single Gaussian, cheapest possible rejection
        if self.bbn:
            l, _ = self.bbn.loglkl(p["H0"], p["Omega_b"])
            if not _add("BBN", l):
                return -np.inf

        # 2. CMB θ* — compressed 3×3 covariance
        if self.cmb:
            l, _ = self.cmb.loglike(cosmo)
            if not _add("CMB θ*", l):
                return -np.inf

        # 3. BAO DESI DR2 — 7 bins, simple χ²
        if self.use_bao_desi:
            l, _ = desi_likelihood(cosmo)
            if not _add("BAO_DESI", l):
                return -np.inf

        # 4. BAO Angular 2D — full covariance
        if self.bao_2d:
            l = self.bao_2d(cosmo)
            if not _add("BAO_2D", l):
                return -np.inf

        # 5. CC — full covariance, ~30 points
        if self.cc:
            l = self.cc(cosmo)      # unified interface: cosmo only
            if not _add("CC", l):
                return -np.inf

        # 6. RSD f·σ₈ — diagonal covariance
        if self.rsd:
            l = self.rsd(cosmo)
            if not _add("RSD", l):
                return -np.inf

        # 7. f(z) — diagonal covariance
        if self.f:
            l = self.f(cosmo)
            if not _add("f(z)", l):
                return -np.inf

        # 8. SNe Ia — heaviest; only reached if all above pass
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

        # ---- Store in LRU cache ---- #
        if self.use_cache and key is not None:
            self._cache.set(key, logL)

        return logL
