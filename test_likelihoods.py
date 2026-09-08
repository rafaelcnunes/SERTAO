"""
test_likelihoods.py
-------------------
Functional tests for all SERTAO likelihoods.

Each test verifies that:
  1. The likelihood loads without error
  2. It returns a finite value at a fiducial Planck cosmology
  3. The value is negative (log-likelihood ≤ 0)
  4. It is sensitive to parameter shifts: moving parameters away from
     best-fit worsens the fit (logL decreases)
  5. The chi2 = -2*logL is in a physically reasonable range for the
     number of data points

No CLASS, no HybridCosmology, no Pantheon+/Union3 covariance matrices
needed for the core tests.  Pantheon+/Union3/BAO_2D tests use the real
data files from the SERTAO data/ directory if available.

Run
---
    python test_likelihoods.py               # from the SERTAO root dir
    python test_likelihoods.py -v            # verbose
"""

import sys
import os
import numpy as np

# ── Locate SERTAO root automatically ─────────────────────────────
# The script can live anywhere; it walks up from its location to find
# the directory that contains both 'likelihoods/' and 'data/'.
def _find_sertao_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [here, os.path.dirname(here)]:
        if (os.path.isdir(os.path.join(candidate, "likelihoods")) and
                os.path.isdir(os.path.join(candidate, "data"))):
            return candidate
    # Fallback: try common SERTAO project structure
    for candidate in [here, os.path.dirname(here)]:
        sub = os.path.join(candidate, "SERTAO")
        if (os.path.isdir(os.path.join(sub, "likelihoods")) and
                os.path.isdir(os.path.join(sub, "data"))):
            return sub
    return here

SERTAO_ROOT = _find_sertao_root()
sys.path.insert(0, SERTAO_ROOT)
sys.path.insert(0, os.path.join(SERTAO_ROOT, "SERTAO"))  # for engine/cosmology

VERBOSE = "-v" in sys.argv

DATA_DIR = os.path.join(SERTAO_ROOT, "data")

# ── Colours ───────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"


# ================================================================
# Mock cosmology (no CLASS needed)
# ================================================================

class MockCosmology:
    """
    Analytic flat LCDM for offline likelihood testing.
    Mimics the GenericCosmology interface used by all likelihoods.
    """
    c = 299792.458

    def __init__(self, H0=67.36, Omega_m=0.315, Omega_b=0.0493,
                 sigma8=0.811, Neff=3.044):
        self.H0      = H0
        self.h       = H0 / 100.0
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.Omega_L = 1.0 - Omega_m
        self.sigma8_0 = sigma8
        self.Neff    = Neff
        self.omega_m = Omega_m * self.h**2
        self.omega_b = Omega_b * self.h**2

        # Distance cache
        self._zg = np.linspace(0.0, 1200.0, 50_000)
        Hg = self._Hfunc(self._zg)
        dc = self.c / Hg
        self._cg = np.zeros_like(self._zg)
        self._cg[1:] = np.cumsum(0.5*(dc[:-1]+dc[1:])*np.diff(self._zg))

        # Growth
        self._precompute_growth()

    def _Hfunc(self, z):
        return self.H0 * np.sqrt(self.Omega_m*(1+z)**3 + self.Omega_L)

    def H(self, z):     return float(self._Hfunc(z))
    def Hubble(self, z):return float(self._Hfunc(z))

    def DM(self, z):
        return float(np.interp(z, self._zg, self._cg))

    def DL(self, z):
        return (1.0 + z) * self.DM(z)

    def DA(self, z):
        return self.DM(z) / (1.0 + z)

    def DH(self, z):
        return self.c / self._Hfunc(z)

    def DV(self, z):
        DM = self.DM(z); DH = self.DH(z)
        return (DM**2 * z * DH)**(1/3)

    def comoving_distance(self, z):
        return self.DM(z)

    def distance_modulus(self, z):
        return 5.0 * np.log10(self.DL(z)) + 25.0

    def rd_sound_horizon(self):
        return (147.05
                * (self.omega_m / 0.1432)**(-0.23)
                * (self.Neff    / 3.044) **(-0.10)
                * (self.omega_b / 0.02236)**(-0.13))

    def _precompute_growth(self):
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d
        a_ini = 1/(1+50)
        ag = np.logspace(np.log10(a_ini), 0, 300)
        def ode(a, y):
            D, dD = y
            z  = 1/a - 1
            H  = self._Hfunc(z)
            Om = self.Omega_m*(1+z)**3*(self.H0/H)**2
            return [dD, -(3/a - 1.5*Om/a)*dD + 1.5*Om*D/a**2]
        sol = solve_ivp(ode, [a_ini, 1.0], [a_ini, 1.0], t_eval=ag,
                        method='DOP853', rtol=1e-8, atol=1e-10)
        D = sol.y[0] / sol.y[0][-1]
        f = ag * sol.y[1] / sol.y[0]
        from scipy.interpolate import interp1d
        zg = (1/ag - 1)[::-1]
        self._D_interp = interp1d(zg, D[::-1], kind='cubic',
                                  bounds_error=False, fill_value=(D[0], D[-1]))
        self._f_interp = interp1d(zg, f[::-1], kind='cubic',
                                  bounds_error=False, fill_value=(f[0], f[-1]))

    def growth_factor(self, z):  return self._D_interp(np.atleast_1d(z))
    def growth_rate(self, z):    return self._f_interp(np.atleast_1d(z))
    def sigma8(self, z):         return self.sigma8_0 * self._D_interp(np.atleast_1d(z))
    def fsigma8(self, z):        return self._f_interp(np.atleast_1d(z)) * self.sigma8(z)
    def Omega_m_z(self, z):
        return self.Omega_m*(1+z)**3*(self.H0/self._Hfunc(z))**2


# ================================================================
# Test runner
# ================================================================

_results = []

def _run(name, fn):
    try:
        msg = fn()
        status = PASS
        _results.append((name, True, msg or ""))
    except AssertionError as e:
        status = FAIL
        _results.append((name, False, str(e)))
        msg = str(e)
    except Exception as e:
        status = FAIL
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        msg = f"{type(e).__name__}: {e}"

    label = f"  {status}  {name}"
    if VERBOSE or status == FAIL:
        detail = f"  → {msg}" if msg else ""
        print(f"{label}{detail}")
    else:
        print(label)


def _skip(name, reason):
    _results.append((name, None, reason))
    print(f"  {SKIP}  {name}  ({reason})")


def assert_finite(val, name="logL"):
    assert np.isfinite(val), f"{name} is not finite: {val}"

def assert_negative(val, name="logL"):
    assert val <= 0.0, f"{name} should be ≤ 0, got {val:.4f}"

def assert_worsens(logL_good, logL_bad, label=""):
    assert logL_bad < logL_good, (
        f"{label} logL did not worsen when moving away from best-fit: "
        f"good={logL_good:.4f}, bad={logL_bad:.4f}"
    )

def assert_chi2_range(logL, n_data, name=""):
    chi2 = -2 * logL
    # chi2/n_data should be O(1): allow [0, 10]
    assert 0.0 <= chi2 <= 10 * n_data, (
        f"{name} chi2={chi2:.2f} unreasonable for n_data={n_data}"
    )


# ================================================================
# Fiducial cosmology
# ================================================================

PLANCK = dict(H0=67.36, Omega_m=0.315, Omega_b=0.0493,
              sigma8=0.811, Neff=3.044)
cosmo_fid  = MockCosmology(**PLANCK)
cosmo_bad  = MockCosmology(H0=50.0, Omega_m=0.5, Omega_b=0.07,
                            sigma8=0.6, Neff=3.044)


# ================================================================
# 1. BBN
# ================================================================

def test_bbn_fiducial():
    from likelihoods.BBN import BBNLikelihood
    bbn = BBNLikelihood()
    logL, omegabh2 = bbn.loglkl(PLANCK['H0'], PLANCK['Omega_b'])
    assert_finite(logL)
    assert_negative(logL)
    assert_chi2_range(logL, 1, "BBN")
    return f"logL={logL:.4f}, omega_b*h2={omegabh2:.5f}"

def test_bbn_sensitivity():
    from likelihoods.BBN import BBNLikelihood
    bbn = BBNLikelihood()
    logL_good, _ = bbn.loglkl(PLANCK['H0'], PLANCK['Omega_b'])
    logL_bad,  _ = bbn.loglkl(50.0, 0.10)
    assert_worsens(logL_good, logL_bad, "BBN")
    return f"Δ logL = {logL_good - logL_bad:.2f}"

def test_bbn_peak():
    """logL is maximised at omega_b*h2 = 0.02230."""
    from likelihoods.BBN import BBNLikelihood
    bbn = BBNLikelihood()
    H0 = 100.0  # h=1 so omega_b*h2 = Omega_b
    logL_peak, _ = bbn.loglkl(H0, 0.02230)
    logL_off,  _ = bbn.loglkl(H0, 0.02500)
    assert logL_peak > logL_off, "BBN peak not at omega_b*h2 = 0.02230"
    return f"peak logL={logL_peak:.4f}"


# ================================================================
# 2. CMB theta*
# ================================================================

def test_cmb_fiducial():
    from likelihoods.CMBTheta import CMBThetaLikelihood
    cmb = CMBThetaLikelihood()
    logL, chi2 = cmb.loglike(cosmo_fid)
    assert_finite(logL)
    assert_negative(logL)
    # CMB theta* chi2 can be large (tight compressed covariance)
    assert np.isfinite(logL) and logL <= 0, f"CMBTheta logL={logL}"
    return f"logL={logL:.4f}, chi2={chi2:.4f}"

def test_cmb_theory_vector():
    """R, l_a, omega_b*h2 should be close to observed values."""
    from likelihoods.CMBTheta import CMBThetaLikelihood
    cmb = CMBThetaLikelihood()
    theory = cmb.get_theory_vector(cosmo_fid)
    obs    = cmb.data
    # Each theory value should be within 5% of observed
    for i, (t, o, name) in enumerate(zip(theory, obs, ['R', 'l_a', 'omega_b_h2'])):
        rel = abs(t/o - 1)
        assert rel < 0.05, f"CMB {name}: theory={t:.4f} obs={o:.4f} diff={rel*100:.1f}%"
    return f"R={theory[0]:.3f}, la={theory[1]:.2f}, wbh2={theory[2]:.5f}"

def test_cmb_sensitivity():
    from likelihoods.CMBTheta import CMBThetaLikelihood
    cmb = CMBThetaLikelihood()
    logL_good, _ = cmb.loglike(cosmo_fid)
    logL_bad,  _ = cmb.loglike(cosmo_bad)
    assert_worsens(logL_good, logL_bad, "CMBTheta")
    return f"Δ logL = {logL_good - logL_bad:.2f}"


# ================================================================
# 3. BAO DESI
# ================================================================

def test_bao_desi_fiducial():
    from likelihoods.BAO_desi import desi_likelihood
    logL, chi2 = desi_likelihood(cosmo_fid)
    assert_finite(logL)
    assert_negative(logL)
    assert_chi2_range(logL, 7, "BAO_DESI")
    return f"logL={logL:.4f}, chi2={chi2:.4f}"

def test_bao_desi_sensitivity():
    from likelihoods.BAO_desi import desi_likelihood
    logL_good, _ = desi_likelihood(cosmo_fid)
    logL_bad,  _ = desi_likelihood(cosmo_bad)
    assert_worsens(logL_good, logL_bad, "BAO_DESI")
    return f"Δ logL = {logL_good - logL_bad:.2f}"

def test_bao_desi_rd_sensitivity():
    """r_d shift should change logL."""
    from likelihoods.BAO_desi import desi_likelihood
    logL_1, _ = desi_likelihood(cosmo_fid)
    # Shift omega_b to change r_d
    cosmo_rd = MockCosmology(H0=67.36, Omega_m=0.315, Omega_b=0.070)
    logL_2, _ = desi_likelihood(cosmo_rd)
    assert logL_1 != logL_2, "BAO_DESI not sensitive to r_d change"
    return f"logL Planck={logL_1:.3f}, shifted Ob={logL_2:.3f}"


# ================================================================
# 4. BAO 2D
# ================================================================

def test_bao_2d_fiducial():
    if not os.path.exists(os.path.join(DATA_DIR, "bao_t_on_data.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.BAO_2D import BAOAngular
    bao = BAOAngular(DATA_DIR)
    logL = bao(cosmo_fid)
    assert_finite(logL)
    assert_negative(logL)
    n = len(bao.z_eff) + len(bao.z_ON)
    assert_chi2_range(logL, n, "BAO_2D")
    return f"logL={logL:.4f}, n_data={n}"

def test_bao_2d_sensitivity():
    if not os.path.exists(os.path.join(DATA_DIR, "bao_t_on_data.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.BAO_2D import BAOAngular
    bao = BAOAngular(DATA_DIR)
    logL_good = bao(cosmo_fid)
    logL_bad  = bao(cosmo_bad)
    assert_worsens(logL_good, logL_bad, "BAO_2D")
    return f"Δ logL = {logL_good - logL_bad:.2f}"


# ================================================================
# 5. Cosmic Chronometers (CC)
# ================================================================

def test_cc_fiducial():
    if not os.path.exists(os.path.join(DATA_DIR, "HzTable_MM_BC03.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.CC import CC
    cc = CC(DATA_DIR)
    # CC uses the original interface: (theta, priors, H_model)
    priors = {"H0": (40,90), "Omega_cdm": (0.1,0.5), "Omega_b": (0.02,0.06)}
    theta  = np.array([PLANCK["H0"], 0.2642, PLANCK["Omega_b"]])
    def H_model(z, p):
        H0 = p["H0"]; Om = p["Omega_cdm"] + p["Omega_b"]
        return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))
    logL = cc(theta, priors, H_model)
    assert_finite(logL)
    assert_negative(logL)
    assert_chi2_range(logL, cc.n_data, "CC")
    return f"logL={logL:.4f}, n_data={cc.n_data}"

def test_cc_sensitivity():
    if not os.path.exists(os.path.join(DATA_DIR, "HzTable_MM_BC03.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.CC import CC
    cc = CC(DATA_DIR)
    priors = {"H0": (40,90), "Omega_cdm": (0.1,0.5), "Omega_b": (0.02,0.06)}
    def H_model(z, p):
        H0 = p["H0"]; Om = p["Omega_cdm"] + p["Omega_b"]
        return H0 * np.sqrt(Om*(1+z)**3 + (1-Om))
    t_good = np.array([PLANCK["H0"], 0.2642, PLANCK["Omega_b"]])
    t_bad  = np.array([50.0, 0.45, 0.07])
    logL_good = cc(t_good, priors, H_model)
    logL_bad  = cc(t_bad,  priors, H_model)
    assert_worsens(logL_good, logL_bad, "CC")
    return f"Δ logL = {logL_good - logL_bad:.2f}"


# ================================================================
# 6. RSD
# ================================================================

def test_rsd_fiducial():
    if not os.path.exists(os.path.join(DATA_DIR, "RSD.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.rsd_likelihood import RSDFastLikelihood
    rsd = RSDFastLikelihood(DATA_DIR)
    logL = rsd(cosmo_fid)
    assert_finite(logL)
    assert_negative(logL)
    assert_chi2_range(logL, rsd.n_data, "RSD")
    return f"logL={logL:.4f}, n_data={rsd.n_data}"

def test_rsd_sensitivity():
    if not os.path.exists(os.path.join(DATA_DIR, "RSD.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.rsd_likelihood import RSDFastLikelihood
    rsd = RSDFastLikelihood(DATA_DIR)
    logL_good = rsd(cosmo_fid)
    logL_bad  = rsd(cosmo_bad)
    assert_worsens(logL_good, logL_bad, "RSD")
    return f"Δ logL = {logL_good - logL_bad:.2f}"

def test_rsd_fsigma8_range():
    """fsigma8 values should be in [0.2, 0.7] for reasonable cosmologies."""
    if not os.path.exists(os.path.join(DATA_DIR, "RSD.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.rsd_likelihood import RSDFastLikelihood
    rsd = RSDFastLikelihood(DATA_DIR)
    fs8 = cosmo_fid.fsigma8(rsd.z)
    assert np.all(fs8 > 0.2) and np.all(fs8 < 0.7), \
        f"fsigma8 out of range: {fs8}"
    return f"fsigma8 in [{fs8.min():.3f}, {fs8.max():.3f}]"


# ================================================================
# 7. f(z)
# ================================================================

def test_f_fiducial():
    if not os.path.exists(os.path.join(DATA_DIR, "f.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.f_likelihood import fFastLikelihood
    fl = fFastLikelihood(DATA_DIR)
    logL = fl(cosmo_fid)
    assert_finite(logL)
    assert_negative(logL)
    assert_chi2_range(logL, fl.n_data, "f(z)")
    return f"logL={logL:.4f}, n_data={fl.n_data}"

def test_f_sensitivity():
    if not os.path.exists(os.path.join(DATA_DIR, "f.txt")):
        raise FileNotFoundError("data file missing")
    from likelihoods.f_likelihood import fFastLikelihood
    fl = fFastLikelihood(DATA_DIR)
    logL_good = fl(cosmo_fid)
    logL_bad  = fl(cosmo_bad)
    assert_worsens(logL_good, logL_bad, "f(z)")
    return f"Δ logL = {logL_good - logL_bad:.2f}"


# ================================================================
# 8. Engine integration test
# ================================================================

def test_engine_bbn_bao():
    """End-to-end: engine assembles logL correctly for BBN + BAO_DESI."""
    from engine import LikelihoodEngine
    from cosmology import GenericCosmology

    engine = LikelihoodEngine(
        datasets=["BBN", "BAO_DESI"],
        data_dir=DATA_DIR,
        use_cache=True,
    )

    priors = {
        "H0":        (40.0, 90.0),
        "Omega_cdm": (0.10, 0.50),
        "Omega_b":   (0.02, 0.06),
    }

    def H_model(z, p):
        H0 = p["H0"]; h = H0/100
        Om = p["Omega_cdm"] + p["Omega_b"]
        OL = 1.0 - Om
        return H0 * np.sqrt(Om*(1+z)**3 + OL)

    # Planck-like point
    theta_good = np.array([67.36, 0.2642, 0.0493])
    theta_bad  = np.array([50.0,  0.45,   0.07])

    logL_good = engine.log_likelihood(theta_good, priors, H_model)
    logL_bad  = engine.log_likelihood(theta_bad,  priors, H_model)

    assert_finite(logL_good)
    assert_negative(logL_good)
    assert_worsens(logL_good, logL_bad, "Engine BBN+BAO_DESI")

    # Test cache: second call must return same value
    logL_cached = engine.log_likelihood(theta_good, priors, H_model)
    assert logL_cached == logL_good, "Cache returned different value"

    if hasattr(engine, "cache_info"):
        ci = engine.cache_info
        assert ci.get("hits", 1) >= 0

    return f"logL(Planck)={logL_good:.4f}  logL(bad)={logL_bad:.4f}"


# ================================================================
# 9. Consistency: logL(Planck) > logL(perturbed)
# ================================================================

def test_gradient_bbn():
    """logL is highest near the BBN mean and falls on both sides."""
    from likelihoods.BBN import BBNLikelihood
    bbn = BBNLikelihood()
    H0  = 67.36

    logL_peak, _ = bbn.loglkl(H0, 0.04930)   # Omega_b ~ 0.0493
    logL_hi,   _ = bbn.loglkl(H0, 0.06000)
    logL_lo,   _ = bbn.loglkl(H0, 0.02000)

    assert logL_peak > logL_hi,  "BBN: logL did not fall above peak"
    assert logL_peak > logL_lo,  "BBN: logL did not fall below peak"
    return f"peak={logL_peak:.4f} | hi={logL_hi:.4f} | lo={logL_lo:.4f}"

def test_gradient_bao_desi():
    """logL degrades smoothly as H0 moves away from best-fit."""
    from likelihoods.BAO_desi import desi_likelihood
    logLs = []
    for H0 in [55, 60, 67.36, 75, 82]:
        cosmo = MockCosmology(H0=H0, Omega_m=0.315, Omega_b=0.0493)
        logL, _ = desi_likelihood(cosmo)
        logLs.append(logL)
    # Planck value (index 2) should be best or close to best
    best_idx = int(np.argmax(logLs))
    assert best_idx in [1, 2, 3], \
        f"BAO_DESI best H0 index={best_idx}, expected near Planck"
    return "  ".join(f"H0={H0}:{L:.2f}" for H0, L in
                     zip([55,60,67.36,75,82], logLs))


# ================================================================
# Run all tests
# ================================================================

TESTS = [
    # BBN
    ("BBN / fiducial logL",       test_bbn_fiducial),
    ("BBN / parameter sensitivity",test_bbn_sensitivity),
    ("BBN / peak at correct value",test_bbn_peak),
    ("BBN / gradient shape",       test_gradient_bbn),

    # CMB theta*
    ("CMBTheta / fiducial logL",   test_cmb_fiducial),
    ("CMBTheta / theory vector",   test_cmb_theory_vector),
    ("CMBTheta / sensitivity",     test_cmb_sensitivity),

    # BAO DESI
    ("BAO_DESI / fiducial logL",   test_bao_desi_fiducial),
    ("BAO_DESI / sensitivity",     test_bao_desi_sensitivity),
    ("BAO_DESI / r_d sensitivity", test_bao_desi_rd_sensitivity),
    ("BAO_DESI / gradient H0",     test_gradient_bao_desi),

    # BAO 2D (needs data file)
    ("BAO_2D / fiducial logL",     test_bao_2d_fiducial),
    ("BAO_2D / sensitivity",       test_bao_2d_sensitivity),

    # CC (needs data file)
    ("CC / fiducial logL",         test_cc_fiducial),
    ("CC / sensitivity",           test_cc_sensitivity),

    # RSD (needs data file)
    ("RSD / fiducial logL",        test_rsd_fiducial),
    ("RSD / sensitivity",          test_rsd_sensitivity),
    ("RSD / fsigma8 range",        test_rsd_fsigma8_range),

    # f(z)
    ("f(z) / fiducial logL",       test_f_fiducial),
    ("f(z) / sensitivity",         test_f_sensitivity),

    # Engine
    ("Engine / BBN + BAO_DESI",    test_engine_bbn_bao),
]


if __name__ == "__main__":
    sep = "=" * 65
    print(sep)
    print("  SERTAO — likelihood test suite")
    print(f"  Data dir : {DATA_DIR}")
    print(f"  Verbose  : {VERBOSE}")
    print(sep)
    print()

    for name, fn in TESTS:
        _run(name, fn)

    # Summary
    passed = sum(1 for _, ok, _ in _results if ok is True)
    failed = sum(1 for _, ok, _ in _results if ok is False)
    skipped= sum(1 for _, ok, _ in _results if ok is None)
    total  = len(_results)

    print()
    print(sep)
    print(f"  Results: {BOLD}{passed} passed{RESET}  "
          f"{RED}{failed} failed{RESET}  "
          f"{YELLOW}{skipped} skipped{RESET}  "
          f"/ {total} total")
    print(sep)

    if failed > 0:
        print()
        print(f"{RED}  Failed tests:{RESET}")
        for name, ok, msg in _results:
            if ok is False:
                print(f"    • {name}: {msg}")
        sys.exit(1)
