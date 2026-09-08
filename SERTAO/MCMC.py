"""
Nested sampling engine for SERTAO, built on dynesty.
----------------------------------
During the run, a table is printed every print_interval seconds:

  iter     H0  Omega_cdm  Omega_b     M_B  chi2_eff     logZ  ΔlogZ    calls
  -----------------------------------------------------------------------
   150   67.41     0.2638   0.0493  -19.32    23.87   -12.44  0.821    6,240

Fields: iter = dead points collected; params = values of the ejected
point; chi2_eff = -2 ln L; logZ = running evidence; ΔlogZ = remaining
evidence (stops when < dlogz); calls = total likelihood evaluations.
"""

import numpy as np
import dynesty
import os
import sys
import contextlib
import threading
import time
from datetime import datetime, timedelta


# ================================================================
# Stdout suppression
# ================================================================

@contextlib.contextmanager
def _suppress_stdout():
    """
    Redirect sys.stdout to /dev/null for the duration of the block.

    IMPORTANT: the _LiveMonitor thread must capture the real stdout
    BEFORE this context manager is entered, otherwise its print()
    calls are also silenced.
    """
    with open(os.devnull, "w") as devnull:
        old, sys.stdout = sys.stdout, devnull
        try:
            yield
        finally:
            sys.stdout = old


# ================================================================
# Pickle-safe likelihood wrapper
# ================================================================

class _SERTAOLikelihood:
    """
    Module-level callable so dynesty can pickle it into checkpoints.
    A closure inside main() raises AttributeError on restore.
    """
    def __init__(self, engine, priors, H_model, pre_check=None, debug=False):
        self._engine    = engine
        self._priors    = priors
        self._H_model   = H_model
        self._pre_check = pre_check
        self._debug     = debug

    def __call__(self, theta):
        return self._engine.log_likelihood(
            theta, self._priors, self._H_model,
            pre_check=self._pre_check, debug=self._debug,
        )


# ================================================================
# Live monitor — MontePython-style table
# ================================================================

class _LiveMonitor:
    """
    Background thread that prints a live parameter table.

    Fix for stdout suppression
    --------------------------
    _suppress_stdout() swaps sys.stdout to /dev/null while dynesty
    runs.  Any print() inside the thread would therefore be silenced.

    The fix: capture the real stdout object at construction time and
    write directly to it — bypassing whatever sys.stdout points to
    at the moment each row is printed.
    """

    COL_WIDTH = 9

    def __init__(self, sampler, param_names, nlive, dlogz,
                 real_stdout, print_interval=30.0):
        self._sampler       = sampler
        self._param_names   = param_names
        self._nlive         = nlive
        self._dlogz         = dlogz
        self._out           = real_stdout      # captured before suppress
        self._print_interval = print_interval
        self._ndim          = len(param_names)

        self._stop          = threading.Event()
        self._thread        = threading.Thread(target=self._run, daemon=True)
        self._last_print    = 0.0
        self._last_iter     = -1
        self._header_done   = False
        self._t0            = None

    def start(self):
        self._t0 = time.monotonic()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def print_final(self):
        state = self._read()
        if state is not None:
            self._print_row(*state)

    # ── internals ─────────────────────────────────────────────────

    def _read(self):
        try:
            res   = self._sampler.results
            it    = int(res.niter)
            if it == 0 or it == self._last_iter:
                return None
            params = res.samples[-1]
            logl   = float(res.logl[-1])
            logz   = float(res.logz[-1])
            dlogz  = float(res.logzerr[-1])
            ncall  = int(np.sum(res.ncall)) if hasattr(res, "ncall") else it
            return it, params, logl, logz, dlogz, ncall
        except Exception:
            return None

    def _write(self, text):
        """Write directly to the real stdout, bypassing any redirect."""
        try:
            self._out.write(text + "\n")
            self._out.flush()
        except Exception:
            pass

    def _header(self):
        w    = self.COL_WIDTH
        sep  = "-" * (9 + self._ndim * (w + 2) + 42)
        names = [p[:w] for p in self._param_names]
        head  = f"{'iter':>8}  " + "  ".join(f"{n:>{w}}" for n in names)
        head += f"  {'chi2_eff':>9}  {'logZ':>8}  {'ΔlogZ':>6}  {'calls':>8}"
        self._write(sep)
        self._write(head)
        self._write(sep)
        self._header_done = True

    def _print_row(self, it, params, logl, logz, dlogz, ncall):
        if not self._header_done:
            self._header()
        w    = self.COL_WIDTH
        chi2 = -2.0 * logl
        row  = f"{it:>8d}  " + "  ".join(f"{v:>{w}.4f}" for v in params)
        row += f"  {chi2:>9.3f}  {logz:>8.4f}  {dlogz:>6.4f}  {ncall:>8,}"
        self._write(row)
        self._last_iter  = it
        self._last_print = time.monotonic()

    def _run(self):
        # First row: as soon as first dead point exists
        while not self._stop.is_set():
            state = self._read()
            if state is not None:
                self._print_row(*state)
                break
            self._stop.wait(0.5)

        # Subsequent rows: every print_interval seconds
        while not self._stop.is_set():
            self._stop.wait(1.0)
            state = self._read()
            if state is None:
                continue
            if time.monotonic() - self._last_print >= self._print_interval:
                self._print_row(*state)


# ================================================================
# GetDist output
# ================================================================

_LATEX = {
    "H0":         r"H_0",
    "Omega_m":    r"\Omega_m",
    "Omega_cdm":  r"\Omega_{\rm cdm}",
    "Omega_b":    r"\Omega_b",
    "M_B":        r"M_B",
    "Mcal":       r"M_{\rm cal}",
    "sigma8":     r"\sigma_8",
    "Neff":       r"N_{\rm eff}",
    "sum_m":      r"\Sigma m_\nu",
    "lambda_phi": r"\lambda_\phi",
    "n_pot":      r"n_{\rm pot}",
    "w0":         r"w_0",
    "wa":         r"w_a",
}


def save_getdist(samples, weights, loglikes, param_names,
                 root_name="LCDM", chain_id=1, output_dir="chains"):
    """Save chains in GetDist format: weight | loglike | params..."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"{root_name}_{chain_id}")
    np.savetxt(f"{base}.txt",
               np.column_stack([weights, loglikes, samples]),
               fmt="%.8e")
    if chain_id == 1:
        with open(os.path.join(output_dir, f"{root_name}.paramnames"), "w") as f:
            for p in param_names:
                f.write(f"{p}\t{_LATEX.get(p, p)}\n")
        with open(os.path.join(output_dir, f"{root_name}.ranges"), "w") as f:
            for p in param_names:
                f.write(f"{p}\t-1e6 1e6\n")
    return base


# ================================================================
# Uniform prior transform
# ================================================================

class UniformPrior:
    """Maps [0,1]^n to physical prior ranges."""

    def __init__(self, priors: dict):
        self.names = list(priors.keys())
        self.mins  = np.array([priors[p][0] for p in self.names], dtype=np.float64)
        self.maxs  = np.array([priors[p][1] for p in self.names], dtype=np.float64)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return self.mins + u * (self.maxs - self.mins)


# ================================================================
# run_mcmc
# ================================================================

def run_mcmc(
    engine,
    H_model,
    priors: dict,
    analysis_name: str = "LCDM",
    datasets=None,
    nlive: int = 300,
    dlogz: float = 0.1,
    output_dir: str = "chains",
    resume: bool = True,
    pre_check=None,
    debug: bool = False,
    print_interval: float = 30.0,
):
    """
    Run a nested-sampling analysis and save GetDist chains.

    Parameters
    ----------
    engine         : LikelihoodEngine
    H_model        : callable  H(z, p) in km/s/Mpc
    priors         : dict  {param: (min, max)}
    analysis_name  : str
    datasets       : list[str], optional
    nlive          : int   live points  (>=500 for publication)
    dlogz          : float stopping delta log Z
    output_dir     : str
    resume         : bool  try to restore checkpoint
    pre_check      : callable  pre_check(p) -> bool
    debug          : bool  print rejection reasons
    print_interval : float  seconds between live-monitor rows (default 30)

    Returns
    -------
    dict  samples, weights, loglikes, param_names, logZ, logZerr,
          results, chain_file, chain_id, rank
    """

    # ── MPI ──────────────────────────────────────────────────────
    try:
        from mpi4py import MPI
        comm    = MPI.COMM_WORLD
        rank    = comm.Get_rank()
        size    = comm.Get_size()
        use_mpi = size > 1
    except ImportError:
        rank, size, use_mpi = 0, 1, False

    chain_id = rank + 1

    # ── Capture real stdout NOW before any suppression ────────────
    # _suppress_stdout() replaces sys.stdout with /dev/null.
    # The monitor thread must hold a reference to the real file object
    # so its writes bypass the redirect.
    real_stdout = sys.stdout

    # ── Setup ─────────────────────────────────────────────────────
    prior_transform = UniformPrior(priors)
    param_names     = prior_transform.names
    ndim            = len(param_names)

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir,
                                   f"{analysis_name}_checkpoint_rank{rank}.pkl")
    evidence_file   = os.path.join(output_dir, f"{analysis_name}_logZ.txt")

    likelihood = _SERTAOLikelihood(engine, priors, H_model, pre_check, debug)

    # ── Restore or create sampler ─────────────────────────────────
    resuming, sampler = False, None

    if resume and os.path.exists(checkpoint_file):
        try:
            sampler, resuming = dynesty.NestedSampler.restore(checkpoint_file), True
        except Exception as exc:
            if rank == 0:
                print(f"  [SERTAO] Checkpoint unreadable ({type(exc).__name__})"
                      f" — starting fresh.")

    if sampler is None:
        sampler = dynesty.NestedSampler(
            loglikelihood=likelihood,
            prior_transform=prior_transform,
            ndim=ndim,
            nlive=nlive,
            bound="multi",
            sample="rwalk",
            walks=20,
        )

    # ── Header ────────────────────────────────────────────────────
    if rank == 0:
        sep = "=" * 65
        print(sep)
        print("  SERTAO  ·  Cosmological parameter estimation")
        print(sep)
        print(f"  Analysis   : {analysis_name}")
        print(f"  Parameters : {', '.join(param_names)}")
        if datasets:
            print(f"  Datasets   : {', '.join(datasets)}")
        print(f"  Method     : Nested Sampling (dynesty)")
        print(f"  Live pts   : {nlive}   |   stop ΔlogZ < {dlogz}")
        print(f"  Processes  : {size}")
        print(f"  Checkpoint : {'resuming' if resuming else 'fresh run'}")
        print(f"  Output     : {output_dir}/")
        print(sep)
        print(f"  Rows every {print_interval:.0f} s  "
              f"|  chi2_eff = -2 ln L  |  ΔlogZ stops at {dlogz}")
        print()
        sys.stdout.flush()

    # ── Start monitor BEFORE entering suppress context ────────────
    monitor = None
    if rank == 0:
        monitor = _LiveMonitor(
            sampler=sampler,
            param_names=param_names,
            nlive=nlive,
            dlogz=dlogz,
            real_stdout=real_stdout,   # captured above, before any redirect
            print_interval=print_interval,
        )
        monitor.start()

    # ── Run (stdout suppressed so dynesty is silent) ──────────────
    t0 = datetime.now()

    with _suppress_stdout():
        sampler.run_nested(
            dlogz=dlogz,
            print_progress=False,
            checkpoint_file=checkpoint_file,
            resume=resuming,
        )

    if rank == 0 and monitor is not None:
        monitor.stop()
        monitor.print_final()   # always show last state after run

    elapsed = datetime.now() - t0

    # ── Results ───────────────────────────────────────────────────
    res      = sampler.results
    samples  = res.samples
    loglikes = res.logl
    weights  = np.exp(res.logwt - res.logz[-1])
    logZ     = float(res.logz[-1])
    logZerr  = float(res.logzerr[-1])

    chain_file = save_getdist(
        samples=samples, weights=weights, loglikes=loglikes,
        param_names=param_names, root_name=analysis_name,
        chain_id=chain_id, output_dir=output_dir,
    )

    if rank == 0:
        with open(evidence_file, "w") as f:
            f.write("# Bayesian evidence — Nested Sampling (dynesty)\n")
            f.write(f"logZ   = {logZ:.6f}\n")
            f.write(f"sigmaZ = {logZerr:.6f}\n")

    if use_mpi:
        comm.Barrier()

    # ── Summary ───────────────────────────────────────────────────
    if rank == 0:
        ibest  = int(np.argmax(loglikes))
        p_best = samples[ibest]

        sep = "=" * 65
        print()
        print(sep)
        print("  Run complete ✔")
        print(f"  Wall time  : {elapsed}")
        print(f"  log Z      : {logZ:.4f} ± {logZerr:.4f}")
        print(f"  Iterations : {res.niter:,}")
        print(f"  Calls      : {int(np.sum(res.ncall)):,}")
        print()
        print("  Best-fit parameters  (max log L point):")
        for name, val in zip(param_names, p_best):
            print(f"    {name:<14} = {val:.6f}")
        print(f"    {'chi2_eff':<14} = {-2*loglikes[ibest]:.4f}")
        print()
        print(f"  Chains     : {chain_file}.txt")
        print(f"  Evidence   : {evidence_file}")
        try:
            ci = engine.cache_info
            print(f"  Cache      : {ci['hits']:,} hits / {ci['misses']:,} misses"
                  f"  ({ci['hit_rate']} hit rate)")
        except Exception:
            pass
        print(sep)

    return {
        "samples":     samples,
        "weights":     weights,
        "loglikes":    loglikes,
        "param_names": param_names,
        "logZ":        logZ,
        "logZerr":     logZerr,
        "results":     res,
        "chain_file":  chain_file,
        "chain_id":    chain_id,
        "rank":        rank,
    }
