"""
Entry point for SERTAO cosmological analyses.

Loads a model from theory_models/, runs the nested sampling
analysis, and saves chains to the output directory.

Usage
-----
    python run.py --model LCDM
    python run.py --model LCDM_neutrinos
    python run.py --model LCDM --nlive 500 --output results/
    mpirun -np 4 python run.py --model LCDM_neutrinos

Arguments
---------
--model     Name of the module inside theory_models/ (without .py)
            Available: LCDM, LCDM_neutrinos
--nlive     Number of live points (default: from model or 300)
--dlogz     Stopping criterion on Δ log Z (default: 0.1)
--output    Output directory for chains (default: chains/)
--no-resume Start fresh even if a checkpoint exists
--debug     Print which dataset rejected a point

Adding a new model
------------------
1. Create  theory_models/my_model.py  with:
       DATASETS, PRIORS, H_model, ANALYSIS_NAME
2. Run:
       python run.py --model my_model
"""

import argparse
import importlib
import sys
import os

# ── Make sure SERTAO package and theory_models are importable ────
# run.py lives next to the SERTAO/ package and theory_models/ folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def load_model(name: str):
    """
    Import theory_models/<name>.py and return the module.
    Raises a clear error if the model is not found.
    """
    try:
        mod = importlib.import_module(f"theory_models.{name}")
    except ModuleNotFoundError:
        available = [
            f.stem for f in
            __import__("pathlib").Path(_HERE, "theory_models").glob("*.py")
            if not f.name.startswith("_")
        ]
        print(f"\n  [SERTAO] Model '{name}' not found in theory_models/")
        print(f"  Available models: {', '.join(sorted(available))}")
        sys.exit(1)

    # Validate required attributes
    required = ["DATASETS", "PRIORS", "H_model"]
    missing  = [a for a in required if not hasattr(mod, a)]
    if missing:
        print(f"\n  [SERTAO] Model '{name}' is missing: {missing}")
        sys.exit(1)

    return mod


def main():
    parser = argparse.ArgumentParser(
        description="SERTAO — cosmological parameter estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name (module in theory_models/, without .py)",
    )
    parser.add_argument(
        "--nlive", type=int, default=None,
        help="Number of live points (default: 300)",
    )
    parser.add_argument(
        "--dlogz", type=float, default=0.1,
        help="Stopping Δ log Z (default: 0.1)",
    )
    parser.add_argument(
        "--output", default="chains",
        help="Output directory for chains (default: chains/)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignoring any existing checkpoint",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print which dataset rejected each point",
    )

    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────
    model = load_model(args.model)

    datasets      = model.DATASETS
    priors        = model.PRIORS
    H_model       = model.H_model
    analysis_name = getattr(model, "ANALYSIS_NAME", args.model)
    nlive         = args.nlive or getattr(model, "NLIVE", 300)

    # ── Engine and MCMC ──────────────────────────────────────────
    from SERTAO.engine import LikelihoodEngine
    from SERTAO.MCMC   import run_mcmc

    engine = LikelihoodEngine(
        datasets=datasets,
        data_dir=os.path.join(_HERE, "data"),
        use_cache=True,
    )

    run_mcmc(
        engine        = engine,
        H_model       = H_model,
        priors        = priors,
        analysis_name = analysis_name,
        datasets      = datasets,
        nlive         = nlive,
        dlogz         = args.dlogz,
        output_dir    = args.output,
        resume        = not args.no_resume,
        debug         = args.debug,
    )


if __name__ == "__main__":
    main()
