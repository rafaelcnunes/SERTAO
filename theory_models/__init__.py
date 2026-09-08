"""
theory_models/
--------------
Each module defines a cosmological model as a self-contained config:
    DATASETS  — observational data to use
    PRIORS    — parameter ranges
    H_model   — H(z, p) callable
    ANALYSIS_NAME — label for output files

Import a model and pass it to run.py, or run directly.
"""
