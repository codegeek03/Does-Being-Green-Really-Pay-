# ESG CAPM Analysis Repo

This repository is a cleaned, GitHub-ready version of the ESG-CAPM project.

## Structure

- `src/analysis/` - Python analysis pipeline and plotting scripts
- `data/raw/` - source CSV inputs
- `data/processed/` - engineered panel data used by the analysis
- `results/tables/` - generated CSV tables
- `results/figures/` - generated publication figures
- `thesis/` - LaTeX thesis sources and build files

## Main scripts

- `src/analysis/esg_capm_analysis.py` - end-to-end empirical pipeline
- `src/analysis/extra_plots.py` - publication-quality figures
- `src/analysis/stochastic_esg_simulator.py` - stochastic interpolation of annualised ESG data

## How to run

From the repository root:

```powershell
python src/analysis/esg_capm_analysis.py
python src/analysis/extra_plots.py
```

If you use the included virtual environment from the original workspace, activate it before running the scripts. The repo itself does not bundle the environment.

## Notes

- The current outputs in `results/` are included so the repository is immediately inspectable.
- The `thesis/` folder contains the LaTeX source tree used for the written report.
- Build artifacts, caches, and virtual environments are ignored through `.gitignore`.
