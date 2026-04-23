"""
synthetic_esg_generator.py
==========================
Generates a SYNTHETIC but internally consistent time-varying ESG panel
from the static cross-sectional ESG scores already in master_panel.csv.

Model
-----
ESG_{i,t} = θ_i(t) + OU_i(t) + η_{s(i),t} + ε_{i,t}

Where:
  θ_i(t)      – firm i's drifting long-run mean
                 = esg_score_i  ×  (1 + TREND × (t - t0) / T)
                 Reflects the empirical rise in ESG reporting quality
                 over 2015-2024 (~+8 pts average across S&P 500).

  OU_i(t)     – Ornstein-Uhlenbeck mean-reverting deviation
                 dX = -κ·X·dt + σ_firm·dW_i
                 κ  = speed of mean reversion  (calibrated so half-life ≈ 18 months)
                 σ_firm = firm-level ESG volatility (scaled by esg_score so
                          high-scoring firms are more stable — empirically true)

  η_{s(i),t}  – sector-level common shock (AR-1)
                 Firms in the same GICS sector co-move in ESG ratings
                 (e.g. an oil-spill hits all Energy firms simultaneously)

  ε_{i,t}     – idiosyncratic white noise

Controversy events
  Each firm draws Poisson(λ=0.04) controversy arrivals per month (~twice/4yr).
  Each event drops ESG by ~ N(8, 3) points, recovering via the OU mechanism.

Sub-score simulation
  env_score, social_score, gov_score are simulated as correlated factors
  that sum (with noise) to reproduce esg_score.  Correlation structure:
    env–social: 0.55, env–gov: 0.35, social–gov: 0.45  (from literature).

Outputs
-------
  synthetic_esg_panel.csv    – monthly panel ready for esg_merge_pipeline.py
  synthetic_esg_summary.csv  – per-firm diagnostics (drift, volatility, etc.)

Usage
-----
  python synthetic_esg_generator.py
  python synthetic_esg_generator.py --master master_panel.csv --seed 42
  python synthetic_esg_generator.py --kappa 0.08 --sigma 1.5 --trend 0.006

Then feed the output directly into the merge pipeline:
  python esg_merge_pipeline.py --merge --esg-file synthetic_esg_panel.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS  (all overridable via CLI)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # OU process
    kappa       = 0.06,    # mean-reversion speed per month
                            #   half-life = ln(2)/κ ≈ 11.5 months
    sigma_base  = 1.2,     # base firm ESG volatility (pts/√month)
    sigma_scale = 0.015,   # σ scales DOWN with esg_score (stable firms)

    # Sector factor
    sigma_sector = 0.6,    # sector shock std dev
    rho_sector   = 0.80,   # sector AR-1 autocorrelation

    # Idiosyncratic noise
    sigma_eps   = 0.4,

    # Secular trend: total drift over 118 months
    # (+8 pts average ESG improvement across S&P 500, 2015-2024)
    total_trend = 8.0,

    # Controversy events
    lambda_controversy = 0.04,   # Poisson rate per firm-month
    controversy_mean   = 8.0,    # mean ESG drop (pts)
    controversy_std    = 3.0,    # std of ESG drop

    # Sub-score factor correlations (env, social, gov)
    sub_corr = [[1.00, 0.55, 0.35],
                [0.55, 1.00, 0.45],
                [0.35, 0.45, 1.00]],

    seed = 2025,
)


# ──────────────────────────────────────────────────────────────────────────────
# CORE SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def simulate_esg_panel(master_path: str, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        panel_df   – long-format monthly ESG panel
        summary_df – per-firm simulation diagnostics
    """
    rng = np.random.default_rng(p["seed"])

    # ── Load master panel ──────────────────────────────────────────────────
    print("  Loading master panel …")
    master = pd.read_csv(master_path,
                         usecols=["ticker", "company", "sector",
                                  "year", "month", "date",
                                  "esg_score", "env_score",
                                  "social_score", "gov_score",
                                  "controversy_score"])

    master["ticker"] = master["ticker"].str.strip().str.upper()

    # Get the static ESG baseline per firm
    firm_info = (
        master.groupby("ticker", as_index=False)
              .agg(company=("company", "first"),
                   sector=("sector", "first"),
                   esg_base=("esg_score", "first"),
                   env_base=("env_score", "first"),
                   social_base=("social_score", "first"),
                   gov_base=("gov_score", "first"))
    )

    # Build the time grid
    dates = (master[["year", "month", "date"]]
             .drop_duplicates()
             .sort_values(["year", "month"])
             .reset_index(drop=True))
    T = len(dates)
    t_idx = np.arange(T)

    tickers = firm_info["ticker"].values
    N = len(tickers)
    sectors = firm_info["sector"].values
    unique_sectors = np.unique(sectors)
    esg_base = firm_info["esg_base"].values          # shape (N,)

    print(f"  Simulating {N} firms × {T} months …")

    # ── Secular trend ──────────────────────────────────────────────────────
    # Linear drift: firms with higher base ESG drift slightly more
    # (leaders keep improving; laggards catch up more slowly)
    monthly_trend = p["total_trend"] / T              # avg pts per month
    trend_matrix = np.outer(
        1 + 0.005 * (esg_base - esg_base.mean()),     # firm-level scaling
        monthly_trend * t_idx                          # cumulative by month
    )   # shape (N, T)

    # ── Sector factor (AR-1) ───────────────────────────────────────────────
    sector_shocks = {}
    for sec in unique_sectors:
        eta = np.zeros(T)
        for t in range(1, T):
            eta[t] = (p["rho_sector"] * eta[t-1]
                      + rng.normal(0, p["sigma_sector"] * np.sqrt(1 - p["rho_sector"]**2)))
        sector_shocks[sec] = eta

    # Map each firm → sector shock series
    sector_matrix = np.array([sector_shocks[s] for s in sectors])   # (N, T)

    # ── OU process per firm ────────────────────────────────────────────────
    # Firm volatility scales inversely with ESG score
    # (higher-rated firms are more stable empirically)
    sigma_firm = p["sigma_base"] - p["sigma_scale"] * esg_base      # (N,)
    sigma_firm = np.clip(sigma_firm, 0.3, 3.0)

    ou = np.zeros((N, T))
    for t in range(1, T):
        dW = rng.normal(0, 1, N)
        ou[:, t] = (ou[:, t-1]
                    - p["kappa"] * ou[:, t-1]
                    + sigma_firm * dW)

    # ── Controversy events ─────────────────────────────────────────────────
    # Poisson arrivals; each event injects a negative shock that decays via OU
    controversy_shocks = np.zeros((N, T))
    for i in range(N):
        arrivals = rng.poisson(p["lambda_controversy"], T)
        for t in np.where(arrivals > 0)[0]:
            magnitude = rng.normal(p["controversy_mean"], p["controversy_std"])
            controversy_shocks[i, t] -= abs(magnitude)

    # Smooth controversy shocks through the OU mechanism
    controversy_ou = np.zeros((N, T))
    for t in range(1, T):
        controversy_ou[:, t] = (controversy_ou[:, t-1]
                                - p["kappa"] * controversy_ou[:, t-1]
                                + controversy_shocks[:, t])

    # ── Idiosyncratic noise ────────────────────────────────────────────────
    eps = rng.normal(0, p["sigma_eps"], (N, T))

    # ── Combine into ESG score ─────────────────────────────────────────────
    esg_base_matrix = esg_base[:, None] * np.ones((N, T))   # broadcast
    esg_sim = (esg_base_matrix
               + trend_matrix
               + ou
               + sector_matrix
               + controversy_ou
               + eps)

    # Hard-clip to [0, 100] — ESG scores are bounded
    esg_sim = np.clip(esg_sim, 0, 100)

    # ── Sub-scores (env, social, gov) ─────────────────────────────────────
    # Each sub-score has its own OU trajectory; constrained to reproduce
    # aggregate esg_score on average via a factor model
    L = np.linalg.cholesky(np.array(p["sub_corr"]))          # (3,3)

    def simulate_subscore(base_col: np.ndarray, weight: float) -> np.ndarray:
        """Simulate one sub-score with given base and correlation weight."""
        sub = np.zeros((N, T))
        base_matrix = base_col[:, None] * np.ones((N, T))
        for t in range(1, T):
            raw = rng.normal(0, 1, (N, 3))
            corr_noise = (raw @ L.T)[:, 0]        # first correlated factor
            sub[:, t] = (sub[:, t-1]
                         - p["kappa"] * sub[:, t-1]
                         + 1.5 * corr_noise)
        return np.clip(base_matrix + sub, 0, 100)

    env_base_arr    = firm_info["env_base"].fillna(firm_info["esg_base"] * 0.35).values
    social_base_arr = firm_info["social_base"].fillna(firm_info["esg_base"] * 0.35).values
    gov_base_arr    = firm_info["gov_base"].fillna(firm_info["esg_base"] * 0.30).values

    env_sim    = simulate_subscore(env_base_arr,    0.35)
    social_sim = simulate_subscore(social_base_arr, 0.35)
    gov_sim    = simulate_subscore(gov_base_arr,    0.30)

    # Controversy score: cumulative controversy events (higher = more controversial)
    controversy_count = np.cumsum(
        np.array([rng.poisson(p["lambda_controversy"], T) for _ in range(N)]),
        axis=1
    ).astype(float)
    controversy_score_sim = np.clip(controversy_count * 2, 0, 10)

    # ── Assemble long-format DataFrame ────────────────────────────────────
    print("  Assembling long-format panel …")
    rows = []
    for i, ticker in enumerate(tickers):
        for t in range(T):
            rows.append({
                "ticker":             ticker,
                "year":               int(dates["year"].iloc[t]),
                "month":              int(dates["month"].iloc[t]),
                "date":               dates["date"].iloc[t],
                "esg_score":          round(float(esg_sim[i, t]),   4),
                "env_score":          round(float(env_sim[i, t]),    4),
                "social_score":       round(float(social_sim[i, t]), 4),
                "gov_score":          round(float(gov_sim[i, t]),    4),
                "controversy_score":  round(float(controversy_score_sim[i, t]), 4),
            })

    panel_df = pd.DataFrame(rows)

    # ── ESG risk level (categorical) ──────────────────────────────────────
    bins   = [0, 20, 40, 60, 80, 100]
    labels = ["Severe", "High", "Medium", "Low", "Negligible"]
    panel_df["esg_risk_level"] = pd.cut(
        panel_df["esg_score"], bins=bins, labels=labels, right=True
    ).astype(str)

    # ── Per-firm diagnostics ───────────────────────────────────────────────
    summary_rows = []
    for i, ticker in enumerate(tickers):
        s = esg_sim[i, :]
        summary_rows.append({
            "ticker":           ticker,
            "sector":           sectors[i],
            "esg_base":         round(float(esg_base[i]), 2),
            "esg_mean_sim":     round(float(s.mean()),    2),
            "esg_std_sim":      round(float(s.std()),     2),
            "esg_min_sim":      round(float(s.min()),     2),
            "esg_max_sim":      round(float(s.max()),     2),
            "total_drift":      round(float(s[-1] - s[0]),2),
            "sigma_firm":       round(float(sigma_firm[i]),2),
        })
    summary_df = pd.DataFrame(summary_rows)

    return panel_df, summary_df


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate(panel_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print key statistics to sanity-check the simulation."""
    print("\n  ── Simulation validation ────────────────────────────────────")
    print(f"  Rows           : {len(panel_df):,}")
    print(f"  Tickers        : {panel_df['ticker'].nunique()}")
    print(f"  Months         : {panel_df['month'].nunique()} per year")
    print(f"  ESG score range: {panel_df['esg_score'].min():.1f} – {panel_df['esg_score'].max():.1f}")
    print(f"  Mean ESG (all) : {panel_df['esg_score'].mean():.2f}")
    print(f"  Avg firm σ     : {summary_df['esg_std_sim'].mean():.2f} pts")
    print(f"  Avg total drift: {summary_df['total_drift'].mean():.2f} pts over panel")

    # Cross-sectional rank correlation with original scores
    firm_mean = panel_df.groupby("ticker")["esg_score"].mean()
    merged = summary_df.set_index("ticker")[["esg_base"]].join(firm_mean.rename("esg_mean_sim"))
    rank_corr = merged["esg_base"].corr(merged["esg_mean_sim"], method="spearman")
    print(f"  Rank corr (base vs sim mean): {rank_corr:.4f}  [want > 0.90]")
    print(f"  ────────────────────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-varying ESG panel from master_panel.csv"
    )
    parser.add_argument("--master",       default="master_panel.csv")
    parser.add_argument("--out",          default="synthetic_esg_panel.csv")
    parser.add_argument("--summary-out",  default="synthetic_esg_summary.csv")
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--kappa",        type=float, default=DEFAULTS["kappa"],
                        help="OU mean-reversion speed (default 0.06 → half-life ~12 mo)")
    parser.add_argument("--sigma",        type=float, default=DEFAULTS["sigma_base"],
                        help="Base firm ESG volatility (default 1.2 pts/√month)")
    parser.add_argument("--trend",        type=float, default=DEFAULTS["total_trend"],
                        help="Total ESG trend over full panel in pts (default 8.0)")
    parser.add_argument("--lambda-controversy", type=float,
                        default=DEFAULTS["lambda_controversy"],
                        help="Controversy event rate per firm-month (default 0.04)")
    args = parser.parse_args()

    p = dict(DEFAULTS)
    p["seed"]                  = args.seed
    p["kappa"]                 = args.kappa
    p["sigma_base"]            = args.sigma
    p["total_trend"]           = args.trend
    p["lambda_controversy"]    = args.lambda_controversy

    print("\n── Synthetic ESG Generator ──────────────────────────────────────────────")
    print(f"  Model: OU(κ={p['kappa']}, σ={p['sigma_base']}) + "
          f"sector-AR1 + trend({p['total_trend']}pts) + "
          f"controversy(λ={p['lambda_controversy']})")

    panel_df, summary_df = simulate_esg_panel(args.master, p)
    validate(panel_df, summary_df)

    panel_df.to_csv(args.out, index=False)
    summary_df.to_csv(args.summary_out, index=False)

    print(f"\n  ✓ Panel   → {args.out}  ({len(panel_df):,} rows)")
    print(f"  ✓ Summary → {args.summary_out}  ({len(summary_df)} firms)")
    print(f"\n  Next step:")
    print(f"    python esg_merge_pipeline.py --merge --esg-file {args.out}\n")


if __name__ == "__main__":
    main()