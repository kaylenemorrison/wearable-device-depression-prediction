"""
Estimate the effective sample size for clustered (repeated-measures) data and
derive a recommended range for the number of model features.

The raw assessment count overstates available information because repeated
PHQ-8 assessments from the same participant are correlated. This script
computes the intraclass correlation (ICC) of the outcome, applies the
design-effect correction to get an effective sample size, and translates that
into an observations-per-predictor range.

Reads the PHQ-8 source directly (../data_cleaned/phq8_data.csv): the effective
sample size depends only on the outcomes and their per-participant grouping,
not on the merge window size.

Run from scripts/:
    python effective_sample_size.py --outcome phq8_score
"""
import argparse

import numpy as np
import pandas as pd


def icc_oneway_unbalanced(df, group_col, value_col):
    """ICC(1), one-way random-effects, adjusted for unbalanced cluster sizes.

    Uses the Donner (1986) adjusted mean cluster size m0 so unequal numbers of
    windows per participant don't bias the estimate.
    """
    g = df[[group_col, value_col]].dropna()
    grand_mean = g[value_col].mean()    

    grouped = g.groupby(group_col)[value_col]
    n_i = grouped.size().to_numpy() # vector of per-participant window counts
    means = grouped.mean().to_numpy() #vector of per-participant outcome means
    N = int(n_i.sum()) # total windows across all participants
    k = len(n_i) # number of participants/clusters
    if k < 2:
        raise ValueError("Need at least 2 participants to estimate ICC.")

    # Between- and within-group sums of squares
    ssb = np.sum(n_i * (means - grand_mean) ** 2) # between-group sum of squares
    within_mean = grouped.transform("mean").to_numpy() # vector of per-row participant means
    ssw = np.sum((g[value_col].to_numpy() - within_mean) ** 2) # within-group sum of squares

    msb = ssb / (k - 1) # between-group mean square; denominator is k-1 (degrees of freedom)
    msw = ssw / (N - k) # within-group mean square; denominator is N-k (degrees of freedom)

    # Adjusted average cluster size for unbalanced designs
    m0 = (N - np.sum(n_i**2) / N) / (k - 1) # Donner (1986) formula for adjusted cluster size

    icc = (msb - msw) / (msb + (m0 - 1) * msw) # ICC(1) formula with m0 adjustment for unbalanced cluster sizes
    return {
        "icc": float(icc),
        "N_total": N,
        "n_participants": k,
        "mean_windows_per_participant": float(n_i.mean()),
        "adjusted_cluster_size_m0": float(m0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="phq8_score",
                    help="Outcome column to compute the ICC on.")
    ap.add_argument("--group", default="participant_name",
                    help="Participant/cluster identifier column.")
    args = ap.parse_args()

    # N_eff depends only on the PHQ-8 assessments and their grouping, not on
    # window size (window size changes the feature lookback, not the number or
    # distribution of outcomes), so read the PHQ-8 source directly.
    path = "../data_cleaned/phq8_data.csv"
    df = pd.read_csv(path)

    if args.outcome not in df.columns:
        raise SystemExit(
            f"Outcome '{args.outcome}' not found. Columns: {list(df.columns)}"
        )

    stats = icc_oneway_unbalanced(df, args.group, args.outcome)

    icc = stats["icc"]
    m_bar = stats["mean_windows_per_participant"]
    N = stats["N_total"]

    deff = 1 + (m_bar - 1) * max(icc, 0.0)
    n_eff = N / deff

    print(f"\nDataset: {path}")
    print(f"  Total assessments (N_total) .... {N}")
    print(f"  Participants (k) ............... {stats['n_participants']}")
    print(f"  Mean assessments/participant (m̄) {m_bar:.2f}")
    print(f"  Outcome ICC (ρ) ................ {icc:.3f}")
    print(f"  Design effect (1+(m̄-1)ρ) ....... {deff:.2f}")
    print(f"  Effective sample size (N_eff) .. {n_eff:.0f}")

    print("\nRecommended feature-count range (predictors p):")
    print(f"  Conservative (N_eff/15) ........ {n_eff / 15:.0f}")
    print(f"  Moderate     (N_eff/10) ........ {n_eff / 10:.0f}")
    print(f"  Liberal      (N_eff/5, regularized) {n_eff / 5:.0f}")
    print(
        "\nNote: with elastic net / XGBoost the liberal bound is a guide, not a "
        "hard cap — regularization handles p > n, but stability degrades past it.\n"
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# References for the formulas used above
# ---------------------------------------------------------------------------
#
# Intraclass correlation, unbalanced one-way random-effects model
# (icc_oneway_unbalanced: MSB, MSW, the adjusted cluster size m0, and the
#  ICC = (MSB - MSW) / (MSB + (m0 - 1)*MSW) estimator):
#   Donner A. "A review of inference procedures for the intraclass correlation
#   coefficient in the one-way random effects model." International Statistical
#   Review 1986;54(1):67-82.  [verified]
#   (Note: Donner recommends the ML estimator over this ANOVA estimator when
#    ICC is very small <=0.1 or very large >=0.8; our estimate ~0.715 is within
#    the range where the ANOVA estimator is appropriate.)
#
# One-way random-effects ICC(1,1) (MSB/MSW formulation):
#   Shrout PE, Fleiss JL. "Intraclass correlations: uses in assessing rater
#   reliability." Psychological Bulletin 1979;86(2):420-428.  [verified]
#
# Design effect and effective sample size
# (DEFF = 1 + (m_bar - 1)*rho; N_eff = N / DEFF):
#   Kish L. "Survey Sampling." New York: John Wiley & Sons, 1965 (xvi+643 pp.).
#   [verified]
#
# Observations-per-predictor heuristics (the N_eff/15, /10, /5 ranges):
#   Harrell FE. "Regression Modeling Strategies: With Applications to Linear
#   Models, Logistic and Ordinal Regression, and Survival Analysis." 2nd ed.
#   Springer Series in Statistics, 2015. DOI: 10.1007/978-3-319-19425-7;
#   ISBN 978-3-319-19424-0.  [verified; heuristic, not a derived bound]
#
# Rigorous prediction-model sample-size calculation (the recommended
# alternative to the observations-per-predictor heuristics):
#   Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, Moons KGM,
#   Collins G, van Smeden M. "Calculating the sample size required for
#   developing a clinical prediction model." BMJ 2020;368:m441.
#   DOI: 10.1136/bmj.m441  [verified]
# ---------------------------------------------------------------------------
