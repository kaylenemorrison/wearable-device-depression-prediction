"""
Riley et al. (2020) minimum-sample-size calculation for a continuous-outcome
prediction model, applied to the PHQ-8 modeling problem.

This is the rigorous counterpart to effective_sample_size.py: instead of an
observations-per-predictor rule of thumb, it uses the four criteria of Riley
et al. (small overfitting / shrinkage >= 0.9, small apparent-vs-adjusted R^2
difference, precise residual SD, precise mean outcome) to either

  (a) report the minimum N required for a given number of predictor
      parameters and anticipated R^2  (--parameters), or
  (b) invert that: report the largest number of predictor parameters that a
      given (effective) sample size can support  (--max-features-for-n),
      e.g. pass the N_eff that effective_sample_size.py reports.

The mean and SD of the outcome (needed by criteria 3 and 4) are read from the
PHQ-8 source data, so only the anticipated R^2 must be assumed.

IMPORTANT — runs in the dedicated `pmsampsize` env, NOT `thesis`
(pmsampsize pins pandas<3, which conflicts with the main pipeline):

    conda env create -f environment-pmsampsize.yml   # once
    conda run -n pmsampsize python scripts/pmsampsize_feature_count.py \
        --rsquared 0.15 --max-features-for-n 817

Note on "parameters": Riley's `parameters` counts predictor *parameters*
(model degrees of freedom), not raw features. A k-level categorical encoded by
one-hot contributes k-1 parameters, so this count corresponds to the expanded
design matrix, not the number of columns in effective_sample_size.py's sense.
"""
import argparse
import io
from contextlib import redirect_stdout

import pandas as pd
from pmsampsize.pmsampsize import pmsampsize


def required_n(parameters, rsquared, intercept, sd, shrinkage, mmoe):
    """Riley minimum sample size for a continuous outcome, table suppressed."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = pmsampsize(
            type="c",
            parameters=parameters,
            rsquared=rsquared,
            intercept=intercept,
            sd=sd,
            shrinkage=shrinkage,
            mmoe=mmoe,
        )
    return res


def max_parameters_for_n(target_n, rsquared, intercept, sd, shrinkage, mmoe,
                         hard_cap=2000):
    """Largest number of predictor parameters whose required N <= target_n.

    Required N is non-decreasing in `parameters` (criteria 1/2 grow with p;
    criteria 3/4 set a p-independent floor), so we scan upward until the
    requirement exceeds target_n. Returns (max_p, floor_n) where floor_n is the
    requirement at p=1 (dominated by the mean/residual-SD criteria)."""
    floor_n = required_n(1, rsquared, intercept, sd, shrinkage, mmoe)["sample_size"]
    if floor_n > target_n:
        return 0, floor_n
    best = 1
    for p in range(2, hard_cap + 1):
        if required_n(p, rsquared, intercept, sd, shrinkage, mmoe)["sample_size"] > target_n:
            break
        best = p
    return best, floor_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsquared", type=float, required=True,
                    help="Anticipated (adjusted) R^2 of the developed model.")
    ap.add_argument("--parameters", type=int, default=None,
                    help="Number of predictor parameters; report the minimum N "
                         "Riley's criteria require for this many.")
    ap.add_argument("--max-features-for-n", type=float, default=None,
                    dest="target_n",
                    help="An (effective) sample size; report the largest number "
                         "of predictor parameters it can support. Pass the N_eff "
                         "from effective_sample_size.py.")
    ap.add_argument("--shrinkage", type=float, default=0.9,
                    help="Target global shrinkage factor (default 0.9).")
    ap.add_argument("--mmoe", type=float, default=1.1,
                    help="Acceptable multiplicative margin of error for the "
                         "intercept / residual SD criteria (default 1.1).")
    ap.add_argument("--outcome", default="phq8_score",
                    help="Outcome column.")
    ap.add_argument("--group", default="participant_name",
                    help="Participant/cluster identifier column (unused in the "
                         "calculation; kept for interface parity).")
    args = ap.parse_args()

    if args.parameters is None and args.target_n is None:
        ap.error("provide --parameters and/or --max-features-for-n")

    path = "../data_cleaned/phq8_data.csv"
    df = pd.read_csv(path)
    if args.outcome not in df.columns:
        raise SystemExit(
            f"Outcome '{args.outcome}' not found. Columns: {list(df.columns)}"
        )

    y = df[args.outcome].dropna()
    mean, sd = float(y.mean()), float(y.std())

    print(f"\nOutcome '{args.outcome}' from {path}")
    print(f"  mean (intercept) = {mean:.3f}")
    print(f"  sd               = {sd:.3f}")
    print(f"  anticipated R^2  = {args.rsquared}")
    print(f"  shrinkage target = {args.shrinkage}   mmoe = {args.mmoe}")

    if args.parameters is not None:
        res = required_n(args.parameters, args.rsquared, mean, sd,
                         args.shrinkage, args.mmoe)
        print(f"\nMinimum N required for {args.parameters} predictor parameters "
              f"(Riley et al. 2020):")
        print(f"  required N = {res['sample_size']}  "
              f"(shrinkage {res['final_shrinkage']}, "
              f"SPP {res['SPP']})")

    if args.target_n is not None:
        max_p, floor_n = max_parameters_for_n(
            args.target_n, args.rsquared, mean, sd, args.shrinkage, args.mmoe)
        print(f"\nGiven a (effective) sample size of {args.target_n:.0f}:")
        if max_p == 0:
            print(f"  even 1 parameter requires N = {floor_n} (driven by the "
                  f"mean/residual-SD criteria) — sample size is insufficient at "
                  f"R^2 = {args.rsquared}.")
        else:
            print(f"  supports up to {max_p} predictor parameters "
                  f"at R^2 = {args.rsquared} (p-independent floor: N = {floor_n}).")
    print()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
# Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, Moons KGM,
# Collins G, van Smeden M. "Calculating the sample size required for developing
# a clinical prediction model." BMJ 2020;368:m441. DOI: 10.1136/bmj.m441
#
# Implemented via the `pmsampsize` Python package (pip), a port of the R
# package of the same name by Riley, Ensor, et al.
# ---------------------------------------------------------------------------
