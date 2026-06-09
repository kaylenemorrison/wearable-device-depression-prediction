#!/usr/bin/env python3
"""
catch22 short-time-series feature validity test
===============================================

Question this answers:
    For a time series of length N, does each catch22 feature produce a
    VALID and INFORMATIVE value? On very short series some features
    (a) error / return NaN, or (b) "degenerate": collapse to a single
    constant value regardless of the input, making them useless as
    features even though they don't crash.

Method:
    For each length N, generate many random series of several different
    shapes (white noise, uniform noise, noisy sinusoid, trend+noise,
    random walk). Compute every catch22 feature on each. Then per
    (feature, length) summarise:
        - NaN/inf rate
        - whether the feature actually VARIES across different inputs
          (std of finite outputs). If it never varies it is degenerate.

Classification per (feature, length):
    OK    : no NaN/inf AND output varies across inputs        -> usable
    DEGEN : no NaN/inf but output is constant for every input -> useless
    NaN   : produced NaN/inf on >0% of inputs (number = % NaN)
"""

import warnings
import numpy as np
import pycatch22

warnings.filterwarnings("ignore")

LENGTHS = [3, 5, 7, 14, 21, 28]
TRIALS_PER_SHAPE = 300
SEED = 0

# The 22 canonical features (catch24 adds DN_Mean & DN_Spread_Std; excluded here).
# Ordered to match the catch22 gitbook feature-overview table. Each entry is
# (pycatch22 callable name, "Feature Name" from the gitbook table). A few gitbook
# Feature Names differ from the pycatch22 function name (e.g. first1e_acf_tau ->
# CO_f1ecac); the callable name is what getattr(pycatch22, ...) needs.
FEATURES = [
    ("DN_HistogramMode_5", "DN_HistogramMode_5"),
    ("DN_HistogramMode_10", "DN_HistogramMode_10"),
    ("DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_p_001_mdrmd"),
    ("DN_OutlierInclude_n_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd"),
    ("CO_f1ecac", "first1e_acf_tau"),
    ("CO_FirstMin_ac", "firstMin_acf"),
    ("SP_Summaries_welch_rect_area_5_1", "SP_Summaries_welch_rect_area_5_1"),
    ("SP_Summaries_welch_rect_centroid", "SP_Summaries_welch_rect_centroid"),
    ("FC_LocalSimple_mean3_stderr", "FC_LocalSimple_mean3_stderr"),
    ("FC_LocalSimple_mean1_tauresrat", "FC_LocalSimple_mean1_tauresrat"),
    ("MD_hrv_classic_pnn40", "MD_hrv_classic_pnn40"),
    ("SB_BinaryStats_mean_longstretch1", "SB_BinaryStats_mean_longstretch1"),
    ("SB_BinaryStats_diff_longstretch0", "SB_BinaryStats_diff_longstretch0"),
    ("SB_MotifThree_quantile_hh", "SB_MotifThree_quantile_hh"),
    ("CO_HistogramAMI_even_2_5", "CO_HistogramAMI_even_2_5"),
    ("CO_trev_1_num", "CO_trev_1_num"),
    ("IN_AutoMutualInfoStats_40_gaussian_fmmi", "IN_AutoMutualInfoStats_40_gaussian_fmmi"),
    ("SB_TransitionMatrix_3ac_sumdiagcov", "SB_TransitionMatrix_3ac_sumdiagcov"),
    ("PD_PeriodicityWang_th0_01", "PD_PeriodicityWang_th001"),
    ("CO_Embed2_Dist_tau_d_expfit_meandiff", "CO_Embed2_Dist_tau_d_expfit_meandiff"),
    ("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1"),
    ("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1"),
]

LABEL_WIDTH = 46  # width of the feature-name column in printouts

rng = np.random.default_rng(SEED)


def make_series(shape, n):
    if shape == "gauss":
        return rng.standard_normal(n)
    if shape == "uniform":
        return rng.uniform(0, 1, n)
    if shape == "sine":
        t = np.arange(n)
        period = rng.uniform(2, max(2.1, n))
        return np.sin(2 * np.pi * t / period) + 0.3 * rng.standard_normal(n)
    if shape == "trend":
        t = np.arange(n)
        return 0.5 * t + rng.standard_normal(n)
    if shape == "walk":
        return np.cumsum(rng.standard_normal(n))
    raise ValueError(shape)


SHAPES = ["gauss", "uniform", "sine", "trend", "walk"]


def run():
    # results[longform_label][length] = list of output values
    labels = [label for _, label in FEATURES]
    results = {label: {n: [] for n in LENGTHS} for label in labels}

    for n in LENGTHS:
        for shape in SHAPES:
            for _ in range(TRIALS_PER_SHAPE):
                x = list(map(float, make_series(shape, n))) # pycatch22 wants lists, not numpy arrays
                for func_name, label in FEATURES:
                    fn = getattr(pycatch22, func_name)
                    try:
                        v = fn(x)
                        if v is None:
                            v = np.nan
                    except Exception:
                        v = np.nan
                    results[label][n].append(v)

    # ---- classify ----
    def classify(vals):
        arr = np.asarray(vals, dtype=float)
        total = arr.size
        bad = np.sum(~np.isfinite(arr)) # count of NaN/inf
        finite = arr[np.isfinite(arr)]
        bad_rate = bad / total
        if bad_rate > 0:
            return ("NaN", bad_rate, finite.std() if finite.size else 0.0)
        spread = finite.std()
        if spread < 1e-10:
            return ("DEGEN", 0.0, spread)
        return ("OK", 0.0, spread)

    status = {label: {} for label in labels}
    for label in labels:
        for n in LENGTHS:
            status[label][n] = classify(results[label][n])

    # ---- print matrix ----
    def cell(s):
        kind, rate, _ = s
        if kind == "OK":
            return "  OK "
        if kind == "DEGEN":
            return "DEGEN"
        return f"{int(round(rate*100)):>3d}%N"  # % NaN

    header = "FEATURE".ljust(LABEL_WIDTH) + "".join(f"{n:>7d}" for n in LENGTHS)
    print(header)
    print("-" * len(header))
    for label in labels:
        row = label.ljust(LABEL_WIDTH) + "".join(f"{cell(status[label][n]):>7}" for n in LENGTHS)
        print(row)

    print("\nLegend:  OK = valid & varies | DEGEN = constant for every input "
          "(useless) | NN%N = NN% of inputs returned NaN/inf")

    # ---- per-length usable counts ----
    print("\nUsable (OK) feature count by length:")
    for n in LENGTHS:
        ok = sum(1 for label in labels if status[label][n][0] == "OK")
        degen = sum(1 for label in labels if status[label][n][0] == "DEGEN")
        nan = sum(1 for label in labels if status[label][n][0] == "NaN")
        print(f"  N={n:>2d}:  OK={ok:>2d}   DEGEN={degen:>2d}   NaN-prone={nan:>2d}")

    # ---- features that are never problematic across all tested lengths ----
    always_ok = [label for label in labels if all(status[label][n][0] == "OK" for n in LENGTHS)]
    print(f"\nFeatures OK at EVERY tested length ({len(always_ok)}/22):")
    for label in always_ok:
        print("  +", label)

    problem = [label for label in labels if label not in always_ok]
    print(f"\nFeatures with issues at one or more lengths ({len(problem)}/22):")
    for label in problem:
        bad_at = [f"N={n}({status[label][n][0]}"
                  + (f" {int(round(status[label][n][1]*100))}%" if status[label][n][0] == 'NaN' else "")
                  + ")"
                  for n in LENGTHS if status[label][n][0] != "OK"]
        print(f"  - {label}: " + ", ".join(bad_at))


if __name__ == "__main__":
    run()
