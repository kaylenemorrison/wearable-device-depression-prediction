import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import shap
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from lightgbm import LGBMClassifier, LGBMRegressor
from missforest import MissForest
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import yaml

from classes.repeated_groupkfold import RepeatedGroupKFold

# os.environ["TQDM_DISABLE"] = "True"

SEED = 42
PATH_TO_YAML = "../feature_spec.yaml"

# Outer CV is RepeatedGroupKFold: OUTER_SPLITS folds per repetition, repeated
# --n-repeats times with a different participant partition each time. Within a
# repetition every window is held out exactly once, so the repetition's pooled
# out-of-fold predictions give ONE performance estimate over the full dataset;
# the spread across repetitions is what the repeats exist to measure. Tuning is
# re-done inside every outer training set with INNER_SPLITS-fold grouped CV.
OUTER_SPLITS = 10
INNER_SPLITS = 5

# Optuna budget per outer fold, overridable with --n-trials.
DEFAULT_TRIALS = {"elastic_net": 20, "xgboost": 40}

# Repetition whose out-of-fold pass backs the calibration scatter and the SHAP
# beeswarm. One repetition is one complete partition of the dataset, which is
# exactly what those plots depict; running them over all repetitions would
# re-plot the same rows --n-repeats times for no added information.
PLOT_REPEAT = 0

def catch22_enabled_windows():
    """Window sizes for which the catch22 panel is produced, read from the
    feature spec so modeling.py and merge_datasets.py agree on the gating (and
    thus on the c22/noc22 filename tag) without hardcoding the set in two places."""
    with open(PATH_TO_YAML) as fh:
        spec = yaml.safe_load(fh)
    return spec.get("catch22", {}).get("enabled_window_sizes", [])

def dataset_stem(window_size, min_present_fraction, catch22_enabled):
    """Filename stem for a merged dataset, mirroring merge_datasets.dataset_stem.
    `mp` = the single present-fraction floor shared by the simple aggregations
    and the catch22 panel; the `c22`/`noc22` tag records whether the catch22
    panel is present for this window. Must stay in sync with merge_datasets.py so
    the right parquet/manifest load.
    """
    c22 = "c22" if catch22_enabled else "noc22"
    return f"final_dataset_w{window_size}_mp{min_present_fraction:g}_{c22}"

def prepare_data(df, categorical_cols, boolean_cols, encode_as_codes=True):
    """Split df into (X, y, groups) and encode the categorical/boolean columns.

    encode_as_codes=True (elastic net + MissForest): cat.codes -> integer codes
    left as float64 (NaN for missing). MissForest accepts float input directly,
    and the OHE consumes the numeric codes fine. encode_as_codes=False (XGBoost,
    enable_categorical=True): keep the
    original labels as 'category' dtype so they are treated as unordered nominal
    — no cat.codes. XGBoost fits on X_train only, so a level appearing only in
    test-fold participants is unseen at train time and routed to the
    missing/default branch at predict; the elastic-net OHE behaves consistently
    via handle_unknown="ignore" (see create_elastic_net_pipeline).
    """
    df_new = df.copy()
    X = df_new.drop(columns=['phq8_score', 'participant_name'])
    y = df_new['phq8_score']
    groups = df_new['participant_name']
    for col in categorical_cols + boolean_cols:
        if encode_as_codes:
            X[col] = X[col].cat.codes
            X[col] = X[col].replace(-1, np.nan)
        elif X[col].dtype.name != 'category':
            X[col] = X[col].astype('category')
    return X, y, groups

def missforest_impute(X_train, X_test, categorical_features, max_iter=3):
    """Leakage-free MissForest imputation: fit on train only, apply to test.

    The categorical columns arrive as float64 integer codes (cat.codes with NaN
    for missing; see prepare_data), which missforest 4.x accepts directly -- no
    dtype coercion needed. The imputer's output is returned as-is: the elastic-net
    pipeline's OneHotEncoder and boolean passthrough consume the numeric codes
    directly. A LGBMClassifier is used for the categorical columns and a
    LGBMRegressor for the numeric ones.
    Returns X_train_imp, or (X_train_imp, X_test_imp) when X_test is given.

    The LightGBM learners are left at their default thread count on purpose:
    called from an outer-CV worker they inherit OMP_NUM_THREADS=1 from joblib's
    inner_max_num_threads, and called from the driver (the final full-data fit)
    they get the whole machine. Pinning n_jobs=1 here would cost the latter."""
    imputer = MissForest(
        clf=LGBMClassifier(linear_tree=True, verbosity=-1, random_state=SEED),
        rgr=LGBMRegressor(linear_tree=True, verbosity=-1, random_state=SEED),
        categorical=categorical_features,
        max_iter=max_iter,
        verbose=0,
    )

    X_train_imp = imputer.fit_transform(X_train)
    if X_test is None:
        return X_train_imp
    X_test_imp = imputer.transform(X_test)
    return X_train_imp, X_test_imp

def impute_inner_folds(X_train, y_train, groups_train, inner_cv,
                       categorical_features, iterations=3):
    """Precompute leakage-free imputed (train, val) pairs for each inner fold.
    MissForest is fit only on inner-train; inner-val is imputed via transform.
    Returns a list of dicts, one per inner fold."""
    cached_folds = []
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train, groups_train):
        # .iloc[...] preserves the original (non-contiguous) labels; reset the
        # index on every slice so X/y stay positionally aligned with the
        # imputer output and downstream indexing.
        X_itr = X_train.iloc[inner_train_idx].reset_index(drop=True)
        X_ival = X_train.iloc[inner_val_idx].reset_index(drop=True)
        y_itr = y_train.iloc[inner_train_idx].reset_index(drop=True)
        y_ival = y_train.iloc[inner_val_idx].reset_index(drop=True)

        X_itr_imp, X_ival_imp = missforest_impute(
            X_itr, X_ival, categorical_features, max_iter=iterations
        )

        cached_folds.append({
            "X_train": X_itr_imp,
            "y_train": y_itr,
            "X_val": X_ival_imp,
            "y_val": y_ival,
        })
    return cached_folds

def create_elastic_net_pipeline(categorical_cols, boolean_cols, numerical_cols,
                                include_imputers=True):
    # include_imputers=False drops the in-pipeline SimpleImputers, used when the
    # input is already complete (imputed upstream by MissForest) so they would
    # only be no-ops. The `simple` strategy keeps them True: it has no upstream
    # imputation, so the SimpleImputers ARE the imputation (fit per Optuna trial).
    numeric_steps = []
    categorical_steps = []
    if include_imputers:
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    numeric_steps.append(("scaler", StandardScaler()))
    # OHE infers categories from the data it is fitted on
    categorical_steps.append(
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    )

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=categorical_steps)
    # The boolean branch only ever held an imputer; with none needed, the
    # integer-coded columns pass straight through.
    boolean_transformer = (
        Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
        if include_imputers else "passthrough"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("bool", boolean_transformer, boolean_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("elasticnet", ElasticNet(random_state=42, max_iter=20000)),
    ])

def run_optuna_search_simple_imputation_elastic_net(X, y, gkf, groups, categorical_cols, all_boolean_cols, numerical_cols, n_trials=20, n_jobs=1, show_progress=False):
    # n_jobs=1 / show_progress=False are the defaults because these searches
    # normally run *inside* a joblib worker that is already one of --n-jobs
    # parallel outer folds: nesting a second parallel layer there oversubscribes
    # the cores, and 1000 concurrent progress bars are unreadable. The full-data
    # final search at the end of main() is the one caller that gets both.
    base_pipeline = create_elastic_net_pipeline(categorical_cols, all_boolean_cols, numerical_cols)
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)

        pipeline = clone(base_pipeline)

        pipeline.set_params(
            elasticnet__alpha=alpha,
            elasticnet__l1_ratio=l1_ratio
        )

        neg_mses = cross_val_score(
            pipeline, X, y,
            cv=gkf,                     # grouped cross validiation
            groups = groups,
            scoring="neg_mean_squared_error", # MSE
            n_jobs=n_jobs,
        )
        return (-1) * neg_mses.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="depression_elastic_net_tuning_simple_imputation",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study

def run_optuna_search_missforest_imputation_elastic_net(cached_folds, categorical_cols,
                      all_boolean_cols, numerical_cols, n_trials=20,
                      show_progress=False):
    base_pipeline = create_elastic_net_pipeline(
        categorical_cols, all_boolean_cols, numerical_cols, include_imputers=False
    )

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)

        fold_mses = []
        for fold in cached_folds:
            pipeline = clone(base_pipeline)
            pipeline.set_params(
                elasticnet__alpha=alpha,
                elasticnet__l1_ratio=l1_ratio,
            )
            pipeline.fit(fold["X_train"], fold["y_train"])
            pred = pipeline.predict(fold["X_val"])
            fold_mses.append(mean_squared_error(fold["y_val"], pred))
        return np.mean(fold_mses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="depression_elastic_net_tuning_missforest_imputation",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study

def run_optuna_search_xgboost(X, y, gkf, groups, n_trials=40, n_jobs=1,
                              show_progress=False):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500, step=100),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 1, 10),
            "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 64),
        }

        model = xgb.XGBRegressor(
            tree_method="hist",  # histogram-based approximate splits; required for enable_categorical
            enable_categorical=True,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
            **params,
        )

        neg_mses = cross_val_score(
            model, X, y,
            cv=gkf,
            groups=groups,
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
        )
        return (-1) * neg_mses.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="depression_xgb_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study

# --- parallel outer CV: identity, checkpointing, and the per-fold worker -----

def run_key_for(args):
    """Timestamp-free identity of a run's *configuration*.

    `run_tag` (the .txt and plot filenames) ends in a per-invocation timestamp so
    successive runs of one config sit side by side and summary_grid.py can take
    the latest. Checkpoints need the opposite: a resume after a crash, or a
    second invocation extending the repetition count, must land in the SAME
    directory or nothing is ever reused. Hence the tag minus the timestamp --
    still carrying model, imputation, feature set, window and mp/c22, so no two
    configurations can ever share a directory.
    """
    imputation_label = "simple_imputation" if args.imputation == "simple" else args.imputation
    return (f"{args.model}_{imputation_label}_{args.feature_set_tag}"
            f"_w{args.window_size}_{args.fraction_tag}")

def code_fingerprint():
    """Hash of the code that produces a fold result, for checkpoint validation.

    Reusing checkpoints written by a different version of the modeling code
    would silently blend incompatible results into one summary. Hashing the two
    files that decide what a fold *is* catches that; a git SHA would not, since
    the tree is usually dirty mid-experiment.
    """
    h = hashlib.sha256()
    for path in (__file__, "classes/repeated_groupkfold.py"):
        try:
            with open(path, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:16]

def run_identity(args):
    """Fields that must match for a checkpoint directory to be reusable.

    `n_repeats` is deliberately absent: RepeatedGroupKFold under an int seed
    makes repetition r the same partition regardless of how many repetitions
    were requested, so a run with more repeats is a strict superset of a
    shorter one and can reuse every fold already on disk. The tuning budget and
    fold counts *do* belong here -- they change what a fold result means.
    """
    return {
        "code": code_fingerprint(),
        "n_trials": args.n_trials,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "seed": SEED,
    }

def prepare_run_dir(args, run_key, X, y, groups, log):
    """Create/validate the checkpoint directory and stage the data for workers.

    Returns (run_dir, data_dir). Workers read X/y/groups from `data_dir` once per
    process rather than receiving them pickled with each of the ~1000 tasks.
    """
    run_dir = f"../results/{args.model}/repeats/{run_key}"
    data_dir = f"{run_dir}/_data"
    manifest_path = f"{run_dir}/manifest.json"
    identity = run_identity(args)

    if args.fresh and os.path.isdir(run_dir):
        log(f"--fresh: discarding existing checkpoints in {run_dir}")
        shutil.rmtree(run_dir)

    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            previous = json.load(fh)
        stale = {k: (previous.get(k), v) for k, v in identity.items()
                 if previous.get(k) != v}
        if stale:
            detail = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in stale.items())
            sys.exit(
                f"Checkpoints in {run_dir} were written under a different "
                f"configuration ({detail}).\nRe-run with --fresh to discard them, "
                f"or restore the previous settings to resume."
            )

    os.makedirs(data_dir, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump(identity, fh, indent=2)

    # Staged to parquet rather than passed through joblib: loky pickles task
    # arguments per task, so shipping the frame 1000 times would cost more than
    # the fitting. Workers cache it per process (see _worker_data).
    X.to_parquet(f"{data_dir}/X.parquet")
    pd.DataFrame({"y": y, "groups": groups}).to_parquet(f"{data_dir}/meta.parquet")
    return run_dir, data_dir

_WORKER_DATA = {}

def _worker_data(data_dir):
    """Load the staged X/y/groups once per worker process, then reuse.

    loky reuses its worker processes across tasks, so this module-level cache is
    populated once per worker (n_jobs loads) instead of once per fold.
    """
    cached = _WORKER_DATA.get(data_dir)
    if cached is None:
        X = pd.read_parquet(f"{data_dir}/X.parquet")
        meta = pd.read_parquet(f"{data_dir}/meta.parquet")
        cached = (X, meta["y"], meta["groups"])
        _WORKER_DATA[data_dir] = cached
    return cached

def fold_paths(run_dir, repeat, fold):
    """(predictions path, record path) for one outer fold.

    One file per fold, never a shared append target: with --n-jobs workers
    finishing concurrently, appending to a single log risks interleaved partial
    writes. The .json is written last and is the completion marker.
    """
    stem = f"{run_dir}/rep{repeat:03d}_fold{fold:02d}"
    return f"{stem}.npz", f"{stem}.json"

def run_outer_fold(repeat, fold, train_idx, test_idx, data_dir, run_dir,
                   categorical_cols, boolean_cols, numerical_cols,
                   model, imputation, n_trials, collect_shap):
    """Tune, fit, and score ONE outer fold, then checkpoint it.

    This is the unit of parallelism: every outer fold of every repetition is
    independent given (X, y, groups) and its own index arrays, so all
    OUTER_SPLITS * n_repeats of them go into a single joblib work queue rather
    than being sharded by repetition (a queue load-balances; fixed shards leave
    workers idle at the end). Hyperparameter search happens here, inside the
    outer training set, so the outer test fold stays untouched by selection.

    Returns the fold's record; predictions go to the .npz so a resumed run can
    reassemble folds it did not refit.
    """
    started = time.time()
    X, y, groups = _worker_data(data_dir)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    inner_cv = GroupKFold(n_splits=INNER_SPLITS)
    categorical_features = categorical_cols + boolean_cols
    shap_values = None

    if model == "elastic_net":
        if imputation == "simple":
            # imputation happens inside the pipeline, per trial
            study = run_optuna_search_simple_imputation_elastic_net(
                X_train, y_train, inner_cv, groups_train,
                categorical_cols, boolean_cols, numerical_cols, n_trials=n_trials,
            )
            best_params = study.best_params

            best_model = create_elastic_net_pipeline(
                categorical_cols, boolean_cols, numerical_cols
            )
            best_model.set_params(
                elasticnet__alpha=best_params["alpha"],
                elasticnet__l1_ratio=best_params["l1_ratio"],
            )
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            X_train_used = X_train

        else:  # missforest
            # impute each inner fold once (leakage-free), search over cache
            cached_folds = impute_inner_folds(
                X_train, y_train, groups_train, inner_cv,
                categorical_features, iterations=3
            )
            study = run_optuna_search_missforest_imputation_elastic_net(
                cached_folds, categorical_cols, boolean_cols, numerical_cols,
                n_trials=n_trials,
            )
            best_params = study.best_params

            # impute outer train/test for the final fold model (fit on
            # outer-train only, applied to outer-test)
            X_train_imp, X_test_imp = missforest_impute(
                X_train, X_test, categorical_features, max_iter=3
            )

            best_model = create_elastic_net_pipeline(
                categorical_cols, boolean_cols, numerical_cols,
                include_imputers=False
            )
            best_model.set_params(
                elasticnet__alpha=best_params["alpha"],
                elasticnet__l1_ratio=best_params["l1_ratio"],
            )
            best_model.fit(X_train_imp, y_train)
            y_pred = best_model.predict(X_test_imp)
            X_train_used = X_train_imp

    elif model == "xgboost":
        study = run_optuna_search_xgboost(
            X_train, y_train, inner_cv, groups_train, n_trials=n_trials
        )
        best_params = study.best_params

        best_model = xgb.XGBRegressor(
            tree_method="hist",  # histogram-based approximate splits; required for enable_categorical
            enable_categorical=True,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,  # the outer joblib pool owns the cores
            **best_params
        )
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        X_train_used = X_train

        if collect_shap:
            # Out-of-sample SHAP on this fold's held-out rows. The default
            # tree_path_dependent perturbation reads the tree structure (via
            # XGBoost's native pred_contribs), so it needs no background data
            # and handles enable_categorical=True.
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

    # Apparent (in-sample) R²: the model re-scored on the data it was fit on.
    # The gap apparent - test R² is the optimism that pmsampsize/shrinkage
    # bounds; a large gap is the direct signature of overfitting.
    record = {
        "repeat": repeat,
        "fold": fold,
        "params": best_params,
        "train_r2": float(r2_score(y_train, best_model.predict(X_train_used))),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "seconds": round(time.time() - started, 1),
    }

    # Write predictions first, the record second: the .json appearing is what
    # marks the fold done, so an interrupted write can never look complete.
    pred_path, record_path = fold_paths(run_dir, repeat, fold)
    arrays = {"test_idx": np.asarray(test_idx), "y_pred": np.asarray(y_pred)}
    if shap_values is not None:
        arrays["shap_values"] = np.asarray(shap_values)
    np.savez(f"{pred_path}.tmp.npz", **arrays)
    os.replace(f"{pred_path}.tmp.npz", pred_path)
    with open(f"{record_path}.tmp", "w") as fh:
        json.dump(record, fh)
    os.replace(f"{record_path}.tmp", record_path)
    return record

def print_model_summary(model_type, model, best_params):
    if model_type == "elastic_net":
        feature_names = model.named_steps["preprocessing"].get_feature_names_out()
        coefs = model.named_steps["elasticnet"].coef_
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

        print(f"\n  alpha:     {best_params['alpha']:.6f}")
        if "l1_ratio" in best_params:
            print(f"  l1_ratio:  {best_params['l1_ratio']:.6f}")
        print(f"\n  {'Feature':<45} Coefficient")
        print("  " + "-" * 60)
        for _, row in coef_df.iterrows():
            print(f"  {row['feature']:<45} {row['coefficient']:>10.6f}")
    elif model_type == "xgboost":
        print("\n  Hyperparameters:")
        for k, v in best_params.items():
            print(f"    {k:<30} {v}")


def summarise_repeats(values):
    """(mean, 2.5th percentile, 97.5th percentile) of one metric over repetitions.

    The interval is the central 95% of the *repetition* estimates: it says how
    far the reported number moves when the participant partition is re-drawn.
    It is a stability interval, not a confidence interval for performance in a
    new sample -- it carries no information about sampling error beyond this
    cohort, and will be narrower than one that did.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)

def print_cv_results(fold_records, repeat_records, n_repeats):
    """Report the outer CV per repetition, then summarise across repetitions.

    Per-fold lines are deliberately not printed: at OUTER_SPLITS x n_repeats
    they run to a thousand rows and say little individually. Each fold's
    parameters and metrics are in its checkpoint .json if needed; what is
    printed here is the per-repetition estimate and its spread.
    """
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    print(f"\nOuter CV:  RepeatedGroupKFold, {OUTER_SPLITS} folds x {n_repeats} "
          f"repetitions ({OUTER_SPLITS * n_repeats} fits)")
    print(f"Inner CV:  GroupKFold, {INNER_SPLITS} folds (tuning, per outer fold)")
    total_hours = sum(r["seconds"] for r in fold_records) / 3600
    print(f"Compute:   {total_hours:.2f} core-hours across all folds")

    # One line per repetition. Every window is held out exactly once within a
    # repetition, so these metrics are scored over the full dataset from a
    # single pooled out-of-fold prediction vector -- directly comparable across
    # repetitions, and free of the unequal-denominator problem that averaging
    # per-fold metrics would carry (RepeatedGroupKFold balances folds by
    # participant count, not by window count).
    print(f"\nOut-of-fold metrics per repetition (pooled over the repetition's "
          f"{OUTER_SPLITS} folds):")
    for rec in repeat_records:
        print(f"  Repeat {rec['repeat']:>3}: RMSE = {rec['rmse']:.4f}, "
              f"MAE = {rec['mae']:.4f}, test R² = {rec['r2']:>7.4f}, "
              f"train R² = {rec['train_r2']:>7.4f}, "
              f"optimism = {rec['train_r2'] - rec['r2']:>7.4f}")

    rows = [
        ("RMSE", [r["rmse"] for r in repeat_records]),
        ("MAE", [r["mae"] for r in repeat_records]),
        ("Test R²", [r["r2"] for r in repeat_records]),
        ("Train R²", [r["train_r2"] for r in repeat_records]),
        # Optimism: apparent (train) R² minus held-out (test) R². A large
        # positive gap means the model is fitting noise. Train R² has no pooled
        # analogue (outer training sets overlap), so it is the mean over the
        # repetition's folds.
        ("Optimism", [r["train_r2"] - r["r2"] for r in repeat_records]),
    ]

    print(f"\nSummary (mean [2.5th, 97.5th percentile] over {n_repeats} repetitions):")
    for label, values in rows:
        mean, lo, hi = summarise_repeats(values)
        note = "  (train - test R²)" if label == "Optimism" else ""
        print(f"  {label + ':':<12} {mean:.4f}  [{lo:.4f}, {hi:.4f}]{note}")
    print("="*50 + "\n")

def print_hyperparameter_stability(fold_records):
    """Distribution of the selected hyperparameters over every outer fold.

    With five folds this was noise; over OUTER_SPLITS x n_repeats independent
    tunings it is a result in its own right -- a tight spread means the data
    identify the setting sharply, a wide one means the inner CV is largely
    choosing at random and the model is insensitive to the choice.
    """
    print("="*50)
    print("HYPERPARAMETER STABILITY (over all outer folds)")
    print("="*50)
    keys = list(dict.fromkeys(k for rec in fold_records for k in rec["params"]))
    print(f"\n  {'Parameter':<22} {'median':>12} {'IQR':>26} {'min':>12} {'max':>12}")
    print("  " + "-" * 86)
    for key in keys:
        values = [rec["params"][key] for rec in fold_records if key in rec["params"]]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        arr = np.asarray(values, dtype=float)
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        print(f"  {key:<22} {q50:>12.5g} {f'[{q25:.5g}, {q75:.5g}]':>26} "
              f"{arr.min():>12.5g} {arr.max():>12.5g}")
    print()

def plot_calibration(y_true, y_pred, run_tag, model, title="Calibration: Actual vs Predicted"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calibration regression: actual ~ alpha + beta * predicted
    beta, alpha = np.polyfit(y_pred, y_true, 1)   # note: order is (x, y) = (pred, true)

    # Calibration-in-the-large: mean(actual) - mean(predicted)
    # (the intercept you'd get if beta were fixed at 1)
    cil = y_true.mean() - y_pred.mean()

    # Shared plot bounds so the diagonal is a true 45 degrees
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    lims = np.array([lo - pad, hi + pad])

    # Sampled from the "viridis" colormap (same scheme as the SHAP beeswarm)
    # so the two plot types read as one visual system.
    cmap = plt.get_cmap("viridis")
    scatter_color = cmap(0.9)
    diagonal_color = "gray"
    fit_color = cmap(0.2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_pred, y_true, alpha=0.5, color=scatter_color, edgecolor="k", linewidth=0.3)

    # Perfect-calibration diagonal (actual = predicted)
    ax.plot(lims, lims, "--", color=diagonal_color, linewidth=3, label="Perfect calibration")

    # Fitted calibration line: actual = alpha + beta * predicted
    ax.plot(lims, beta * lims + alpha, color=fit_color, linewidth=3,
            label=f"Calibration fit (α={alpha:.3f}, β={beta:.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted PHQ-8 Score", fontsize=16)
    ax.set_ylabel("Actual PHQ-8 Score", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(loc="upper left", fontsize=16)

    # Metrics box
    txt = (f"α (intercept) = {alpha:.3f}\n"
           f"β (slope)     = {beta:.3f}\n"
           f"CIL           = {cil:.3f}\n"
           f"R² = {r2:.3f}   RMSE = {rmse:.3f}   MAE = {mae:.3f}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes,
            ha="right", va="bottom", family="monospace", fontsize=16,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_dir = f"../results/{model}/calibration"
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = f"{out_dir}/calibration_{run_tag}.png"
    out_path_svg = f"{out_dir}/calibration_{run_tag}.svg"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plot saved to {out_path_png}")
    print(f"Calibration plot saved to {out_path_svg}")
    return {"alpha": alpha, "beta": beta, "cil": cil, "r2": r2, "rmse": rmse, "mae": mae}

def print_calibration_stability(y_true, oof_pred):
    """Calibration intercept/slope/CIL across every repetition's out-of-fold pass.

    The scatter plot shows one repetition; these are the same three quantities
    refit on each of the n_repeats pooled prediction vectors, so the spread says
    how much the calibration conclusion depends on the partition drawn. Cheap --
    one polyfit per repetition, no refitting of models.
    """
    y_true = np.asarray(y_true, dtype=float)
    stats = {"alpha": [], "beta": [], "cil": []}
    for r in range(oof_pred.shape[1]):
        pred = oof_pred[:, r]
        beta, alpha = np.polyfit(pred, y_true, 1)  # (x, y) = (pred, true)
        stats["alpha"].append(alpha)
        stats["beta"].append(beta)
        stats["cil"].append(y_true.mean() - pred.mean())

    print("="*50)
    print("CALIBRATION STABILITY (over repetitions)")
    print("="*50)
    labels = {"alpha": "α (intercept)", "beta": "β (slope)", "cil": "CIL"}
    print(f"\n  {'Quantity':<16} {'mean':>10}  [2.5th, 97.5th pct]")
    print("  " + "-" * 52)
    for key, values in stats.items():
        mean, lo, hi = summarise_repeats(values)
        print(f"  {labels[key]:<16} {mean:>10.4f}  [{lo:.4f}, {hi:.4f}]")
    print()

def build_feature_description_map():
    """Build a mapping from feature column names to human-readable descriptions.

    Descriptions come from feature_spec.yaml (wearable features) and hardcoded
    demographics. Used to relabel SHAP plot axes for readability."""
    name_to_desc = {
        "age": "age",
        "gender": "gender",
        "audit_1": "Alcohol consumption frequency",
        "mh_smoking": "Smoking status",
        "depression_medication": "Currently taking depression medication",
    }

    with open(PATH_TO_YAML) as fh:
        spec = yaml.safe_load(fh)

    for stream in ["sleep", "heartrate", "activity", "steps"]:
        stream_config = spec.get(stream, {})
        for feature_name, feature_cfg in stream_config.items():
            desc = feature_cfg.get("desc")
            if desc:
                profile = feature_cfg.get("profile")
                ops = spec.get("profiles", {}).get(profile, [])
                for op in ops:
                    col_name = f"{stream}_{feature_name}_{op}"
                    name_to_desc[col_name] = f"{desc} ({op})"

    return name_to_desc

def plot_shap_beeswarm(shap_values_folds, X_test_folds, run_tag, model,
                       title="SHAP summary (out-of-fold test predictions)"):
    """Beeswarm SHAP summary over the pooled outer test folds (XGBoost only).

    Each (shap_values, X_test) pair comes from one outer fold: the SHAP values
    were computed by a TreeExplainer over the fold model on its held-out rows,
    so every sample is explained out-of-sample. The folds passed in are the
    OUTER_SPLITS folds of a single repetition (PLOT_REPEAT), which partition the
    dataset, so stacking them row-wise reconstructs one SHAP value per
    (row, feature) for the full dataset. Restricting to one repetition is what
    keeps that one-value-per-row property -- pooling every repetition would
    stack n_repeats explanations of the same rows. The blocks share the same feature
    columns, so np.vstack / pd.concat glue them vertically and stay row-aligned
    (both preserve fold order), which is what summary_plot needs to color each
    dot by its feature value.
    """
    shap_matrix = np.vstack(shap_values_folds)
    X_matrix = pd.concat(X_test_folds, ignore_index=True)

    name_to_desc = build_feature_description_map()
    X_renamed = X_matrix.rename(columns=name_to_desc)

    plt.figure(figsize=(10, 7.5))
    shap.summary_plot(shap_matrix, X_renamed, plot_type="dot", max_display=10,
                       cmap="viridis", show=False)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title(title)
    # shap.summary_plot draws its "Feature value" colorbar on a separate axis
    for cax in fig.axes:
        if cax is not ax and cax.get_ylabel() == "Feature value":
            cax.set_ylabel("Feature value")
    plt.tight_layout()
    # Keep the beeswarm dots rasterized (a fully-vector scatter is tens of
    # thousands of path nodes, which PowerPoint's SVG engine refuses to render
    # and shows as a red X). Text/axes stay vector in the SVG regardless, so
    # labels print crisp on A0; only the embedded dot bitmap is resolution-bound,
    # so save the SVG at high DPI to keep the dots sharp at poster scale.
    for cax in fig.axes:
        for artist in cax.collections:
            artist.set_rasterized(True)
    out_dir = f"../results/{model}/shap"
    os.makedirs(out_dir, exist_ok=True)
    out_path_png = f"{out_dir}/shap_beeswarm_{run_tag}.png"
    out_path_svg = f"{out_dir}/shap_beeswarm_{run_tag}.svg"
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_svg, format="svg", bbox_inches="tight", dpi=600)
    plt.close()
    print(f"SHAP beeswarm plot saved to {out_path_png}")
    print(f"SHAP beeswarm plot saved to {out_path_svg}")

    # Global importance ranking = mean |SHAP| per feature, matching how
    # plot_calibration returns its summary numbers.
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    ranking = (pd.Series(mean_abs, index=X_matrix.columns)
               .sort_values(ascending=False))
    return ranking

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"PHQ-8 regression with repeated nested grouped CV "
            f"({OUTER_SPLITS}-fold outer x --n-repeats repetitions, "
            f"{INNER_SPLITS}-fold inner tuning)."
        )
    )
    parser.add_argument(
        "--feature-set",
        nargs="+",
        choices=["demographics", "simple", "slope", "catch22", "adherence"],
        default=["simple"],
        help=(
            "One or more feature families to pool (any non-empty subset of "
            "demographics, simple, slope, catch22, adherence). Default: simple."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        choices=[3, 5, 7, 14, 21, 28],
        default=14,
        help="Window size for the aggregated features. Default: 14.",
    )
    parser.add_argument(
        "--min-present-fraction",
        type=float,
        default=0.75,
        help=(
            "Present-fraction floor of the merged dataset to load (the `mp` in the "
            "filename), shared by the simple aggregations and the catch22 panel. "
            "Selects which merge_datasets.py output to model on; must match an "
            "existing dataset. Default: 0.75."
        ),
    )
    parser.add_argument(
        "--imputation",
        choices=["simple", "missforest"],
        default="simple",
        help="Imputation strategy. Ignored when --model xgboost. Default: simple.",
    )
    parser.add_argument(
        "--model",
        choices=["elastic_net", "xgboost"],
        default="elastic_net",
        help="Predictive model. Default: elastic_net.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=100,
        help=(
            "Number of outer-CV repetitions (independent participant partitions). "
            "Each contributes one pooled out-of-fold performance estimate; the "
            "summary reports their mean and 2.5th/97.5th percentiles. Total outer "
            f"fits = {OUTER_SPLITS} x this. Default: 100."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help=(
            "Worker processes for the outer folds. Folds are queued individually "
            "so the pool self-balances. Inner parallelism is forced to 1 in the "
            "workers, so this is the only parallel layer. Default: 8."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Optuna trials per outer fold. Default: "
            + ", ".join(f"{m} {n}" for m, n in DEFAULT_TRIALS.items()) + "."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Discard existing fold checkpoints for this configuration before "
            "running. Without it, completed folds are reused and only the "
            "missing ones are fitted."
        ),
    )
    args = parser.parse_args()

    if args.n_repeats < 1:
        parser.error("--n-repeats must be at least 1.")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1.")

    # Normalize the selected families to a fixed order and drop duplicates, so
    # the run tag / output filename are deterministic regardless of CLI order.
    family_order = ["demographics", "simple", "slope", "catch22", "adherence"]
    args.feature_set = [f for f in family_order if f in args.feature_set]
    args.feature_set_tag = "+".join(args.feature_set)

    # No numeric range check on the fraction: it only picks which generated
    # dataset to load, so the file-existence check in main() is the real guard
    # (it also handles valid-but-not-yet-built fractions, and lists what exists).
    # The c22/noc22 tag records whether catch22 is produced for this window,
    # matching merge_datasets.dataset_stem so the loaded filename lines up.
    args.catch22_enabled = args.window_size in catch22_enabled_windows()
    _c22 = "c22" if args.catch22_enabled else "noc22"
    args.fraction_tag = f"mp{args.min_present_fraction:g}_{_c22}"

    # Resolved after the model is known so the help text can advertise both
    # defaults while the run records the single number actually used (it is part
    # of the checkpoint identity -- changing it invalidates cached folds).
    if args.n_trials is None:
        args.n_trials = DEFAULT_TRIALS[args.model]

    # XGBoost handles missing values natively; imputation must not be applied.
    if args.model == "xgboost":
        # only complain if the user *explicitly* asked for imputation
        if "--imputation" in sys.argv:
            parser.error(
                "--imputation cannot be used with --model xgboost "
                "(XGBoost handles missing values natively)."
            )
        args.imputation = "none"

    return args


def main(args):

    # Progress to stderr so it shows live in the terminal even though stdout is
    # redirected to the results file. Step names + shapes only, no participant rows.
    _run_start = time.time()
    def log(msg):
        print(f"[{time.time() - _run_start:7.1f}s] {msg}", file=sys.stderr, flush=True)

    print("="*50)
    print("RUN CONFIGURATION")
    print("="*50)
    print(f"  Feature set:  {args.feature_set_tag}  ({', '.join(args.feature_set)})")
    print(f"  Window size:  {args.window_size}")
    print(f"  Present frac: {args.min_present_fraction:g} "
          f"(catch22 {'present' if args.catch22_enabled else 'absent'})")
    if args.model == "xgboost":
        print(f"  Imputation:   none (native NaN handling)")
    else:
        print(f"  Imputation:   {args.imputation}")
    print(f"  Model:        {args.model}")
    print(f"  Outer CV:     RepeatedGroupKFold, {OUTER_SPLITS} folds x "
          f"{args.n_repeats} repeats ({OUTER_SPLITS * args.n_repeats} fits)")
    print(f"  Inner CV:     GroupKFold, {INNER_SPLITS} folds")
    print(f"  Optuna:       {args.n_trials} trials per outer fold")
    print(f"  Workers:      {args.n_jobs}")
    print("="*50)

    log(f"Run: {args.model} | {args.imputation} | {args.feature_set_tag} | "
        f"w{args.window_size} | {args.fraction_tag}")

    # The dataset to load is identified by window + the single present-fraction
    # floor (mp) + the catch22 regime. The fraction only selects a file, so file
    # existence is the guard: a missing dataset (out-of-range OR valid-but-not-yet
    # -built) errors with the list of what exists and the command to generate it.
    data_dir = "../data_merged"
    stem = dataset_stem(args.window_size, args.min_present_fraction,
                        args.catch22_enabled)
    data_path = f"{data_dir}/{stem}.parquet"
    manifest_path = f"{data_dir}/{stem}_columns.json"

    if not os.path.exists(data_path):
        available = sorted(
            os.path.basename(p)
            for p in glob.glob(f"{data_dir}/final_dataset_w*.parquet")
        )
        listing = "\n  ".join(available) if available else "(none found)"
        sys.exit(
            f"Merged dataset not found: {data_path}\n"
            f"Available:\n  {listing}\n"
            f"Generate it with: python merge_datasets.py "
            f"--window-size {args.window_size} "
            f"--min-present-fraction {args.min_present_fraction:g}"
        )

    df = pd.read_parquet(data_path)
    log(f"Loaded {data_path} -> {df.shape}")

    # The column manifest (emitted alongside the parquet) maps each feature
    # family to the columns it produced, post all-NaN-drop and catch22 gating.
    # It is the single source of truth for feature-set selection.
    if not os.path.exists(manifest_path):
        sys.exit(
            f"Column manifest not found: {manifest_path}. "
            f"Re-run merge_datasets.py for this dataset to emit it."
        )
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    # Pool the selected families. catch22 is absent (empty list) at window sizes
    # where the panel is not produced; selecting it there is a hard error.
    feature_cols = []
    for family in args.feature_set:
        fam_cols = manifest.get(family, [])
        if not fam_cols:
            sys.exit(
                f"Feature family '{family}' has no columns for window size "
                f"{args.window_size} (catch22 is only produced for longer "
                f"windows). Drop it from --feature-set or pick a larger window."
            )
        feature_cols += [c for c in fam_cols if c in df.columns]

    print(f"\nSelected {len(feature_cols)} feature columns "
          f"from families [{args.feature_set_tag}]:")
    for col in feature_cols:
        print(f"  {col}")

    # Keep only the selected features plus the keys prepare_data needs.
    df = df[["participant_name", "phq8_score"] + feature_cols]

    # Demographic typing, intersected with the selected columns so the
    # categorical/boolean handling activates only when demographics is included.
    categorical_cols = [c for c in ['mh_smoking', 'audit_1'] if c in feature_cols]
    all_boolean_cols = [c for c in ['depression_medication', 'gender'] if c in feature_cols]

    if args.model == "xgboost":
        # native categoricals: keep original labels, no cat.codes
        X, y, groups = prepare_data(df, categorical_cols, all_boolean_cols,
                                    encode_as_codes=False)
    else:
        X, y, groups = prepare_data(df, categorical_cols, all_boolean_cols)

    # Column families come straight from the explicit demographic lists above:
    # the codes path leaves categoricals as float64 (no 'category' dtype to
    # detect), and these lists are already exactly the categorical/boolean set.
    categorical_features = categorical_cols + all_boolean_cols
    numerical_cols = [c for c in X.columns if c not in categorical_features]

    # Positional indices are the contract between the splitter, the checkpoint
    # files and the OOF matrix, so pin the frames to a clean RangeIndex once.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    outer_cv = RepeatedGroupKFold(
        n_splits=OUTER_SPLITS, n_repeats=args.n_repeats, random_state=SEED
    )

    # Enumerate every (repetition, fold) up front. Index generation is free --
    # only the fitting is expensive -- and a flat task list is what lets one
    # joblib queue load-balance all of them. get_n_splits() reports folds x
    # repeats, so the repetition a fold belongs to comes from outer_cv.n_splits.
    tasks = []
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        repeat, fold = divmod(i, outer_cv.n_splits)
        tasks.append((repeat, fold, train_idx, test_idx))

    n_outer = len(tasks)
    log(f"X={X.shape}, y={len(y)} | {n_outer} outer fits "
        f"({OUTER_SPLITS} folds x {args.n_repeats} repeats)")

    run_key = run_key_for(args)
    run_dir, data_dir = prepare_run_dir(args, run_key, X, y, groups, log)

    # Resume: a fold whose record exists is complete and is not refitted. Its
    # predictions are read back from the .npz at assembly time, so reused and
    # freshly fitted folds are handled identically.
    pending = [t for t in tasks if not os.path.exists(fold_paths(run_dir, t[0], t[1])[1])]
    done = n_outer - len(pending)
    if done:
        log(f"Resuming from {run_dir}: {done}/{n_outer} folds already complete")

    if pending:
        log(f"Fitting {len(pending)} outer folds on {args.n_jobs} workers "
            f"(inner parallelism forced to 1)")
        # inner_max_num_threads=1 stops BLAS/OpenMP from spawning a full thread
        # pool inside every worker; with LightGBM/XGBoost n_jobs=1 and the
        # Optuna searches at n_jobs=1, the outer pool is the only parallel layer.
        # batch_size=1 keeps the queue fine-grained, so a worker that draws a
        # slow fold does not also hold a batch of unstarted ones.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            Parallel(n_jobs=args.n_jobs, batch_size=1, verbose=10)(
                delayed(run_outer_fold)(
                    repeat, fold, train_idx, test_idx, data_dir, run_dir,
                    categorical_cols, all_boolean_cols, numerical_cols,
                    args.model, args.imputation, args.n_trials,
                    collect_shap=(args.model == "xgboost" and repeat == PLOT_REPEAT),
                )
                for repeat, fold, train_idx, test_idx in pending
            )

    # --- assemble the out-of-fold predictions ------------------------------
    log("All folds complete; assembling out-of-fold predictions...")
    # Column r holds repetition r's predictions. Within a repetition the folds
    # partition the dataset, so every cell is written exactly once.
    oof_pred = np.full((len(y), args.n_repeats), np.nan)
    fold_records = []
    # Out-of-fold SHAP: one (shap_values, X_test) block per fold of PLOT_REPEAT,
    # pooled into a single beeswarm (XGBoost only).
    shap_values_all = []
    X_test_all = []

    for repeat, fold, _, _ in tasks:
        pred_path, record_path = fold_paths(run_dir, repeat, fold)
        with open(record_path) as fh:
            fold_records.append(json.load(fh))
        with np.load(pred_path) as arrays:
            oof_pred[arrays["test_idx"], repeat] = arrays["y_pred"]
            if "shap_values" in arrays and repeat == PLOT_REPEAT:
                shap_values_all.append(arrays["shap_values"])
                X_test_all.append(X.iloc[arrays["test_idx"]].reset_index(drop=True))

    missing = np.isnan(oof_pred).sum(axis=0)
    if missing.any():
        sys.exit(
            f"Out-of-fold matrix is incomplete: repetitions "
            f"{np.flatnonzero(missing).tolist()} have unpredicted rows. "
            f"Delete {run_dir} or re-run with --fresh."
        )

    # One performance estimate per repetition, scored once over that
    # repetition's pooled predictions (i.e. over the full dataset).
    repeat_records = []
    for r in range(args.n_repeats):
        pred = oof_pred[:, r]
        fold_train_r2 = [rec["train_r2"] for rec in fold_records if rec["repeat"] == r]
        repeat_records.append({
            "repeat": r,
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "train_r2": float(np.mean(fold_train_r2)),
        })

    print_cv_results(fold_records, repeat_records, args.n_repeats)
    print_hyperparameter_stability(fold_records)

    imputation_label = "simple_imputation" if args.imputation == "simple" else args.imputation
    run_tag = (f"{args.model}_{imputation_label}_{args.feature_set_tag}"
               f"_w{args.window_size}_{args.fraction_tag}_{args.timestamp}")
    # One repetition's complete out-of-fold pass. Averaging predictions across
    # repetitions instead would be an ensemble, and would flatter the slope.
    plot_calibration(
        y, oof_pred[:, PLOT_REPEAT], run_tag, args.model,
        title=f"Calibration: Actual vs Predicted (repetition {PLOT_REPEAT})",
    )
    print_calibration_stability(y, oof_pred)

    if args.model == "xgboost" and shap_values_all:
        shap_ranking = plot_shap_beeswarm(shap_values_all, X_test_all, run_tag, args.model)
        print("\n" + "="*50)
        print("SHAP FEATURE IMPORTANCE (mean |SHAP| over out-of-fold test rows)")
        print("="*50)
        print(f"\n  {'Feature':<45} Mean |SHAP|")
        print("  " + "-" * 60)
        for feature, value in shap_ranking.items():
            print(f"  {feature:<45} {value:>10.6f}")

    print("\n" + "="*50)
    print("FINAL MODEL (tuned and fit on full dataset)")
    print("="*50)

    log("Fitting final model on full dataset...")
    # Same grouped tuning CV as the outer folds use internally. This search runs
    # in the driver with no worker pool competing for cores, so it takes the
    # cores (n_jobs=-1) and shows its progress bar.
    final_cv = GroupKFold(n_splits=INNER_SPLITS)

    if args.model == "elastic_net":
        if args.imputation == "simple":
            final_study = run_optuna_search_simple_imputation_elastic_net(
                X, y, final_cv, groups,
                categorical_cols, all_boolean_cols, numerical_cols,
                n_trials=args.n_trials, n_jobs=-1, show_progress=True,
            )
            final_best_params = final_study.best_params
            print(f"  Best params: {final_best_params}")

            final_model = create_elastic_net_pipeline(
                categorical_cols, all_boolean_cols, numerical_cols
            )
            final_model.set_params(
                elasticnet__alpha=final_best_params["alpha"],
                elasticnet__l1_ratio=final_best_params["l1_ratio"],
            )
            final_model.fit(X, y)

        elif args.imputation == "missforest":
            final_cached_folds = impute_inner_folds(
                X, y, groups, final_cv, categorical_features, iterations=3
            )
            final_study = run_optuna_search_missforest_imputation_elastic_net(
                final_cached_folds, categorical_cols,
                all_boolean_cols, numerical_cols,
                n_trials=args.n_trials, show_progress=True,
            )
            final_best_params = final_study.best_params
            print(f"  Best params: {final_best_params}")

            X_imp = missforest_impute(X, None, categorical_features, max_iter=3)

            final_model = create_elastic_net_pipeline(
                categorical_cols, all_boolean_cols, numerical_cols,
                include_imputers=False
            )
            final_model.set_params(
                elasticnet__alpha=final_best_params["alpha"],
                elasticnet__l1_ratio=final_best_params["l1_ratio"],
            )
            final_model.fit(X_imp, y)

    elif args.model == "xgboost":
        final_study = run_optuna_search_xgboost(
            X, y, final_cv, groups,
            n_trials=args.n_trials, n_jobs=-1, show_progress=True,
        )
        final_best_params = final_study.best_params
        print(f"  Best params: {final_best_params}")
        final_model = xgb.XGBRegressor(
            tree_method="hist",  # histogram-based approximate splits; required for enable_categorical
            enable_categorical=True,
            objective="reg:squarederror",
            random_state=SEED,
            **final_best_params
        )
        final_model.fit(X, y)

    print_model_summary(args.model, final_model, final_best_params)
    print("\n" + "="*50)

    log("Done.")

if __name__ == "__main__":
    args = parse_args()
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    imputation_label = "simple_imputation" if args.imputation == "simple" else args.imputation
    text_dir = f"../results/{args.model}/text"
    os.makedirs(text_dir, exist_ok=True)
    out_path = (f"{text_dir}/{args.model}_{imputation_label}_{args.feature_set_tag}"
                f"_w{args.window_size}_{args.fraction_tag}_{args.timestamp}.txt")
    sys.stdout = open(out_path, "w", buffering=1)
    main(args)
