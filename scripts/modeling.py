import argparse
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
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

# os.environ["TQDM_DISABLE"] = "True"

SEED = 42

def prepare_data(df, categorical_cols, boolean_cols, encode_as_codes=True):
    """Split df into (X, y, groups) and encode the categorical/boolean columns.

    encode_as_codes=True (elastic net + MissForest): cat.codes -> integer codes
    carried as 'category' dtype. MissForest requires int/float/bool input
    (`missforest_impute` casts these codes to Int64), and the OHE consumes them
    fine. encode_as_codes=False (XGBoost, enable_categorical=True): keep the
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
            X[col] = X[col].astype('category')
        elif X[col].dtype.name != 'category':
            X[col] = X[col].astype('category')
    return X, y, groups

def missforest_impute(X_train, X_test, categorical_features, max_iter=3):
    """Leakage-free MissForest imputation: fit on train only, apply to test.

    missforest 4.x accepts only int/float/bool columns, so the integer-coded
    categorical columns (cat.codes / 0-1 flags, carried as 'category' dtype)
    are cast to nullable Int64 on the way in. The imputer's output is returned
    as-is (categoricals come back as Int64): the elastic-net pipeline's
    OneHotEncoder and boolean passthrough consume those integer codes directly,
    so no dtype restore is needed (and casting back to 'category' here would
    actually break the OneHotEncoder). A LGBMClassifier is used for the
    categorical columns and a LGBMRegressor for the numeric ones.
    Returns X_train_imp, or (X_train_imp, X_test_imp) when X_test is given."""
    imputer = MissForest(
        clf=LGBMClassifier(linear_tree=True, verbosity=-1, random_state=SEED),
        rgr=LGBMRegressor(linear_tree=True, verbosity=-1, random_state=SEED),
        categorical=categorical_features,
        max_iter=max_iter,
        verbose=0,
    )

    def _to_input(X):
        X = X.copy()
        for col in categorical_features:
            X[col] = X[col].astype("Int64")
        return X

    X_train_imp = imputer.fit_transform(_to_input(X_train))
    if X_test is None:
        return X_train_imp
    X_test_imp = imputer.transform(_to_input(X_test))
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

def run_optuna_search_simple_imputation_elastic_net(X, y, gkf, groups, categorical_cols, all_boolean_cols, numerical_cols, n_trials=40):
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
            n_jobs=-1,                  # parallelize
        )
        return (-1) * neg_mses.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="depression_elastic_net_tuning_simple_imputation",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def run_optuna_search_missforest_imputation_elastic_net(cached_folds, categorical_cols,
                      all_boolean_cols, numerical_cols, n_trials=40):
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def run_optuna_search_xgboost(X, y, gkf, groups, n_trials=80):
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
            n_jobs=-1,
        )
        return (-1) * neg_mses.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        study_name="depression_xgb_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

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


def print_cv_results(fold_params, test_rmse_scores, test_r2_scores, test_mae_scores,
                     train_r2_scores):
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)

    print("\nBest parameters in each fold:")
    for i, params in enumerate(fold_params, 1):
        print(f"  Fold {i}: {params}")

    print("\nTest metrics per fold:")
    for i, (rmse, mae, r2) in enumerate(zip(test_rmse_scores, test_mae_scores, test_r2_scores), 1):
        print(f"  Fold {i}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

    # Overfitting check: apparent (train) R² vs held-out (test) R². The optimism
    # is their difference; a large positive gap means the model is fitting noise.
    print("\nOverfitting check (apparent vs held-out R² per fold):")
    optimism = []
    for i, (tr, te) in enumerate(zip(train_r2_scores, test_r2_scores), 1):
        gap = tr - te
        optimism.append(gap)
        print(f"  Fold {i}: train R² = {tr:.4f}, test R² = {te:.4f}, optimism = {gap:.4f}")

    print("\nSummary:")
    print(f"  RMSE:        {np.mean(test_rmse_scores):.4f} ± {np.std(test_rmse_scores):.4f}")
    print(f"  MAE:         {np.mean(test_mae_scores):.4f} ± {np.std(test_mae_scores):.4f}")
    print(f"  Test R²:     {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
    print(f"  Train R²:    {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
    print(f"  Optimism:    {np.mean(optimism):.4f} ± {np.std(optimism):.4f}  (train - test R²)")
    print("="*50 + "\n")

def plot_calibration(y_true, y_pred, run_tag, title="Calibration: Actual vs Predicted"):
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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, y_true, alpha=0.5, edgecolor="k", linewidth=0.3)

    # Perfect-calibration diagonal (actual = predicted)
    ax.plot(lims, lims, "--", color="gray", label="Perfect calibration")

    # Fitted calibration line: actual = alpha + beta * predicted
    ax.plot(lims, beta * lims + alpha, color="red",
            label=f"Calibration fit (α={alpha:.3f}, β={beta:.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted PHQ-8 Score")
    ax.set_ylabel("Actual PHQ-8 Score")
    ax.set_title(title)
    ax.legend(loc="upper left")

    # Metrics box
    txt = (f"α (intercept) = {alpha:.3f}\n"
           f"β (slope)     = {beta:.3f}\n"
           f"CIL           = {cil:.3f}\n"
           f"R² = {r2:.3f}   RMSE = {rmse:.3f}   MAE = {mae:.3f}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes,
            ha="right", va="bottom", family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = f"../results/calibration_{run_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plot saved to {out_path}")
    return {"alpha": alpha, "beta": beta, "cil": cil, "r2": r2, "rmse": rmse, "mae": mae}

def parse_args():
    parser = argparse.ArgumentParser(
        description="PHQ-8 regression with nested grouped CV."
    )
    parser.add_argument(
        "--feature-set",
        choices=["demographics", "simple_aggregations", "catch22"],
        default="simple_aggregations",
        help="Which feature set to use. Default: simple_aggregations.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        choices=[3, 5, 7, 14],
        default=14,
        help="Window size for the aggregated features. Default: 14.",
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
    args = parser.parse_args()

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

    print("="*50)
    print("RUN CONFIGURATION")
    print("="*50)
    print(f"  Feature set:  {args.feature_set}")
    print(f"  Window size:  {args.window_size}")
    if args.model == "xgboost":
        print(f"  Imputation:   none (native NaN handling)")
    else:
        print(f"  Imputation:   {args.imputation}")
    print(f"  Model:        {args.model}")
    print("="*50)

    data_path = f"../data_merged/final_dataset_w{args.window_size}.parquet"
    df = pd.read_parquet(data_path)

    print(f"\nColumns in final_dataset_w{args.window_size}.parquet:")
    for col in df.columns:
        print(f"  {col}")

    df["any_disorder"] = df["any_disorder"].astype('category')

    demographic_cols = [
        'age',
        'gender',
        'mh_smoking',
        'audit_1',
        'depression_medication'
    ]

    categorical_cols = [
        'mh_smoking',
        'audit_1',
    ]

    boolean_cols_demographics = [
        'depression_medication',
        'gender'
    ]

    boolean_cols_sleep = [
        'any_disorder'
    ]

    all_boolean_cols = boolean_cols_demographics + boolean_cols_sleep

    if args.model == "xgboost":
        # native categoricals: keep original labels, no cat.codes
        X, y, groups = prepare_data(df, categorical_cols, all_boolean_cols,
                                    encode_as_codes=False)
    else:
        X, y, groups = prepare_data(df, categorical_cols, boolean_cols_demographics)

    if args.feature_set == "demographics":
        X = X[demographic_cols]
        all_boolean_cols = boolean_cols_demographics
    elif args.feature_set == "simple_aggregations":
        pass  # all aggregated sleep/activity features + demographics
    elif args.feature_set == "catch22":
        raise NotImplementedError("catch22 features not yet implemented.")

    # prepare_data drops phq8_score/participant_name and encodes every
    # categorical/boolean column to 'category' dtype, so X's columns partition
    # cleanly by dtype: the category columns are the categorical/boolean inputs,
    # everything else is numeric. Deriving both lists from dtype (rather than a
    # hand-maintained denylist) keeps them in sync with the column set and with
    # each other, and adapts automatically to feature-set subsetting.
    categorical_features = X.select_dtypes(include="category").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="category").columns.tolist()

    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=3)

    test_rmse_scores = []
    test_r2_scores = []
    test_mae_scores = []
    train_r2_scores = []
    fold_params = []
    y_true_all = []
    y_pred_all = []

    for _, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), 1
    ):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]

        # Reset indices once so X/y remain positionally aligned with the
        # imputer output and any downstream positional indexing.
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        groups_train = groups_train.reset_index(drop=True)

        # --- model selection -----------------------------------------------
        if args.model == "elastic_net":

            if args.imputation == "simple":
                # imputation happens inside the pipeline, per trial
                study = run_optuna_search_simple_imputation_elastic_net(
                    X_train, y_train, inner_cv, groups_train,
                    categorical_cols, all_boolean_cols, numerical_cols
                )
                best_params = study.best_params

                best_model = create_elastic_net_pipeline(
                    categorical_cols, all_boolean_cols, numerical_cols
                )
                best_model.set_params(
                    elasticnet__alpha=best_params["alpha"],
                    elasticnet__l1_ratio=best_params["l1_ratio"]
                )
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)
                X_train_used = X_train

            elif args.imputation == "missforest":
                # impute each inner fold once (leakage-free), search over cache
                cached_folds = impute_inner_folds(
                    X_train, y_train, groups_train, inner_cv,
                    categorical_features, iterations=3
                )
                study = run_optuna_search_missforest_imputation_elastic_net(
                    cached_folds, categorical_cols,
                    all_boolean_cols, numerical_cols
                )
                best_params = study.best_params

                # impute outer train/test for the final fold model (fit on
                # outer-train only, applied to outer-test)
                X_train_imp, X_test_imp = missforest_impute(
                    X_train, X_test, categorical_features, max_iter=3
                )

                best_model = create_elastic_net_pipeline(
                    categorical_cols, all_boolean_cols, numerical_cols,
                    include_imputers=False
                )
                best_model.set_params(
                    elasticnet__alpha=best_params["alpha"],
                    elasticnet__l1_ratio=best_params["l1_ratio"]
                )
                best_model.fit(X_train_imp, y_train)
                y_pred = best_model.predict(X_test_imp)
                X_train_used = X_train_imp

        elif args.model == "xgboost":
            study = run_optuna_search_xgboost(X_train, y_train, inner_cv, groups_train)
            best_params = study.best_params

            best_model = xgb.XGBRegressor(
                tree_method="hist",  # histogram-based approximate splits; required for enable_categorical
                enable_categorical = True,
                objective="reg:squarederror",
                random_state=SEED,
                **best_params
            )

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            X_train_used = X_train


        fold_params.append(best_params)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Apparent (in-sample) R²: the model re-scored on the data it was fit on.
        # The gap apparent - test R² is the optimism that pmsampsize/shrinkage
        # bounds; a large gap is the direct signature of overfitting.
        train_r2 = r2_score(y_train, best_model.predict(X_train_used))

        test_rmse_scores.append(rmse)
        test_r2_scores.append(r2)
        test_mae_scores.append(mae)
        train_r2_scores.append(train_r2)

    print_cv_results(
        fold_params,
        test_rmse_scores,
        test_r2_scores,
        test_mae_scores,
        train_r2_scores
    )

    imputation_label = "simple_imputation" if args.imputation == "simple" else args.imputation
    run_tag = f"{args.model}_{imputation_label}_{args.feature_set}_w{args.window_size}"
    plot_calibration(y_true_all, y_pred_all, run_tag)

    print("\n" + "="*50)
    print("FINAL MODEL (tuned and fit on full dataset)")
    print("="*50)

    final_cv = GroupKFold(n_splits=5)

    if args.model == "elastic_net":
        if args.imputation == "simple":
            final_study = run_optuna_search_simple_imputation_elastic_net(
                X, y, final_cv, groups,
                categorical_cols, all_boolean_cols, numerical_cols,
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
        final_study = run_optuna_search_xgboost(X, y, final_cv, groups)
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

if __name__ == "__main__":
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    imputation_label = "simple_imputation" if args.imputation == "simple" else args.imputation
    out_path = f"../results/{args.model}_{imputation_label}_{args.feature_set}_w{args.window_size}_{timestamp}.txt"
    sys.stdout = open(out_path, "w", buffering=1)
    main(args)
