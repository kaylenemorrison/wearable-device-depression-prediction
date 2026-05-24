import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    cross_val_score,
    GroupKFold
)
from sklearn.base import clone
import optuna
from optuna.samplers import TPESampler
import numpy as np
import os
#os.environ["TQDM_DISABLE"] = "True"
import warnings

SEED = 42
DATA_PATH = "../data_cleaned/final_dataset.parquet"

def prepare_data(df, categorical_cols, boolean_cols):
    df_new = df.drop(columns = ["timeInterval_ID"])
    X = df_new.drop(columns=['phq8_score', 'participant_name'])
    y = df_new['phq8_score']
    groups = df_new['participant_name']
    cols_to_encode = categorical_cols + boolean_cols
    for col in cols_to_encode:
        X[col] = X[col].cat.codes
        X[col] = X[col].replace(-1, np.nan)
        X[col] = X[col].astype('category')
    return X, y, groups

def get_oh_encoder(all_cat_cols, X_global):
    categorical_cols_dict = {
        col: sorted(X_global[col].dropna().unique().tolist())
        for col in all_cat_cols
    }
    categories_list = list(categorical_cols_dict.values())
    oh_encoder = OneHotEncoder(
        categories=categories_list,
        sparse_output=False,
    )
    return oh_encoder

def create_pipeline(X_global, categorical_cols, boolean_cols, numerical_cols):

    oh_encoder = get_oh_encoder(categorical_cols, X_global)

    # numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", oh_encoder)
    ])

    # boolean pipeline
    boolean_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
            ("bool", boolean_pipeline, boolean_cols),
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("elasticnet", ElasticNet(random_state=42, max_iter=20000))
    ])

    return pipeline

def run_optuna_search(X_global, X, y, gkf, groups, categorical_cols, all_boolean_cols, numerical_cols, n_trials=40):
    base_pipeline = create_pipeline(X_global, categorical_cols, all_boolean_cols, numerical_cols)
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
        study_name="depression_elastic_net_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

def print_cv_results(fold_params, test_rmse_scores, test_r2_scores, test_mae_scores):
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)

    print("\nBest parameters in each fold:")
    for i, params in enumerate(fold_params, 1):
        print(f"  Fold {i}: {params}")

    print("\nTest metrics per fold:")
    for i, (rmse, mae, r2) in enumerate(zip(test_rmse_scores, test_mae_scores, test_r2_scores), 1):
        print(f"  Fold {i}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

    print("\nSummary:")
    print(f"  RMSE: {np.mean(test_rmse_scores):.4f} ± {np.std(test_rmse_scores):.4f}")
    print(f"  MAE:  {np.mean(test_mae_scores):.4f} ± {np.std(test_mae_scores):.4f}")
    print(f"  R²:   {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")
    

    print("="*50 + "\n")

def plot_calibration(y_true, y_pred, title="Calibration: Actual vs Predicted"):
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
    ax.set_xlabel("Predicted")          # x-axis predicted
    ax.set_ylabel("Actual")             # y-axis actual
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
    plt.show()
    return {"alpha": alpha, "beta": beta, "cil": cil, "r2": r2, "rmse": rmse, "mae": mae}

def main():
    df = pd.read_parquet(DATA_PATH)

    df["any_disorder"] = df["any_disorder"].astype('category')

    demographic_cols = [
        'age',
        'gender',
        'marital_status',
        'children',
        'ethnicity_1',
        'drop_school_1',
        'accommodation',
        'mh_smoking',
        'audit_1',
        'depression_medication'
    ]

    categorical_cols = [
        'marital_status',
        'ethnicity_1',
        'accommodation',
        'mh_smoking',
        'audit_1',
    ]

    boolean_cols_demographics = [
        'children',
        'drop_school_1',
        'depression_medication',
        'gender'
    ]

    boolean_cols_sleep = [
        'any_disorder'
    ]

    all_boolean_cols = boolean_cols_demographics + boolean_cols_sleep

    numerical_cols = ['age']

    group_cols = ['participant_name', 'timeInterval_ID']

    outcome_col = ['phq8_score']
        
    cols_to_exclude_numerical = demographic_cols + boolean_cols_sleep + group_cols + outcome_col

    numerical_cols += [
        col for col in df.columns
        if col not in cols_to_exclude_numerical
    ]

    X, y, groups = prepare_data(df, categorical_cols, boolean_cols_demographics)

    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=3)

    test_rmse_scores = []
    test_r2_scores = []
    test_mae_scores = []
    fold_params = []
    y_true_all = []
    y_pred_all = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), 1
    ):

        print(f"\nStarting outer fold {fold_idx}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]

        study = run_optuna_search(X, X_train, y_train, inner_cv, groups_train, categorical_cols, all_boolean_cols, numerical_cols)

        best_params = study.best_params

        print(f"Best params: {best_params}")

        fold_params.append(best_params)

        best_model = create_pipeline(X, categorical_cols, all_boolean_cols, numerical_cols)

        best_model.set_params(
            elasticnet__alpha=best_params["alpha"],
            elasticnet__l1_ratio=best_params["l1_ratio"]
        )

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        test_rmse_scores.append(rmse)
        test_r2_scores.append(r2)
        test_mae_scores.append(mae)

    plot_calibration(y_true_all, y_pred_all)

    print_cv_results(
        fold_params,
        test_rmse_scores,
        test_r2_scores,
        test_mae_scores
    )

if __name__ == "__main__":
    main()
