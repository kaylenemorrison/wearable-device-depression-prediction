import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV

from classes.ImputerMerger import ImputerMerger

from sklearn.metrics import mean_squared_error, r2_score

from joblib import Memory

import numpy as np

DATA_PATH = "../data_cleaned/final_dataset.parquet"

def prepare_data(df, categorical_cols, ordinal_cols, boolean_cols):
    phq8_demographics = df.drop(columns=[col for col in df.columns if col.startswith('sleep_')])
    phq8_demographics = phq8_demographics.drop(columns = ["timeInterval_ID"])

    X = phq8_demographics.drop(columns=['phq8_score'])
    y = phq8_demographics['phq8_score']
    groups = phq8_demographics['participant_name']

    cols_to_encode = categorical_cols + ordinal_cols + boolean_cols

    for col in cols_to_encode:
        X[col] = X[col].cat.codes
        X[col] = X[col].replace(-1, np.nan)

    return X, y, groups

def print_cv_results(fold_params, test_rmse_scores, test_r2_scores):
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)

    print("\nBest parameters in each fold:")
    for i, params in enumerate(fold_params, 1):
        print(f"  Fold {i}: {params}")

    print("\nTest metrics per fold:")
    for i, (rmse, r2) in enumerate(zip(test_rmse_scores, test_r2_scores), 1):
        print(f"  Fold {i}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

    print("\nSummary:")
    print(f"  RMSE: {np.mean(test_rmse_scores):.4f} ± {np.std(test_rmse_scores):.4f}")
    print(f"  R²:   {np.mean(test_r2_scores):.4f} ± {np.std(test_r2_scores):.4f}")

    print("="*50 + "\n")


def main():
    df = pd.read_parquet(DATA_PATH)

    demographic_cols = [
    'age',
    'gender',
    'marital_status',
    'children',
    'ethnicity_1',
    'drop_school_1',
    'income_gbp',
    'accommodation',
    'mh_smoking',
    'audit_1',
    'depression_medication'
    ]

    categorical_cols = ['marital_status', 'ethnicity_1', 'accommodation']

    boolean_cols = ['children', 'drop_school_1', 'depression_medication', 'gender']

    numerical_cols = ['age']

    ordinal_cols = ['mh_smoking', 'audit_1', 'income_gbp']

    X, y, groups = prepare_data(df, categorical_cols, ordinal_cols, boolean_cols)

    all_cat_cols = categorical_cols + boolean_cols

    categorical_cols_dict = {
        col: sorted(X[col].dropna().unique().tolist())
        for col in all_cat_cols
    }

    categories_list = list(categorical_cols_dict.values())

    oh_encoder = OneHotEncoder(
        categories=categories_list,
        sparse_output=False,
    ).set_output(transform="pandas")

    column_processor = ColumnTransformer(
        transformers=[
            # One-Hot Encode Categorical and Boolean columns
            ('cat', oh_encoder, all_cat_cols),
            
            # Scale Numerical and Ordinal columns
            ('num', StandardScaler(), numerical_cols + ordinal_cols),
            
            # Explicitly drop the participant_name now that merging is done
            ('drop_id', 'drop', ['participant_name'])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False 
    ).set_output(transform="pandas")

    memory = Memory(location="cache_dir", verbose=0)

    pipeline = Pipeline(steps=[
        ('imputer_merger', ImputerMerger(
            demographic_cols=demographic_cols,
            categorical_cols=categorical_cols + boolean_cols
        )),
        ('preprocessing', column_processor),
        ('elasticnet', ElasticNet(random_state=42, max_iter=5000))
        ],
    memory=memory
    )

    outer_cv = GroupKFold(n_splits=5)
    inner_cv = GroupKFold(n_splits=3)

    test_rmse_scores = []
    test_r2_scores = []
    fold_params = []
  
    alphas = np.logspace(-4, 1, 20)
    l1_ratios = np.linspace(0.05, 0.95, 10)

    param_grid = {
    'elasticnet__alpha': alphas,
    'elasticnet__l1_ratio': l1_ratios
    }


    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        #print(f"Outer Fold {fold_idx}\n")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        # Precompute inner CV splits respecting groups
        inner_cv_splits = list(inner_cv.split(X_train, y_train, groups_train))
        
        # Hyperparameter tuning with GridSearchCV on precomputed inner folds
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv_splits, scoring='neg_mean_squared_error', n_jobs=-1,)
        grid_search.fit(X_train, y_train)  # Note: no pre-scaling, pipeline handles it
        
        # Best model
        best_model = grid_search.best_estimator_
        #print(f"Best params for fold: {grid_search.best_params_}")
        fold_params.append(grid_search.best_params_)
        
        # Evaluate on test set (pipeline includes scaler, so transform X_test)
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        test_rmse_scores.append(rmse)
        test_r2_scores.append(r2)

    # print best parameters for each fold
    print_cv_results(fold_params, test_rmse_scores, test_r2_scores)


if __name__ == "__main__":
    main()

# question: how important is preventing data leakage for hyperparameter tuning (specifically preventing leakage between train_inner and validate, not between train_outer and test)
    # imputation
    # scaling

# cache memory so data isn't re-imputed for a new set of hyperparameters in the inner loop

# reduce number of inner folds to 3

# is it ok to treat ordinal features as continuous 