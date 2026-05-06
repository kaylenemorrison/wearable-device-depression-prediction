import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_parquet("../data_cleaned/final_dataset.parquet")

physiological_phq8 = df[['participant_name', 'timeInterval_ID', 'phq8_score', 'sleep_missing_last14', 'sleep_min', 'sleep_max', 'sleep_mean','sleep_median', 'sleep_std', 'sleep_iqr']]

# Separate features and target
X = physiological_phq8.drop(columns=['participant_name', 'timeInterval_ID', 'phq8_score'])
y = physiological_phq8['phq8_score']
groups = physiological_phq8['participant_name']

# Nested cross-validation
outer_cv = GroupKFold(n_splits=5)
inner_cv = GroupKFold(n_splits=3)

test_rmse_scores = []
test_r2_scores = []

alphas = np.logspace(-4, 1, 20)
l1_ratios = np.linspace(0.05, 0.95, 10)

param_grid = {
    'elasticnet__alpha': alphas,
    'elasticnet__l1_ratio': l1_ratios
}

# Define pipeline: scaler + elasticnet
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNet(random_state=42, max_iter=5000))
])

for train_idx, test_idx in outer_cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    # Precompute inner CV splits respecting groups
    inner_cv_splits = list(inner_cv.split(X_train, y_train, groups_train))
    
    # Hyperparameter tuning with GridSearchCV on precomputed inner folds
    grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv_splits, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)  # Note: no pre-scaling, pipeline handles it
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best params for fold: {grid_search.best_params_}")
    
    # Evaluate on test set (pipeline includes scaler, so transform X_test)
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    test_rmse_scores.append(rmse)
    test_r2_scores.append(r2)

# print all test rmse scores and r2 scores
print(f"Test RMSE scores: {test_rmse_scores}")
print(f"Test R² scores: {test_r2_scores}")

print(f"Test RMSE: {np.mean(test_rmse_scores):.4f} (+/- {np.std(test_rmse_scores):.4f})")
print(f"Test R²: {np.mean(test_r2_scores):.4f} (+/- {np.std(test_r2_scores):.4f})")

