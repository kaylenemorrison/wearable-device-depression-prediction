import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer

import numpy as np

import os
os.environ["TQDM_DISABLE"] = "True"

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

def get_oh_encoder(all_cat_cols, X):
    categorical_cols_dict = {
        col: sorted(X[col].dropna().unique().tolist())
        for col in all_cat_cols
    }
    categories_list = list(categorical_cols_dict.values())
    oh_encoder = OneHotEncoder(
        categories=categories_list,
        sparse_output=False,
    )
    return oh_encoder

def create_pipeline(X, categorical_cols, boolean_cols, numerical_cols):

    oh_encoder = get_oh_encoder(categorical_cols, X)

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
        ("model", ElasticNet(random_state=42, max_iter=20000))
    ])

    return pipeline

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

pipeline = create_pipeline(X, categorical_cols, all_boolean_cols, numerical_cols)

pipeline.fit(X, y)