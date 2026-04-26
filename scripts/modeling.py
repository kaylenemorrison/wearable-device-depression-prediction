import pandas as pd

demographics = pd.read_csv("../data_cleaned/totaldataset.csv")
physiological_phq8 = pd.read_csv("../data_cleaned/physiological_PHQ8.csv")

# demographics columns: ['participant_name', 'age', 'gender_at_birth', 'marital_status', 'children', 'ethnicity_1', 'eduction_finish_age', 'drop_school_1','degree_1', 'income_gbp', 'accommodation', 'mh_smoking', 'audit_1', 'depression_medication']

categorical_cols = ['marital_status, ethnicity_1',  'accommodation' ]

boolean_cols = ['gender_at_birth', 'children', 'drop_school_1', 'depression_medication']

ordinal_cols = ['degree_1', 'income_gbp', 'mh_smoking', 'audit_1']

numeric_cols = ['age', 'eduction_finish_age']

# use miceforest for imputation
# one hot encode categorical variables 