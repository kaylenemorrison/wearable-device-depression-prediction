import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from missforest import MissForest

class ImputerMerger(BaseEstimator, TransformerMixin):
    """
    Single transformer that:
    1. Extracts unique demographic rows (one per participant)
    2. Imputes using MissForest (handles mixed data types automatically)
    3. Merges with physiological data
    """
    
    def __init__(self, demographic_cols, categorical_cols,
                 participant_id_col='participant_name'):
        self.demographic_cols = demographic_cols
        self.categorical_cols = categorical_cols 
        self.participant_id_col = participant_id_col
        
        # Will be fitted
        self.imputer_ = None
        
    def fit(self, X):
        """Fit MissForest on unique demographic rows"""
        # Extract unique demographic rows (one per participant)
        demo_unique = X.groupby(self.participant_id_col)[self.demographic_cols].first()
        
        # Create and fit MissForest imputer
        self.imputer_ = MissForest (
            categorical=self.categorical_cols
        )

        # Fit on demographic data
        self.imputer_.fit(demo_unique)
        
        return self
    
    def transform(self, X):
        """Impute demographics and merge back"""
        
        # Extract unique demographic rows with participant IDs
        demo_unique = X.groupby(self.participant_id_col)[self.demographic_cols].first()
        
        # Impute
        demo_imputed = self.imputer_.transform(demo_unique)
        
        # Drop original demographic columns from X
        X_no_demo = X.drop(columns=self.demographic_cols)
        
        # Merge imputed demographics back
        X_merged = X_no_demo.merge(
            demo_imputed,
            left_on=self.participant_id_col,
            right_index=True,
            how='left'
        )
        
        return X_merged