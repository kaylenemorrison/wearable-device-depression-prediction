import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Unsupervised redundancy prune for highly collinear feature blocks.

    Clusters features by absolute correlation (distance = 1 - |corr|) using
    average-linkage hierarchical clustering, cuts the dendrogram at
    `1 - threshold`, and keeps one representative per cluster. Because it uses
    only X (never y), it is leakage-safe; fit() learns the keep-set from the
    training rows alone, so refitting per fold inside an sklearn Pipeline carries
    no leakage. Intended for the catch22 feature set, where the same statistic is
    computed across many correlated sub-signals, but it is harmless on any
    numeric block.

    Parameters
    ----------
    threshold : float, default=0.9
        Features with absolute correlation above this are treated as redundant
        and collapsed into a single cluster. Higher -> prunes less.
    method : {"spearman", "pearson", "kendall"}, default="spearman"
        Correlation used to build the distance matrix. Spearman is the default
        because catch22 outputs are skewed/heavy-tailed, so rank correlation
        captures redundancy more robustly than Pearson.
    representative : {"medoid", "variance"}, default="medoid"
        How to pick the survivor from each cluster. "medoid" keeps the member
        with the highest mean |corr| to its cluster-mates (most central, best
        summarizes what it replaces); "variance" keeps the highest-variance
        member.

    Attributes
    ----------
    keep_columns_ : list
        Column labels retained by the prune, in cluster order.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, threshold=0.9, method="spearman", representative="medoid"):
        self.threshold = threshold
        self.method = method
        self.representative = representative

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]

        # |corr| -> distance. NaN correlations (e.g. a constant column) become
        # distance 1 (treated as fully dissimilar) rather than crashing linkage.
        corr = X.corr(method=self.method).abs()
        corr = corr.fillna(0.0)
        dist = 1.0 - corr.values
        np.fill_diagonal(dist, 0.0)

        # A single feature has nothing to cluster against; keep it as-is.
        if self.n_features_in_ < 2:
            self.keep_columns_ = list(corr.columns)
            return self

        Z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - self.threshold, criterion="distance")

        keep = []
        for c in np.unique(labels):
            members = corr.columns[labels == c]
            if len(members) == 1:
                keep.append(members[0])
            elif self.representative == "variance":
                keep.append(X[members].var().idxmax())
            else:  # medoid: most central member of the cluster
                sub = corr.loc[members, members]
                keep.append(sub.mean(axis=1).idxmax())
        self.keep_columns_ = list(keep)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_columns_)
