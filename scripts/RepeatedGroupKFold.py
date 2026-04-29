import numpy as np
from sklearn.model_selection import GroupKFold

class RepeatedGroupKFold:
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_repeats):
            gkf = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=rng.randint(np.iinfo(np.int32).max))
            yield from gkf.split(X, y, groups=groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats