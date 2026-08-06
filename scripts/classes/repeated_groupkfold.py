from sklearn.model_selection import GroupKFold
from sklearn.model_selection._split import GroupsConsumerMixin, _RepeatedSplits

__all__ = ["RepeatedGroupKFold"]


class RepeatedGroupKFold(GroupsConsumerMixin, _RepeatedSplits):
    """Repeated K-fold iterator variant with non-overlapping groups.

    Repeats :class:`~sklearn.model_selection.GroupKFold` ``n_repeats`` times,
    reshuffling the groups before each repetition. Within a repetition every
    group appears in exactly one test fold, so a group is never split across
    train and test, and each sample is tested exactly once. Across repetitions
    each sample is therefore tested exactly ``n_repeats`` times.

    Folds hold an equal number of *groups*, not an equal number of *samples*:
    the underlying ``GroupKFold`` balances folds by sample count only when
    ``shuffle=False``, which is incompatible with reshuffling per repetition.
    With unevenly sized groups the test folds vary in size accordingly, so
    prefer pooling out-of-fold predictions and scoring once over averaging
    per-fold metrics with unequal denominators.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds per repetition. Must be at least 2 and at most the
        number of distinct groups.

    n_repeats : int, default=10
        Number of times the cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the group permutation drawn for each repetition. Pass an int
        for reproducible output across multiple calls to ``split``: the
        underlying generator is re-seeded on entry to every call, so repetition
        *r* always yields the same partition, and a run with ``n_repeats=n`` is
        a prefix of a run with any larger ``n_repeats``. With ``None`` or a
        RandomState instance the generator state carries across calls, so
        successive calls give different partitions.

    Notes
    -----
    With few groups relative to ``n_splits``, distinct repetitions can produce
    the same partition by chance; the number of possible partitions is finite
    and no de-duplication is attempted.

    ``GroupKFold(shuffle=...)`` needs scikit-learn >= 1.6, but ``environment``
    pins 1.9: the partition a given seed produces is only stable within a
    version, since it depends on how many draws the private ``_RepeatedSplits``
    and ``GroupKFold._iter_test_indices`` take from the generator. Those
    private internals are also what supply the per-repetition reshuffling and
    the per-call re-seeding described above.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(8).reshape(-1, 1)
    >>> groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> cv = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=0)
    >>> cv.get_n_splits()
    4
    >>> for train, test in cv.split(X, groups=groups):
    ...     assert not set(groups[train]) & set(groups[test])
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    @property
    def n_splits(self):
        """Number of folds per repetition.

        ``_RepeatedSplits`` sweeps ``n_splits`` into ``self.cvargs`` and leaves
        only ``n_repeats`` as an attribute, so the plain sklearn splitters
        expose ``n_splits`` and this one would not. Reading it back out keeps
        that parity, and matters because ``get_n_splits()`` reports folds *times
        repeats* -- the wrong number for anything that slices the split sequence
        into repeats. Deliberately read-only: it is a view onto the dict that
        ``split`` actually passes to ``GroupKFold``, so the two cannot drift.
        """
        return self.cvargs["n_splits"]
