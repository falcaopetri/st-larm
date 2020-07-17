import numpy as np
import pandas as pd
from itertools import product

class TrajectoryData:
    """Trajectory data wrapper.

    Parameters
    ----------
    attributes : array-like
        The names of attributes/features describing trajectory points in the
        dataset.
    data : array-like, shape: (n_trajectories, n_points, n_features)
        The trajectory data.
    tids : array-like
        The corresponding trajectory IDs of trajectories in ``data``.
    labels : array-like (default=None)
        The corresponding labels of trajectories in ``data``.
    """

    def __init__(self, attributes, data, tids, labels=None):
        attributes = np.array(attributes)
        tids = np.array(tids)
        data = [(tid, *point) for tid, point in zip(tids, data)]
        columns = np.concatenate([['tid'], attributes])
        self.data = (pd.DataFrame(data, columns=columns)
                       .groupby('tid')
                       .apply(lambda g: g.assign(pid=np.arange(len(g))))
                       .set_index(['tid', 'pid']))
        
        self.labels = None if labels is None else pd.Series(labels, index=tids, name='label')
        self._stats = None
        
    @classmethod
    def from_dataframe(cls, attrs_df: pd.DataFrame, labels: np.array):
        return cls(attrs_df.columns, 
                   attrs_df.to_numpy().tolist(),
                   attrs_df.index.to_numpy(), 
                   labels)

    def get_attributes(self):
        """Retrieves the attributes in the dataset.

        Returns
        -------
        attributes : array
            An array of length `n_features`.
        """
        return self.data.columns.to_numpy()

    def get_tids(self, label=None):
        """Retrieves the trajectory IDs in the dataset.

        Parameters
        ----------
        label : int (default=None)
            If `None`, then retrieves all trajectory IDs. Otherwise, returns
            the IDs corresponding to the given label.

        Returns
        -------
        attributes : array
            An array of length `n_trajectories`.
        """
        if not label or self.labels is None:
            return self.data.index.unique(level='tid')

        return self.labels[self.labels == label].index.to_numpy()

    def get_label(self, tid):
        """Retrieves the label for the corresponding tid.

        Parameters
        ----------
        tid : int
            The trajectory ID.

        Returns
        -------
        label : int or str
            The corresponding label.
        """
        return self.labels[tid]

    def get_labels(self, unique=False):
        """Retrieves the labels of the trajectories in the dataset.

        Parameters
        ----------
        unique : bool (default=False)
            If ``True``, then the set of unique labels is returned. Otherwise,
            an array with the labels of each individual trajectory is returned.

        Returns
        -------
        labels : array
            An array of length `n_trajectories` if `unique=False`, and of
            length `n_labels` otherwise.
        """
        if unique and self.labels is not None:
            return np.sort(self.labels.unique())

        return self.labels.to_numpy()

    def get_trajectory(self, tid):
        """Retrieves a trajectory from the dataset.

        Parameters
        ----------
        tid : int
            The trajectory ID.

        Returns
        -------
        trajectory : array, shape: (n_points, n_features)
            The corresponding trajectory.
        """
        return self.data.loc[tid].to_numpy().tolist()

    def get_trajectories(self, label=None):
        """Retrieves multiple trajectories from the dataset.

        Parameters
        ----------
        label : int (default=None)
            The label of the trajectories to be retrieved. If ``None``, then
            all trajectories are retrieved.

        Returns
        -------
        trajectories : array
            The trajectories of the given label. If `label=None` or if the
            dataset does not contain labels, then all trajectories are
            returned.
        """
        if not label or self.labels is None:
            return (self.data.groupby(level='tid')
                             .apply(lambda g: g.to_numpy().tolist())
                             .to_numpy().tolist())

        tids = self.labels[self.labels == label].index
        return (self.data.loc[tids].groupby(level='tid')
                         .apply(lambda g: g.to_numpy().tolist())
                         .to_numpy())

    def length(self):
        """Returns the number of trajectories in the dataset.

        Returns
        -------
        length : int
            Number of trajectories in the dataset.
        """
        return self.data.index.get_level_values('tid').nunique()

    def merge(self, other, ignore_duplicates=True, inplace=True):
        """Merges this trajectory data with another one. Notice that this
        method only works if the datasets have the same set of attributes.

        Parameters
        ----------
        other : :class:`trajminer.TrajectoryData`
            The dataset to be merged with.
        ignore_duplicates : bool (default=True)
            If `True`, then trajectory IDs in `other` that already exist in
            `self` are ignored. Otherwise, raises an exception when a duplicate
            is found.
        inplace : bool (default=True)
            If `True` modifies the current object, otherwise returns a new
            object.

        Returns
        -------
        dataset : :class:`trajminer.TrajectoryData`
            The merged dataset. If `inplace=True`, then returns the modified
            current object.
        """
        if set(self.attributes) != set(other.attributes):
            raise Exception("Cannot merge datasets with different sets of " +
                            "attributes!")

        n_attributes = self.attributes
        n_tids = self.tids.tolist()
        n_labels = self.labels.tolist()
        n_data = self.data.tolist()

        for tid in other.tids:
            if tid in n_tids:
                if ignore_duplicates:
                    continue
                raise Exception("tid", tid, "already exists in 'self'!")
            n_tids.append(tid)
            n_data.append(other.get_trajectory(tid))

            if n_labels is not None:
                n_labels.append(other.get_label(tid))

        if inplace:
            self._update(n_attributes, n_data, n_tids, n_labels)
            return self

        return TrajectoryData(n_attributes, n_data, n_tids, n_labels)

    def to_file(self, file, file_type='csv'):
        """Persists the dataset to a file.

        Parameters
        ----------
        file : str
            The output file.
        file_type : str (default='csv')
            The file type. Must be one of `{csv}`.
        """
        if file_type == 'csv':
            self._to_csv(file)

    def stats(self, print_stats=False):
        """Computes statistics for the dataset.

        Parameters
        ----------
        print_stats : bool (default=False)
            If `True`, stats are printed.

        Returns
        -------
        stats : dict
            A dictionary containing the dataset statistics.
        """
        if self._stats:
            if print_stats:
                self._print_stats()
            return self._stats

        traj_lengths = np.array([len(x) for x in self.data])
        points = np.concatenate(self.data)

        def count_not_none(arr):
            return sum([1 if x is not None else 0 for x in arr])

        attr_count = np.array([count_not_none(p) for p in points])

        self._stats = {
            'attribute': {
                'count': len(self.attributes),
                'min': attr_count.min(),
                'avg': attr_count.mean(),
                'std': attr_count.std(),
                'max': attr_count.max()
            },
            'point': {
                'count': traj_lengths.sum()
            },
            'trajectory': {
                'count': len(self.data),
                'length': {
                    'min': traj_lengths.min(),
                    'avg': traj_lengths.mean(),
                    'std': traj_lengths.std(),
                    'max': traj_lengths.max()
                }
            }
        }

        if self.labels is not None:
            unique, counts = np.unique(self.labels, return_counts=True)
            self._stats['label'] = {
                'count': len(unique),
                'min': counts.min(),
                'avg': counts.mean(),
                'std': counts.std(),
                'max': counts.max()
            }

        if print_stats:
            self._print_stats()
        return self._stats

    def _update(self, attributes, data, tids, labels):
        self.tids = np.array(tids)
        self.labels = np.array(labels)
        self.data = np.array(data)
        self.tidToIdx = dict(zip(tids, np.r_[0:len(tids)]))
        self.labelToIdx = TrajectoryData._get_label_to_idx(labels)
        self._stats = None

    def _to_csv(self, file):
        df = pd.merge(self.data, self.labels, validate='many_to_one')
        df.to_csv(file)

    def _print_stats(self):
        print('==========================================================')
        print('                           STATS                          ')
        print('==========================================================')
        print('ATTRIBUTE')
        print('  Count:           ', self._stats['attribute']['count'])
        print('  Min:             ', self._stats['attribute']['min'])
        print('  Max:             ', self._stats['attribute']['max'])
        print('  Avg ± Std:        %.4f ± %.4f' % (
            self._stats['attribute']['avg'], self._stats['attribute']['std']))

        print('\nPOINT')
        print('  Count:           ', self._stats['point']['count'])

        print('\nTRAJECTORY')
        print('  Count:           ', self._stats['trajectory']['count'])
        print('  Min length:      ',
              self._stats['trajectory']['length']['min'])
        print('  Max lenght:      ',
              self._stats['trajectory']['length']['max'])
        print('  Avg length ± Std: %.4f ± %.4f' %
              (self._stats['trajectory']['length']['avg'],
               self._stats['trajectory']['length']['std']))

        if self.labels is not None:
            print('\nLABEL')
            print('  Count:           ', self._stats['label']['count'])
            print('  Min:             ', self._stats['label']['min'])
            print('  Max:             ', self._stats['label']['max'])
            print('  Avg ± Std:        %.4f ± %.4f' % (
                self._stats['label']['avg'], self._stats['label']['std']))
            print('==========================================================')
        else:
            print('==========================================================')
