from typing import List, Union

import pandas as pd
import numpy as np

from ..trajectory_data import TrajectoryData


class TrajectoryLoader(object):
    """Base class for trajectory loaders.
    """

    def load(self):
        """Loads trajectories according to the specific approach.

        Returns
        -------
        data : :class:`trajminer.TrajectoryData`
            A :class:`trajminer.TrajectoryData` containing the loaded dataset.
        """
        pass


class CSVTrajectoryLoader(TrajectoryLoader):
    """A trajectory data loader from a CSV file.

    Parameters
    ----------
    file : str
        The CSV file from which to read the data.
    sep : str (default=',')
        The CSV separator.
    tid_cols : Union[str, List[str]] (default='tid')
        The column(s) in the CSV file corresponding to the trajectory IDs.
    label_col : str (default='label')
        The column in the CSV file corresponding to the trajectory labels. If
        `None`, labels are not loaded.
    lat : str (default='lat')
        The column in the CSV file corresponding to the latitude of the
        trajectory points.
    lon : str (default='lon')
        The column in the CSV file corresponding to the longitude of the
        trajectory points.
    drop_col : array-like (default=None)
        List of columns to drop when reading the data from the file.

    Examples
    --------
    >>> from trajminer.utils import CSVTrajectoryLoader
    >>> loader = CSVTrajectoryLoader('my_data.csv')
    >>> dataset = loader.load()
    >>> dataset.get_attributes()
    ['poi', 'day', 'time']
    """

    def __init__(self, file, sep=',', tid_cols: Union[str, List[str]] = 'tid', label_col: str = 'label',
                 lat='lat', lon='lon', drop_col=None):
        
        if isinstance(tid_cols, str):
            tid_cols = [tid_cols]
                
        self.file = file
        self.sep = sep
        self.tid_cols = tid_cols
        self.label_col = label_col
        self.lat = lat
        self.lon = lon
        self.drop_col = drop_col if drop_col is not None else []

    def load(self):
        df = pd.read_csv(self.file, sep=self.sep, usecols=lambda c: c not in self.drop_col)
        
        not_attributes = set(self.tid_cols) | set(self.label_col)
        attributes = [x for x in df.columns if x not in not_attributes]
        
        df = df.sort_values(by=self.tid_cols)
        df = df.set_index(self.tid_cols, drop=False)
        
        attrs_df = df[attributes].copy()
        
        if self.label_col:
            labels = df[self.label_col].copy()
        else:
            labels = None
            
        return TrajectoryData.from_dataframe(attrs_df, labels)
