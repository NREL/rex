# -*- coding: utf-8 -*-
"""
Resource Statistics Extraction
"""
import os
import pandas as pd
from warnings import warn

from rex.rechunk_h5.rechunk_h5 import get_chunk_slices
from rex.resource import Resource


class ResourceStats:
    """
    Temporal Statistics from Resource Data
    """
    STATS = ['mean', 'median', 'std', 'stdev']

    def __init__(self, res_h5, statistics=('mean'), max_workers=None,
                 res_cls=Resource):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        """
        self._res_h5 = res_h5
        self._stats = None
        self.statistics = statistics

        self._max_workers = None
        self.max_workers = max_workers
        self._res_cls = res_cls

        with res_cls(res_h5) as f:
            self._time_index = f.time_index
            self._meta = f.meta

    @property
    def res_h5(self):
        """
        Path to resource h5 file(s)

        Returns
        -------
        str
            [description]
        """
        return self._res_h5

    @property
    def statistics(self):
        """
        Statistics to extract:
        - mean
        - median
        - std or stdev

        Returns
        -------
        tuple
        """
        return self._stats

    @statistics.setter
    def statistics(self, statistics):
        """
        Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev'

        Parameters
        ----------
        statistics : str | tuple | list
        """
        if isinstance(statistics, str):
            statistics = [statistics]

        stats = []
        for s in statistics:
            if s.lower() in self.STATS:
                if s.lower.startswith('std'):
                    s = 'std'

                stats.append(s)
            else:
                msg = ("{} is not a valid statistic, must be one of:\n{}"
                       .format(s, self.STATS))
                warn(msg)

        stats = list(set(stats))
        if not stats:
            msg = ('No valid statistics were supplied!')
            raise ValueError(msg)

        self._stats = stats

    @property
    def max_workers(self):
        """
        Number of workers to use, if 1 run in serial

        Returns
        -------
        int
        """
        return self._max_workers

    @max_workers.setter
    def max_workers(self, max_workers):
        """
        Number of workers to use, if 1 run in serial, if None use all
        available cores

        Parameters
        ----------
        max_workers : None | int
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        self._max_workers = max_workers

    @property
    def res_cls(self):
        """
        Resource class to use to access res_h5

        Returns
        -------
        Class
        """
        return self._res_cls

    @property
    def time_index(self):
        """
        Resource Datetimes

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def meta(self):
        """
        Resource meta-data table

        Returns
        -------
        pandas.DataFrame
        """
        return self._meta

    @staticmethod
    def _create_columns(index, statistic):
        """
        Generate statistics columns

        Parameters
        ----------
        index : pandas.Index | pandas.MultiIndex
            Temporal index, either month, hour, or (month, hour)
        statistic : str
            Statistic that was computed

        Returns
        -------
        columns : list
            List of column names to use
        """
        columns = []
        for i in index:
            if isinstance(i, tuple):
                i = '-'.join(['{:02d}'.format(j) for j in i])
            else:
                i = '{:02d}'.format(i)

            columns.append('{}_{}'.format(i, statistic))

        return columns

    @staticmethod
    def _compute_stats(res_data, statistics, diurnal=False, month=False):
        """
        Compute desired stats for desired time intervals from res_data

        Parameters
        ----------
        res_data : ndarray
            Resource data array
        statistics : tuple | list
            Statistics to extract, must be 'mean', 'median', and/or 'std
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        groupby = []
        if month:
            groupby.append(res_data.index.month)

        if diurnal:
            groupby.append(res_data.index.hour)

        if groupby:
            res_data = res_data.groupby(groupby)

        res_stats = []
        for s in statistics:
            if s == 'mean':
                s_data = res_data.mean()
            elif s == 'median':
                s_data = res_data.median()
            elif s == 'std':
                s_data = res_data.std()
            else:
                msg = ("{} is not a valid statistic, must be one of:\n{}"
                       .format(s, ['mean', 'median', 'std']))
                warn(msg)

            if groupby:
                s_data = s_data.T
                s_data.columns = ResourceStats._create_columns(s_data.index, s)
            else:
                s_data = s_data.to_frame(name=s)

            res_stats.append(s_data)

        res_stats = pd.concat(res_stats, axis=1)

        return res_stats

    @staticmethod
    def _extract_stats(res_h5, res_cls, statistics, dataset, time_index=None,
                       site_slice=None, diurnal=False, month=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        statistics : tuple | list
            Statistics to extract, must be 'mean', 'median', and/or 'std
        dataset : str
            Dataset to extract stats for
        time_index : pandas.DatatimeIndex | None, optional
            Resource DatetimeIndex, if None extract from res_h5,
            by default None
        site_slice : slice | None, optional
            Sites to extract, if None all, by default None
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        if site_slice is None:
            site_slice = slice(None, None, None)

        with res_cls(res_h5) as f:
            if time_index is None:
                time_index = f.time_index

            res_data = pd.DataFrame(f[dataset, :, site_slice],
                                    index=time_index)

        res_stats = ResourceStats._compute_stats(res_data, statistics,
                                                 diurnal=diurnal, month=month)

        if site_slice.stop:
            res_stats.index = list(range(*site_slice.indices(site_slice.stop)))

        return res_stats

    def _get_slices(self, dataset, chunks_per_slice=10):
        """
        Get slices to extract

        Parameters
        ----------
        dataset : str
            Dataset to extract data from
        chunks_per_slice : int, optional
            Number of chunks to extract in each slice, by default 10

        Returns
        -------
        slices : list
            List of slices to extract
        """
        with self.res_cls(self.res_h5) as f:
            shape, _, chunks = f.get_dset_properties(dataset)

        if len(shape) != 2:
            msg = ('Cannot extract temporal stats for dataset {}, as it is '
                   'not a timeseries dataset!'.format(dataset))
            raise RuntimeError(msg)

        sites = shape[1]
        slice_size = chunks[1] * chunks_per_slice
        slices = [slice(s, e, None) for s, e
                  in get_chunk_slices(sites, slice_size)]

        return slices

    def compute_statistics(self, dataset, diurnal=False, month=False,
                           chunks_per_worker=10):
        """
        Compute statistics

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 10
        """
