# -*- coding: utf-8 -*-
"""
Temporal Statistics Extraction
"""
from concurrent.futures import as_completed
import logging
import numpy as np
import os
import pandas as pd
from warnings import warn

from rex.rechunk_h5.rechunk_h5 import get_chunk_slices
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool


logger = logging.getLogger(__name__)


class TemporalStats:
    """
    Temporal Statistics from Resource Data
    """
    STATS = ['mean', 'median', 'std', 'stdev']

    def __init__(self, res_h5, statistics=('mean'), max_workers=None,
                 res_cls=Resource, hsds=False):
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
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        self._res_h5 = res_h5
        self._stats = None
        self.statistics = statistics

        self._max_workers = None
        self.max_workers = max_workers
        self._res_cls = res_cls
        self._hsds = hsds

        with res_cls(res_h5, hsds=self._hsds) as f:
            self._time_index = f.time_index
            self._meta = f.meta

    @property
    def res_h5(self):
        """
        Path to resource h5 file(s)

        Returns
        -------
        str
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
                if s.lower().startswith('std'):
                    s = 'std'

                stats.append(s)
            else:
                msg = ("{} is not a valid statistic, must be one of:\n{}"
                       .format(s, self.STATS))
                warn(msg)
                logger.warning(msg)

        stats = list(set(stats))
        if not stats:
            msg = ('No valid statistics were supplied!')
            logger.error(msg)
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

    @property
    def lat_lon(self):
        """
        Resource (lat, lon) coordinates

        Returns
        -------
        pandas.DataFrame
        """
        lat_lon_cols = ['latitude', 'longitude']
        for c in self.meta.columns:
            if c.lower() in ['lat', 'latitude']:
                lat_lon_cols[0] = c
            elif c.lower() in ['lon', 'long', 'longitude']:
                lat_lon_cols[1] = c

        return self.meta[lat_lon_cols]

    @staticmethod
    def _format_index_value(index, stat, month_map=None):
        """
        Format groupby index value

        Parameters
        ----------
        index : int | tuple
            hour, month, or (month, hour) groupby index value
        stat : str
            Statistic that was computed
        month_map : dict | None, optional
            Mapping of month int to str, by default None

        Returns
        -------
        out : str

        """
        if isinstance(index, np.ndarray):
            m, h = index
            if month_map is not None:
                m = month_map[m]
            else:
                m = '{:02d}'.format(m)

            out = "{}-{:02d}".format(m, h)
        else:
            if month_map is not None:
                out = month_map[index]
            else:
                out = '{:02d}'.format(index)

        out += '_{}'.format(stat)

        return out

    @staticmethod
    def _create_names(index, stat):
        """
        Generate statistics names

        Parameters
        ----------
        index : pandas.Index | pandas.MultiIndex
            Temporal index, either month, hour, or (month, hour)
        stat : str
            Statistic that was computed

        Returns
        -------
        columns : list
            List of column names to use
        """
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                     6: 'June', 7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct',
                     11: 'Nov', 12: 'Dec'}
        index = np.array(index.to_list())
        if len(index.shape) != 2 and index.max() > 12:
            month_map = None

        columns = [TemporalStats._format_index_value(i, stat,
                                                     month_map=month_map)
                   for i in index]

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
                logger.warning(msg)

            if groupby:
                columns = TemporalStats._create_names(s_data.index, s)
                s_data = s_data.T
                s_data.columns = columns
            else:
                s_data = s_data.to_frame(name=s)

            res_stats.append(s_data)

        res_stats = pd.concat(res_stats, axis=1)

        return res_stats

    @staticmethod
    def _extract_stats(res_h5, res_cls, statistics, dataset, hsds=False,
                       time_index=None, site_slice=None, diurnal=False,
                       month=False, combinations=False):
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
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        time_index : pandas.DatatimeIndex | None, optional
            Resource DatetimeIndex, if None extract from res_h5,
            by default None
        site_slice : slice | None, optional
            Sites to extract, if None all, by default None
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        if site_slice is None:
            site_slice = slice(None, None, None)

        with res_cls(res_h5, hsds=hsds) as f:
            if time_index is None:
                time_index = f.time_index

            res_data = pd.DataFrame(f[dataset, :, site_slice],
                                    index=time_index)
        if combinations:
            res_stats = [TemporalStats._compute_stats(res_data, statistics)]
            if month:
                res_stats.append(TemporalStats._compute_stats(res_data,
                                                              statistics,
                                                              month=True))

            if diurnal:
                res_stats.append(TemporalStats._compute_stats(res_data,
                                                              statistics,
                                                              diurnal=True))
            if month and diurnal:
                res_stats.append(TemporalStats._compute_stats(res_data,
                                                              statistics,
                                                              month=True,
                                                              diurnal=True))

            res_stats = pd.concat(res_stats, axis=1)
        else:
            res_stats = TemporalStats._compute_stats(res_data, statistics,
                                                     diurnal=diurnal,
                                                     month=month)

        if site_slice.stop:
            res_stats.index = list(range(*site_slice.indices(site_slice.stop)))

        res_stats.index.name = 'gid'

        return res_stats

    def _get_slices(self, dataset, chunks_per_slice=5):
        """
        Get slices to extract

        Parameters
        ----------
        dataset : str
            Dataset to extract data from
        chunks_per_slice : int, optional
            Number of chunks to extract in each slice, by default 5

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
            logger.error(msg)
            raise RuntimeError(msg)

        sites = shape[1]
        if chunks is not None:
            slice_size = chunks[1] * chunks_per_slice
        else:
            slice_size = chunks_per_slice

        slices = [slice(s, e, None) for s, e
                  in get_chunk_slices(sites, slice_size)]

        return slices

    def compute_statistics(self, dataset, diurnal=False, month=False,
                           combinations=False, chunks_per_worker=5,
                           lat_lon_only=True):
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
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        if self.max_workers > 1:
            msg = ('Extracting {} for {} in parallel using {} workers'
                   .format(self.statistics, dataset, self.max_workers))
            logger.info(msg)

            slices = self._get_slices(dataset,
                                      chunks_per_slice=chunks_per_worker)
            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=self.max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for site_slice in slices:
                    future = exe.submit(TemporalStats._extract_stats,
                                        self.res_h5, self.res_cls,
                                        self.statistics, dataset,
                                        hsds=self._hsds,
                                        time_index=self.time_index,
                                        site_slice=site_slice,
                                        diurnal=diurnal,
                                        month=month,
                                        combinations=combinations)
                    futures.append(future)

                res_stats = []
                for i, future in enumerate(as_completed(futures)):
                    res_stats.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

            res_stats = pd.concat(res_stats).sort_index()
        else:
            msg = ('Extracting {} for {} in serial'
                   .format(self.statistics, dataset))
            logger.info(msg)
            res_stats = TemporalStats._extract_stats(
                self.res_h5, self.res_cls, self.statistics, dataset,
                hsds=self._hsds, time_index=self.time_index, diurnal=diurnal,
                month=month, combinations=combinations)

        if lat_lon_only:
            meta = self.lat_lon
        else:
            meta = self.meta

        res_stats = meta.join(res_stats, how='inner')

        return res_stats

    def annual_stats(self, dataset, chunks_per_worker=5, lat_lon_only=True):
        """
        Compute annual stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        annual_stats : pandas.DataFrame
            DataFrame of annual statistics
        """
        annual_stats = self.compute_statistics(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return annual_stats

    def monthly_stats(self, dataset, chunks_per_worker=5, lat_lon_only=True):
        """
        Compute monthly stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        monthly_stats : pandas.DataFrame
            DataFrame of monthly statistics
        """
        monthly_stats = self.compute_statistics(
            dataset, month=True,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return monthly_stats

    def diurnal_stats(self, dataset, chunks_per_worker=5, lat_lon_only=True):
        """
        Compute diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        diurnal_stats : pandas.DataFrame
            DataFrame of diurnal statistics
        """
        diurnal_stats = self.compute_statistics(
            dataset, diurnal=True,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return diurnal_stats

    def monthly_diurnal_stats(self, dataset, chunks_per_worker=5,
                              lat_lon_only=True):
        """
        Compute monthly-diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        monthly_diurnal_stats : pandas.DataFrame
            DataFrame of monthly-diurnal statistics
        """
        diurnal_stats = self.compute_statistics(
            dataset, month=True, diurnal=True,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return diurnal_stats

    def all_stats(self, dataset, chunks_per_worker=5, lat_lon_only=True):
        """
        Compute annual, monthly, monthly-diurnal, and diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        all_diurnal_stats : pandas.DataFrame
            DataFrame of temporal statistics
        """
        all_stats = self.compute_statistics(
            dataset, month=True, diurnal=True, combinations=True,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return all_stats

    def save_stats(self, res_stats, out_path):
        """
        Save statistics to disk

        Parameters
        ----------
        res_stats : pandas.DataFrame
            Table of statistics to save
        out_path : str
            Directory, .csv, or .json path to save statistics too
        """
        if os.path.isdir(out_path):
            out_fpath = os.path.splitext(os.path.basename(self.res_h5))[0]
            out_fpath = os.path.join(out_path, out_fpath + '.csv')
        else:
            out_fpath = out_path

        if out_fpath.endswith('.csv'):
            res_stats.to_csv(out_fpath)
        elif out_fpath.endswith('.json'):
            res_stats.to_json(out_fpath)
        else:
            msg = ("Cannot save statistics, expecting a directory, .csv, or "
                   ".json path, but got: {}".format(out_path))
            logger.error(msg)
            raise OSError(msg)

    @classmethod
    def annual(cls, res_h5, dataset, statistics=('mean'), max_workers=None,
               res_cls=Resource, hsds=False, chunks_per_worker=5,
               lat_lon_only=True, out_path=None):
        """
        Compute annual stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        annual_stats : pandas.DataFrame
            DataFrame of annual statistics
        """
        res_stats = cls(res_h5, statistics=statistics, max_workers=max_workers,
                        res_cls=res_cls, hsds=hsds)
        annual_stats = res_stats.annual_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(annual_stats, out_path)

        return annual_stats

    @classmethod
    def monthly(cls, res_h5, dataset, statistics=('mean'), max_workers=None,
                res_cls=Resource, hsds=False, chunks_per_worker=5,
                lat_lon_only=True, out_path=None):
        """
        Compute monthly stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        monthly_stats : pandas.DataFrame
            DataFrame of monthly statistics
        """
        res_stats = cls(res_h5, statistics=statistics, max_workers=max_workers,
                        res_cls=res_cls, hsds=hsds)
        monthly_stats = res_stats.monthly_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(monthly_stats, out_path)

        return monthly_stats

    @classmethod
    def diurnal(cls, res_h5, dataset, statistics=('mean'), max_workers=None,
                res_cls=Resource, hsds=False, chunks_per_worker=5,
                lat_lon_only=True, out_path=None):
        """
        Compute diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        diurnal_stats : pandas.DataFrame
            DataFrame of diurnal statistics
        """
        res_stats = cls(res_h5, statistics=statistics, max_workers=max_workers,
                        res_cls=res_cls, hsds=hsds)
        diurnal_stats = res_stats.diurnal_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(diurnal_stats, out_path)

        return diurnal_stats

    @classmethod
    def monthly_diurnal(cls, res_h5, dataset, statistics=('mean'),
                        max_workers=None, res_cls=Resource, hsds=False,
                        chunks_per_worker=5, lat_lon_only=True,
                        out_path=None):
        """
        Compute monthly-diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        monthly_diurnal_stats : pandas.DataFrame
            DataFrame of monthly-diurnal statistics
        """
        res_stats = cls(res_h5, statistics=statistics, max_workers=max_workers,
                        res_cls=res_cls, hsds=hsds)
        monthly_diurnal_stats = res_stats.monthly_diurnal_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(monthly_diurnal_stats, out_path)

        return monthly_diurnal_stats

    @classmethod
    def all(cls, res_h5, dataset, statistics=('mean'), max_workers=None,
            res_cls=Resource, hsds=False, chunks_per_worker=5,
            lat_lon_only=True, out_path=None):
        """
        Compute annual, monthly, monthly-diurnal, and diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default ('mean')
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        all_stats : pandas.DataFrame
            DataFrame of temporal statistics
        """
        res_stats = cls(res_h5, statistics=statistics, max_workers=max_workers,
                        res_cls=res_cls, hsds=hsds)
        all_stats = res_stats.all_stats(
            dataset,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(all_stats, out_path)

        return all_stats
