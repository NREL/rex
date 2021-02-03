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
    STATS = {'mean': {'func': np.mean, 'kwargs': {'axis': 0}},
             'median': {'func': np.median, 'kwargs': {'axis': 0}},
             'std': {'func': np.std, 'kwargs': {'axis': 0}}}

    def __init__(self, res_h5, statistics='mean', res_cls=Resource,
                 hsds=False):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        self._res_h5 = res_h5
        self._stats = None
        self.statistics = statistics

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
        Dictionary of statistic functions/kwargs to run

        Returns
        -------
        dict
        """
        return self._stats

    @statistics.setter
    def statistics(self, statistics):
        """
         Statistics to extract, either a key or tuple of keys in
        cls.STATS, or a dictionary of the form
        {'stat_name': {'func': *, 'kwargs: {**}}}

        Parameters
        ----------
        statistics : dict
        """
        self._stats = self._check_stats(statistics)

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

    @classmethod
    def _check_stats(cls, statistics):
        """
        check desired statistics to make sure inputs are valid

        Parameters
        ----------
        statistics : str | tuple | dict
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}}

        Returns
        -------
        stats : dict
            Dictionary of statistic functions/kwargs to run
        """
        if isinstance(statistics, str):
            statistics = (statistics, )

        if isinstance(statistics, (tuple, list)):
            statistics = {s: cls.STATS[s] for s in statistics}

        for stat in statistics.values():
            msg = 'A "func"(tion) must be provided for each statistic'
            assert 'func' in stat, msg
            if 'kwargs' in stat:
                msg = 'statistic function kwargs must be a dictionary '
                assert isinstance(stat['kwargs'], dict), msg

        return statistics

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

    @classmethod
    def _create_names(cls, index, stat):
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

        columns = [cls._format_index_value(i, stat, month_map=month_map)
                   for i in index]

        return columns

    @classmethod
    def _compute_stats(cls, res_data, statistics, diurnal=False, month=False):
        """
        Compute desired stats for desired time intervals from res_data

        Parameters
        ----------
        res_data : pandas.DataFrame
            DataFrame or resource data. Index is time_index, columns are sites
        statistics : dict
            Dictionary of statistic functions/kwargs to run
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
        for name, stat in statistics.items():
            func = stat['func']
            kwargs = stat.get('kwargs', {})
            s_data = res_data.aggregate(func, **kwargs)

            if groupby:
                columns = cls._create_names(s_data.index, name)
                s_data = s_data.T
                s_data.columns = columns
            else:
                s_data = s_data.to_frame(name=name)

            res_stats.append(s_data)

        res_stats = pd.concat(res_stats, axis=1)

        return res_stats

    @classmethod
    def _extract_stats(cls, res_h5, res_cls, statistics, dataset, hsds=False,
                       time_index=None, sites_slice=None, diurnal=False,
                       month=False, combinations=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        statistics : dict
            Dictionary of statistic functions/kwargs to run
        dataset : str
            Dataset to extract stats for
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        time_index : pandas.DatatimeIndex | None, optional
            Resource DatetimeIndex, if None extract from res_h5,
            by default None
        sites_slice : slice | None, optional
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
        if sites_slice is None:
            sites_slice = slice(None, None, None)

        with res_cls(res_h5, hsds=hsds) as f:
            if time_index is None:
                time_index = f.time_index

            res_data = pd.DataFrame(f[dataset, :, sites_slice],
                                    index=time_index)
        if combinations:
            res_stats = [cls._compute_stats(res_data, statistics)]
            if month:
                res_stats.append(cls._compute_stats(res_data, statistics,
                                                    month=True))

            if diurnal:
                res_stats.append(cls._compute_stats(res_data, statistics,
                                                    diurnal=True))
            if month and diurnal:
                res_stats.append(cls._compute_stats(res_data, statistics,
                                                    month=True, diurnal=True))

            res_stats = pd.concat(res_stats, axis=1)
        else:
            res_stats = cls._compute_stats(res_data, statistics,
                                           diurnal=diurnal, month=month)

        if isinstance(sites_slice, slice) and sites_slice.stop:
            res_stats.index = \
                list(range(*sites_slice.indices(sites_slice.stop)))
        elif isinstance(sites_slice, (list, np.ndarray)):
            res_stats.index = sites_slice

        res_stats.index.name = 'gid'

        return res_stats

    @staticmethod
    def _slice_sites(sites_slice, n_sites, slice_size):
        """
        Break up sites_slice into slices of size slice_size

        Parameters
        ----------
        sites_slice : slice
            Sites to extract as a slice object to extract
        n_sites : int
            Total number of sites to extract
        slice_size : int
            Number of sites in each slice to extract either on each worker,
            or in series

        Returns
        -------
        slices : list
            List of slices to extract
        """
        stop = sites_slice.stop
        if stop is None:
            stop = n_sites

        step = sites_slice.step
        if step is not None:
            slice_size *= step

        if slice_size >= n_sites:
            msg = ('The slice_size {} is >= the number of sites to be '
                   'extracted {}! A single slice will be extracted.'
                   .format(slice_size, n_sites))
            logger.warning(msg)
            warn(msg)
            slices = [sites_slice]
        else:
            # Create slices of size slice_size
            slices = [slice(s, e, step) for s, e
                      in get_chunk_slices(stop, slice_size)]

        return slices

    @staticmethod
    def _split_sites(sites, slice_size):
        """
        Split sites into sub-lists of ~ size slice_size

        Parameters
        ----------
        sites : list
            Sites to extract as a list or numpy object to extract
        slice_size : int
            Number of sites in each slice to extract either on each worker,
            or in series

        Returns
        -------
        slices : list
            List of slices to extract
        """
        if slice_size >= len(sites):
            msg = ('The slice_size {} is >= the number of sites to be '
                   'extracted {}! A single slice will be extracted.'
                   .format(slice_size, len(sites)))
            logger.warning(msg)
            warn(msg)
            slices = [sites]
        else:
            slices = np.array_split(sites, len(sites) // slice_size)

        return slices

    def _get_slices(self, dataset, sites=None, chunks_per_slice=5):
        """
        Get slices to extract

        Parameters
        ----------
        dataset : str
            Dataset to extract data from
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
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

        if chunks is not None:
            slice_size = chunks[1] * chunks_per_slice
        else:
            slice_size = chunks_per_slice

        if sites is None:
            sites = slice(None)

        if isinstance(sites, slice):
            slices = self._slice_sites(sites, shape[1], slice_size)
        elif isinstance(sites, (list, tuple, np.ndarray)):
            slices = self._split_sites(sites, slice_size)
        else:
            msg = ('sites must be of type "None", "slice", "list", "tuple", '
                   'or "np.ndarray", but {} was provided'.format(type(sites)))
            raise TypeError(msg)

        return slices

    def compute_statistics(self, dataset, sites=None,
                           diurnal=False, month=False, combinations=False,
                           max_workers=None, chunks_per_worker=5,
                           lat_lon_only=True):
        """
        Compute statistics

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of desired statistics at desired time intervals
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        slices = self._get_slices(dataset, sites,
                                  chunks_per_slice=chunks_per_worker)
        if len(slices) == 1:
            max_workers = 1

        if max_workers > 1:
            msg = ('Extracting {} for {} in parallel using {} workers'
                   .format(list(self.statistics), dataset, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites_slice in slices:
                    future = exe.submit(self._extract_stats,
                                        self.res_h5, self.res_cls,
                                        self.statistics, dataset,
                                        hsds=self._hsds,
                                        time_index=self.time_index,
                                        sites_slice=sites_slice,
                                        diurnal=diurnal,
                                        month=month,
                                        combinations=combinations)
                    futures.append(future)

                res_stats = []
                for i, future in enumerate(as_completed(futures)):
                    res_stats.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

            res_stats = pd.concat(res_stats)
        else:
            msg = ('Extracting {} for {} in serial'
                   .format(self.statistics.keys(), dataset))
            logger.info(msg)
            if chunks_per_worker is not None:
                res_stats = []
                for sites_slice in slices:
                    res_stats.append(self._extract_stats(
                        self.res_h5, self.res_cls, self.statistics, dataset,
                        hsds=self._hsds, time_index=self.time_index,
                        sites_slice=sites_slice, diurnal=diurnal, month=month,
                        combinations=combinations))

                res_stats = pd.concat(res_stats)
            else:
                res_stats = self._extract_stats(
                    self.res_h5, self.res_cls, self.statistics, dataset,
                    hsds=self._hsds, time_index=self.time_index,
                    diurnal=diurnal, month=month,
                    combinations=combinations)

        if lat_lon_only:
            meta = self.lat_lon
        else:
            meta = self.meta

        res_stats = meta.join(res_stats.sort_index(), how='inner')

        return res_stats

    def full_stats(self, dataset, sites=None, max_workers=None,
                   chunks_per_worker=5, lat_lon_only=True):
        """
        Compute stats for entire temporal extent of file

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True

        Returns
        -------
        full_stats : pandas.DataFrame
            DataFrame of statistics for the entire temporal extent of file
        """
        full_stats = self.compute_statistics(
            dataset, sites=sites,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return full_stats

    def monthly_stats(self, dataset, sites=None, max_workers=None,
                      chunks_per_worker=5, lat_lon_only=True):
        """
        Compute monthly stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
            dataset, sites=sites, month=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return monthly_stats

    def diurnal_stats(self, dataset, sites=None, max_workers=None,
                      chunks_per_worker=5, lat_lon_only=True):
        """
        Compute diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
            dataset, sites=sites, diurnal=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return diurnal_stats

    def monthly_diurnal_stats(self, dataset, sites=None,
                              max_workers=None, chunks_per_worker=5,
                              lat_lon_only=True):
        """
        Compute monthly-diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
            dataset, sites=sites, month=True, diurnal=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)

        return diurnal_stats

    def all_stats(self, dataset, sites=None, max_workers=None,
                  chunks_per_worker=5, lat_lon_only=True):
        """
        Compute annual, monthly, monthly-diurnal, and diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
            dataset, sites=sites, month=True, diurnal=True, combinations=True,
            max_workers=max_workers,
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
    def run(cls, res_h5, dataset, sites=None, statistics='mean',
            diurnal=False, month=False, combinations=False,
            res_cls=Resource, hsds=False, max_workers=None,
            chunks_per_worker=5, lat_lon_only=True, out_path=None):
        """
        Compute temporal stats, by default full temporal extent stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default 'mean'
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        res_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        res_stats = cls(res_h5, statistics=statistics, res_cls=res_cls,
                        hsds=hsds)
        res_stats = res_stats.compute_statistics(
            dataset, sites=sites,
            diurnal=diurnal, month=month, combinations=combinations,
            max_workers=max_workers, chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only)
        if out_path is not None:
            res_stats.save_stats(res_stats, out_path)

        return res_stats

    @classmethod
    def monthly(cls, res_h5, dataset, sites=None, statistics='mean',
                res_cls=Resource, hsds=False, max_workers=None,
                chunks_per_worker=5, lat_lon_only=True, out_path=None):
        """
        Compute monthly stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
        monthly_stats = cls.run(res_h5, dataset, sites=sites,
                                statistics=statistics, diurnal=False,
                                month=True, combinations=False,
                                res_cls=res_cls, hsds=hsds,
                                max_workers=max_workers,
                                chunks_per_worker=chunks_per_worker,
                                lat_lon_only=lat_lon_only, out_path=out_path)

        return monthly_stats

    @classmethod
    def diurnal(cls, res_h5, dataset, sites=None, statistics='mean',
                res_cls=Resource, hsds=False, max_workers=None,
                chunks_per_worker=5, lat_lon_only=True, out_path=None):
        """
        Compute diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
        diurnal_stats = cls.run(res_h5, dataset, sites=sites,
                                statistics=statistics, diurnal=True,
                                month=False, combinations=False,
                                res_cls=res_cls, hsds=hsds,
                                max_workers=max_workers,
                                chunks_per_worker=chunks_per_worker,
                                lat_lon_only=lat_lon_only, out_path=out_path)

        return diurnal_stats

    @classmethod
    def monthly_diurnal(cls, res_h5, dataset, sites=None,
                        statistics='mean', res_cls=Resource, hsds=False,
                        max_workers=None, chunks_per_worker=5,
                        lat_lon_only=True, out_path=None):
        """
        Compute monthly-diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
        monthly_diurnal_stats = cls.run(res_h5, dataset, sites=sites,
                                        statistics=statistics, diurnal=True,
                                        month=True, combinations=False,
                                        res_cls=res_cls, hsds=hsds,
                                        max_workers=max_workers,
                                        chunks_per_worker=chunks_per_worker,
                                        lat_lon_only=lat_lon_only,
                                        out_path=out_path)

        return monthly_diurnal_stats

    @classmethod
    def all(cls, res_h5, dataset, sites=None, statistics='mean',
            res_cls=Resource, hsds=False, max_workers=None,
            chunks_per_worker=5, lat_lon_only=True, out_path=None):
        """
        Compute annual, monthly, monthly-diurnal, and diurnal stats

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        statistics : str | tuple, optional
            Statistics to extract, must be 'mean', 'median', 'std',
            and/or 'stdev', by default 'mean'
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
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
        all_stats = cls.run(res_h5, dataset, sites=sites,
                            statistics=statistics, diurnal=True,
                            month=True, combinations=True,
                            res_cls=res_cls, hsds=hsds,
                            max_workers=max_workers,
                            chunks_per_worker=chunks_per_worker,
                            lat_lon_only=lat_lon_only, out_path=out_path)

        return all_stats
