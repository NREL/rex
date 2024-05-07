# -*- coding: utf-8 -*-
"""
Temporal Statistics Extraction
"""
from concurrent.futures import as_completed
import gc
import logging
import numpy as np
import os
import pandas as pd

from rex.resource import Resource
from rex.utilities.bc_utils import (sample_q_linear, sample_q_log,
                                    sample_q_invlog)
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem, log_versions, create_dirs
from rex.utilities.utilities import get_lat_lon_cols, slice_sites

logger = logging.getLogger(__name__)


def circular_mean(data, weights=None, degrees=True, axis=0,
                  exponential_weights=True):
    """
    Computed the ciruclar average. if provided compute the weighed average
    with the given weights. For example, if averaging wind direction with wind
    speed as weights, wind directions that occur at higher wind speeds will
    have a larger weight of the final mean value.

    Parameters
    ----------
    data : ndarray
        Data to average
    weights : ndarray, optional
        Weights to apply to data during averaging, must be of the same
        shape as data, by default None
    degree : bool, optional
        Flag indicating that data is in degrees and needs to be converted
        to/from radians during averaging. By default True
    axis : int, optional
        Axis to compute average along, by default 0 which will produce
        site averages
    norm_weights: : bool, optional
        Flag to normalize weights, by default True
    exponential_weights : bool
        Flag to convert weights to exponential, by default True

    Returns
    -------
    mean : ndarray
        Weighted circular mean along the given axis
    """
    if degrees:
        data = np.radians(data, dtype=np.float32)

    sin = np.sin(data)
    cos = np.cos(data)
    if weights is None:
        sin = np.nanmean(sin, axis=axis)
        cos = np.nanmean(cos, axis=axis)
    else:
        if exponential_weights:
            weights = np.exp(weights)

        if weights.shape != data.shape:
            msg = ('The shape of weights {} does not match the shape of the '
                   'data {} to which it is to be applied!'
                   .format(weights.shape, data.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        n_weights = np.expand_dims(np.nansum(weights, axis=axis), axis)
        sin = np.nansum(sin * weights, axis=axis) / n_weights
        cos = np.nansum(cos * weights, axis=axis) / n_weights

    mean = np.arctan2(sin, cos)
    if degrees:
        mean = np.degrees(mean)
        mask = mean < 0
        if isinstance(mask, np.ndarray):
            mean[mask] += 360
        elif mask:
            mean += 360

    return mean


def cdf(data, n_samples=50, sampling='linear', log_base=10, decimals=None):
    """Get a number of x-values that define a CDF for the input data.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to make a CDF for
    n_samples : int
        Number of points to fit the CDF
    sampling : str
        Option for quantile sampling (see sampling functions in
        ``rex.utilities.bc_utils``), e.g., how to sample the y-axis of the
        distribution. "linear" will do even spacing, "log" will concentrate
        samples near quantile=0, and "invlog" will concentrate samples near
        quantile=1
    log_base : int | float
        Log base value if sampling is "log" or "invlog". A higher value will
        concentrate more samples at the extreme sides of the distribution.
    decimals : int | None
        Precision to round output to (see docstring for np.round). None will
        not round outputs (default).

    Returns
    -------
    x_values : np.ndarray
        1D array of values with shape (n_samples,). Each value is in the same
        units as the input data argument. The x_values[0] is the minimum value
        of data (0th percentile) and x_values[-1] is the maximum
        (100th percentile). The values are spaced in quantile space (y-axis of
        the CDF) according to the sampling option (e.g., evenly spaced if
        sampling='linear').
    """

    nan_mask = np.isnan(data)
    if nan_mask.all():
        return np.zeros(n_samples)

    sampling = sampling.casefold()
    if sampling == 'linear':
        quantiles = sample_q_linear(n_samples)
    elif sampling == 'log':
        quantiles = sample_q_log(n_samples, log_base)
    elif sampling == 'invlog':
        quantiles = sample_q_invlog(n_samples, log_base)
    else:
        msg = ('sampling option must be linear, log, or invlog, but received: '
               '{}'.format(sampling))
        logger.error(msg)
        raise KeyError(msg)

    x_values = np.interp(quantiles, np.linspace(0, 1, len(data[~nan_mask])),
                         sorted(data[~nan_mask]))

    msg = (f'First and last x-value points defining the CDF '
           '({x_values[0]}, {x_values[-1]}) '
           f'were not the min and max data values '
           f'({np.nanmin(data)}, {np.nanmin(data)}).')
    assert x_values[0] == np.nanmin(data), msg
    assert x_values[-1] == np.nanmax(data), msg

    if decimals is not None:
        x_values = np.round(x_values, decimals=decimals)

    return x_values


class TemporalStats:
    """
    Temporal Statistics from Resource Data
    """
    STATS = {'mean': {'func': np.nanmean, 'kwargs': {'axis': 0}},
             'median': {'func': np.nanmedian, 'kwargs': {'axis': 0}},
             'std': {'func': np.nanstd, 'kwargs': {'axis': 0}}}

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
        log_versions(logger)
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
        lat_lon_cols = get_lat_lon_cols(self.meta)

        return self.meta[lat_lon_cols]

    @staticmethod
    def _format_grp_names(grp_names):
        """
        Format groupby index values

        Parameters
        ----------
        grp_names : list
            Group by index values, these correspond to each unique group in
            the groupby

        Returns
        -------
        out : ndarray
            2D array of grp index values properly formatted as strings
        """
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                     6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                     11: 'Nov', 12: 'Dec'}

        # pylint: disable=unnecessary-lambda-assignment, unnecessary-lambda
        year = lambda s: "{}".format(s)
        month = lambda s: "{}".format(month_map[s])
        hour = lambda s: "{:02d}:00UTC".format(s)

        grp_names = np.array(grp_names).T
        if len(grp_names.shape) == 1:
            grp_names = np.expand_dims(grp_names, 0)

        out = []
        for grp_i in grp_names:  # pylint: disable=not-an-iterable
            grp_max = grp_i.max()
            if grp_max <= 12:
                out.append(list(map(month, grp_i)))
            elif grp_max <= 23:
                out.append(list(map(hour, grp_i)))
            else:
                out.append(list(map(year, grp_i)))

        return np.array(out).T

    @classmethod
    def _create_names(cls, groups, stats):
        """
        Generate statistics names

        Parameters
        ----------
        groups : list
            List of group names, some combination of year, month, hour
        stats : list
            Statistics to be computed

        Returns
        -------
        columns_map : dict
            Dictionary of column names to use for each statistic
        """
        group_names = cls._format_grp_names(groups)

        columns_map = {}
        for s in stats:
            # pylint: disable=not-an-iterable
            cols = ['{}_{}'.format('-'.join(n), s) for n
                    in group_names]
            columns_map[s] = cols

        return columns_map

    @staticmethod
    def _compute_weighted_stats(func, res_data, weights, column_names,
                                **kwargs):
        """
        Computed the weighted means using given function and kwargs

        Parameters
        ----------
        func : object
            Function to use to compute the weighted means
        res_data : pandas.DataFrame | pandas.GroupBy
            Resource data to compute weighted stats from
        weights : pandas.DataFrame | pandas.GroupBy
            Weights to use for weighted stats calculation
            Column names based on group by names, by default None
        column_names : list | str
            Either the state name or the list of output stat names,
            used out output column names.
        kwargs : dict
            Function kwargs
        """
        if isinstance(column_names, list):
            s_data = {}
            for c_name, (grp_name, res_grp) in zip(column_names, res_data):
                if weights is not None:
                    grp_w = weights.get_group(grp_name[0])
                else:
                    grp_w = None

                grp_s = func(res_grp, weights=grp_w, **kwargs)
                if grp_s.shape[0] == 1:
                    grp_s = grp_s[0]

                s_data[c_name] = grp_s

            s_data = pd.DataFrame(s_data)
        else:
            s_data = func(res_data, weights=weights, **kwargs)
            if s_data.shape[0] == 1:
                s_data = s_data[0]

            s_data = pd.DataFrame(s_data.flatten(), columns=[column_names])

        return s_data

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
        column_names = None
        if month:
            groupby.append(res_data.index.month)

        if diurnal:
            groupby.append(res_data.index.hour)

        if groupby:
            res_data = res_data.groupby(groupby)
            column_names = cls._create_names(list(res_data.groups),
                                             list(statistics))

        res_stats = []
        for name, stat in statistics.copy().items():
            func = stat['func']
            kwargs = stat.get('kwargs', {}).copy()
            if name.lower().startswith('weight'):
                weights = kwargs.pop('weights').copy()
                if groupby:
                    weights = weights.groupby(groupby)
                    weight_names = column_names[name]
                else:
                    weight_names = name

                s_data = cls._compute_weighted_stats(func, res_data, weights,
                                                     weight_names, **kwargs)
            else:
                axis = kwargs.pop('axis', 0)
                s_data = res_data.aggregate(func, axis=axis, **kwargs)

                if groupby:
                    columns = column_names[name]
                    s_data = s_data.T
                    s_data.columns = columns
                elif not isinstance(s_data, pd.DataFrame):
                    s_data = s_data.to_frame(name=name)
                elif isinstance(s_data, pd.DataFrame) and len(s_data) > 1:
                    # e.g., if func is scipy.stats.beta.fit(), this collapses
                    # multiple output parameters into list
                    s_data['name'] = name
                    s_data = s_data.groupby('name').agg(list).T

            res_stats.append(s_data)

        res_stats = pd.concat(res_stats, axis=1)

        return res_stats

    @staticmethod
    def _create_index(sites_slice):
        """
        Create index from site slice

        Parameters
        ----------
        sites_slice : slice | list | ndarray
            Sites to build index from

        Returns
        -------
        idx : list
            site gids
        """
        if isinstance(sites_slice, slice) and sites_slice.stop:
            idx = list(range(*sites_slice.indices(sites_slice.stop)))
        elif isinstance(sites_slice, (list, np.ndarray)):
            idx = sites_slice

        return idx

    @staticmethod
    def _extract_weights(res, weights_dsets, sites_slice, time_index):
        """
        Extract weights datasets from resource and combine into weights
        to use for weighted stats

        Parameters
        ----------
        res : rex.Resource
            Open Resource class or sub-class to extract datasets from
        weights_dsets : str | list | tuple
            List of weight(s) datasets to extract and combine
        sites_slice : slice
            Subslice of sites to extract weights for
        time_index : pandas.DatatimeIndex
            Resource DatetimeIndex, needed to output DataFrame Index

        Returns
        -------
        weights : pandas.DataFrame
            Weights DataFrame to match res_data
        """
        if not isinstance(weights_dsets, (list, tuple)):
            weights_dsets = [weights_dsets]

        weights = None
        for dset in weights_dsets:
            if weights is None:
                weights = res[dset, :, sites_slice]
            else:
                weights *= res[dset, :, sites_slice]

        return pd.DataFrame(weights, index=time_index)

    @classmethod
    def _extract_stats(cls, res_h5, statistics, dataset, res_cls=Resource,
                       hsds=False, time_index=None, sites_slice=None,
                       diurnal=False, month=False, combinations=False,
                       mask_zeros=False):
        """
        Extract stats for given dataset, sites, and temporal extent

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        statistics : dict
            Statistics to extract a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}}
        dataset : str
            Dataset to extract stats for
        res_cls : Class, optional
            Resource class to use to access res_h5, by default Resource
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
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

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

            if mask_zeros:
                res_data[(res_data == 0)] = np.nan

            for s, s_dict in statistics.items():
                weights = s_dict.get('kwargs', {}).get('weights')
                if weights is not None:
                    weights = cls._extract_weights(f, weights, sites_slice,
                                                   time_index)
                    statistics[s]['kwargs']['weights'] = weights

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

        res_stats.index = cls._create_index(sites_slice)
        res_stats.index.name = 'gid'

        return res_stats

    def _get_slices(self, dataset, sites=None, chunks_per_slice=5):
        """
        Get slices to extract

        Parameters
        ----------
        dataset : str
            Dataset to extract data from
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
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

        slices = slice_sites(shape, chunks, sites=sites,
                             chunks_per_slice=chunks_per_slice)

        return slices

    def _check_stats(self, statistics):
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
            statistics = {s: self.STATS[s] for s in statistics}

        for stat in statistics.values():
            msg = 'A "func"(tion) must be provided for each statistic'
            assert 'func' in stat, msg
            if 'kwargs' in stat:
                msg = 'statistic function kwargs must be a dictionary '
                assert isinstance(stat['kwargs'], dict), msg

        return statistics

    def compute_statistics(self, dataset, sites=None, diurnal=False,
                           month=False, combinations=False, max_workers=None,
                           chunks_per_worker=5, lat_lon_only=True,
                           mask_zeros=False):
        """
        Compute statistics

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        diurnal : bool, optional
            Extract diurnal stats, by default False
        month : bool, optional
            Extract monthly stats, by default False
        combinations : bool, optional
            Extract all combinations of temporal stats, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

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
                                        self.res_h5, self.statistics, dataset,
                                        res_cls=self.res_cls,
                                        hsds=self._hsds,
                                        time_index=self.time_index,
                                        sites_slice=sites_slice,
                                        diurnal=diurnal,
                                        month=month,
                                        combinations=combinations,
                                        mask_zeros=mask_zeros)
                    futures.append(future)

                res_stats = []
                for i, future in enumerate(as_completed(futures)):
                    res_stats.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))
        else:
            msg = ('Extracting {} for {} in serial'
                   .format(self.statistics.keys(), dataset))
            logger.info(msg)
            res_stats = []
            for i, sites_slice in enumerate(slices):
                res_stats.append(self._extract_stats(
                    self.res_h5, self.statistics, dataset,
                    res_cls=self.res_cls, hsds=self._hsds,
                    time_index=self.time_index, sites_slice=sites_slice,
                    diurnal=diurnal, month=month,
                    combinations=combinations,
                    mask_zeros=mask_zeros))
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        gc.collect()
        log_mem(logger)
        res_stats = pd.concat(res_stats)

        if lat_lon_only:
            meta = self.lat_lon
        else:
            meta = self.meta

        res_stats = meta.join(res_stats.sort_index(), how='inner')

        return res_stats

    def full_stats(self, dataset, sites=None, max_workers=None,
                   chunks_per_worker=5, lat_lon_only=True, mask_zeros=False):
        """
        Compute stats for entire temporal extent of file

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

        Returns
        -------
        full_stats : pandas.DataFrame
            DataFrame of statistics for the entire temporal extent of file
        """
        full_stats = self.compute_statistics(
            dataset, sites=sites,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only,
            mask_zeros=mask_zeros)

        return full_stats

    def monthly_stats(self, dataset, sites=None, max_workers=None,
                      chunks_per_worker=5, lat_lon_only=True,
                      mask_zeros=False):
        """
        Compute monthly stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

        Returns
        -------
        monthly_stats : pandas.DataFrame
            DataFrame of monthly statistics
        """
        monthly_stats = self.compute_statistics(
            dataset, sites=sites, month=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only,
            mask_zeros=mask_zeros)

        return monthly_stats

    def diurnal_stats(self, dataset, sites=None, max_workers=None,
                      chunks_per_worker=5, lat_lon_only=True,
                      mask_zeros=False):
        """
        Compute diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

        Returns
        -------
        diurnal_stats : pandas.DataFrame
            DataFrame of diurnal statistics
        """
        diurnal_stats = self.compute_statistics(
            dataset, sites=sites, diurnal=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only,
            mask_zeros=mask_zeros)

        return diurnal_stats

    def monthly_diurnal_stats(self, dataset, sites=None,
                              max_workers=None, chunks_per_worker=5,
                              lat_lon_only=True, mask_zeros=False):
        """
        Compute monthly-diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

        Returns
        -------
        monthly_diurnal_stats : pandas.DataFrame
            DataFrame of monthly-diurnal statistics
        """
        diurnal_stats = self.compute_statistics(
            dataset, sites=sites, month=True, diurnal=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only,
            mask_zeros=mask_zeros)

        return diurnal_stats

    def all_stats(self, dataset, sites=None, max_workers=None,
                  chunks_per_worker=5, lat_lon_only=True, mask_zeros=False):
        """
        Compute annual, monthly, monthly-diurnal, and diurnal stats

        Parameters
        ----------
        dataset : str
            Dataset to extract stats for
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).

        Returns
        -------
        all_diurnal_stats : pandas.DataFrame
            DataFrame of temporal statistics
        """
        all_stats = self.compute_statistics(
            dataset, sites=sites, month=True, diurnal=True, combinations=True,
            max_workers=max_workers,
            chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only,
            mask_zeros=mask_zeros)

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

        # Drop any wild card values
        out_fpath = out_fpath.replace('*', '')

        out_dir = os.path.dirname(out_fpath)
        create_dirs(out_dir)

        logger.info('Writing temporal statistics to {}'.format(out_fpath))
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
            chunks_per_worker=5, lat_lon_only=True, mask_zeros=False,
            out_path=None):
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
            (sites is synonymous with gids aka spatial indices)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).
        out_path : str, optional
            Directory, .csv, or .json path to save statistics too,
            by default None

        Returns
        -------
        out_stats : pandas.DataFrame
            DataFrame of resource statistics
        """
        logger.info('Computing temporal stats for {} in {}'
                    .format(dataset, res_h5))
        logger.debug('Computing {} using:'
                     '\n-diurnal={}'
                     '\n-month={}'
                     '\n-combinations={}'
                     '\n-max workers={}'
                     '\n-chunks per worker={}'
                     '\n-output lat lons only={}'
                     .format(statistics, diurnal, month, combinations,
                             max_workers, chunks_per_worker, lat_lon_only))
        res_stats = cls(res_h5, statistics=statistics, res_cls=res_cls,
                        hsds=hsds)
        out_stats = res_stats.compute_statistics(
            dataset, sites=sites,
            diurnal=diurnal, month=month, combinations=combinations,
            max_workers=max_workers, chunks_per_worker=chunks_per_worker,
            lat_lon_only=lat_lon_only, mask_zeros=mask_zeros)
        if out_path is not None:
            res_stats.save_stats(out_stats, out_path)

        return out_stats

    @classmethod
    def monthly(cls, res_h5, dataset, sites=None, statistics='mean',
                res_cls=Resource, hsds=False, max_workers=None,
                chunks_per_worker=5, lat_lon_only=True, mask_zeros=False,
                out_path=None):
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
            (sites is synonymous with gids aka spatial indices)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).
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
                                lat_lon_only=lat_lon_only,
                                mask_zeros=mask_zeros,
                                out_path=out_path)

        return monthly_stats

    @classmethod
    def diurnal(cls, res_h5, dataset, sites=None, statistics='mean',
                res_cls=Resource, hsds=False, max_workers=None,
                chunks_per_worker=5, lat_lon_only=True, mask_zeros=False,
                out_path=None):
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
            (sites is synonymous with gids aka spatial indices)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).
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
                                lat_lon_only=lat_lon_only,
                                mask_zeros=mask_zeros,
                                out_path=out_path)

        return diurnal_stats

    @classmethod
    def monthly_diurnal(cls, res_h5, dataset, sites=None,
                        statistics='mean', res_cls=Resource, hsds=False,
                        max_workers=None, chunks_per_worker=5,
                        lat_lon_only=True, mask_zeros=False, out_path=None):
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
            (sites is synonymous with gids aka spatial indices)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).
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
                                        mask_zeros=mask_zeros,
                                        out_path=out_path)

        return monthly_diurnal_stats

    @classmethod
    def all(cls, res_h5, dataset, sites=None, statistics='mean',
            res_cls=Resource, hsds=False, max_workers=None,
            chunks_per_worker=5, lat_lon_only=True, mask_zeros=False,
            out_path=None):
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
            (sites is synonymous with gids aka spatial indices)
        statistics : str | tuple | dict, optional
            Statistics to extract, either a key or tuple of keys in
            cls.STATS, or a dictionary of the form
            {'stat_name': {'func': *, 'kwargs: {**}}},
            by default 'mean'
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
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        lat_lon_only : bool, optional
            Only append lat, lon coordinates to stats, by default True
        mask_zeros : bool
            Flag to only calculate stats when all data is > 0 (useful for
            global horizontal irradiance).
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
                            lat_lon_only=lat_lon_only, out_path=out_path,
                            mask_zeros=mask_zeros)

        return all_stats
