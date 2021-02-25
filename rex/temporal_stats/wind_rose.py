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
from warnings import warn

from rex.renewable_resource import WindResource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem
from rex.utilities.utilities import slice_sites

logger = logging.getLogger(__name__)


class WindRose:
    """
    Compute wind rose at desired hub-height
    """
    def __init__(self, wind_h5, res_cls=WindResource, hsds=False):
        """
        Parameters
        ----------
        wind_h5 : str
            Path to resource h5 file(s)
        res_cls : Class, optional
            Resource handler class to use to access wind_h5,
            by default WindResource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        self._wind_h5 = wind_h5

        self._res_cls = res_cls
        self._hsds = hsds

    @property
    def wind_h5(self):
        """
        Path to resource h5 file(s)

        Returns
        -------
        str
        """
        return self._wind_h5

    @property
    def res_cls(self):
        """
        Resource class to use to access wind_h5

        Returns
        -------
        Class
        """
        return self._res_cls

    @staticmethod
    def compute_wind_rose(wspd, wdir, wspd_bins, wdir_bins):
        """
        Compute the wind rose using the wspd and wdir vectors using the given
        wspd and wdir bins

        Parameters
        ----------
        wspd : ndarray
            Time-series vector of wind speed
        wdir : ndarray
            Time-series vector of complimentary wind direction
        wspd_bins : ndarray
            Wind speed bin edges
        wdir_bins : ndarray
            Wind direction bin edges

        Returns
        -------
        wind_rose : ndarray
            Vector of wind rose frequencies
        """
        wind_rose = np.histogram2d(wspd, wdir,
                                   bins=(wspd_bins, wdir_bins),
                                   density=True)[0]

        return wind_rose.astype(np.float32)

    @staticmethod
    def _make_bins(start, stop, step):
        """
        Create bin edges from bin range

        Parameters
        ----------
        start : int
            bin range start value
        stop : int
            bin range stop value
        step : int
            bin range step value

        Returns
        -------
        bin_edges : ndarray
            Vector of inclusive bin edges
        """
        bin_edges = np.arange(start, stop + step, step)

        return bin_edges

    @classmethod
    def _compute_multisite_wind_rose(cls, wind_h5, hub_height,
                                     wspd_bins=(0, 30, 1),
                                     wdir_bins=(0, 360, 5),
                                     res_cls=WindResource,
                                     hsds=False, sites_slice=None):
        """
        Compute the wind rose from wind speed and direction at given hub-height
        for given sites

        Parameters
        ----------
        wind_h5 : str
            Path to resource h5 file(s)
        hub_height : str | int
            Hub-height to compute wind rose at
        wspd_bins : tuple
            (start, stop, step) for wind speed bins
        wdir_bins : tuple
            (start, stop, step) for wind direction bins
        res_cls : Class, optional
            Resource handler class to use to access wind_h5,
            by default WindResource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        sites_slice : slice | None, optional
            Sites to extract, if None all, by default None

        Returns
        -------
        wind_rose : pandas.DataFrame
            DataFrame of wind rose data at desired hub-height for given sites
        """
        if sites_slice is None:
            sites_slice = slice(None, None, None)

        wspd = 'windspeed_{}m'.format(hub_height)
        wdir = 'winddirection_{}m'.format(hub_height)
        with res_cls(wind_h5, hsds=hsds) as f:
            wspd = f[wspd, :, sites_slice]
            wdir = f[wdir, :, sites_slice]

        wspd_bins = cls._make_bins(*wspd_bins)
        wdir_bins = cls._make_bins(*wdir_bins)

        index = np.meshgrid(wspd_bins[:-1], wdir_bins[:-1], indexing='ij')
        index = np.array(index).T.reshape(-1, 2).astype(np.int16)
        index = pd.MultiIndex.from_arrays(index.T, names=('wspd', 'wdir'))

        if isinstance(sites_slice, slice) and sites_slice.stop:
            gids = list(range(*sites_slice.indices(sites_slice.stop)))
        elif isinstance(sites_slice, (list, np.ndarray)):
            gids = sites_slice

        wind_rose = {}
        for i, (ws, wd) in enumerate(zip(wspd.T, wdir.T)):
            wind_rose[gids[i]] = cls.compute_wind_rose(
                ws, wd, wspd_bins, wdir_bins).flatten(order='F')

        wind_rose = pd.DataFrame(wind_rose, index=index)

        return wind_rose

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
        with self.res_cls(self.wind_h5) as f:
            shape, _, chunks = f.get_dset_properties(dataset)

        slices = slice_sites(shape, chunks, sites=sites,
                             chunks_per_slice=chunks_per_slice)

        return slices

    def compute(self, hub_height, sites=None, wspd_bins=(0, 30, 1),
                wdir_bins=(0, 360, 5), max_workers=None,
                chunks_per_worker=5):
        """
        Compute statistics

        Parameters
        ----------
        hub_height : str | int
            Hub-height to compute wind rose at
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        wspd_bins : tuple
            (start, stop, step) for wind speed bins
        wdir_bins : tuple
            (start, stop, step) for wind direction bins
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5

        Returns
        -------
        wind_rose : pandas.DataFrame
            DataFrame of wind rose frequencies at desired hub-height
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        dataset = 'windspeed_{}m'.format(hub_height)
        slices = self._get_slices(dataset, sites,
                                  chunks_per_slice=chunks_per_worker)
        if len(slices) == 1:
            max_workers = 1

        if max_workers > 1:
            msg = ('Computing wind rose for {}m wind in parallel using {} '
                   'workers'.format(hub_height, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites_slice in slices:
                    future = exe.submit(self._compute_multisite_wind_rose,
                                        self.wind_h5, hub_height,
                                        wspd_bins=wspd_bins,
                                        wdir_bins=wdir_bins,
                                        res_cls=self.res_cls,
                                        hsds=self._hsds,
                                        sites_slice=sites_slice)
                    futures.append(future)

                wind_rose = []
                for i, future in enumerate(as_completed(futures)):
                    wind_rose.append(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

        else:
            msg = ('Computing wind rose for {}m wind in serial'
                   .format(hub_height))
            logger.info(msg)
            wind_rose = []
            for i, sites_slice in enumerate(slices):
                wind_rose.append(self._compute_multisite_wind_rose(
                    self.wind_h5, hub_height,
                    wspd_bins=wspd_bins,
                    wdir_bins=wdir_bins,
                    res_cls=self.res_cls,
                    hsds=self._hsds,
                    sites_slice=sites_slice))
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        gc.collect()
        log_mem(logger)
        wind_rose = pd.concat(wind_rose).sort_index(axis=1)

        return wind_rose

    def save_wind_rose(self, wind_rose, out_path):
        """
        Save wind rose to disk

        Parameters
        ----------
        wind_rose : pandas.DataFrame
            Table of wind_rose frequencies to save
        out_path : str
            Directory, .csv, or .json path to save wind rose and site meta too
        """

        if os.path.isdir(out_path):
            out_fpath = os.path.splitext(os.path.basename(self.wind_h5))[0]
            out_fpath = os.path.join(out_path, out_fpath + '.csv')
        else:
            out_fpath = out_path

        # Drop any wild card values
        out_fpath = out_fpath.replace('*', '')

        logger.info('Writing wind rose to {}'.format(out_fpath))
        if out_fpath.endswith('.csv'):
            wind_rose.to_csv(out_fpath)
        elif out_fpath.endswith('.json'):
            wind_rose.to_json(out_fpath)
        else:
            msg = ("Cannot save wind rose, expecting a directory, .csv, or "
                   ".json path, but got: {}".format(out_path))
            logger.error(msg)
            raise OSError(msg)

        out_fpath = out_fpath.split('.')[0] + '_meta.csv'
        if os.path.exists(out_fpath):
            msg = ("Wind rose site meta data already exists at {}!")
            logger.warning(msg)
            warn(msg)
        else:
            logger.info('Writing wind rose site meta data to {}'
                        .format(out_fpath))
            with self.res_cls(self.wind_h5) as f:
                meta = f['meta', wind_rose.columns.values]

            meta.to_csv(out_fpath, index=False)

    @classmethod
    def run(cls, wind_h5, hub_height, sites=None, wspd_bins=(0, 30, 1),
            wdir_bins=(0, 360, 5), res_cls=WindResource, hsds=False,
            max_workers=None, chunks_per_worker=5, out_path=None):
        """
        Compute temporal stats, by default full temporal extent stats

        Parameters
        ----------
        wind_h5 : str
            Path to resource h5 file(s)
        hub_height : str | int
            Hub-height to compute wind rose at
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
        wspd_bins : tuple
            (start, stop, step) for wind speed bins
        wdir_bins : tuple
            (start, stop, step) for wind direction bins
        res_cls : Class, optional
            Resource class to use to access wind_h5, by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5
        out_path : str, optional
            Directory, .csv, or .json path to save wind rose and site meta too,
            by default None

        Returns
        -------
        wind_rose : pandas.DataFrame
            DataFrame of wind rose frequencies at desired hub-height
        """
        logger.info('Computing wind rose for {}m wind in {}'
                    .format(hub_height, wind_h5))
        logger.debug('Computing wind rose using:'
                     '\n-wind speed bins: {}'
                     '\n-wind direction bins: {}'
                     '\n-max workers: {}'
                     '\n-chunks per worker: {}'
                     .format(wspd_bins, wdir_bins, max_workers,
                             chunks_per_worker))
        wind_rose = cls(wind_h5, res_cls=res_cls, hsds=hsds)
        out = wind_rose.compute(hub_height,
                                sites=sites,
                                wspd_bins=wspd_bins,
                                wdir_bins=wdir_bins,
                                max_workers=max_workers,
                                chunks_per_worker=chunks_per_worker)
        if out_path is not None:
            wind_rose.save_wind_rose(out, out_path)

        return out
