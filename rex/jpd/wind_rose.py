# -*- coding: utf-8 -*-
"""
Wind Rose (wspd - wdir JPD) calculator
"""
from concurrent.futures import as_completed
import gc
import logging
import numpy as np
import os
import pandas as pd

from rex.jpd.jpd import JPD
from rex.renewable_resource import WindResource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem

logger = logging.getLogger(__name__)


class WindRose(JPD):
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
        super().__init__(wind_h5, res_cls=WindResource, hsds=hsds)

    def compute(self, hub_height, sites=None, wspd_bins=(0, 30, 1),
                wdir_bins=(0, 360, 5), max_workers=None,
                chunks_per_worker=5):
        """
        Compute wind rose at given hubheight

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

        wspd = 'windspeed_{}m'.format(hub_height)
        wdir = 'winddirection_{}m'.format(hub_height)
        slices = self._get_slices(wspd, wdir, sites,
                                  chunks_per_slice=chunks_per_worker)
        if len(slices) == 1:
            max_workers = 1

        wind_rose = {}
        if max_workers > 1:
            msg = ('Computing wind rose for {}m wind in parallel using {} '
                   'workers'.format(hub_height, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites_slice in slices:
                    future = exe.submit(self._compute_multisite_jpd,
                                        self.res_h5, wspd, wdir,
                                        wspd_bins, wdir_bins,
                                        res_cls=self.res_cls,
                                        hsds=self._hsds,
                                        sites_slice=sites_slice)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    wind_rose.update(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

        else:
            msg = ('Computing wind rose for {}m wind in serial'
                   .format(hub_height))
            logger.info(msg)
            for i, sites_slice in enumerate(slices):
                wind_rose.update(self._compute_multisite_jpd(
                    self.res_h5, wspd, wdir, wspd_bins, wdir_bins,
                    res_cls=self.res_cls,
                    hsds=self._hsds,
                    sites_slice=sites_slice))
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        gc.collect()
        log_mem(logger)
        wspd_bins = self._make_bins(*wspd_bins)
        wdir_bins = self._make_bins(*wdir_bins)
        index = np.meshgrid(wspd_bins[:-1], wdir_bins[:-1], indexing='ij')
        index = np.array(index).T.reshape(-1, 2).astype(np.int16)
        index = pd.MultiIndex.from_arrays(index.T, names=('wspd', 'wdir'))
        wind_rose = pd.DataFrame(wind_rose, index=index).sort_index(axis=1)

        return wind_rose

    @classmethod
    def run(cls, wind_h5, hub_height, sites=None, wspd_bins=(0, 30, 1),
            wdir_bins=(0, 360, 5), res_cls=WindResource, hsds=False,
            max_workers=None, chunks_per_worker=5, out_fpath=None):
        """
        Compute wind rose at given hub height

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
        out_fpath : str, optional
            .csv, or .h5 file path to save wind rose to

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
        if out_fpath is not None:
            wind_rose.save(out, out_fpath)

        return out
