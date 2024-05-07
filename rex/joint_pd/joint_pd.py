# -*- coding: utf-8 -*-
"""
General Joint Probabilty Distribution calculator
"""
from concurrent.futures import as_completed
import gc
import logging
import h5py
import numpy as np
import os
import pandas as pd
from warnings import warn

from rex.version import __version__
from rex.renewable_resource import WindResource
from rex.resource import Resource
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_mem, log_versions
from rex.utilities.utilities import slice_sites, to_records_array

logger = logging.getLogger(__name__)


class JointPD:
    """
    Compute the joint probability distribution between the desired variables
    """
    def __init__(self, res_h5, res_cls=Resource, hsds=False):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        res_cls : Class, optional
            Resource handler class to use to access res_h5,
            by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        """
        log_versions(logger)
        self._res_h5 = res_h5
        self._res_cls = res_cls
        self._hsds = hsds

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
    def res_cls(self):
        """
        Resource class to use to access wind_h5

        Returns
        -------
        Class
        """
        return self._res_cls

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
    def compute_joint_pd(cls, res_h5, dset1, dset2, bins1, bins2,
                         res_cls=Resource, hsds=False,
                         sites_slice=None):
        """
        Compute the joint probability distribution between the two given
        datasets using the given bins for given sites

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dset1 : str
            Dataset 1 to generate joint probability distribution for
        dset2 : str
            Dataset 2 to generate joint probabilty distribution for
        bins1 : tuple
            (start, stop, step) for dataset 1 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        bins2 : tuple
            (start, stop, step) for dataset 2 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        res_cls : Class, optional
            Resource handler class to use to access res_h5,
            by default Resource
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        sites_slice : slice | None, optional
            Sites to extract, if None all, by default None
            (sites is synonymous with gids aka spatial indices)

        Returns
        -------
        jpd : dict
            Dictionary of joint probabilty distribution densities for given
            sites
        """
        if sites_slice is None:
            sites_slice = slice(None, None, None)
        elif isinstance(sites_slice, int):
            sites_slice = [sites_slice]

        with res_cls(res_h5, hsds=hsds) as f:
            arr1 = f[dset1, :, sites_slice]
            arr2 = f[dset2, :, sites_slice]

        bins1 = cls._make_bins(*bins1)
        bins2 = cls._make_bins(*bins2)

        if isinstance(sites_slice, slice) and sites_slice.stop:
            gids = list(range(*sites_slice.indices(sites_slice.stop)))
        elif isinstance(sites_slice, (list, np.ndarray)):
            gids = sites_slice

        jpd = {}
        for i, (a1, a2) in enumerate(zip(arr1.T, arr2.T)):
            jpd[gids[i]] = np.histogram2d(a1, a2,
                                          bins=(bins1, bins2),
                                          density=True)[0].astype(np.float32)

        return jpd

    def _get_slices(self, dset1, dset2, sites=None, chunks_per_slice=5):
        """
        Get slices to extract, ensure the shapes of dset1 and 2 match.

        Parameters
        ----------
        dset1 : str
            Dataset 1 to generate joint probability distribution for
        dset2 : str
            Dataset 2 to generate joint probabilty distribution for
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
            shape, _, chunks = f.get_dset_properties(dset1)
            shape2, _, _ = f.get_dset_properties(dset2)

        if shape != shape2:
            msg = ("The shape of {}: {}, does not match the shape of {}: {}!"
                   .format(dset1, shape, dset2, shape2))
            logger.error(msg)
            raise RuntimeError(msg)

        slices = slice_sites(shape, chunks, sites=sites,
                             chunks_per_slice=chunks_per_slice)

        return slices

    def compute(self, dset1, dset2, bins1, bins2, sites=None, max_workers=None,
                chunks_per_worker=5):
        """
        Compute joint probability distribution between given datasets using
        given bins for all sites.

        Parameters
        ----------
        dset1 : str
            Dataset 1 to generate joint probability distribution for
        dset2 : str
            Dataset 2 to generate joint probabilty distribution for
        bins1 : tuple
            (start, stop, step) for dataset 1 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        bins2 : tuple
            (start, stop, step) for dataset 2 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_worker : int, optional
            Number of chunks to extract on each worker, by default 5

        Returns
        -------
        jpd: pandas.DataFrame
            DataFrame of joint probability distribution between given datasets
            with given bins
        """
        if max_workers is None:
            max_workers = os.cpu_count()

        slices = self._get_slices(dset1, dset2, sites,
                                  chunks_per_slice=chunks_per_worker)
        if len(slices) == 1:
            max_workers = 1

        jpd = {}
        if max_workers > 1:
            msg = ('Computing the joint probability distribution between {} '
                   'and {} in parallel using {} workers'
                   .format(dset1, dset2, max_workers))
            logger.info(msg)

            loggers = [__name__, 'rex']
            with SpawnProcessPool(max_workers=max_workers,
                                  loggers=loggers) as exe:
                futures = []
                for sites_slice in slices:
                    future = exe.submit(self.compute_joint_pd,
                                        self.res_h5, dset1, dset2,
                                        bins1, bins2,
                                        res_cls=self.res_cls,
                                        hsds=self._hsds,
                                        sites_slice=sites_slice)
                    futures.append(future)

                for i, future in enumerate(as_completed(futures)):
                    jpd.update(future.result())
                    logger.debug('Completed {} out of {} workers'
                                 .format((i + 1), len(futures)))

        else:
            msg = ('Computing the joint probability distribution between {} '
                   'and {} in serial.'
                   .format(dset1, dset2))
            logger.info(msg)
            for i, sites_slice in enumerate(slices):
                jpd.update(self.compute_joint_pd(
                    self.res_h5, dset1, dset2,
                    bins1, bins2,
                    res_cls=self.res_cls,
                    hsds=self._hsds,
                    sites_slice=sites_slice))
                logger.debug('Completed {} out of {} sets of sites'
                             .format((i + 1), len(slices)))

        gc.collect()
        log_mem(logger)
        bins1 = self._make_bins(*bins1)
        bins2 = self._make_bins(*bins2)
        index = np.meshgrid(bins1[:-1], bins2[:-1], indexing='ij')
        index = np.array(index).T.reshape(-1, 2).astype(np.int16)
        index = pd.MultiIndex.from_arrays(index.T, names=(dset1, dset2))
        jpd = pd.DataFrame({k: v.flatten(order='F') for k, v
                            in jpd.items()}, index=index).sort_index(axis=1)

        return jpd

    def save(self, jpd, out_fpath):
        """
        Save joint probability distribution to disk

        Parameters
        ----------
        jpd : pandas.DataFrame
            Table of joint probability distribution densities to save
        out_fpath : str
            .csv, or .h5 file path to save joint probability
            distribution to
        """
        with self.res_cls(self.res_h5) as f:
            meta = f['meta', jpd.columns.values]

        logger.info('Writing joint probability distribution to {}'
                    .format(out_fpath))
        if out_fpath.endswith('.csv'):
            jpd.to_csv(out_fpath)
            meta_fpath = out_fpath.split('.')[0] + '_meta.csv'
            if os.path.exists(meta_fpath):
                msg = ("Site meta data already exists at {}!")
                logger.warning(msg)
                warn(msg)
            else:
                logger.debug('Writing site meta data to {}'
                             .format(meta_fpath))
                meta.to_csv(meta_fpath, index=False)
        elif out_fpath.endswith('.h5'):
            with h5py.File(out_fpath, mode='w') as f:
                f.attrs['rex version'] = __version__
                for i, n in enumerate(jpd.index.names):
                    logger.info('')
                    data = np.array(jpd.index.get_level_values(i))
                    dset = '{}-bins'.format(n)
                    logger.debug('Writing {}'.format(dset))
                    f.create_dataset(dset, shape=data.shape, dtype=data.dtype,
                                     data=data)

                logger.debug('Writing joint probability density values to jpd')
                data = jpd.values
                f.create_dataset('jpd', shape=data.shape, dtype=data.dtype,
                                 data=data)

                logger.debug('Writing site meta data to meta')
                meta = to_records_array(meta)
                f.create_dataset('meta', shape=meta.shape, dtype=meta.dtype,
                                 data=meta)
        else:
            msg = ("Cannot save joint probability distribution, expecting "
                   ".csv or .h5 path, but got: {}".format(out_fpath))
            logger.error(msg)
            raise OSError(msg)

    @staticmethod
    def plot_joint_pd(jpd, site=None, **kwargs):
        """
        Plot the mean joint probability distribution accross all sites
        (site=None), or the distribution for the single given site

        Parameters
        ----------
        jpd: pandas.DataFrame
            DataFrame of joint probability distribution between given datasets
            with given bins
        site : int, optional
            Site to plot distribution for, if None plot mean distribution
            across all sites, by default None
        """
        x, y = jpd.index.names
        if site is not None:
            msg = ("Can only plot the joint probabilty distribution for a "
                   "single site or the mean probability distribution accross "
                   "all sites (site=None), you provided: {}".format(site))
            assert isinstance(site), msg
            plt = jpd.loc[:, [site]].reset_index()
        else:
            site = 'mean_jpd'
            plt = jpd.mean(axis=1)
            plt.name = site
            plt = plt.reset_index()

        plt.plot.scatter(x=x, y=y, c=site, **kwargs)

    @classmethod
    def run(cls, res_h5, dset1, dset2, bins1, bins2,
            sites=None, res_cls=Resource, hsds=False,
            max_workers=None, chunks_per_worker=5, out_fpath=None):
        """
        Compute joint probability distribution between given datasets using
        given bins

        Parameters
        ----------
        res_h5 : str
            Path to resource h5 file(s)
        dset1 : str
            Dataset 1 to generate joint probability distribution for
        dset2 : str
            Dataset 2 to generate joint probabilty distribution for
        bins1 : tuple
            (start, stop, step) for dataset 1 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        bins2 : tuple
            (start, stop, step) for dataset 2 bins. The stop value is
            inclusive, so (0, 6, 2) would yield three bins with edges (0, 2, 4,
            6). If the stop value is not perfectly divisible by the step, the
            last bin will overshoot the stop value.
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
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
        out_fpath : str, optional
            .csv, or .h5 file path to save joint probability
            distribution to

        Returns
        -------
        out : pandas.DataFrame
            DataFrame of joint probability distribution between given datasets
            with given bins
        """
        logger.info('Computing joint probability distribution between {} and '
                    '{} in {}'
                    .format(dset1, dset2, res_h5))
        logger.debug('Computing joint probability distribution using:'
                     '\n-{} bins: {}'
                     '\n-{} bins: {}'
                     '\n-max workers: {}'
                     '\n-chunks per worker: {}'
                     .format(dset1, bins1, dset2, bins2, max_workers,
                             chunks_per_worker))
        jpd = cls(res_h5, res_cls=res_cls, hsds=hsds)
        out = jpd.compute(dset1, dset2, bins1, bins2,
                          sites=sites,
                          max_workers=max_workers,
                          chunks_per_worker=chunks_per_worker)
        if out_fpath is not None:
            jpd.save(out, out_fpath)

        return out

    @classmethod
    def wind_rose(cls, wind_h5, hub_height, wspd_bins=(0, 30, 1),
                  wdir_bins=(0, 360, 5), sites=None, res_cls=WindResource,
                  hsds=False, max_workers=None, chunks_per_worker=5,
                  out_fpath=None):
        """
        Compute wind rose at given hub height

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
        sites : list | slice, optional
            Subset of sites to extract, by default None or all sites
            (sites is synonymous with gids aka spatial indices)
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
        wspd_dset = 'windspeed_{}m'.format(hub_height)
        wdir_dset = 'winddirection_{}m'.format(hub_height)
        out = wind_rose.compute(wspd_dset, wdir_dset, wspd_bins, wdir_bins,
                                sites=sites,
                                max_workers=max_workers,
                                chunks_per_worker=chunks_per_worker)
        if out_fpath is not None:
            wind_rose.save(out, out_fpath)

        return out
