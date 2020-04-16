# -*- coding: utf-8 -*-
"""
Resource Extraction Tools
"""
import gzip
import logging
import numpy as np
import os
import pandas as pd
import pickle
from scipy.spatial import cKDTree

from rex.resource import Resource, MultiFileResource
from rex.renewable_resource import (MultiFileWTK, MultiFileNSRDB, NSRDB,
                                    SolarResource, WindResource)

TREE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trees')
logger = logging.getLogger(__name__)


class ResourceX(Resource):
    """
    Resource data extraction tool
    """
    def __init__(self, res_h5, tree=None, unscale=True, hsds=False,
                 str_decode=True, group=None):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource .h5 file of interest
        tree : str
            Path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        super().__init__(res_h5, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, group=group)
        self._tree = tree

    @property
    def tree(self):
        """
        Returns
        -------
        tree : cKDTree
            Lat, lon coordinates cKDTree
        """
        if not isinstance(self._tree, cKDTree):
            self._tree = self._init_tree(tree=self._tree)

        return self._tree

    @property
    def countries(self):
        """
        Returns
        -------
        countries : ndarray
            Countries available in .h5 file
        """
        if 'country' in self.meta:
            countries = self.meta['country'].unique()
        else:
            countries = None

        return countries

    @property
    def states(self):
        """
        Returns
        -------
        states : ndarray
            States available in .h5 file
        """
        if 'state' in self.meta:
            states = self.meta['state'].unique()
        else:
            states = None

        return states

    @property
    def counties(self):
        """
        Returns
        -------
        counties : ndarray
            Counties available in .h5 file
        """
        if 'county' in self.meta:
            counties = self.meta['county'].unique()
        else:
            counties = None

        return counties

    @staticmethod
    def _load_tree(tree_pickle):
        """
        Load tree from pickle file

        Parameters
        ----------
        tree_pickle : str
            Pickle (.pkl, .pickle) or compressed pickle (.pgz, .pgzip) file
            containing precomputed cKDTree

        Returns
        -------
        tree : cKDTree
            Precomputed tree of lat, lon coordinates
        """
        try:
            if tree_pickle.endswith(('.pkl', '.pickle')):
                with open(tree_pickle, 'rb') as f:
                    tree = pickle.load(f)
            elif tree_pickle.endswith(('.pgz', '.pgzip', '.gz', '.gzip')):
                with gzip.open(tree_pickle, 'r') as f:
                    tree = pickle.load(f)
            else:
                logger.warning('Cannot parse files of type "{}"'
                               .format(tree_pickle))
                tree = None
        except Exception as e:
            logger.warning('Could not extract tree from {}: {}'
                           .format(tree_pickle, e))
            tree = None

        return tree

    def _init_tree(self, tree=None, compute_tree=False):
        """
        Inititialize cKDTree of lat, lon coordinates

        Parameters
        ----------
        tree : str | cKDTree | NoneType
            Path to .pgz file containing pre-computed tree
            If None search bin for .pgz file matching h5 file
            else compute tree
        compute_tree : bool
            Force the computation of the cKDTree

        Returns
        -------
        tree : cKDTree
            cKDTree of lat, lon coordinate from wtk .h5 file
        """
        if compute_tree:
            tree = None
        else:
            if not isinstance(tree, (cKDTree, str, type(None))):
                tree = None
                logger.warning('Precomputed tree must be supplied as a pickle '
                               'file or a cKDTree, not a {}'
                               .format(type(tree)))

            if tree is None:
                pgz_files = [file for file in os.listdir(TREE_DIR)
                             if file.endswith('.pgz')]
                for pgz in pgz_files:
                    prefix = pgz.split('_tree')[0]
                    if self.h5_file.startswith(prefix):
                        tree = os.path.join(TREE_DIR, pgz)
                        break

            if isinstance(tree, str):
                tree = self._load_tree(tree)

        if tree is None:
            lat_lon = self.meta[['latitude', 'longitude']].values
            tree = cKDTree(lat_lon)

        return tree

    def _get_nearest(self, lat_lon):
        """
        Get nearest gid to given (lat, lon) pair or pairs

        Parameters
        ----------
        lat_lon : ndarray
            Either a single (lat, lon) pair or series of (lat, lon) pairs

        Returns
        -------
        gids : int | ndarray
            Nearest gid(s) to given (lat, lon) pair(s)
        """
        _, gids = self.tree.query(lat_lon)
        return gids

    def _get_region(self, region, region_col='state'):
        """
        Get the gids for given region

        Parameters
        ----------
        region : str
            Region to search for
        region_col : str
            Region column to search

        Returns
        -------
        gids : ndarray
            Vector of gids in given region
        """
        gids = (self.meta[region_col] == region).index.values
        return gids

    def _get_timestep_idx(self, timestep):
        """
        Get the index of the desired timestep

        Parameters
        ----------
        timestep : str
            Timestep of interest

        Returns
        -------
        ts_idx : int
            Time index value
        """
        timestep = pd.to_datetime(timestep)
        idx = np.where(self.time_index == timestep)[0][0]

        return idx

    def get_gid_ts(self, ds_name, gid):
        """
        Extract timeseries of site(s) neareset to given lat_lon(s)

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        gid : int | list
            Resource gid(s) of interset

        Return
        ------
        site_ts : ndarray
            Time-series for given site(s) and dataset
        """
        site_ts = self[ds_name, :, gid]

        return site_ts

    def get_gid_df(self, ds_name, gid):
        """
        Extract timeseries of site(s) nearest to given lat_lon(s) and return
        as a DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        gid : int | list
            Resource gid(s) of interset

        Return
        ------
        site_df : pandas.DataFrame
            Time-series DataFrame for given site and dataset
        """
        if isinstance(gid, int):
            site_df = pd.DataFrame({ds_name: self[ds_name, :, gid]},
                                   index=self.time_index)
            site_df.name = gid
            site_df.index.name = 'time_index'
        else:
            site_df = pd.DataFrame(self[ds_name, :, gid], columns=gid,
                                   index=self.time_index)
            site_df.name = ds_name
            site_df.index.name = 'time_index'

        return site_df

    def get_lat_lon_ts(self, ds_name, lat_lon):
        """
        Extract timeseries of site(s) neareset to given lat_lon(s)

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple | list
            (lat, lon) coordinate of interest or pairs of coordinates

        Return
        ------
        site_ts : ndarray
            Time-series for given site(s) and dataset
        """
        gid = self._get_nearest(lat_lon)
        site_ts = self.get_gid_ts(ds_name, gid)

        return site_ts

    def get_lat_lon_df(self, ds_name, lat_lon):
        """
        Extract timeseries of site(s) nearest to given lat_lon(s) and return
        as a DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple
            (lat, lon) coordinate of interest

        Return
        ------
        site_df : pandas.DataFrame
            Time-series DataFrame for given site and dataset
        """
        gid = self._get_nearest(lat_lon)
        site_df = self.get_gid_df(ds_name, gid)

        return site_df

    def get_region_ts(self, ds_name, region, region_col='state'):
        """
        Extract timeseries of of all sites in given region

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        region : str
            Region to search for
        region_col : str
            Region column to search

        Return
        ------
        region_ts : ndarray
            Time-series array of desired dataset for all sites in desired
            region
        """
        gids = self._get_region(region, region_col=region_col)
        region_ts = self[ds_name, :, gids]

        return region_ts

    def get_region_df(self, ds_name, region, region_col='state'):
        """
        Extract timeseries of of all sites in given region and return as a
        DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        region : str
            Region to extract all pixels for
        region_col : str
            Region column to search

        Return
        ------
        region_df : pandas.DataFrame
            Time-series array of desired dataset for all sites in desired
            region
        """
        gids = self._get_region(region, region_col=region_col)
        region_df = pd.DataFrame(self[ds_name, :, gids], columns=gids,
                                 index=self.time_index)
        region_df.name = ds_name
        region_df.index.name = 'time_index'

        return region_df

    def get_SAM_gid(self, gid):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given resource gid

        Parameters
        ----------
        gid : int | list
            Resource gid(s) of interset

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        if isinstance(gid, int):
            gid = [gid, ]

        SAM_df = []
        for id in gid:
            SAM_df.append(self['SAM', id])

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_SAM_lat_lon(self, lat_lon):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon

        Parameters
        ----------
        lat_lon : tuple
            (lat, lon) coordinate of interest

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gid = self._get_nearest(lat_lon)
        SAM_df = self.get_SAM_gid(gid)

        return SAM_df

    def get_timestep_map(self, ds_name, timestep, region=None,
                         region_col='state'):
        """
        Extract a map of the given dataset at the given timestep for the
        given region if supplied

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        timestep : str
            Timestep of interest
        region : str
            Region to extract all pixels for
        region_col : str
            Region column to search

        Returns
        -------
        ts_map : pandas.DataFrame
            DataFrame of map values
        """
        lat_lons = self.meta[['latitude', 'longitude']].values
        ts_idx = self._get_timestep_idx(timestep)
        gids = slice(None)
        if region is not None:
            gids = self._get_region(region, region_col=region_col)
            lat_lons = lat_lons[gids]

        ts_map = self[ds_name, ts_idx, gids]
        ts_map = pd.DataFrame({'longitude': lat_lons[:, 1],
                               'latitude': lat_lons[:, 0],
                               ds_name: ts_map})

        return ts_map


class MultiFileResourceX(MultiFileResource, ResourceX):
    """
    Multi-File resource extraction class
    """

    def __init__(self, resource_path, tree=None, compute_tree=False,
                 unscale=True, str_decode=True):
        """
        Parameters
        ----------
        resource_path : str
            Path to resource .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        super().__init__(resource_path, unscale=unscale, str_decode=str_decode)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)


class SolarX(SolarResource, ResourceX):
    """
    Solar Resource extraction class
    """
    def __init__(self, solar_h5, tree=None, compute_tree=False, unscale=True,
                 hsds=False, str_decode=True, group=None):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        super().__init__(solar_h5, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, group=group)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)


class NSRDBX(NSRDB, ResourceX):
    """
    NSRDB extraction class
    """
    def __init__(self, nsrdb_h5, tree=None, compute_tree=False, unscale=True,
                 hsds=False, str_decode=True, group=None):
        """
        Parameters
        ----------
        nsrdb_h5 : str
            Path to NSRDB .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        """
        super().__init__(nsrdb_h5, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, group=group)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)


class MultiFileNSRDBX(MultiFileNSRDB, ResourceX):
    """
    Multi-File NSRDB extraction class
    """
    def __init__(self, nsrdb_path, tree=None, compute_tree=False,
                 unscale=True, str_decode=True):
        """
        Parameters
        ----------
        nsrdb_path : str
            Path to NSRDB .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        super().__init__(nsrdb_path, unscale=unscale, str_decode=str_decode)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)


class WindX(WindResource, ResourceX):
    """
    Wind Resource extraction class
    """
    def __init__(self, wind_h5, tree=None, compute_tree=False, unscale=True,
                 hsds=False, str_decode=True, group=None):
        """
        Parameters
        ----------
        wind_h5 : str
            Path to Wind .h5 file of interest
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        kwargs : dict
            Kwargs for Resource
        """
        super().__init__(wind_h5, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, group=group)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)

    def get_SAM_gid(self, hub_height, gid, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given resource gid and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        gid : int | list
            Resource gid(s) of interset
        kwargs : dict
            Internal kwargs for _get_SAM_df:
            - require_wind_dir
            - icing

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        ds_name = 'SAM_{}m'.format(hub_height)
        if isinstance(gid, int):
            gid = [gid, ]

        SAM_df = []
        for id in gid:
            SAM_df.append(self._get_SAM_df(ds_name, id, **kwargs))

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_SAM_lat_lon(self, hub_height, lat_lon):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        gid : int | list
            Resource gid(s) of interset
        kwargs : dict
            Internal kwargs for _get_SAM_df:
            - require_wind_dir
            - icing

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gid = self._get_nearest(lat_lon)
        SAM_df = self.get_SAM_gid(hub_height, gid)

        return SAM_df


class MultiFileWindX(MultiFileWTK, WindX):
    """
    Multi-File Wind Resource extraction class
    """
    def __init__(self, wtk_path, tree=None, compute_tree=False,
                 unscale=True, str_decode=True):
        """
        Parameters
        ----------
        wtk_path : str
            Path to five minute WTK .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str
            path to .pgz file containing pickled cKDTree of lat, lon
            coordinates
        compute_tree : bool
            Force the computation of the cKDTree
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        """
        super().__init__(wtk_path, unscale=unscale, str_decode=str_decode)
        self._tree = self._init_tree(tree=tree, compute_tree=compute_tree)
