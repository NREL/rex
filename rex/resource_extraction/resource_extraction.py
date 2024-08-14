# -*- coding: utf-8 -*-
"""
Resource Extraction Tools
"""
import copy
import logging
import os
import pickle
from tempfile import TemporaryDirectory
from warnings import warn

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from rex.multi_file_resource import (
    MultiFileNSRDB,
    MultiFileResource,
    MultiFileWTK,
)
from rex.multi_time_resource import MultiTimeResource
from rex.multi_year_resource import MultiYearResource
from rex.renewable_resource import (
    NSRDB,
    SolarResource,
    WaveResource,
    WindResource,
)
from rex.resource import Resource, ResourceDataset, BaseDatasetIterable
from rex.temporal_stats.temporal_stats import TemporalStats
from rex.utilities.exceptions import ResourceValueError, ResourceWarning
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.loggers import log_versions
from rex.utilities.utilities import check_tz, parse_year, res_dist_threshold

# pylint: disable=consider-using-with
TREE_DIR = TemporaryDirectory()
logger = logging.getLogger(__name__)


class ResourceX(BaseDatasetIterable):
    """
    Resource data extraction tool
    """

    DEFAULT_RES_CLS = Resource

    def __init__(self, res_h5, res_cls=None, tree=None, unscale=True,
                 str_decode=True, group=None, hsds=False, hsds_kwargs=None,
                 log_vers=True):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource .h5 file of interest
        res_cls : obj, optional
            Resource class to use to open and access resource data,
            by default Resource (default changes for subclasses like NSRDBX)
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        log_vers : bool
            Flag to log rex versions, True by default. Disable this if wrapping
            in a parallel process (logs get very verbose).
        """

        if log_vers:
            log_versions(logger)

        res_cls = self.DEFAULT_RES_CLS if res_cls is None else res_cls
        self._res = res_cls(res_h5, unscale=unscale, str_decode=str_decode,
                            group=group, hsds=hsds, hsds_kwargs=hsds_kwargs)
        self._dist_thresh = None
        self._tree = tree

    def __repr__(self):
        msg = "{} extractor for {}".format(self._res.__class__.__name__,
                                           self.resource.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return len(self.resource)

    def __getitem__(self, keys):
        return self.resource[keys]

    def __contains__(self, dset):
        return dset in self.datasets

    @property
    def resource(self):
        """
        Open res_cls instance to access res_h5 data

        Returns
        -------
        res_cls : rex.resource.Resource | rex.renewable_resource.*
        """
        return self._res

    @property
    def h5(self):
        """
        Open h5py File instance. If _group is not None return open Group

        Returns
        -------
        h5 : h5py.File | h5py.Group
        """
        return self.resource.h5

    @property
    def datasets(self):
        """
        Datasets available

        Returns
        -------
        list
        """
        return self.resource.datasets

    @property
    def dsets(self):
        """
        Datasets available

        Returns
        -------
        list
        """
        return self.datasets

    @property
    def resource_datasets(self):
        """
        Available resource datasets

        Returns
        -------
        list
        """
        return self.resource.resource_datasets

    @property
    def res_dsets(self):
        """
        Available resource datasets

        Returns
        -------
        list
        """
        return self.resource_datasets

    @property
    def groups(self):
        """
        Groups available

        Returns
        -------
        groups : list
        """
        return self.resource.groups

    @property
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
        """
        return self.resource.shape

    @property
    def meta(self):
        """
        Resource meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
        """
        return self.resource.meta

    @property
    def time_index(self):
        """
        Resource DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
        """
        return self.resource.time_index

    @property
    def coordinates(self):
        """
        Coordinates: (lat, lon) pairs

        Returns
        -------
        lat_lon : ndarray
        """
        return self.resource.lat_lon

    @property
    def lat_lon(self):
        """
        Extract (latitude, longitude) pairs

        Returns
        -------
        lat_lon : ndarray
        """
        return self.resource.lat_lon

    @property
    def data_version(self):
        """
        Get the version attribute of the data. None if not available.

        Returns
        -------
        version : str | None
        """
        return self.resource.data_version

    @property
    def global_attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return self.resource.global_attrs

    @property
    def attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        attrs : dict
        """
        return self.resource.attrs

    @property
    def tree(self):
        """
        Pre-initialized cKDTree on the resource lat, lon coordinates

        Returns
        -------
        tree : cKDTree
        """
        if not isinstance(self._tree, cKDTree):
            self._tree = self._init_tree(self._tree)

        return self._tree

    @property
    def distance_threshold(self):
        """
        Distance threshold, calculated as half of the diagonal between closest
        resource points, with an extra 5% margin

        Returns
        -------
        float
        """
        if self._dist_thresh is None:
            self._dist_thresh = res_dist_threshold(self._res.lat_lon,
                                                   tree=self.tree)

        return self._dist_thresh

    @property
    def countries(self):
        """
        Available Countires

        Returns
        -------
        countries : ndarray
        """
        if 'country' in self.meta:
            countries = self.meta['country'].unique()
        else:
            countries = None

        return countries

    @property
    def states(self):
        """
        Available states

        Returns
        -------
        states : ndarray
        """
        if 'state' in self.meta:
            states = self.meta['state'].unique()
        else:
            states = None

        return states

    @property
    def counties(self):
        """
        Available Counties

        Returns
        -------
        counties : ndarray
        """
        if 'county' in self.meta:
            counties = self.meta['county'].unique()
        else:
            counties = None

        return counties

    @staticmethod
    def _get_tree_file(h5_file):
        """
        Create path to pre-computed tree from h5_file by splitting file name
        at year if available, else replacing the .h5 suffix

        Parameters
        ----------
        h5_file : str
            Path to source .h5 file

        Returns
        -------
        tree_file : str
            Path to pre-comupted tree .pkl file name
        """
        f_name = os.path.basename(h5_file)
        try:
            year = parse_year(f_name)
            tree_file = f_name.split(str(year))[0] + 'tree.pkl'
        except RuntimeError:
            tree_file = f_name.replace('.h5', '_tree.pkl')

        return tree_file

    @staticmethod
    def _load_tree(tree_path):
        """
        Load tree from pickle file

        Parameters
        ----------
        tree_path : str
            Pickle (.pkl, .pickle) file containing precomputed cKDTree

        Returns
        -------
        tree : cKDTree
            Precomputed tree of lat, lon coordinates
        """
        try:
            with open(tree_path, 'rb') as f:
                tree = pickle.load(f)
        except Exception as e:
            logger.warning('Could not extract tree from {}: {}'
                           .format(tree_path, e))
            tree = None

        return tree

    @staticmethod
    def _save_tree(tree, tree_path):
        """
        Save pre-computed Tree to TEMP_DIR as a pickle file

        Parameters
        ----------
        tree : cKDTree
            pre-computed cKDTree
        tree_path : str
            Path to pickle file in TEMP_DIR to save tree too
        """
        try:
            with open(tree_path, 'wb') as f:
                pickle.dump(tree, f)
        except Exception as e:
            logger.warning('Could not save tree to {}: {}'
                           .format(tree_path, e))

    @staticmethod
    def _get_ds_slice(dset, gids):
        """
        Get dataset region slice

        Parameters
        ----------
        dset : str
            Dataset to extract region from
        gids : ndarray | list
            Gids associated with region

        Returns
        -------
        ds_slice : tuple
            ds slice tuple to properly extract region from given dataset
        """
        if dset == 'time_index':
            ds_slice = (slice(None), )
        elif dset in ['coordinates', 'meta']:
            ds_slice = (gids, )
        else:
            ds_slice = (slice(None), gids)

        return ds_slice

    @staticmethod
    def _to_SAM_csv(sam_df, site_meta, out_path, write_time=True):
        """
        Save SAM dataframe to disk and add meta data to header to make
        SAM compliant

        Parameters
        ----------
        sam_df : pandas.DataFrame
            rex SAM DataFrame
        site_meta : pandas.DataFrame
            Site meta data
        out_path : str
            Path to .csv file to save data too
        write_time : bool
            Flag to write the time columns (Year, Month, Day, Hour, Minute)
        """
        if not out_path.endswith('.csv'):
            if os.path.isfile(out_path):
                out_path = os.path.basename(out_path)

            out_path = os.path.join(out_path, "{}.csv".format(sam_df.name))

        if write_time:
            sam_df.to_csv(out_path, index=False)
        else:
            time_cols = ('year', 'month', 'day', 'hour', 'minute')
            cols = [c for c in sam_df if c.lower() not in time_cols]
            sam_df[cols].to_csv(out_path, index=False)

        if 'gid' not in site_meta:
            site_meta.index.name = 'gid'
            site_meta = site_meta.reset_index()

        col_map = {}
        for c in site_meta.columns:
            if c.lower() == 'timezone':
                col_map[c] = 'Time Zone'
            elif c.lower() == 'gid':
                col_map[c] = 'Location ID'
            elif c.islower():
                col_map[c] = c.capitalize()

        site_meta = site_meta.rename(columns=col_map)
        cols = ','.join(site_meta.columns)
        values = site_meta.values[0].astype(str)
        values = ','.join([value.replace(',', '') for value in values])
        values = values.replace('\n', '').replace('\r', '').replace('\t', '')

        with open(out_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(cols + '\n' + values + '\n' + content)

    def _init_tree(self, tree):
        """
        Inititialize cKDTree of lat, lon coordinates

        Parameters
        ----------
        tree : str | cKDTree | NoneType
            Path to .pgz file containing pre-computed tree
            If None search bin for .pgz file matching h5 file
            else compute tree

        Returns
        -------
        tree : cKDTree
            cKDTree of lat, lon coordinate from wtk .h5 file
        """
        tree_path = self._get_tree_file(self.resource.h5_file)
        if not isinstance(tree, (cKDTree, str, type(None))):
            tree = None
            logger.warning('Precomputed tree must be supplied as a pickle '
                           'file or a cKDTree, not a {}'
                           .format(type(tree)))

        if tree is None:
            if tree_path in os.listdir(TREE_DIR.name):
                tree = os.path.join(TREE_DIR.name, tree_path)

        if isinstance(tree, str):
            tree = self._load_tree(tree)

        if tree is None:
            lat_lon = self.lat_lon
            tree = cKDTree(lat_lon)  # pylint: disable=not-callable
            self._save_tree(tree, os.path.join(TREE_DIR.name, tree_path))

        return tree

    def _check_lat_lon(self, lat_lon):
        """
        Check lat lon coordinates against domain

        Parameters
        ----------
        lat_lon : ndarray
            Either a single (lat, lon) pair or series of (lat, lon) pairs
        """
        lat_min, lat_max = np.sort(self.lat_lon[:, 0])[[0, -1]]
        lon_min, lon_max = np.sort(self.lat_lon[:, 1])[[0, -1]]

        lat = lat_lon[:, 0]
        check = lat < lat_min
        check |= lat > lat_max

        lon = lat_lon[:, 1]
        check |= lon < lon_min
        check |= lon > lon_max

        if any(check):
            bad_coords = lat_lon[check]
            msg = ("Latitude, longitude coordinates ({}) are outsides of the "
                   "resource domain: (({}, {}), ({}, {}))"
                   .format(bad_coords, lat_min, lon_min, lat_max, lon_max))
            raise ResourceValueError(msg)

    def lat_lon_gid(self, lat_lon, check_lat_lon=True):
        """
        Get nearest gid to given (lat, lon) pair or pairs

        Parameters
        ----------
        lat_lon : ndarray
            Either a single (lat, lon) pair or series of (lat, lon) pairs
        check_lat_lon : bool, optional
            Flag to check to make sure the requested lat lons are inside the
            resource grid. This is done by comparing with the bounding box of
            the resource coordinates and by ensuring the nearest neighbor
            distance are below the distance threshold to ensure that requested
            lat, lon coordinates are within the resource grid, by default True

        Returns
        -------
        gids : int | ndarray
            Nearest gid(s) to given (lat, lon) pair(s)
        """
        if not isinstance(lat_lon, np.ndarray):
            lat_lon = np.array(lat_lon, dtype=np.float32)

        if len(lat_lon.shape) == 1:
            lat_lon = np.expand_dims(lat_lon, axis=0)

        dist, gids = self.tree.query(lat_lon)

        if check_lat_lon:
            self._check_lat_lon(lat_lon)
            dist_check = dist > self.distance_threshold
            if np.any(dist_check):
                msg = ("Latitude, longitude coordinates ({}) do not sit within"
                       " resource grid!".format(lat_lon[dist_check]))
                logger.error(msg)
                raise ResourceValueError(msg)

        if len(gids) == 1:
            gids = int(gids[0])

        return gids

    def region_gids(self, region, region_col='state'):
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
        gids = self.meta
        gids = gids[gids[region_col] == region].index.values

        return gids

    def box_gids(self, lat_lon_1, lat_lon_2):
        """
        Get gids within bounding lat_lon coordinates

        Parameters
        ----------
        lat_lon_1 : list | tuple
            One corner of the bounding box
        lat_lon_2 : list | tuple
            The other corner of the bounding box

        Returns
        -------
        gids : ndarray
            Gids in bounding box
        """
        self._check_lat_lon(np.vstack((lat_lon_1, lat_lon_2)))
        lat_min, lat_max = sorted([lat_lon_1[0], lat_lon_2[0]])
        lon_min, lon_max = sorted([lat_lon_1[1], lat_lon_2[1]])

        coords = self.lat_lon
        gids = coords[:, 0] >= lat_min
        gids &= coords[:, 0] <= lat_max
        gids &= coords[:, 1] >= lon_min
        gids &= coords[:, 1] <= lon_max

        return np.where(gids)[0]

    def timestep_idx(self, timestep):
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
        timestep = check_tz(pd.to_datetime(timestep))
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
        ts : ndarray
            Time-series for given site(s) and dataset
        """
        ts = self[ds_name, :, gid]

        return ts

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
        df : pandas.DataFrame
            Time-series DataFrame for given site(s) and dataset
        """
        index = pd.Index(data=self.time_index, name='time_index')
        if isinstance(gid, (int, np.integer)):
            columns = [gid]
        else:
            columns = gid

        df = pd.DataFrame(self[ds_name, :, gid], columns=columns,
                          index=index)
        df.name = ds_name

        return df

    def get_lat_lon_ts(self, ds_name, lat_lon, check_lat_lon=True):
        """
        Extract timeseries of site(s) neareset to given lat_lon(s)

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple | list
            (lat, lon) coordinate of interest or pairs of coordinates
        check_lat_lon : bool, optional
            Flag to check to make sure the requested lat lons are inside the
            resource grid. This is done by comparing with the bounding box of
            the resource coordinates and by ensuring the nearest neighbor
            distance are below the distance threshold to ensure that requested
            lat, lon coordinates are within the resource grid, by default True

        Return
        ------
        ts : ndarray
            Time-series for given site(s) and dataset
        """
        gid = self.lat_lon_gid(lat_lon, check_lat_lon=check_lat_lon)
        ts = self.get_gid_ts(ds_name, gid)

        return ts

    def get_lat_lon_df(self, ds_name, lat_lon, check_lat_lon=True):
        """
        Extract timeseries of site(s) nearest to given lat_lon(s) and return
        as a DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon : tuple
            (lat, lon) coordinate of interest
        check_lat_lon : bool, optional
            Flag to check to make sure the requested lat lons are inside the
            resource grid. This is done by comparing with the bounding box of
            the resource coordinates and by ensuring the nearest neighbor
            distance are below the distance threshold to ensure that requested
            lat, lon coordinates are within the resource grid, by default True

        Return
        ------
        df : pandas.DataFrame
            Time-series DataFrame for given site(s) and dataset
        """
        gid = self.lat_lon_gid(lat_lon, check_lat_lon=check_lat_lon)
        df = self.get_gid_df(ds_name, gid)

        return df

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
        gids = self.region_gids(region, region_col=region_col)
        region_ts = self.get_gid_ts(ds_name, gids)

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
        gids = self.region_gids(region, region_col=region_col)
        region_df = self.get_gid_df(ds_name, gids)

        return region_df

    def get_box_ts(self, ds_name, lat_lon_1, lat_lon_2):
        """
        Extract timeseries of of all sites in given bounding box

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon_1 : list | tuple
            One corner of the bounding box
        lat_lon_2 : list | tuple
            The other corner of the bounding box

        Return
        ------
        box_ts : ndarray
            Time-series array of desired dataset for all sites in desired
            bounding box
        """
        gids = self.box_gids(lat_lon_1, lat_lon_2)
        box_ts = self.get_gid_ts(ds_name, gids)

        return box_ts

    def get_box_df(self, ds_name, lat_lon_1, lat_lon_2):
        """
        Extract timeseries of of all sites in given bounding box and return as
        a DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        lat_lon_1 : list | tuple
            One corner of the bounding box
        lat_lon_2 : list | tuple
            The other corner of the bounding box

        Return
        ------
        box_df : pandas.DataFrame
            Time-series array of desired dataset for all sites in desired
            bounding box
        """
        gids = self.box_gids(lat_lon_1, lat_lon_2)
        box_df = self.get_gid_df(ds_name, gids)

        return box_df

    def get_SAM_gid(self, gid, out_path=None, write_time=True,
                    extra_meta_data=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given resource gid

        Parameters
        ----------
        gid : int | list
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        write_time : bool
            Flag to write the time columns (Year, Month, Day, Hour, Minute)
        extra_meta_data : dict, optional
            Dictionary that maps the names and values of extra meta
            info. For example, extra_meta_data={'TMY Year': '2020'}
            will add a column 'TMY Year' to the meta data with
            a value of '2020'.
        kwargs : dict
            Internal kwargs for get_SAM_df

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        if isinstance(gid, (int, np.integer)):
            gid = [gid]

        SAM_df = []
        for res_id in gid:
            # pylint: disable=E1111
            df = self.resource.get_SAM_df(res_id, **kwargs)
            SAM_df.append(df)

            if out_path is not None:
                assert out_path.endswith('.csv'), 'out_path must be .csv!'
                i_out_path = out_path
                if len(gid) > 1:
                    tag = '_{}.csv'.format(res_id)
                    i_out_path = i_out_path.replace('.csv', tag)

                site_meta = self['meta', res_id]

                extra_meta_data = extra_meta_data or {}
                for col_name, val in extra_meta_data.items():
                    site_meta[col_name] = val

                if self.data_version is not None:
                    # pylint: disable=unsupported-assignment-operation
                    site_meta['Version'] = self.data_version

                self._to_SAM_csv(df, site_meta, i_out_path,
                                 write_time=write_time)

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_SAM_lat_lon(self, lat_lon, check_lat_lon=True, out_path=None,
                        **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon

        Parameters
        ----------
        lat_lon : tuple
            (lat, lon) coordinate of interest
        check_lat_lon : bool, optional
            Flag to check to make sure the requested lat lons are inside the
            resource grid. This is done by comparing with the bounding box of
            the resource coordinates and by ensuring the nearest neighbor
            distance are below the distance threshold to ensure that requested
            lat, lon coordinates are within the resource grid, by default True
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        kwargs : dict
            Internal kwargs for get_SAM_df

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gid = self.lat_lon_gid(lat_lon, check_lat_lon=check_lat_lon)
        SAM_df = self.get_SAM_gid(gid, out_path=out_path, **kwargs)

        return SAM_df

    def get_timestep_map(self, ds_name, timestep, region=None,
                         region_col='state', box=None):
        """
        Extract a map of the given dataset at the given timestep for the
        given region if supplied

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        timestep : str
            Timestep of interest
        region : str, optional
            Region to extract all pixels for, by default None
        region_col : str, optional
            Region column to search, by default 'state'
        box : tuple, optional
            Bounding corners of box to extract pixels for

        Returns
        -------
        ts_map : pandas.DataFrame
            DataFrame of map values
        """
        lat_lons = self.lat_lon
        ts_idx = self.timestep_idx(timestep)
        gids = slice(None)

        if region is not None and box is not None:
            msg = 'Can only process a region OR a set of box corners!'
            raise RuntimeError(msg)

        if region is not None:
            gids = self.region_gids(region, region_col=region_col)
            lat_lons = lat_lons[gids]
        elif box is not None:
            gids = self.box_gids(*box)
            lat_lons = lat_lons[gids]

        ts_map = self[ds_name, ts_idx, gids]
        ts_map = pd.DataFrame({'longitude': lat_lons[:, 1],
                               'latitude': lat_lons[:, 0],
                               ds_name: ts_map})

        return ts_map

    def get_grid_vectors(self, target, meta=None):
        """Get vectors representing pure horizontal/vertical movements in the
        meta data coordinate system. Note that this can break down if a target
        is requested outside of the main grid area.

        Parameters
        ----------
        target : tuple
            Starting coordinate (latitude, longitude) in decimal degrees for
            the bottom left hand corner of the raster grid.
        meta : pd.DataFrame | None
            Optional meta data input with latitude, longitude fields. Default
            is None which extracts self.meta from the resource data.

        Returns
        -------
        gid_target : np.ndarray
            1D array of shape (2,) with (latitude, longitude) corresponding to
            the meta data grid cell closest to the requested target.
        vector_x : np.ndarray
            1D array of shape (2,) with (delta_latitude, delta_longitude)
            corresponding to the vector for pure positive horizontal movement
            in the meta data
        vector_y : np.ndarray
            1D array of shape (2,) with (delta_latitude, delta_longitude)
            corresponding to the vector for pure positive vertical movement in
            the meta data
        close : np.ndarray
            Meta data index values corresponding to the 3x3 box of pixels
            closest to gid_target.
        """

        meta = meta if meta is not None else self.meta

        out_of_bounds = ((target[0] > meta['latitude']).all()
                         | (target[0] < meta['latitude']).all()
                         | (target[1] > meta['longitude']).all()
                         | (target[1] < meta['longitude']).all())
        if out_of_bounds:
            msg = ('Target {} is outside of meta data extent with latitude '
                   'range {} to {} and longitude range {} to {}'
                   .format(target, meta['latitude'].min(),
                           meta['latitude'].max(), meta['longitude'].min(),
                           meta['longitude'].max()))
            raise RuntimeError(msg)

        # find the actual meta data point closest to the target
        dist = ((meta['latitude'] - target[0])**2
                + (meta['longitude'] - target[1])**2)
        target_loc = meta.index.values[np.argmin(dist)]
        gid_target = np.array([meta.at[target_loc, 'latitude'],
                               meta.at[target_loc, 'longitude']])

        # find the 3x3 box of points around the target
        dy = meta['latitude'] - gid_target[0]
        dx = meta['longitude'] - gid_target[1]
        dist = np.sqrt(dx**2 + dy**2)
        close = meta.index.values[np.argsort(dist)[:9]]

        # get the vectors closest to pure horizontal/vertical movement
        theta = np.arctan2(dy.loc[close].values, dx.loc[close].values)
        dx_loc = close[np.argsort(np.abs(theta))[1]]
        dy_loc = close[np.argmin(np.abs(theta - np.pi / 2))]

        # get (delta_latitude, delta_longitude) vectors
        # for pure horizontal/vertical movements
        vector_dx = np.array([dy.loc[dx_loc], dx.loc[dx_loc]])
        vector_dy = np.array([dy.loc[dy_loc], dx.loc[dy_loc]])
        vector_dx[1] = 1e-6 if vector_dx[1] == 0 else vector_dx[1]
        vector_dy[1] = 1e-6 if vector_dy[1] == 0 else vector_dy[1]

        return gid_target, vector_dx, vector_dy, close

    @staticmethod
    def _order_raster_index(raster_index, meta, shape,
                            vec_dy, lat_descending=True):
        """Ensure that the raster index is propertly sorted.

        Parameters
        ----------
        raster_index : np.ndarray
            2D array of meta data index values that form a 2D rectangular grid
        meta : pd.DataFrame
            Resource meta data with latitude and longitude columns
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        vec_dy : np.ndarray
            1D array that represents a (lat, lon) vector
        lat_descending : bool
            Flat to have descending latitudes (this is how the raster would
            appear on the map with north upwards). This option can be changed
            for ease of vertical chunking / indexing.

        Returns
        -------
        raster_index : np.ndarray
            2D array of meta data index values that form a 2D rectangular grid
        """

        iflat = raster_index.flatten()
        lats_raw = meta.loc[iflat, 'latitude'].values
        lons_raw = meta.loc[iflat, 'longitude'].values

        # need to rotate the coordinates to unskew them before sorting lat/lons
        theta = np.arctan2(vec_dy[0], vec_dy[1])
        delta = (np.pi / 2) - theta
        lons = lons_raw * np.cos(delta) - lats_raw * np.sin(delta)
        lats = lats_raw * np.cos(delta) + lons_raw * np.sin(delta)

        # sorting by lat/lons ensures the reshape order
        df = pd.DataFrame({'lats': lats, 'lons': lons}, index=iflat)
        df = df.sort_values(['lons', 'lats'])

        # you need to make sure all the lons in a column are equal otherwise
        # imperfect grid sorting happens
        lons = df['lons'].values.reshape(shape, order='F')
        lons[:] = lons.mean(axis=0)
        df['lons'] = lons.flatten(order='F')
        df = df.sort_values(['lons', 'lats'])

        iflat = df.index.values
        raster_index = iflat.reshape(shape, order='F')

        lons = df['lons'].values.reshape(shape, order='F')
        lats = df['lats'].values.reshape(shape, order='F')

        # make sure lons are ordered correctly
        if (np.diff(lons.mean(axis=0)) < 0).sum() > 0.5 * lons.shape[1]:
            raster_index = raster_index[:, ::-1]

        # make sure lats are ordered correctly
        if (np.diff(lats.mean(axis=1)) < 0).sum() > 0.5 * lats.shape[0]:
            raster_index = raster_index[::-1, :]

        if lat_descending:
            raster_index = raster_index[::-1]
            lats = lats[::-1]

        return raster_index

    @classmethod
    def _get_raster_index(cls, meta, gid_target, vec_dx, vec_dy,
                          shape, lat_descending=True):
        """Get meta data index values that correspond to a 2D rectangular grid
        of the requested shape. This is a hidden compute method that can be
        called iteratively for adaptive sampling.

        Parameters
        ----------
        meta : pd.DataFrame
            Resource meta data with latitude and longitude columns
        gid_target : tuple
            Actual starting coordinates corresponding to a real gid point in
            meta data.
        vector_x : np.ndarray
            1D array of shape (2,) with (delta_latitude, delta_longitude)
            corresponding to the vector for pure positive horizontal movement
            in the meta data
        vector_y : np.ndarray
            1D array of shape (2,) with (delta_latitude, delta_longitude)
            corresponding to the vector for pure positive vertical movement in
            the meta data
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        lat_descending : bool
            Flat to have descending latitudes (this is how the raster would
            appear on the map with north upwards). This option can be changed
            for ease of vertical chunking / indexing.

        Returns
        -------
        raster_index : np.ndarray
            2D array of meta data index values that form a 2D rectangular grid
            with latitudes descending from top to bottom and longitudes
            ascending from left to right.
        start_xy : np.ndarray
            1D array of shape (2,) coordinates of the starting search point
        point_x : np.ndarray
            1D array of shape (2,) coordinates of the horizonital search point
        point_y : np.ndarray
            1D array of shape (2,) coordinates of the vertical search point
        end_xy : np.ndarray
            1D array of shape (2,) coordinates of the final search point
        """
        n_vert, n_horiz = shape
        # Set points for origin, horizontal/verical movements, and final
        start_xy = copy.deepcopy(gid_target)
        point_x = copy.deepcopy(gid_target)
        point_y = copy.deepcopy(gid_target)
        end_xy = copy.deepcopy(gid_target)

        # add offsets so bounding box is between grid lines
        start_xy -= (0.5 * (vec_dy + vec_dx))
        point_x += (n_horiz - 0.5) * vec_dx - (0.5 * vec_dy)
        point_y += (n_vert - 0.5) * vec_dy - (0.5 * vec_dx)
        end_xy += (n_vert - 0.5) * vec_dy + (n_horiz - 0.5) * vec_dx

        # slopes of horizontal / vertical vectors
        m_horiz = vec_dx[0] / vec_dx[1]
        m_vert = vec_dy[0] / vec_dy[1]

        # horizontal lines (low, high)
        lin_y_1 = (m_horiz * (meta['longitude'].values - start_xy[1])
                   + start_xy[0])
        lin_y_2 = (m_horiz * (meta['longitude'].values - point_y[1])
                   + point_y[0])

        # vertical lines (left, right)
        lin_x_1 = ((meta['latitude'].values - start_xy[0]) / m_vert
                   + start_xy[1])
        lin_x_2 = ((meta['latitude'].values - point_x[0]) / m_vert
                   + point_x[1])

        # get the mask of the bounding box
        mask = ((meta['latitude'] > lin_y_1)
                & (meta['latitude'] < lin_y_2)
                & (meta['longitude'] > lin_x_1)
                & (meta['longitude'] < lin_x_2))

        if mask.sum() != (n_horiz * n_vert):
            msg = ('Found {} gids but should have found {} by {}. '
                   'Gid target was {}, '
                   'bounding points were calculated to be {} {} {} {},'
                   'and the final found coordinates are: \n{}'
                   .format(mask.sum(), n_horiz, n_vert, gid_target,
                           start_xy, point_x, point_y, end_xy, meta[mask]))
            raise RuntimeError(msg)

        raster_index = meta[mask].index.values
        raster_index = cls._order_raster_index(raster_index, meta,
                                               shape, vec_dy,
                                               lat_descending=lat_descending)

        return raster_index, start_xy, point_x, point_y, end_xy

    def get_raster_index(self, target, shape, meta=None, max_delta=50):
        """Get meta data index values that correspond to a 2D rectangular grid
        of the requested shape starting with the target coordinate in the
        bottom left hand corner. Note that this can break down if a target is
        requested outside of the main grid area.

        Parameters
        ----------
        target : tuple
            Starting coordinate (latitude, longitude) in decimal degrees for
            the bottom left hand corner of the raster grid.
        shape : tuple
            Desired raster shape in format (number_rows, number_cols)
        meta : pd.DataFrame | None
            Optional meta data input with latitude, longitude fields. Default
            is None which extracts self.meta from the resource data.
        max_delta : int
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raseter will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances.

        Returns
        -------
        raster_index : np.ndarray
            2D array of meta data index values that form a 2D rectangular grid
            with latitudes descending from top to bottom and longitudes
            ascending from left to right.
        """

        meta = meta if meta is not None else self.meta

        raster_index = np.zeros(shape, dtype=int)

        next_target = None
        gid_target = self.get_grid_vectors(target, meta=meta)[0]

        # chunk the row (i) and columns (j) rasters
        i_split = int(np.ceil(shape[0] / max_delta))
        j_split = int(np.ceil(shape[1] / max_delta))
        i_chunks = np.array_split(np.arange(shape[0]), i_split)
        j_chunks = np.array_split(np.arange(shape[1]), j_split)

        for ii, i_chunk in enumerate(i_chunks):
            i_slice = slice(i_chunk[0], i_chunk[-1] + 1)
            logger.info('Working on row chunk {} out of {}'
                        .format(ii + 1, len(i_chunks)))

            for jj, j_chunk in enumerate(j_chunks):
                logger.debug('Working on column chunk {} out of {}'
                             .format(jj + 1, len(j_chunks)))
                j_slice = slice(j_chunk[0], j_chunk[-1] + 1)
                temp_shape = (len(i_chunk), len(j_chunk))

                # get the grid vectors using the gid_target from the
                # previous raster chunk
                gid_target, vec_dx, vec_dy, _ = self.get_grid_vectors(
                    gid_target, meta=meta)

                # get the raster using the current grid vectors
                temp, _, point_x, point_y, _ = self._get_raster_index(
                    meta, gid_target, vec_dx, vec_dy, temp_shape,
                    lat_descending=False)

                raster_index[i_slice, j_slice] = temp
                gid_target = point_x + (0.5 * (vec_dx + vec_dy))

                if jj == 0:
                    # save the gid_target for the next row
                    next_target = point_y + (0.5 * (vec_dx + vec_dy))
                elif jj == len(j_chunks) - 1:
                    # use the saved gid_target for the next row
                    gid_target = next_target

        raster_index = raster_index[::-1]

        return raster_index

    @classmethod
    def make_SAM_files(cls, res_h5, gids, out_path, write_time=True,
                       extra_meta_data=None, max_workers=1, n_chunks=36,
                       **kwargs):
        """A performant parallel entry point for making many SAM csv
        files for many gids

        Parameters
        ----------
        res_h5 : str
            Filepath to resource h5 file.
        gids : list | tuple | np.ndarray
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format. A gid index
            "*_{gid}.csv" will be appended to the file path
        write_time : bool
            Flag to write the time columns (Year, Month, Day, Hour, Minute)
        extra_meta_data : dict, optional
            Dictionary that maps the names and values of extra meta
            info. For example, extra_meta_data={'TMY Year': '2020'}
            will add a column 'TMY Year' to the meta data with
            a value of '2020'.
        max_workers : int | None
            Number of parallel workers. None for all workers.
        n_chunks : int
            Number of chunks to split gids into for parallelization
        kwargs : dict
            Internal kwargs for get_SAM_df
        """

        if max_workers == 1:
            with cls(res_h5) as res:
                res.get_SAM_gid(gids, out_path=out_path,
                                write_time=write_time,
                                extra_meta_data=extra_meta_data,
                                **kwargs)
        else:
            msg = 'Bad gids dtype: {}'.format(type(gids))
            assert isinstance(gids, (list, tuple, np.ndarray)), msg
            gid_chunks = np.array_split(np.array(gids), n_chunks)
            with SpawnProcessPool(max_workers=max_workers) as spp:
                for chunk in gid_chunks:
                    spp.submit(cls.make_SAM_files, res_h5, chunk, out_path,
                               write_time=write_time,
                               extra_meta_data=extra_meta_data,
                               max_workers=1, **kwargs)

    def close(self):
        """
        Close res_cls instance
        """
        self._res.close()

    def _get_datasets(self, datasets=None):
        """
        Get datasets to extract, if None extract all datasets

        Parameters
        ----------
        datasets : list | str, optional
            Dataset(s) to extract, by default None

        Returns
        -------
        datasets : list
            Unique set of datasets in alphabetical order
        """
        if datasets is None:
            datasets = self.datasets
        else:
            if isinstance(datasets, str):
                datasets = [datasets]
            else:
                datasets = datasets.copy()

            datasets += ['meta', 'time_index', 'coordinates']

        return sorted(set(datasets))

    def save_subset(self, out_fpath, gids, datasets=None):
        """
        Extract desired datasets for given gids and save to a new
        out_fpath .h5 file

        Parameters
        ----------
        out_fpath : str
            Path to .h5 file to save region datasets to
        gids : list
            List of gids to extract data from and save to .h5
        datasets : str | list, optional
            Dataset(s) to extract from given region and save to out_fpath,
            if None extract all datasets, by default None
        """
        scale_attr = self.resource.SCALE_ATTR
        add_attr = self.resource.ADD_ATTR
        unscale = False

        datasets = self._get_datasets(datasets=datasets)
        with h5py.File(out_fpath, mode='w-') as f_out:
            for k, v in self.global_attrs.items():
                try:
                    f_out.attrs[k] = v
                except Exception as ex:
                    msg = ('Could not transfer global attribute {}: {}\n{}'
                           .format(k, v, ex))
                    warn(msg)

            for dset in datasets:
                if dset in self:
                    ds = self.h5[dset]
                    ds_slice = self._get_ds_slice(dset, gids)
                    data = ResourceDataset.extract(ds, ds_slice,
                                                   scale_attr=scale_attr,
                                                   add_attr=add_attr,
                                                   unscale=unscale)

                    ds_out = f_out.create_dataset(dset,
                                                  shape=data.shape,
                                                  dtype=data.dtype,
                                                  data=data)
                    for k, v in ds.attrs.items():
                        try:
                            ds_out.attrs[k] = v
                        except Exception as ex:
                            msg = ('Could not transfer {} attribute {}: {}\n{}'
                                   .format(dset, k, v, ex))
                            warn(msg)
                else:
                    msg = ("Dataset {} is not available in {} and will "
                           "not be saved to {}".format(dset, self, out_fpath))
                    warn(msg, ResourceWarning)

    def save_region(self, out_fpath, region, datasets=None,
                    region_col='state'):
        """
        Extract desired datasets from desired region and save to a new
        out_fpath .h5 file

        Parameters
        ----------
        out_fpath : str
            Path to .h5 file to save region datasets to
        region : str, optional
            Region to extract all pixels for, by default None
        datasets : str | list, optional
            Dataset(s) to extract from given region and save to out_fpath,
            if None extract all datasets, by default None
        region_col : str, optional
            Region column to search, by default 'state'
        """
        gids = self.region_gids(region, region_col=region_col)

        self.save_subset(out_fpath, gids, datasets=datasets)


class MultiFileResourceX(ResourceX):
    """
    Multi-File resource extraction class
    """

    DEFAULT_RES_CLS = MultiFileResource

    def __init__(self, resource_path, res_cls=None, tree=None,
                 unscale=True, str_decode=True, check_files=False):
        """
        Parameters
        ----------
        resource_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets.
        res_cls : obj
            Resource class to use to open and access resource data
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        """
        log_versions(logger)
        res_cls = self.DEFAULT_RES_CLS if res_cls is None else res_cls
        self._res = res_cls(resource_path, unscale=unscale,
                            str_decode=str_decode, check_files=check_files)
        self._dist_thresh = None
        self._tree = tree


class MultiYearResourceX(ResourceX):
    """
    Multi Year resource extraction class
    """

    DEFAULT_RES_CLS = Resource

    def __init__(self, resource_path, years=None, tree=None, unscale=True,
                 str_decode=True, res_cls=None, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        resource_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets.
        years : list, optional
            List of years to access, by default None
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        res_cls : obj
            Resource handler to use to open individual .h5 files
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        log_versions(logger)
        res_cls = self.DEFAULT_RES_CLS if res_cls is None else res_cls
        self._res = MultiYearResource(resource_path, years=years,
                                      unscale=unscale, str_decode=str_decode,
                                      res_cls=res_cls, hsds=hsds,
                                      hsds_kwargs=hsds_kwargs)
        self._dist_thresh = None
        self._tree = tree

    def get_means_map(self, ds_name, year=None, region=None,
                      region_col='state', max_workers=None,
                      chunks_per_worker=5):
        """
        Extract given year(s) and compute means

        Parameters
        ----------
        ds_name : str
            Dataset to extract
        year : str | list, optional
            Year(s) to compute means for, by default None
        region : str
            Region to extract all pixels for
        region_col : str
            Region column to search
        max_workers : None | int, optional
            Number of workers to use, if 1 run in serial, if None use all
            available cores, by default None
        chunks_per_slice : int, optional
            Number of chunks to extract on each worker, by default 5

        Returns
        -------
        ts_map : pandas.DataFrame
            DataFrame of map values
        """
        gids = slice(None)
        if region is not None:
            gids = self.region_gids(region, region_col=region_col)

        if year is None:
            year = slice(None)

        means_map = TemporalStats.run(self.resource.h5_file, ds_name,
                                      sites=gids, statistics='mean',
                                      res_cls=self.resource.__class__,
                                      hsds=self.resource.hsds,
                                      max_workers=max_workers,
                                      chunks_per_worker=chunks_per_worker,
                                      lat_lon_only=True)

        return means_map


class MultiTimeResourceX(ResourceX):
    """
    Resource extraction class for data stored temporaly accross multiple files
    """

    def __init__(self, resource_path, tree=None, unscale=True,
                 str_decode=True, res_cls=None, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        resource_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets.
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        res_cls : obj
            Resource handler to us to open individual .h5 files
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        log_versions(logger)
        res_cls = self.DEFAULT_RES_CLS if res_cls is None else res_cls
        self._res = MultiTimeResource(resource_path, unscale=unscale,
                                      str_decode=str_decode, res_cls=res_cls,
                                      hsds=hsds, hsds_kwargs=hsds_kwargs)
        self._dist_thresh = None
        self._tree = tree


class SolarX(ResourceX):
    """
    Solar Resource extraction class
    """

    DEFAULT_RES_CLS = SolarResource


class NSRDBX(ResourceX):
    """
    NSRDB extraction class
    """

    DEFAULT_RES_CLS = NSRDB


class MultiFileNSRDBX(MultiFileResourceX):
    """
    Multi-File NSRDB extraction class
    """

    DEFAULT_RES_CLS = MultiFileNSRDB


class MultiYearNSRDBX(MultiYearResourceX):
    """
    Multi Year NSRDB extraction class
    """

    DEFAULT_RES_CLS = NSRDB


class MultiTimeNSRDBX(MultiTimeResourceX):
    """
    NSRDB extraction class for data stored temporaly accross multiple files
    """

    DEFAULT_RES_CLS = NSRDB


class WindX(ResourceX):
    """
    Wind Resource extraction class
    """

    DEFAULT_RES_CLS = WindResource

    def get_SAM_gid(self, hub_height, gid, out_path=None, write_time=True,
                    extra_meta_data=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given resource gid and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        gid : int | list
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        write_time : bool
            Flag to write the time columns (Year, Month, Day, Hour, Minute)
        extra_meta_data : dict, optional
            Dictionary that maps the names and values of extra meta
            info. For example, extra_meta_data={'TMY Year': '2020'}
            will add a column 'TMY Year' to the meta data with
            a value of '2020'.
        kwargs : dict
            Internal kwargs for get_SAM_df:
            - require_wind_dir
            - icing

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        kwargs['height'] = hub_height
        if out_path is not None:
            write_time = False
            kwargs.update({'add_header': True})

        SAM_df = super().get_SAM_gid(gid, out_path=out_path,
                                     write_time=write_time,
                                     extra_meta_data=extra_meta_data,
                                     **kwargs)

        return SAM_df

    def get_SAM_lat_lon(self, hub_height, lat_lon, check_lat_lon=True,
                        out_path=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        lat_lon : tuple
            (lat, lon) coordinate of interest
        check_lat_lon : bool, optional
            Flag to check to make sure the requested lat lons are inside the
            resource grid. This is done by comparing with the bounding box of
            the resource coordinates and by ensuring the nearest neighbor
            distance are below the distance threshold to ensure that requested
            lat, lon coordinates are within the resource grid, by default True
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        kwargs : dict
            Internal kwargs for get_SAM_df:
            - require_wind_dir
            - icing

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gid = self.lat_lon_gid(lat_lon, check_lat_lon=check_lat_lon)
        SAM_df = self.get_SAM_gid(hub_height, gid, out_path=out_path, **kwargs)

        return SAM_df

    @classmethod
    def make_SAM_files(cls, hub_height, res_h5, gids, out_path,
                       write_time=True, extra_meta_data=None, max_workers=1,
                       n_chunks=36, **kwargs):
        """A performant parallel entry point for making many SAM csv
        files for many gids

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        res_h5 : str
            Filepath to resource h5 file.
        gids : list | tuple | np.ndarray
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format. A gid index
            "*_{gid}.csv" will be appended to the file path
        write_time : bool
            Flag to write the time columns (Year, Month, Day, Hour, Minute)
        extra_meta_data : dict, optional
            Dictionary that maps the names and values of extra meta
            info. For example, extra_meta_data={'TMY Year': '2020'}
            will add a column 'TMY Year' to the meta data with
            a value of '2020'.
        max_workers : int | None
            Number of parallel workers. None for all workers.
        n_chunks : int
            Number of chunks to split gids into for parallelization
        kwargs : dict
            Internal kwargs for get_SAM_df
        """
        kwargs['height'] = hub_height
        super().get_SAM_gid(res_h5, gids, out_path, write_time=write_time,
                            extra_meta_data=extra_meta_data,
                            max_workers=max_workers, n_chunks=n_chunks,
                            **kwargs)


class MultiFileWindX(MultiFileResourceX):
    """
    Multi-File Wind Resource extraction class
    """

    DEFAULT_RES_CLS = MultiFileWTK


class MultiYearWindX(MultiYearResourceX):
    """
    Multi Year Wind Resource extraction class
    """

    DEFAULT_RES_CLS = WindResource


class MultiTimeWindX(MultiTimeResourceX):
    """
    Wind resource extraction class for data stored temporaly accross multiple
    files
    """

    DEFAULT_RES_CLS = WindResource


class WaveX(ResourceX):
    """
    Wave data extraction class
    """

    DEFAULT_RES_CLS = WaveResource

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
        ts : ndarray
            Time-series for given site(s) and dataset
        """
        if ds_name == 'directional_wave_spectrum':
            ts = self[ds_name, :, :, :, gid]
        else:
            ts = self[ds_name, :, gid]

        return ts

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
        df : pandas.DataFrame
            Time-series DataFrame for given site(s) and dataset
        """
        if ds_name == 'directional_wave_spectrum':
            df = self[ds_name, :, :, :, gid]
            index = pd.MultiIndex.from_product(
                [self.time_index, self['frequency'], self['direction']],
                names=['time_index', 'frequency', 'direction'])
            ax1 = np.prod(df.shape[:3])
            ax2 = df.shape[-1] if len(df.shape) == 4 else 1
            df = df.reshape(ax1, ax2)
        else:
            df = self[ds_name, :, gid]
            index = pd.Index(data=self.time_index, name='time_index')

        if isinstance(gid, (int, np.integer)):
            df = pd.DataFrame(df, columns=[gid],
                              index=index)
            df.name = gid
        else:
            df = pd.DataFrame(df, columns=gid,
                              index=index)
            df.name = ds_name

        return df


class MultiYearWaveX(MultiYearResourceX):
    """
    Multi Year Wave extraction class
    """

    DEFAULT_RES_CLS = WaveResource


class MultiTimeWaveX(MultiTimeResourceX):
    """
    Wave resource extraction class for data stored temporaly accross multiple
    files
    """

    DEFAULT_RES_CLS = WaveResource
