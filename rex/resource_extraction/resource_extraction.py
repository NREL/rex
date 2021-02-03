# -*- coding: utf-8 -*-
"""
Resource Extraction Tools
"""
import h5py
import logging
import numpy as np
import os
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from tempfile import TemporaryDirectory
from warnings import warn

from rex.multi_file_resource import (MultiFileNSRDB, MultiFileResource,
                                     MultiFileWTK)
from rex.multi_time_resource import MultiTimeResource
from rex.multi_year_resource import MultiYearResource
from rex.resource import Resource, ResourceDataset
from rex.renewable_resource import (NSRDB, SolarResource, WaveResource,
                                    WindResource)
from rex.temporal_stats.temporal_stats import TemporalStats
from rex.utilities.exceptions import ResourceValueError, ResourceWarning
from rex.utilities.utilities import parse_year, check_tz

TREE_DIR = TemporaryDirectory()
logger = logging.getLogger(__name__)


class ResourceX:
    """
    Resource data extraction tool
    """
    def __init__(self, res_h5, res_cls=Resource, tree=None, unscale=True,
                 hsds=False, str_decode=True, group=None):
        """
        Parameters
        ----------
        res_h5 : str
            Path to resource .h5 file of interest
        res_cls : obj, optional
            Resource class to use to open and access resource data,
            by default Resource
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        """
        self._res = res_cls(res_h5, unscale=unscale, hsds=hsds,
                            str_decode=str_decode, group=group)
        self._tree = tree
        self._i = 0

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

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

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
    def global_attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return self.resource.attrs

    @property
    def attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        attrs : dict
        """
        return self.global_attrs

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
    def _to_SAM_csv(sam_df, site_meta, out_path):
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
        """
        if not out_path.endswith('.csv'):
            if os.path.isfile(out_path):
                out_path = os.path.basename(out_path)

            out_path = os.path.join(out_path, "{}.csv".format(sam_df.name))

        sam_df.to_csv(out_path, index=False)

        if 'gid' not in site_meta:
            site_meta.index.name = 'gid'
            site_meta = site_meta.reset_index()

        col_map = {}
        for c in site_meta.columns:
            if c == 'timezone':
                col_map[c] = 'Time Zone'
            elif c == 'gid':
                col_map[c] = 'Location ID'
            else:
                col_map[c] = c.capitalize()

        site_meta = site_meta.rename(columns=col_map)
        cols = ','.join(site_meta.columns)
        values = ','.join(site_meta.values[0].astype(str))

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
        lat = self.lat_lon[:, 0]
        lat_min = lat.min()
        lat_max = lat.max()

        lon = self.lat_lon[:, 1]
        lon_min = lon.min()
        lon_max = lon.max()

        if not isinstance(lat_lon, np.ndarray):
            lat_lon = np.array(lat_lon)

        if len(lat_lon.shape) == 1:
            lat_lon = np.expand_dims(lat_lon, axis=0)

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

        return lat_lon

    def lat_lon_gid(self, lat_lon):
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
        lat_lon = self._check_lat_lon(lat_lon)
        _, gids = self.tree.query(lat_lon)

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
        ts : ndarray
            Time-series for given site(s) and dataset
        """
        gid = self.lat_lon_gid(lat_lon)
        ts = self.get_gid_ts(ds_name, gid)

        return ts

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
        df : pandas.DataFrame
            Time-series DataFrame for given site(s) and dataset
        """
        gid = self.lat_lon_gid(lat_lon)
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

    def get_SAM_gid(self, gid, out_path=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given resource gid

        Parameters
        ----------
        gid : int | list
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        kwargs : dict
            Internal kwargs for _get_SAM_df

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        if isinstance(gid, (int, np.integer)):
            gid = [gid, ]

        SAM_df = []
        for res_id in gid:
            # pylint: disable=E1111
            df = self.resource._get_SAM_df('SAM', res_id, **kwargs)
            SAM_df.append(df)
            if out_path is not None:
                site_meta = self['meta', res_id]
                self._to_SAM_csv(df, site_meta, out_path)

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_SAM_lat_lon(self, lat_lon, out_path=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon

        Parameters
        ----------
        lat_lon : tuple
            (lat, lon) coordinate of interest
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
        kwargs : dict
            Internal kwargs for _get_SAM_df

        Return
        ------
        SAM_df : pandas.DataFrame | list
            Time-series DataFrame for given site and dataset
            If multiple lat, lon pairs are given a list of DatFrames is
            returned
        """
        gid = self.lat_lon_gid(lat_lon)
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

    def close(self):
        """
        Close res_cls instance
        """
        self._res.close()

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
        scale_attr = self.resource.SCALE_ATTR
        add_attr = self.resource.ADD_ATTR
        unscale = False

        datasets = self._get_datasets(datasets=datasets)
        gids = self.region_gids(region, region_col=region_col)
        with h5py.File(out_fpath, mode='w-') as f_out:
            for k, v in self.attrs.items():
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


class MultiFileResourceX(ResourceX):
    """
    Multi-File resource extraction class
    """

    def __init__(self, resource_path, res_cls=MultiFileResource, tree=None,
                 unscale=True, str_decode=True, check_files=False):
        """
        Parameters
        ----------
        resource_path : str
            Path to resource .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        self._res = res_cls(resource_path, unscale=unscale,
                            str_decode=str_decode, check_files=check_files)
        self._tree = tree
        self._i = 0


class MultiYearResourceX(ResourceX):
    """
    Multi Year resource extraction class
    """
    def __init__(self, resource_path, years=None, tree=None, unscale=True,
                 str_decode=True, hsds=False, res_cls=Resource):
        """
        Parameters
        ----------
        resource_path : str
            Path to resource .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        res_cls : obj
            Resource handler to use to open individual .h5 files
        """
        self._res = MultiYearResource(resource_path, years=years,
                                      unscale=unscale, str_decode=str_decode,
                                      hsds=hsds, res_cls=res_cls)
        self._tree = tree
        self._i = 0

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
                 str_decode=True, hsds=False, res_cls=Resource):
        """
        Parameters
        ----------
        resource_path : str
            Path to resource .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        res_cls : obj
            Resource handler to us to open individual .h5 files
        """
        self._res = MultiTimeResource(resource_path, unscale=unscale,
                                      str_decode=str_decode, hsds=hsds,
                                      res_cls=res_cls)
        self._tree = tree
        self._i = 0


class SolarX(ResourceX):
    """
    Solar Resource extraction class
    """
    def __init__(self, solar_h5, tree=None, unscale=True, hsds=False,
                 str_decode=True, group=None):
        """
        Parameters
        ----------
        solar_h5 : str
            Path to solar .h5 file of interest
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        """
        super().__init__(solar_h5, unscale=unscale, str_decode=str_decode,
                         group=group, hsds=hsds, tree=tree,
                         res_cls=SolarResource)


class NSRDBX(ResourceX):
    """
    NSRDB extraction class
    """
    def __init__(self, nsrdb_h5, tree=None, unscale=True, hsds=False,
                 str_decode=True, group=None):
        """
        Parameters
        ----------
        nsrdb_h5 : str
            Path to NSRDB .h5 file of interest
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        """
        super().__init__(nsrdb_h5, unscale=unscale, str_decode=str_decode,
                         group=group, hsds=hsds, tree=tree,
                         res_cls=NSRDB)


class MultiFileNSRDBX(MultiFileResourceX):
    """
    Multi-File NSRDB extraction class
    """
    def __init__(self, nsrdb_source, tree=None, unscale=True, str_decode=True,
                 check_files=False):
        """
        Parameters
        ----------
        nsrdb_source : str | list
            Path to directory containing multi-file NSRDB file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source NSRDB .h5 files
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
        super().__init__(nsrdb_source, unscale=unscale, str_decode=str_decode,
                         check_files=check_files, tree=tree,
                         res_cls=MultiFileNSRDB)


class MultiYearNSRDBX(MultiYearResourceX):
    """
    Multi Year NSRDB extraction class
    """
    def __init__(self, nsrdb_path, years=None, tree=None, unscale=True,
                 str_decode=True, hsds=False):
        """
        Parameters
        ----------
        nsrdb_path : str
            Path to NSRDB .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(nsrdb_path, years=years, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds, res_cls=NSRDB)


class MultiTimeNSRDBX(MultiTimeResourceX):
    """
    NSRDB extraction class for data stored temporaly accross multiple files
    """

    def __init__(self, nsrdb_path, tree=None, unscale=True,
                 str_decode=True, hsds=False):
        """
        Parameters
        ----------
        nsrdb_path : str
            Path to NSRDB .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(nsrdb_path, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds, res_cls=NSRDB)


class WindX(ResourceX):
    """
    Wind Resource extraction class
    """
    def __init__(self, wind_h5, tree=None, unscale=True, hsds=False,
                 str_decode=True, group=None):
        """
        Parameters
        ----------
        wind_h5 : str
            Path to Wind .h5 file of interest
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        """
        super().__init__(wind_h5, unscale=unscale, str_decode=str_decode,
                         group=group, hsds=hsds, tree=tree,
                         res_cls=WindResource)

    def get_SAM_gid(self, hub_height, gid, out_path=None, **kwargs):
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
        if isinstance(gid, (int, np.integer)):
            gid = [gid, ]

        SAM_df = []
        for res_id in gid:
            df = self.resource._get_SAM_df(ds_name, res_id, **kwargs)
            SAM_df.append(df)
            if out_path is not None:
                site_meta = self['meta', res_id]
                self._to_SAM_csv(df, site_meta, out_path)

        if len(SAM_df) == 1:
            SAM_df = SAM_df[0]

        return SAM_df

    def get_SAM_lat_lon(self, hub_height, lat_lon, out_path=None, **kwargs):
        """
        Extract time-series of all variables needed to run SAM for nearest
        site to given lat_lon and hub height

        Parameters
        ----------
        hub_height : int
            Hub height of interest
        gid : int | list
            Resource gid(s) of interset
        out_path : str, optional
            Path to save SAM data to in SAM .csv format, by default None
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
        gid = self.lat_lon_gid(lat_lon)
        SAM_df = self.get_SAM_gid(hub_height, gid, out_path=out_path, **kwargs)

        return SAM_df


class MultiFileWindX(MultiFileResourceX):
    """
    Multi-File Wind Resource extraction class
    """
    def __init__(self, wtk_source, tree=None, unscale=True, str_decode=True,
                 check_files=False):
        """
        Parameters
        ----------
        wtk_source : str | list
            Path to directory containing multi-file WTK file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source WTK .h5 files
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
        super().__init__(wtk_source, unscale=unscale, str_decode=str_decode,
                         check_files=check_files, tree=tree,
                         res_cls=MultiFileWTK)


class MultiYearWindX(MultiYearResourceX):
    """
    Multi Year Wind Resource extraction class
    """
    def __init__(self, wtk_path, years=None, tree=None, unscale=True,
                 str_decode=True, hsds=False):
        """
        Parameters
        ----------
        wtk_path : str
            Path to annual WTK .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(wtk_path, years=years, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds,
                         res_cls=WindResource)


class MultiTimeWindX(MultiTimeResourceX):
    """
    Wind resource extraction class for data stored temporaly accross multiple
    files
    """

    def __init__(self, wtk_path, tree=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        wtk_path : str
            Path to annual WTK .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(wtk_path, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds,
                         res_cls=WindResource)


class WaveX(ResourceX):
    """
    Wave data extraction class
    """

    def __init__(self, wave_h5, tree=None, unscale=True, hsds=False,
                 str_decode=True, group=None):
        """
        Parameters
        ----------
        wave_h5 : str
            Path to US_Wave .h5 file of interest
        tree : str | cKDTree, optional
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates, by default None
        unscale : bool, optional
            Boolean flag to automatically unscale variables on extraction,
            by default True
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        str_decode : bool, optional
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
            by default True
        group : str, optional
            Group within .h5 resource file to open, by default None
        """
        super().__init__(wave_h5, unscale=unscale, str_decode=str_decode,
                         group=group, hsds=hsds, tree=tree,
                         res_cls=WaveResource)

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
            ax1 = np.product(df.shape[:3])
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

    def __init__(self, wave_path, years=None, tree=None, unscale=True,
                 str_decode=True, hsds=False):
        """
        Parameters
        ----------
        wave_path : str
            Path to US_Wave .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(wave_path, years=years, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds,
                         res_cls=WaveResource)


class MultiTimeWaveX(MultiTimeResourceX):
    """
    Wave resource extraction class for data stored temporaly accross multiple
    files
    """

    def __init__(self, wave_path, tree=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        wave_path : str
            Path to US_Wave .h5 files
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        tree : str | cKDTree
            cKDTree or path to .pkl file containing pre-computed tree
            of lat, lon coordinates
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(wave_path, tree=tree, unscale=unscale,
                         str_decode=str_decode, hsds=hsds,
                         res_cls=WindResource)
