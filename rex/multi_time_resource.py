# -*- coding: utf-8 -*-
"""
Classes to handle resource data stored over multiple files
"""
import os
from glob import glob
from itertools import chain
from fnmatch import fnmatch
import logging

import s3fs
import numpy as np
import pandas as pd

from rex.renewable_resource import (
    NSRDB,
    SolarResource,
    WaveResource,
    WindResource,
)
from rex.resource import Resource, BaseDatasetIterable
from rex.utilities.exceptions import FileInputError
from rex.utilities.parse_keys import parse_keys, parse_slice
from rex.utilities.utilities import is_hsds_file, is_s3_file


logger = logging.getLogger(__name__)


class MultiTimeH5:
    """
    Class to handle h5 Resources stored over multiple temporal files
    """

    def __init__(self, h5_path, res_cls=Resource, hsds=False, hsds_kwargs=None,
                 **res_cls_kwargs):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        res_cls : obj
            Resource class to use to open and access resource data
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        res_cls_kwargs : dict, optional
            Kwargs for `res_cls`
        """
        self.h5_path = h5_path
        self._file_paths = self._get_file_paths(h5_path, hsds=hsds,
                                                hsds_kwargs=hsds_kwargs)
        res_cls_kwargs.update({'hsds': hsds})
        self._h5_map = self._map_file_instances(self._file_paths,
                                                res_cls=res_cls,
                                                **res_cls_kwargs)

        self._datasets = None
        self._shape = None
        self._time_index = None
        self._time_slice_map = []

    def __repr__(self):
        msg = ("{} for {}:\n Contains data from {} files"
               .format(self.__class__.__name__, self.h5_path, len(self)))
        return msg

    def __getitem__(self, file):
        fn_fp_map = {os.path.basename(fp): fp for fp in self._file_paths}
        if file in self._h5_map['fp'].values:
            iloc = np.where(self._h5_map['fp'] == file)[0][0]
            h5 = self._h5_map.at[iloc, 'h5']
        elif file in fn_fp_map:
            iloc = np.where(self._h5_map['fp'] == fn_fp_map[file])[0][0]
            h5 = self._h5_map.at[iloc, 'h5']
        else:
            raise ValueError('{} is invalid, must be one of: {}'
                             .format(file, self._file_paths))

        return h5

    @property
    def attrs(self):
        """
        Global .h5 file attributes sourced from first .h5 file

        Returns
        -------
        attrs : dict
        """
        attrs = dict(self.h5.attrs)

        return attrs

    @property
    def files(self):
        """
        Available file paths

        Returns
        -------
        list
        """
        return sorted(self._file_paths)

    @property
    def h5_files(self):
        """
        .h5 files data is being sourced from

        Returns
        -------
        list
        """
        return sorted(self._h5_map['fp'])

    @property
    def h5(self):
        """
        open h5 file handler for a single .h5 file

        Returns
        -------
        h5py.File
        """
        return self._h5_map['h5'].values[0]

    @property
    def datasets(self):
        """
        Available datasets

        Returns
        -------
        list
        """
        if self._datasets is None:
            all_dsets = self._h5_map['dsets'].values.tolist()
            dsets = [d for sub in all_dsets for d in sub]
            self._datasets = list(set(dsets))

        return self._datasets

    @property
    def resource_datasets(self):
        """
        Available resource datasets

        Returns
        -------
        list
        """
        res_dsets = [ds for ds in self.datasets
                     if ds not in ['meta', 'time_index', 'coordinates']]

        return res_dsets

    @property
    def shape(self):
        """
        Dataset shape (time, sites)

        Returns
        -------
        tuple
        """
        if self._shape is None:
            self._shape = (len(self.time_index), self.h5.shape[1])

        return self._shape

    @property
    def time_index(self):
        """
        Multi-year datetime index

        Returns
        -------
        pandas.DatatimeIndex
        """
        if self._time_index is None:
            time_slice_map = []
            for _, row in self._h5_map.iterrows():
                h5 = row['h5']
                fp = row['fp']
                ti = h5.time_index
                time_slice_map.append(np.full(len(ti), os.path.basename(fp)))
                if self._time_index is None:
                    self._time_index = ti
                else:
                    self._time_index = self._time_index.append(ti)

            if len(self._time_index) != len(np.unique(self._time_index)):
                unique, duplicates = np.unique(self._time_index,
                                               return_counts=True)
                duplicates = np.where(duplicates > 1)[0]
                duplicates = unique[duplicates]
                msg = ('The combined time_index has {} duplicate values:\n{}'
                       .format(len(duplicates), duplicates))
                raise RuntimeError(msg)

            self._time_slice_map = np.concatenate(time_slice_map, axis=0)

        return self._time_index

    @staticmethod
    def _get_hsds_file_paths(h5_path, hsds_kwargs=None):
        """
        Get a list of h5 filepaths matching the h5_path specification from HSDS

        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes.
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None

        Returns
        -------
        file_paths : list
            List of filepaths for this handler to handle.
        """
        try:
            import h5pyd
        except Exception as e:
            msg = (f'Tried to open hsds file path: "{h5_path}" with '
                   'h5pyd but could not import, try '
                   '`pip install NREL-rex[hsds]`')
            logger.error(msg)
            raise ImportError(msg) from e

        if hsds_kwargs is None:
            hsds_kwargs = {}

        if isinstance(h5_path, (list, tuple)):
            msg = ('HSDS filepath must be a string, possibly with glob '
                   f'pattern, but received list/tuple: {h5_path}')
            logger.error(msg)
            raise TypeError(msg)

        hsds_dir = os.path.dirname(h5_path)
        fn = os.path.basename(h5_path)

        if '*' in hsds_dir:
            msg = ('HSDS path specifications cannot handle wildcards in the '
                   'directory name! The directory must be explicit but the '
                   'filename can have wildcards. This HSDS h5_path input '
                   'cannot be used: {}'.format(h5_path))
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not fn:
            msg = ('h5_path must be a unix shell style pattern with '
                   'wildcard * in order to find files, but received '
                   'directory specification: {}'.format(h5_path))
            logger.error(msg)
            raise FileInputError(msg)

        with h5pyd.Folder(hsds_dir + '/', **hsds_kwargs) as f:
            file_paths = [f'{hsds_dir}/{fn}' for fn in f
                          if fnmatch(f'{hsds_dir}/{fn}', h5_path)]

        return file_paths

    @staticmethod
    def _get_s3_file_paths(h5_path):
        """
        Get a list of h5 filepaths matching the h5_path specification from s3

        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes.

        Returns
        -------
        file_paths : list
            List of filepaths for this handler to handle.
        """
        s3 = s3fs.S3FileSystem(anon=True)

        if isinstance(h5_path, (list, tuple)):
            file_paths = [s3.glob(fp) for fp in h5_path]
            file_paths = list(chain.from_iterable(file_paths))
        elif isinstance(h5_path, str):
            file_paths = s3.glob(h5_path)

        # s3fs glob drops prefix for some reason
        for i, fp in enumerate(file_paths):
            if not fp.startswith('s3://'):
                file_paths[i] = f's3://{fp}'

        return file_paths

    @classmethod
    def _get_file_paths(cls, h5_path, hsds=False, hsds_kwargs=None):
        """
        Get a file list based on the h5_path specification.

        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None

        Returns
        -------
        file_paths : list
            List of filepaths for this handler to handle.
        """

        if is_hsds_file(h5_path) or hsds:
            file_paths = cls._get_hsds_file_paths(h5_path,
                                                  hsds_kwargs=hsds_kwargs)
        elif is_s3_file(h5_path):
            file_paths = cls._get_s3_file_paths(h5_path)
        elif isinstance(h5_path, (list, tuple)):
            file_paths = list(chain.from_iterable(glob(fp) for fp in h5_path))
            for fp in file_paths:
                assert os.path.exists(fp), 'Does not exist: {}'.format(fp)
        elif os.path.isdir(h5_path):
            msg = ('h5_path must be a unix shell style pattern with '
                   'wildcard * in order to find files, but received '
                   'directory specification: {}'.format(h5_path))
            raise FileInputError(msg)
        elif isinstance(h5_path, str):
            file_paths = glob(h5_path)

        if not any(file_paths):
            msg = ('Could not find any file paths with pattern: {}'
                   .format(h5_path))
            raise FileInputError(msg)

        return file_paths

    @staticmethod
    def _map_file_instances(file_paths, res_cls=Resource, **res_cls_kwargs):
        """
        Open all .h5 files and map the open h5py instances to the
        associated file paths

        Parameters
        ----------
        file_paths : list
            List of filepaths for this handler to handle.

        Returns
        -------
        h5_map : pd.DataFrame
            DataFrame mapping file paths to open resource instances and
            datasets per file (columns: fp, h5, and dsets)
        """
        h5_map = pd.DataFrame({'fp': file_paths, 'h5': None,
                               't0': None, 't1': None})
        for i, f_path in enumerate(file_paths):
            h5_map.at[i, 'h5'] = res_cls(f_path, **res_cls_kwargs)
            h5_map.at[i, 't0'] = h5_map.at[i, 'h5'].time_index.values[0]
            h5_map.at[i, 't1'] = h5_map.at[i, 'h5'].time_index.values[1]

        h5_map['dsets'] = [h5.dsets for h5 in h5_map['h5'].values]
        h5_map = h5_map.sort_values('t0').reset_index(drop=True)

        return h5_map

    @staticmethod
    def _check_time_slice(time_slice):
        """
        Check to see if time positions can be represented as a slice

        Parameters
        ----------
        time_slice : ndarray | list
            List of temporal positions

        Returns
        -------
        time_slice : ndarray | list | slice
            Slice covering range of positions to extract if possible
        """
        s = time_slice[0]
        e = time_slice[-1] + 1
        if (e - s) == len(time_slice):
            time_slice = slice(s, e, None)

        return time_slice

    def _map_time_slice(self, time_slice):
        """
        Map timeslices to files

        Parameters
        ----------
        time_slice : int | list | slice
            tuple describing slice of dataset array to extract

        Returns
        -------
        file_times : dict
            Dictionary mapping files to the time_slices to extract
        """
        time_index = self.time_index[time_slice]
        files = self._time_slice_map[time_slice]
        file_times = {}
        for file in np.unique(files):
            ti = self[file].time_index
            file_slice = np.where(ti.isin(time_index))[0]
            file_slice = self._check_time_slice(file_slice)
            file_times[file] = file_slice

        return file_times

    def _get_ds(self, ds_name, ds_slice):
        """
        Extract data from given dataset

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : int | list | slice
            tuple describing slice of dataset array to extract

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        ds_slice = parse_slice(ds_slice)
        out = []
        time_slice = ds_slice[0]
        if isinstance(time_slice, (int, np.integer)):
            time_step = self.time_index[time_slice]
            file = self._time_slice_map[time_slice]
            time_index = self[file].time_index
            time_slice = np.where(time_step == time_index)[0][0]
            file_slice = (time_slice, ) + ds_slice[1:]
            out = self[file]._get_ds(ds_name, file_slice)
        else:
            file_times = self._map_time_slice(ds_slice[0])
            for file, time_slice in file_times.items():
                file_slice = (time_slice, ) + ds_slice[1:]
                out.append(self[file]._get_ds(ds_name, file_slice))

            out = np.concatenate(out, axis=0)

        return out

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map['h5']:
            f.close()


class MultiTimeResource(BaseDatasetIterable):
    """
    Class to handle resource data stored temporally accross multiple
    .h5 files

    Examples
    --------
    Extracting the resource's Datetime Index

    >>> path = '$TESTDATADIR/nsrdb/ri_100_nsrdb_*.h5'
    >>> with MultiTimeResource(path) as res:
    >>>     ti = res.time_index
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:30:00',
                   '2012-01-01 01:00:00', '2012-01-01 01:30:00',
                   '2012-01-01 02:00:00', '2012-01-01 02:30:00',
                   '2012-01-01 03:00:00', '2012-01-01 03:30:00',
                   '2012-01-01 04:00:00', '2012-01-01 04:30:00',
                   ...
                   '2013-12-31 19:00:00', '2013-12-31 19:30:00',
                   '2013-12-31 20:00:00', '2013-12-31 20:30:00',
                   '2013-12-31 21:00:00', '2013-12-31 21:30:00',
                   '2013-12-31 22:00:00', '2013-12-31 22:30:00',
                   '2013-12-31 23:00:00', '2013-12-31 23:30:00'],
                  dtype='datetime64[ns]', length=35088, freq=None)

    NOTE: time_index covers data from 2012 and 2013

    >>> with MultiTimeResource(path) as res:
    >>>     print(res.h5_files)

    ['/Users/mrossol/Git_Repos/rex/tests/data/nsrdb/ri_100_nsrdb_2012.h5',
     '/Users/mrossol/Git_Repos/rex/tests/data/nsrdb/ri_100_nsrdb_2013.h5']

    Data slicing works the same as with "Resource" except axis 0 now covers
    2012 and 2013

    >>> with MultiTimeResource(path) as res:
    >>>     temperature = res['air_temperature']
    >>>
    >>> temperature
    [[ 4.  5.  5. ...  4.  3.  4.]
     [ 4.  4.  5. ...  4.  3.  4.]
     [ 4.  4.  5. ...  4.  3.  4.]
     ...
     [-1. -1.  0. ... -2. -3. -2.]
     [-1. -1.  0. ... -2. -3. -2.]
     [-1. -1.  0. ... -2. -3. -2.]]
    >>> temperature.shape
    (35088, 100)

    >>> with MultiTimeResource(path) as res:
    >>>     temperature = res['air_temperature', ::100] # every 100th timestep
    >>>
    >>> temperature
    [[ 4.  5.  5. ...  4.  3.  4.]
     [ 1.  1.  2. ...  0.  0.  1.]
     [-2. -1. -1. ... -2. -4. -2.]
     ...
     [-3. -2. -2. ... -3. -4. -3.]
     [ 0.  0.  1. ...  0. -1.  0.]
     [ 3.  3.  3. ...  2.  2.  3.]]
    >>> temperature.shape
    (351, 100)
    """

    def __init__(self, h5_path, unscale=True, str_decode=True,
                 res_cls=Resource, hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
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

        self.h5_path = h5_path
        self._time_index = None
        # Map variables to their .h5 files
        cls_kwargs = {'unscale': unscale, 'str_decode': str_decode,
                      'hsds': hsds, 'hsds_kwargs': hsds_kwargs}
        self._h5 = MultiTimeH5(self.h5_path, res_cls=res_cls, **cls_kwargs)
        self.h5_files = self._h5.h5_files
        self.h5_file = self.h5_files[0]

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_path)
        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return len(self.h5.time_index)

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)

        if ds.endswith('time_index'):
            out = self.h5.h5._get_time_index(ds_slice)
        elif ds.endswith('meta'):
            out = self.h5.h5._get_meta(ds, ds_slice)
        elif ds.endswith('coordinates'):
            out = self.h5.h5._get_coords(ds, ds_slice)
        else:
            out = self.h5._get_ds(ds, ds_slice)

        return out

    def __contains__(self, dset):
        return dset in self.datasets

    @property
    def h5(self):
        """
        Open class instance that handles all .h5 files that data is to
        be extracted from

        Returns
        -------
        h5 : MultiTimeH5 | MultiYearH5
        """
        return self._h5

    @property
    def datasets(self):
        """
        Datasets available

        Returns
        -------
        list
        """
        return self.h5.datasets

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
        return self.h5.resource_datasets

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
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
        """
        return self.h5.shape

    @property
    def meta(self):
        """
        Resource meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
        """

        return self.h5.h5.meta

    @property
    def time_index(self):
        """
        Resource DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
        """
        return self.h5.time_index

    @property
    def lat_lon(self):
        """
        Extract (latitude, longitude) pairs

        Returns
        -------
        lat_lon : ndarray
        """
        return self.h5.h5.lat_lon

    @property
    def coordinates(self):
        """
        Coordinates: (lat, lon) pairs

        Returns
        -------
        lat_lon : ndarray
        """
        return self.lat_lon

    @property
    def global_attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return self.get_attrs()

    @property
    def attrs(self):
        """
        Dictionary of all dataset attributes

        Returns
        -------
        attrs : dict
        """
        return self.h5.h5.attrs

    @property
    def shapes(self):
        """
        Dictionary of all dataset shapes

        Returns
        -------
        shapes : dict
        """
        return self.h5.h5.shapes

    @property
    def dtypes(self):
        """
        Dictionary of all dataset dtypes

        Returns
        -------
        dtypes : dict
        """
        return self.h5.h5.dtypes

    @property
    def chunks(self):
        """
        Dictionary of all dataset chunk sizes

        Returns
        -------
        chunks : dict
        """
        return self.h5.h5.chunks

    @property
    def scale_factors(self):
        """
        Dictionary of all dataset scale factors

        Returns
        -------
        scale_factors : dict
        """
        return self.h5.h5.scale_factors

    @property
    def units(self):
        """
        Dictionary of all dataset units

        Returns
        -------
        units : dict
        """
        return self.h5.h5.units

    def get_attrs(self, dset=None):
        """
        Get h5 attributes either from file or dataset

        Parameters
        ----------
        dset : str
            Dataset to get attributes for, if None get file (global) attributes

        Returns
        -------
        attrs : dict
            Dataset or file attributes
        """
        return self.h5.h5.get_attrs(dset=dset)

    def get_dset_properties(self, dset):
        """
        Get dataset properties (shape, dtype, chunks)

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        shape : tuple
            Dataset array shape
        dtype : str
            Dataset array dtype
        chunks : tuple
            Dataset chunk size
        """
        return self.h5.h5.get_dset_properties(dset)

    def get_scale_factor(self, dset):
        """
        Get dataset scale factor

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        float
            Dataset scale factor, used to unscale int values to floats
        """
        return self.h5.h5.get_scale_factor(dset)

    def get_units(self, dset):
        """
        Get dataset units

        Parameters
        ----------
        dset : str
            Dataset to get units for

        Returns
        -------
        str
            Dataset units, None if not defined
        """
        return self.h5.h5.get_units(dset)

    def get_meta_arr(self, rec_name, rows=slice(None)):
        """Get a meta array by name (faster than DataFrame extraction).

        Parameters
        ----------
        rec_name : str
            Named record from the meta data to retrieve.
        rows : slice
            Rows of the record to extract.

        Returns
        -------
        meta_arr : np.ndarray
            Extracted array from the meta data record name.
        """
        meta_arr = self.h5.h5.get_meta_arr(rec_name, rows=rows)

        return meta_arr

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()


class MultiTimeSolarResource:
    """
    Class to handle solar resource data stored temporaly accross multiple .h5
    files
    """

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=SolarResource)


class MultiTimeNSRDB(MultiTimeResource):
    """
    Class to handle NSRDB data stored temporaly accross multiple .h5
    files
    """

    PREFIX = 'nsrdb'

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=NSRDB)


class MultiTimeWindResource(MultiTimeResource):
    """
    Class to handle wind resource data stored temporaly accross multiple .h5
    files
    """

    PREFIX = 'wtk'

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=WindResource)


class MultiTimeWaveResource(MultiTimeResource):
    """
    Class to handle wave resource data stored temporaly accross multiple .h5
    files
    """

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards.
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=WaveResource)
