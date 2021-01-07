# -*- coding: utf-8 -*-
"""
Classes to handle resource data stored over multiple files
"""
import numpy as np
import os
from warnings import warn

from rex.multi_file_resource import MultiH5Path
from rex.resource import Resource
from rex.renewable_resource import (NSRDB, SolarResource, WindResource,
                                    WaveResource)
from rex.utilities.exceptions import ResourceWarning
from rex.utilities.parse_keys import parse_keys, parse_slice


class MultiTimeH5:
    """
    Class to handle h5 Resources stored over multiple temporal files
    """
    def __init__(self, h5_dir, prefix='', suffix='.h5',
                 res_cls=Resource, hsds=False, **res_cls_kwargs):
        """
        Parameters
        ----------
        h5_dir : str
            Path to directory containing 5min .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        res_cls : obj
            Resource class to use to open and access resource data
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self.h5_dir = h5_dir
        self._file_map = self._map_files(h5_dir, prefix=prefix,
                                         suffix=suffix, hsds=hsds)
        res_cls_kwargs.update({'hsds': hsds})
        self._h5_map = self._map_file_instances(list(self._file_map.values()),
                                                res_cls=res_cls,
                                                **res_cls_kwargs)
        self._datasets = None
        self._shape = None
        self._time_index = None
        self._time_slice_map = []
        self._i = 0

    def __repr__(self):
        msg = ("{} for {}:\n Contains data from {} files"
               .format(self.__class__.__name__, self.h5_dir, len(self)))
        return msg

    def __getitem__(self, file):
        if file in self.files:
            path = self._file_map[file]
            h5 = self._h5_map[path]
        else:
            raise ValueError('{} is invalid, must be one of: {}'
                             .format(file, self.files))

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
        Available files

        Returns
        -------
        list
        """
        return sorted(self._file_map)

    @property
    def h5_files(self):
        """
        .h5 files data is being sourced from

        Returns
        -------
        list
        """
        return sorted(self._h5_map)

    @property
    def h5(self):
        """
        open h5 file handler for a single .h5 file

        Returns
        -------
        h5py.File
        """
        return self._h5_map[self.h5_files[0]]

    @property
    def datasets(self):
        """
        Available datasets

        Returns
        -------
        list
        """
        if self._datasets is None:
            self._datasets = self.h5.datasets

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
            for file in self.files:
                path = self._file_map[file]
                h5 = self._h5_map[path]
                ti = h5.time_index
                time_slice_map.append(np.full(len(ti), file))
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
    def _map_local_files(h5_dir, prefix='', suffix='.h5'):
        """
        Map local file paths to file_name

        Parameters
        ----------
        h5_dir : str
            Path to directory containing Resource .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files

        Returns
        -------
        file_map : dict
            Dictionary mapping file names to file paths
        """
        file_map = {}
        for file in sorted(os.listdir(h5_dir)):
            if file.startswith(prefix) and file.endswith(suffix):
                path = os.path.join(h5_dir, file)
                if file not in file_map:
                    file_map[file] = path
                else:
                    msg = ('WARNING: Skipping {} as {} has already been '
                           'parsed'.format(path, file))
                    warn(msg, ResourceWarning)

        return file_map

    @staticmethod
    def _map_hsds_files(hsds_dir, prefix='', suffix='.h5'):
        """
        Map hsds file paths to file names

        Parameters
        ----------
        hsds_dir : str
            HSDS directory containing Resource .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files

        Returns
        -------
        file_map : dict
            Dictionary mapping file names to file paths
        """
        import h5pyd

        file_map = {}
        if not hsds_dir.endswith('/'):
            hsds_dir += '/'

        with h5pyd.Folder(hsds_dir) as f:
            for file in f:
                if file.startswith(prefix) and file.endswith(suffix):
                    path = os.path.join(hsds_dir, file)
                    if file not in file_map:
                        file_map[file] = path
                    else:
                        msg = ('WARNING: Skipping {} as {} has already '
                               ' been parsed'.format(path, file))
                        warn(msg, ResourceWarning)

        return file_map

    @classmethod
    def _map_files(cls, h5_dir, prefix='', suffix='.h5', hsds=False):
        """
        Map file paths to file names

        Parameters
        ----------
        h5_dir : str
            Path to directory containing Resource .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS

        Returns
        -------
        file_map : dict
            Dictionary mapping file name to file paths
        """
        if hsds:
            file_map = cls._map_hsds_files(h5_dir, prefix=prefix,
                                           suffix=suffix)
        else:
            file_map = cls._map_local_files(h5_dir, prefix=prefix,
                                            suffix=suffix)

        return file_map

    @staticmethod
    def _map_file_instances(h5_files, res_cls=Resource, **res_cls_kwargs):
        """
        Open all .h5 files and map the open h5py instances to the
        associated file paths

        Parameters
        ----------
        h5_files : list
            List of .h5 files to open
        Returns
        -------
        h5_map : dict
            Dictionary mapping file paths to open resource instances
        """
        h5_map = {}
        for f_path in h5_files:
            h5_map[f_path] = res_cls(f_path, **res_cls_kwargs)

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
        for f in self._h5_map.values():
            f.close()


class MultiTimeResource:
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
    PREFIX = ''
    SUFFIX = '.h5'

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False,
                 res_cls=Resource):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
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
        self.h5_dir, prefix, suffix = MultiH5Path.multi_file_args(h5_path)
        if prefix is None:
            prefix = self.PREFIX

        if suffix is None:
            suffix = self.SUFFIX

        self._time_index = None
        # Map variables to their .h5 files
        cls_kwargs = {'unscale': unscale, 'str_decode': str_decode,
                      'hsds': hsds}
        self._h5 = MultiTimeH5(self.h5_dir, prefix=prefix, suffix=suffix,
                               res_cls=res_cls, **cls_kwargs)
        self.h5_files = self._h5.h5_files
        self.h5_file = self.h5_files[0]
        self._i = 0

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_dir)
        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return len(self.h5.time_index)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

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
        Open h5py File instance. If _group is not None return open Group

        Returns
        -------
        h5 : h5py.File | h5py.Group
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
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return self.global_attrs

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

    def get_scale(self, dset):
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
        return self.h5.h5.get_scale(dset)

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

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=SolarResource)


class MultiTimeNSRDB(MultiTimeResource):
    """
    Class to handle NSRDB data stored temporaly accross multiple .h5
    files
    """
    PREFIX = 'nsrdb'

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=NSRDB)


class MultiTimeWindResource(MultiTimeResource):
    """
    Class to handle wind resource data stored temporaly accross multiple .h5
    files
    """
    PREFIX = 'wtk'

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=WindResource)


class MultiTimeWaveResource(MultiTimeResource):
    """
    Class to handle wave resource data stored temporaly accross multiple .h5
    files
    """

    def __init__(self, h5_path, unscale=True, str_decode=True, hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=WaveResource)
