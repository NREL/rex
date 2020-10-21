# -*- coding: utf-8 -*-
"""
Classes to handle multiple years of resource data
"""
import numpy as np
import os
from warnings import warn

from rex.multi_file_resource import MultiH5Path
from rex.renewable_resource import (NSRDB, SolarResource, WindResource,
                                    WaveResource)
from rex.resource import Resource
from rex.utilities.exceptions import ResourceWarning
from rex.utilities.parse_keys import parse_keys, parse_slice
from rex.utilities.utilities import parse_year


class MultiYearH5:
    """
    Class to handle multiple years of h5 Resources
    """

    def __init__(self, h5_dir, prefix='', suffix='.h5', years=None,
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
        years : list, optional
            List of years to access, by default None
        res_cls : obj
            Resource class to use to open and access resource data
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self.h5_dir = h5_dir
        self._year_map = self._map_file_years(h5_dir, prefix=prefix,
                                              suffix=suffix, hsds=hsds,
                                              years=years)
        res_cls_kwargs.update({'hsds': hsds})
        self._h5_map = self._map_file_instances(list(self._year_map.values()),
                                                res_cls=res_cls,
                                                **res_cls_kwargs)
        self._datasets = None
        self._shape = None
        self._time_index = None
        self._i = 0

    def __repr__(self):
        msg = ("{} for {}:\n Contains data for {} year"
               .format(self.__class__.__name__, self.h5_dir, len(self)))
        return msg

    def __len__(self):
        return len(self.years)

    def __getitem__(self, year):
        if isinstance(year, str):
            year = int(year)

        if year in self.years:
            path = self._year_map[year]
            h5 = self._h5_map[path]
        else:
            raise ValueError('{} is invalid, must be one of: {}'
                             .format(year, self.years))

        return h5

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.years):
            self._i = 0
            raise StopIteration

        year = self.years[self._i]
        self._i += 1

        return year

    def __contains__(self, year):
        return year in self.years

    @property
    def attrs(self):
        """
        Global .h5 file attributes sourced from first .h5 file

        Returns
        -------
        attrs : dict
            .h5 file attributes sourced from first .h5 file
        """
        attrs = dict(self.h5.attrs)
        return attrs

    @property
    def years(self):
        """
        Available years

        Returns
        -------
        list
            List of dataset present in .h5 files
        """
        return sorted(self._year_map)

    @property
    def h5_files(self):
        """
        .h5 files data is being sourced from

        Returns
        -------
        list
            List of .h5 files data is being sourced form
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
            List of available datasets
        """
        if self._datasets is None:
            self._datasets = self.h5.datasets

        return self._datasets

    @property
    def shape(self):
        """
        Dataset shape (time, sites)

        Returns
        -------
        tuple
        """
        if self._shape is None:
            self._shape = (len(self.time_index), len(self.h5))

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
            for year in self.years:
                path = self._year_map[year]
                h5 = self._h5_map[path]
                if self._time_index is None:
                    self._time_index = h5.time_index
                else:
                    self._time_index = self._time_index.append(h5.time_index)

        return self._time_index

    @staticmethod
    def _map_local_files(h5_dir, prefix='', suffix='.h5'):
        """
        Map local file paths to year for which it contains data

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
        year_map : dict
            Dictionary mapping years to file paths
        """
        year_map = {}
        for file in sorted(os.listdir(h5_dir)):
            if file.startswith(prefix) and file.endswith(suffix):
                try:
                    year = parse_year(file)
                    path = os.path.join(h5_dir, file)
                    if year not in year_map:
                        year_map[year] = path
                    else:
                        msg = ('WARNING: Skipping {} as {} has already been '
                               'parsed'.format(path, year))
                        warn(msg, ResourceWarning)
                except RuntimeError:
                    msg = ('WARNING: Could not find a valid year in {}'
                           .format(file))
                    warn(msg, ResourceWarning)

        return year_map

    @staticmethod
    def _map_hsds_files(hsds_dir, prefix='', suffix='.h5'):
        """
        Map hsds file paths to year for which it contains data

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
        year_map : dict
            Dictionary mapping years to file paths
        """
        import h5pyd

        year_map = {}
        if not hsds_dir.endswith('/'):
            hsds_dir += '/'

        with h5pyd.Folder(hsds_dir) as f:
            for file in f:
                if file.startswith(prefix) and file.endswith(suffix):
                    try:
                        year = parse_year(file)
                        path = os.path.join(hsds_dir, file)
                        if year not in year_map:
                            year_map[year] = path
                        else:
                            msg = ('WARNING: Skipping {} as {} has already '
                                   ' been parsed'.format(path, year))
                            warn(msg, ResourceWarning)
                    except RuntimeError:
                        msg = ('WARNING: Could not find a valid year in {}'
                               .format(file))
                        warn(msg, ResourceWarning)

        return year_map

    @staticmethod
    def _get_years(year_map, years):
        """
        [summary]

        Parameters
        ----------
        year_map : dict
            Dictionary mapping years to file paths
        years : list
            List of years of interest. Should be a subset of years in year_map

        Returns
        -------
        new_map : dict
            Dictionary mapping requested years to file paths
        """
        new_map = {}
        for year in years:
            if not isinstance(year, int):
                year = int(year)

            if year in year_map:
                new_map[year] = year_map[year]
            else:
                msg = ('A file for {} is unavailable!'.format(year))
                warn(msg, ResourceWarning)

        if not new_map:
            msg = ('No files were found for the given years:\n{}'
                   .format(years))
            raise RuntimeError(msg)

        return new_map

    @staticmethod
    def _map_file_years(h5_dir, prefix='', suffix='.h5', hsds=False,
                        years=None):
        """
        Map file paths to year for which it contains data

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
        years : list, optional
            List of years to access, by default None

        Returns
        -------
        year_map : dict
            Dictionary mapping years to file paths
        """
        if hsds:
            year_map = MultiYearH5._map_hsds_files(h5_dir, prefix=prefix,
                                                   suffix=suffix)
        else:
            year_map = MultiYearH5._map_local_files(h5_dir, prefix=prefix,
                                                    suffix=suffix)

        if years is not None:
            year_map = MultiYearH5._get_years(year_map, years)

        return year_map

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

    def year_index(self, year):
        """
        Extract time_index for a specific year

        Parameters
        ----------
        year : int
            Year to extract time_index for

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Resource datetime index for desired year
        """
        return self.time_index[self.time_index.year == year]

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map.values():
            f.close()


class MultiYearResource:
    """
    Class to handle multiple years of resource data stored accross multiple
    .h5 files

    Examples
    --------

    Extracting the resource's Datetime Index

    >>> path = '$TESTDATADIR/nsrdb/ri_100_nsrdb_*.h5'
    >>> with MultiYearResource(path) as res:
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

    >>> with MultiYearResource(path) as res:
    >>>     print(res.h5_files)

    ['/Users/mrossol/Git_Repos/rex/tests/data/nsrdb/ri_100_nsrdb_2012.h5',
     '/Users/mrossol/Git_Repos/rex/tests/data/nsrdb/ri_100_nsrdb_2013.h5']

    Data slicing works the same as with "Resource" except axis 0 now covers
    2012 and 2013

    >>> with MultiYearResource(path) as res:
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

    >>> with MultiYearResource(path) as res:
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

    You can also request a specific year of data using a string representation
    of the year of interest
    NOTE: you can also request a list of years using strings

    >>> with MultiYearResource(path) as res:
    >>>     temperature = res['air_temperature', '2012'] # every 100th timestep
    >>>
    >>> temperature
    [[4. 5. 5. ... 4. 3. 4.]
     [4. 4. 5. ... 4. 3. 4.]
     [4. 4. 5. ... 4. 3. 4.]
     ...
     [1. 1. 2. ... 0. 0. 0.]
     [1. 1. 2. ... 0. 0. 1.]
     [1. 1. 2. ... 0. 0. 1.]]
    >>> temperature.shape
    (17520, 100)
    """
    PREFIX = ''
    SUFFIX = '.h5'

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False, res_cls=Resource):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        years : list, optional
            List of years to access, by default None
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
        self._h5 = MultiYearH5(self.h5_dir, prefix=prefix, suffix=suffix,
                               years=years, res_cls=res_cls, **cls_kwargs)
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
        return self.h5.shape[0]

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
            out = self._get_time_index(ds_slice)
        elif ds.endswith('meta'):
            out = self._get_meta(ds, ds_slice)
        elif ds.endswith('coordinates'):
            out = self._get_coords(ds, ds_slice)
        else:
            out = self._get_ds(ds, ds_slice)

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
            Open h5py File or Group instance
        """
        h5 = self._h5

        return h5

    @property
    def datasets(self):
        """
        Datasets available

        Returns
        -------
        list
            List of datasets
        """
        return self._h5.datasets

    @property
    def dsets(self):
        """
        Datasets available

        Returns
        -------
        list
            List of datasets
        """
        return self.datasets

    @property
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
            Shape of resource variable arrays (timesteps, sites)
        """
        return self.h5.shape

    @property
    def meta(self):
        """
        Meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
            Resource Meta Data
        """

        return self.h5.h5.meta

    @property
    def time_index(self):
        """
        DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Resource datetime index
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
            Array of (lat, lon) pairs for each site in meta
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
        return dict(self.h5.attrs)

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
        if dset is None:
            attrs = dict(self.h5.attrs)
        else:
            attrs = dict(self.h5.h5[dset].attrs)

        return attrs

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
        ds = self.h5.h5[dset]
        shape, dtype, chunks = ds.shape, ds.dtype, ds.chunks
        if isinstance(chunks, dict):
            chunks = tuple(chunks.get('dims', None))

        return shape, dtype, chunks

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
        return self.h5.h5[dset].attrs.get('scale_factor', 1)

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
        return self.h5.h5[dset].attrs.get('units', None)

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

    def _get_time_index(self, ds_slice):
        """
        Extract and convert time_index to pandas Datetime Index

        Parameters
        ----------
        ds_name : str
            Dataset to extract time_index from
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from
            time_index, each arg is for a sequential axis

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Vector of datetime stamps
        """
        ds_slice = parse_slice(ds_slice)
        time_index = self.h5.time_index[ds_slice[0]]

        return time_index

    def _get_meta(self, ds_name, ds_slice):
        """
        Extract and convert meta to a pandas DataFrame

        Parameters
        ----------
        ds_name : str
            Dataset to extract meta from
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray, str) of what sites and columns
            to extract from meta

        Returns
        -------
        meta : pandas.Dataframe
            Dataframe of location meta data
        """
        meta = self.h5.h5._get_meta(ds_name, ds_slice)

        return meta

    def _get_coords(self, ds_name, ds_slice):
        """
        Extract coordinates (lat, lon) pairs

        Parameters
        ----------
        ds_name : str
            Dataset to extract coordinates from
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from
            coordinates, each arg is for a sequential axis

        Returns
        -------
        coords : ndarray
            Array of (lat, lon) pairs for each site in meta
        """
        coords = self.h5.h5._get_coords(ds_name, ds_slice)

        return coords

    def _get_year_ds(self, ds_name, year, year_slice):
        """
        Extract dataset data for given year

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        year : int
            Year to extract data for
        year_slice : int | list | slice
            tuple describing slice of dataset array to extract

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        out = self.h5[year]._get_ds(ds_name, year_slice)

        return out

    @staticmethod
    def _check_year_slice(year_slice):
        """
        Check to see if year positions can be represented as a slice

        Parameters
        ----------
        year_slice : ndarray | list
            List of temporal positions

        Returns
        -------
        year_slice : ndarray | list | slice
            Slice covering range of positions to extract if possible
        """
        s = year_slice[0]
        e = year_slice[-1] + 1
        if (e - s) == len(year_slice):
            year_slice = slice(s, e, None)

        return year_slice

    @staticmethod
    def _check_for_years(time_slice):
        """
        Check to see if temporal slice is a year (str) or list of years (strs)
        to extract data for

        Parameters
        ----------
        time_slice : list | slice | int | str
            Temporal slice to extract

        Returns
        -------
        check : bool
            True if temporal slice is a year (str) or list of years (strs),
            else False
        """
        check = False
        if isinstance(time_slice, (list, tuple)):
            time_slice = time_slice[0]

        if isinstance(time_slice, str):
            check = True

        return check

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
        if self._check_for_years(ds_slice[0]):
            years = ds_slice[0]
            year_slice = (slice(None), ) + ds_slice[1:]
            if isinstance(years, str):
                years = [years]

            for year in years:
                year = int(year)
                out.append(self._get_year_ds(ds_name, year, year_slice))

        else:
            time_index = self.h5.time_index[ds_slice[0]]
            year_map = time_index.year
            for year in year_map.unique():
                year_index = self.h5.year_index(year)
                year_slice = \
                    np.where(year_index.isin(time_index[year_map == year]))[0]
                year_slice = \
                    (self._check_year_slice(year_slice), ) + ds_slice[1:]
                out.append(self._get_year_ds(ds_name, year, year_slice))

        return np.concatenate(out, axis=0)

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()


class MultiYearSolarResource:
    """
    Class to handle multiple years of solar resource data stored accross
    multiple .h5 files
    """
    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=SolarResource)


class MultiYearNSRDB(MultiYearResource):
    """
    Class to handle multiple years of NSRDB data stored accross
    multiple .h5 files
    """
    PREFIX = 'nsrdb'

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=NSRDB)


class MultiYearWindResource(MultiYearResource):
    """
    Class to handle multiple years of wind resource data stored accross
    multiple .h5 files
    """
    PREFIX = 'wtk'

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=WindResource)


class MultiYearWaveResource:
    """
    Class to handle multiple years of wave resource data stored accross
    multiple .h5 files
    """

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, res_cls=WaveResource)
