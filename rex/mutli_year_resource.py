# -*- coding: utf-8 -*-
"""
Classes to handle resource accross multiple files
"""
import numpy as np
import os
from warnings import warn

from rex.resource import MultiH5, Resource
from rex.utilities.exceptions import ResourceKeyError
from rex.utilities.parse_keys import parse_keys, parse_slice
from rex.utilities.utilities import parse_year


class MultiYearH5:
    """
    Class to handle multiple years of h5 Resources
    """

    def __init__(self, h5_dir, prefix='', suffix='.h5', res_cls=Resource,
                 **res_cls_kwargs):
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
        """
        self.h5_dir = h5_dir
        self._year_map = self._map_file_years(h5_dir, prefix=prefix,
                                              suffix=suffix)
        self._h5_map = self._map_file_instances(set(self._year_map.values()),
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
        [type]
            [description]
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
    def _map_file_years(h5_dir, prefix='', suffix='.h5'):
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
                except RuntimeError:
                    msg = ('Could not file a valid year in {}'
                           .format(file))
                    warn(msg)

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

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map.values():
            f.close()


class MultiYearResource:
    """
    Class to handle fine spatial resolution resource data stored in
    multiple .h5 files


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
        """
        self.h5_dir, prefix, suffix = MultiH5.multi_file_args(h5_path)
        if prefix is None:
            prefix = self.PREFIX

        if suffix is None:
            suffix = self.SUFFIX

        self._meta = None
        self._time_index = None
        self._coords = None
        # Map variables to their .h5 files
        cls_kwargs = {'unscale': unscale, 'str_decode': str_decode,
                      'hsds': hsds}
        self._h5 = MultiYearH5(self.h5_dir, prefix=prefix, suffix=suffix,
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
        return self.h5.shape[0]

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

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

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
        if self._meta is None:
            if 'meta' in self.h5:
                self._meta = self._get_meta('meta', slice(None))
            else:
                raise ResourceKeyError("'meta' is not a valid dataset")

        return self._meta

    @property
    def time_index(self):
        """
        DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Resource datetime index
        """
        if self._time_index is None:
            if 'time_index' in self.h5:
                self._time_index = self._get_time_index('time_index',
                                                        slice(None))
            else:
                raise ResourceKeyError("'time_index' is not a valid dataset!")

        return self._time_index

    @property
    def coordinates(self):
        """
        Coordinates: (lat, lon) pairs

        Returns
        -------
        coords : ndarray
            Array of (lat, lon) pairs for each site in meta
        """
        if self._coords is None:
            if 'coordinates' in self.h5:
                self._coords = self._get_coords('coordinates', slice(None))
            else:
                raise ResourceKeyError("'coordinates' is not a valid dataset!")

        return self._coords

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
        ds_slice : int | list | slice
            tuple describing slice of time_index to extract

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
        ds_slice : int | list | slice
            Pandas slicing describing which sites and columns to extract

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
        ds_slice : int | list | slice
            tuple describing slice of coordinates to extract

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
        [summary]

        Parameters
        ----------
        year_slice : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        s = year_slice[0]
        e = year_slice[-1] + 1
        if (e - s) == len(year_slice):
            year_slice = slice(s, e, None)

        return year_slice

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
        if ds_name not in self.datasets:
            raise ResourceKeyError('{} not in {}'
                                   .format(ds_name, self.datasets))

        ds_slice = parse_slice(ds_slice)
        year_map = self.h5.time_index[ds_slice[0]].year
        out = []
        for year in np.unique(year_map):
            year_slice = np.where(year_map == year)[0]
            year_slice = (self._check_year_slice(year_slice), ) + ds_slice[1:]
            out.append(self._get_year_ds(ds_name, year, year_slice))

        return np.vstack(out)

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()
