# -*- coding: utf-8 -*-
"""
Classes to handle resource data
"""
import h5py
import numpy as np
import os
import pandas as pd

from rex.sam_resource import SAMResource
from rex.utilities.parse_keys import parse_keys, parse_slice
from rex.utilities.exceptions import ResourceKeyError, ResourceRuntimeError
from rex.utilities.utilities import check_tz


class ResourceDataset:
    """
    h5py.Dataset wrapper for Resource .h5 files
    """
    def __init__(self, ds, scale_attr='scale_factor', add_attr='add_offset',
                 unscale=True):
        """
        Parameters
        ----------
        ds : h5py.dataset
            Open .h5 dataset instance to extract data from
        scale_attr : str, optional
            Name of scale factor attribute, by default 'scale_factor'
        add_attr : str, optional
            Name of add offset attribute, by default 'add_offset'
        unscale : bool, optional
            Flag to unscale dataset data, by default True
        """
        self._ds = ds
        self._scale_factor = self.ds.attrs.get(scale_attr, 1)
        self._adder = self.ds.attrs.get(add_attr, 0)
        self._unscale = unscale

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.ds.name)

        return msg

    def __getitem__(self, ds_slice):
        ds_slice = parse_slice(ds_slice)

        return self._get_ds_slice(ds_slice)

    @property
    def ds(self):
        """
        Open Dataset instance

        Returns
        -------
        h5py(d).Dataset
        """
        return self._ds

    @property
    def shape(self):
        """
        Dataset shape

        Returns
        -------
        tuple
        """
        return self.ds.shape

    @property
    def size(self):
        """
        Dataset size

        Returns
        -------
        int
        """
        return self.ds.size

    @property
    def dtype(self):
        """
        Dataset dtype

        Returns
        -------
        str | numpy.dtype
        """
        return self.ds.dtype

    @property
    def chunks(self):
        """
        Dataset chunk size

        Returns
        -------
        tuple
        """
        chunks = self.ds.chunks
        if isinstance(chunks, dict):
            chunks = tuple(chunks.get('dims', None))

        return chunks

    @property
    def scale_factor(self):
        """
        Dataset scale factor

        Returns
        -------
        float
        """
        return self._scale_factor

    @property
    def adder(self):
        """
        Dataset add offset

        Returns
        -------
        float
        """
        return self._adder

    @staticmethod
    def _check_slice(ds_slice):
        """
        Check ds_slice for lists, ensure lists are of the same len

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        list_len : int | None
            List lenght, None if none of the args are a list | ndarray
        multi_list : bool
            Flag if multiple list are provided in ds_slice
        """
        multi_list = False
        list_len = []
        for s in ds_slice:
            if isinstance(s, (list, np.ndarray)):
                list_len.append(len(s))

        if list_len:
            if len(list_len) > 1:
                multi_list = True

            list_len = list(set(list_len))
            if len(list_len) > 1:
                msg = ('shape mismatch: indexing arrays could not be '
                       'broadcast together with shapes {}'
                       .format(['({},)'.format(ln) for ln in list_len]))
                raise IndexError(msg)
            else:
                list_len = list_len[0]
        else:
            list_len = None

        return list_len, multi_list

    @staticmethod
    def _make_list_slices(ds_slice, list_len):
        """
        Duplicate slice arguements to enable zipping of list slices with
        non-list slices

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis
        list_len : int
            List lenght

        Returns
        -------
        zip_slices : list
            List of slices to extract for each entry in list slice
        """
        zip_slices = []
        for s in ds_slice:
            if not isinstance(s, (list, np.ndarray)):
                zip_slices.append([s] * list_len)
            else:
                zip_slices.append(s)

        return zip_slices

    @staticmethod
    def _list_to_slice(ds_slice):
        """
        Check ds_slice to see if it is an int, slice, or list. Return
        pieces required for fancy indexing based on input type.

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        ds_slice : slice
            Slice that encompasses the entire range
        ds_idx : ndarray
            Adjusted list to extract points of interest from sliced array
        """
        ds_idx = None
        if isinstance(ds_slice, (list, np.ndarray)):
            in_slice = np.array(ds_slice)
            if np.issubdtype(in_slice.dtype, np.dtype(bool)):
                in_slice = np.where(in_slice)[0]

            s = in_slice.min()
            e = in_slice.max() + 1
            ds_slice = slice(s, e, None)
            ds_idx = in_slice - s
        elif isinstance(ds_slice, slice):
            ds_idx = slice(None)

        return ds_slice, ds_idx

    @staticmethod
    def _get_out_arr_slice(arr_slice, start):
        """
        Determine slice of pre-build output array that is being filled

        Parameters
        ----------
        arr_slice : tuple
            Tuple of (int, slice, list, ndarray) for section of output array
            being extracted
        start : int
            Start of slice, used for list gets

        Returns
        -------
        out_slice : tuple
            Slice arguments of portion of output array to insert arr_slice
            into
        stop : int
            Stop of slice, used for list gets, will be new start upon
            iteration
        """
        out_slice = ()
        int_slice = ()
        int_start = start
        int_stop = start
        stop = start
        for s in arr_slice:
            if isinstance(s, slice):
                out_slice += (slice(None), )
                int_slice += (slice(None), )
            elif isinstance(s, int):
                if int_start == int_stop:
                    int_slice += (int_start, )
                    int_stop += 1
            elif isinstance(s, (list, tuple, np.ndarray)):
                list_len = len(s)
                if list_len == 1:
                    stop += 1
                    out_slice += ([start], )
                else:
                    stop += len(s)
                    out_slice += (slice(start, stop), )

        if not out_slice:
            out_slice += (start, )
            stop += 1
        elif all(s == slice(None) for s in out_slice):
            out_slice = int_slice
            stop = int_stop

        return out_slice, stop

    def _get_out_arr_shape(self, ds_slice):
        """
        Determine shape of output array

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        out_shape : tuple
            Shape of output array
        """
        ds_shape = self.shape
        out_shape = ()
        contains_list = False

        ds_slice += (slice(None), ) * (len(ds_shape) - len(ds_slice))
        for i, ax_slice in enumerate(ds_slice):
            if isinstance(ax_slice, slice):
                stop = ax_slice.stop
                if stop is None:
                    stop = ds_shape[i]

                out_shape += (len(range(*ax_slice.indices(stop))), )

            if isinstance(ax_slice, (list, tuple, np.ndarray)):
                if not contains_list:
                    out_shape += (len(ax_slice), )

                contains_list = True

        return out_shape

    def _extract_list_slice(self, ds_slice):
        """
        Optimize and extract list slice request along a single dimension

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            Extracted array of data from ds
        """
        out_slices = []
        chunks = self.chunks
        sort_idx = []
        list_len = None
        if chunks:
            for i, ax_slice in enumerate(ds_slice):
                c = chunks[i]
                if isinstance(ax_slice, (list, np.ndarray)):
                    if not isinstance(ax_slice, np.ndarray):
                        ax_slice = np.array(ax_slice)

                    idx = np.argsort(ax_slice)
                    sort_idx.append(np.argsort(idx))
                    ax_slice = ax_slice[idx]
                    diff = np.diff(ax_slice) > c
                    if np.any(diff):
                        pos = np.where(diff)[0] + 1
                        ax_slice = np.split(ax_slice, pos)
                        list_len = len(ax_slice)
                elif isinstance(ax_slice, slice):
                    sort_idx.append(slice(None))

                out_slices.append(ax_slice)
        else:
            out_slices = ds_slice

        if list_len is not None:
            out_shape = self._get_out_arr_shape(ds_slice)
            out_slices = self._make_list_slices(out_slices, list_len)

            out = np.zeros(out_shape, dtype=self.dtype)
            start = 0
            for s in zip(*out_slices):
                arr_slice, stop = self._get_out_arr_slice(s, start)
                out[arr_slice] = self._extract_ds_slice(s)
                start = stop

            out = out[tuple(sort_idx)]
        else:
            out = self._extract_ds_slice(ds_slice)

        return out

    def _extract_multi_list_slice(self, ds_slice, list_len):
        """
        Extract ds_slice that has multiple lists

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis
        list_len : int
            List lenght

        Returns
        -------
        out : ndarray
            Extracted array of data from ds
        """
        zip_slices = self._make_list_slices(ds_slice, list_len)

        out_shape = self._get_out_arr_shape(ds_slice)

        out = np.zeros(out_shape, dtype=self.dtype)
        start = 0
        for s in zip(*zip_slices):
            arr_slice, stop = self._get_out_arr_slice(s, start)
            arr = self._extract_ds_slice(s)
            out[arr_slice] = arr

            start = stop

        return out

    def _extract_ds_slice(self, ds_slice):
        """
        Extact ds_slice from ds using slices where possible

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            Extracted array of data from ds
        """
        slices = ()
        idx_slice = ()
        for ax_slice in ds_slice:
            ax_slice, ax_idx = self._list_to_slice(ax_slice)
            slices += (ax_slice,)
            if ax_idx is not None:
                idx_slice += (ax_idx,)

        out = self.ds[slices]

        # check to see if idx_slice needs to be applied
        if any(s != slice(None) if isinstance(s, slice) else True
               for s in idx_slice):
            out = out[idx_slice]

        return out

    def _unscale_data(self, data):
        """
        Unscale dataset data

        Parameters
        ----------
        data : ndarray
            Native dataset array

        Returns
        -------
        data : ndarray
            Unscaled dataset array
        """
        data = data.astype('float32')

        if self.adder != 0:
            data *= self.scale_factor
            data += self.adder
        else:
            data /= self.scale_factor

        return data

    def _get_ds_slice(self, ds_slice):
        """
        Get ds_slice from ds as efficiently as possible, unscale if desired

        Parameters
        ----------
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            Extracted array of data from ds
        """
        list_len, multi_list = self._check_slice(ds_slice)
        if list_len is not None:
            if multi_list:
                out = self._extract_multi_list_slice(ds_slice, list_len)
            else:
                out = self._extract_list_slice(ds_slice)
        else:
            out = self._extract_ds_slice(ds_slice)

        if self._unscale:
            out = self._unscale_data(out)

        return out

    @classmethod
    def extract(cls, ds, ds_slice, scale_attr='scale_factor',
                add_attr='add_offset', unscale=True):
        """
        Extract data from Resource Dataset

        Parameters
        ----------
        ds : h5py.dataset
            Open .h5 dataset instance to extract data from
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis
        scale_attr : str, optional
            Name of scale factor attribute, by default 'scale_factor'
        add_attr : str, optional
            Name of add offset attribute, by default 'add_offset'
        unscale : bool, optional
            Flag to unscale dataset data, by default True
        """
        dset = cls(ds, scale_attr=scale_attr, add_attr=add_attr,
                   unscale=unscale)

        return dset[ds_slice]


class Resource:
    """
    Base class to handle resource .h5 files

    Examples
    --------

    Extracting the resource's Datetime Index

    >>> file = '$TESTDATADIR/nsrdb/ri_100_nsrdb_2012.h5'
    >>> with Resource(file) as res:
    >>>     ti = res.time_index
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:30:00',
                   '2012-01-01 01:00:00', '2012-01-01 01:30:00',
                   '2012-01-01 02:00:00', '2012-01-01 02:30:00',
                   '2012-01-01 03:00:00', '2012-01-01 03:30:00',
                   '2012-01-01 04:00:00', '2012-01-01 04:30:00',
                   ...
                   '2012-12-31 19:00:00', '2012-12-31 19:30:00',
                   '2012-12-31 20:00:00', '2012-12-31 20:30:00',
                   '2012-12-31 21:00:00', '2012-12-31 21:30:00',
                   '2012-12-31 22:00:00', '2012-12-31 22:30:00',
                   '2012-12-31 23:00:00', '2012-12-31 23:30:00'],
                  dtype='datetime64[ns]', length=17568, freq=None)

    Efficient slicing of the Datetime Index

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', 1]
    >>>
    >>> ti
    2012-01-01 00:30:00

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', :10]
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:00:00', '2012-01-01 00:30:00',
                   '2012-01-01 01:00:00', '2012-01-01 01:30:00',
                   '2012-01-01 02:00:00', '2012-01-01 02:30:00',
                   '2012-01-01 03:00:00', '2012-01-01 03:30:00',
                   '2012-01-01 04:00:00', '2012-01-01 04:30:00'],
                  dtype='datetime64[ns]', freq=None)

    >>> with Resource(file) as res:
    >>>     ti = res['time_index', [1, 2, 4, 8, 9]
    >>>
    >>> ti
    DatetimeIndex(['2012-01-01 00:30:00', '2012-01-01 01:00:00',
                   '2012-01-01 02:00:00', '2012-01-01 04:00:00',
                   '2012-01-01 04:30:00'],
                  dtype='datetime64[ns]', freq=None)

    Extracting resource's site metadata

    >>> with Resource(file) as res:
    >>>     meta = res.meta
    >>>
    >>> meta
            latitude  longitude   elevation  timezone    country ...
    0      41.29     -71.86    0.000000        -5           None ...
    1      41.29     -71.82    0.000000        -5           None ...
    2      41.25     -71.82    0.000000        -5           None ...
    3      41.33     -71.82   15.263158        -5  United States ...
    4      41.37     -71.82   25.360000        -5  United States ...
    ..       ...        ...         ...       ...            ... ...
    95     41.25     -71.66    0.000000        -5           None ...
    96     41.89     -71.66  153.720000        -5  United States ...
    97     41.45     -71.66   35.440000        -5  United States ...
    98     41.61     -71.66  140.200000        -5  United States ...
    99     41.41     -71.66   35.160000        -5  United States ...
    [100 rows x 10 columns]

    Efficient slicing of the metadata

    >>> with Resource(file) as res:
    >>>     meta = res['meta', 1]
    >>>
    >>> meta
       latitude  longitude  elevation  timezone country state county urban ...
    1     41.29     -71.82        0.0        -5    None  None   None  None ...

    >>> with Resource(file) as res:
    >>>     meta = res['meta', :5]
    >>>
    >>> meta
       latitude  longitude  elevation  timezone        country ...
    0     41.29     -71.86   0.000000        -5           None ...
    1     41.29     -71.82   0.000000        -5           None ...
    2     41.25     -71.82   0.000000        -5           None ...
    3     41.33     -71.82  15.263158        -5  United States ...
    4     41.37     -71.82  25.360000        -5  United States ...

    >>> with Resource(file) as res:
    >>>     tz = res['meta', :, 'timezone']
    >>>
    >>> tz
    0    -5
    1    -5
    2    -5
    3    -5
    4    -5
         ..
    95   -5
    96   -5
    97   -5
    98   -5
    99   -5
    Name: timezone, Length: 100, dtype: int64

    >>> with Resource(file) as res:
    >>>     lat_lon = res['meta', :, ['latitude', 'longitude']]
    >>>
    >>> lat_lon
        latitude  longitude
    0      41.29     -71.86
    1      41.29     -71.82
    2      41.25     -71.82
    3      41.33     -71.82
    4      41.37     -71.82
    ..       ...        ...
    95     41.25     -71.66
    96     41.89     -71.66
    97     41.45     -71.66
    98     41.61     -71.66
    99     41.41     -71.66
    [100 rows x 2 columns]

    Extracting resource variables (datasets)

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed']
    >>>
    >>> wspd
    [[12. 12. 12. ... 12. 12. 12.]
     [12. 12. 12. ... 12. 12. 12.]
     [12. 12. 12. ... 12. 12. 12.]
     ...
     [14. 14. 14. ... 14. 14. 14.]
     [15. 15. 15. ... 15. 15. 15.]
     [15. 15. 15. ... 15. 15. 15.]]

    Efficient slicing of variables

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed', :2]
    >>>
    >>> wspd
    [[12. 12. 12. 12. 12. 12. 53. 53. 53. 53. 53. 12. 53.  1.  1. 12. 12. 12.
       1.  1. 12. 53. 53. 53. 12. 12. 12. 12. 12.  1. 12. 12.  1. 12. 12. 53.
      12. 53.  1. 12.  1. 53. 53. 12. 12. 12. 12.  1.  1.  1. 12. 12.  1.  1.
      12. 12. 53. 53. 53. 12. 12. 53. 53. 12. 12. 12. 12. 12. 12.  1. 53.  1.
      53. 12. 12. 53. 53.  1.  1.  1. 53. 12.  1.  1. 53. 53. 53. 12. 12. 12.
      12. 12. 12. 12.  1. 12.  1. 12. 12. 12.]
     [12. 12. 12. 12. 12. 12. 53. 53. 53. 53. 53. 12. 53.  1.  1. 12. 12. 12.
       1.  1. 12. 53. 53. 53. 12. 12. 12. 12. 12.  1. 12. 12.  1. 12. 12. 53.
      12. 53.  1. 12.  1. 53. 53. 12. 12. 12. 12.  1.  1.  1. 12. 12.  1.  1.
      12. 12. 53. 53. 53. 12. 12. 53. 53. 12. 12. 12. 12. 12. 12.  1. 53.  1.
      53. 12. 12. 53. 53.  1.  1.  1. 53. 12.  1.  1. 53. 53. 53. 12. 12. 12.
      12. 12. 12. 12.  1. 12.  1. 12. 12. 12.]]

    >>> with Resource(file) as res:
    >>>     wspd = res['wind_speed', :, [2, 3]]
    >>>
    >>> wspd
    [[12. 12.]
     [12. 12.]
     [12. 12.]
     ...
     [14. 14.]
     [15. 15.]
     [15. 15.]]
    """
    SCALE_ATTR = 'scale_factor'
    ADD_ATTR = 'add_offset'
    UNIT_ATTR = 'units'

    def __init__(self, h5_file, unscale=True, hsds=False, str_decode=True,
                 group=None):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
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
        self.h5_file = h5_file
        if hsds:
            import h5pyd
            self._h5 = h5pyd.File(self.h5_file, 'r')
        else:
            self._h5 = h5py.File(self.h5_file, 'r')

        self._group = group
        self._unscale = unscale
        self._meta = None
        self._time_index = None
        self._lat_lon = None
        self._str_decode = str_decode
        self._i = 0

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self.h5_file)

        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return self.h5['meta'].shape[0]

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        _, ds_name = os.path.split(ds)
        if ds_name.startswith('time_index'):
            out = self._get_time_index(ds, ds_slice)
        elif ds_name.startswith('meta'):
            out = self._get_meta(ds, ds_slice)
        elif ds_name.startswith('coordinates'):
            out = self._get_coords(ds, ds_slice)
        elif 'SAM' in ds_name:
            site = ds_slice[0]
            if isinstance(site, int):
                out = self._get_SAM_df(ds, site)  # pylint: disable=E1111
            else:
                msg = "Can only extract SAM DataFrame for a single site"
                raise ResourceRuntimeError(msg)
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

    @classmethod
    def _get_datasets(cls, h5_obj, group=None):
        """
        Search h5 file instance for Datasets

        Parameters
        ----------
        h5_obj : h5py.File | h5py.Group
            Open h5py File or Group instance to search

        Returns
        -------
        dsets : list
            List of datasets in h5_obj
        """
        dsets = []
        for name in h5_obj:
            sub_obj = h5_obj[name]
            if isinstance(sub_obj, h5py.Group):
                dsets.extend(cls._get_datasets(sub_obj, group=name))
            else:
                dset_name = name
                if group is not None:
                    dset_name = "{}/{}".format(group, dset_name)

                dsets.append(dset_name)

        return dsets

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
        if self._group is not None:
            h5 = h5[self._group]

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
        return self._get_datasets(self.h5)

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
    def groups(self):
        """
        Groups available

        Returns
        -------
        groups : list
            List of groups
        """
        groups = []
        for name in self.h5:
            if isinstance(self.h5[name], h5py.Group):
                groups.append(name)

        return groups

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
        _shape = (self.h5['time_index'].shape[0], self.h5['meta'].shape[0])
        return _shape

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
        lat_lon : ndarray
            Array of (lat, lon) pairs for each site in meta
        """
        return self.lat_lon

    @property
    def lat_lon(self):
        """
        Extract (latitude, longitude) pairs

        Returns
        -------
        lat_lon : ndarray
        """
        if self._lat_lon is None:
            if 'coordinates' in self:
                self._lat_lon = self._get_coords('coordinates', slice(None))
            else:
                self._lat_lon = self.meta
                lat_lon_cols = ['latitude', 'longitude']
                for c in self.meta.columns:
                    if c.lower() in ['lat', 'latitude']:
                        lat_lon_cols[0] = c
                    elif c.lower() in ['lon', 'long', 'longitude']:
                        lat_lon_cols[1] = c

                self._lat_lon = self._lat_lon[lat_lon_cols].values

        return self._lat_lon

    @property
    def global_attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        global_attrs : dict
        """
        return dict(self.h5.attrs)

    @property
    def attrs(self):
        """
        Global (file) attributes

        Returns
        -------
        attrs : dict
            .h5 file attributes sourced from first .h5 file
        """
        return self.global_attrs

    @staticmethod
    def df_str_decode(df):
        """Decode a dataframe with byte string columns into ordinary str cols.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with some columns being byte strings.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with str columns instead of byte str columns.
        """
        for col in df:
            if (np.issubdtype(df[col].dtype, np.object_)
                    and isinstance(df[col].values[0], bytes)):
                df[col] = df[col].copy().str.decode('utf-8', 'ignore')

        return df

    def open_dataset(self, ds_name):
        """
        Open resource dataset

        Parameters
        ----------
        ds_name : str
            Dataset name to open

        Returns
        -------
        ds : ResourceDataset
            Resource for open resource dataset
        """
        if ds_name not in self.datasets:
            raise ResourceKeyError('{} not in {}'
                                   .format(ds_name, self.datasets))

        ds = ResourceDataset(self.h5[ds_name], scale_attr=self.SCALE_ATTR,
                             add_attr=self.ADD_ATTR, unscale=self._unscale)

        return ds

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
            attrs = dict(self.h5[dset].attrs)

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
        ds = self.h5[dset]
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
        return self.h5[dset].attrs.get(self.SCALE_ATTR, 1)

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
        return self.h5[dset].attrs.get(self.UNIT_ATTR, None)

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
        if 'meta' in self.h5:
            meta_arr = self.h5['meta'][rec_name, rows]
            if self._str_decode and np.issubdtype(meta_arr.dtype, np.bytes_):
                meta_arr = np.char.decode(meta_arr, encoding='utf-8')
        else:
            raise ResourceKeyError("'meta' is not a valid dataset")

        return meta_arr

    def _get_time_index(self, ds_name, ds_slice):
        """
        Extract and convert time_index to pandas Datetime Index

        Parameters
        ----------
        ds_name : str
            Dataset to extract time_index from
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from
            time_index

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Vector of datetime stamps
        """
        ds_slice = parse_slice(ds_slice)
        time_index = self.h5[ds_name]
        time_index = ResourceDataset.extract(time_index, ds_slice[0],
                                             unscale=False)

        time_index = check_tz(pd.to_datetime(time_index.astype(str)))

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
        ds_slice = parse_slice(ds_slice)
        sites = ds_slice[0]
        if isinstance(sites, int):
            sites = slice(sites, sites + 1)

        meta = self.h5[ds_name]
        meta = ResourceDataset.extract(meta, sites, unscale=False)

        if isinstance(sites, slice):
            stop = sites.stop
            if stop is None:
                stop = len(meta)

            sites = list(range(*sites.indices(stop)))

        meta = pd.DataFrame(meta, index=sites)
        if 'gid' not in meta:
            meta.index.name = 'gid'

        if self._str_decode:
            meta = self.df_str_decode(meta)

        if len(ds_slice) == 2:
            meta = meta[ds_slice[1]]

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
        ds_slice = parse_slice(ds_slice)
        coords = self.h5[ds_name]
        coords = ResourceDataset.extract(coords, ds_slice[0],
                                         unscale=False)
        return coords

    def _get_SAM_df(self, ds_name, site):
        """
        Placeholder for get_SAM_df method that it resource specific

        Parameters
        ----------
        ds_name : str
            'Dataset' name == SAM
        site : int
            Site to extract SAM DataFrame for
        """

    def _get_ds(self, ds_name, ds_slice):
        """
        Extract data from given dataset

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from ds,
            each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        if ds_name not in self.datasets:
            raise ResourceKeyError('{} not in {}'
                                   .format(ds_name, self.datasets))

        ds = self.h5[ds_name]
        ds_slice = parse_slice(ds_slice)
        out = ResourceDataset.extract(ds, ds_slice, scale_attr=self.SCALE_ATTR,
                                      add_attr=self.ADD_ATTR,
                                      unscale=self._unscale)

        return out

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()

    def _preload_SAM(self, sites, tech, time_index_step=None, means=False):
        """
        Placeholder method to pre-load project_points for SAM

        Parameters
        ----------
        sites : list
            List of sites to be provided to SAM
        tech : str
            Technology to be run by SAM
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False
        """
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, tech, self['time_index', time_slice],
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = self['meta', sites]

        for var in SAM_res.var_list:
            if var in self.datasets:
                SAM_res[var] = self[var, time_slice, sites]

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, tech, unscale=True, hsds=False,
                    str_decode=True, group=None, time_index_step=None,
                    means=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        sites : list
            List of sites to be provided to SAM
        tech : str
            Technology to be run by SAM
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
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(sites, tech,
                                       time_index_step=time_index_step,
                                       means=means)

        return SAM_res
