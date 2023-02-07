# -*- coding: utf-8 -*-
"""
Classes to handle resource data
"""
import os
from glob import glob
import h5py
import numpy as np

from rex.renewable_resource import (NSRDB, WindResource,
                                    AbstractInterpolatedResource)
from rex.resource import Resource
from rex.utilities.exceptions import FileInputError, ResourceRuntimeError
from rex.utilities.utilities import unstupify_path


class MultiH5:
    """
    Class to handle multiple h5 file Resources
    """

    def __init__(self, h5_files, check_files=False):
        """
        Parameters
        ----------
        h5_files : list
            List of .h5 files to source data from
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        """
        self._dset_map = self._map_file_dsets(h5_files)
        self._h5_map = self._map_file_instances(set(self._dset_map.values()))

        self._i = 0

        if check_files:
            self._preflight_check()

    def __repr__(self):
        msg = ("{} contains {} files and {} datasets"
               .format(self.__class__.__name__, len(self),
                       len(self._dset_map)))
        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return len(self._h5_map)

    def __getitem__(self, dset):
        if dset in self:
            path = self._dset_map[dset]
            h5 = self._h5_map[path]
            ds = h5[dset]
        else:
            raise ValueError('{} is invalid must be one of: {}'
                             .format(dset, self.datasets))

        return ds

    def __next__(self):
        if self._i >= len(self.datasets):
            self._i = 0
            raise StopIteration

        dset = self.datasets[self._i]
        self._i += 1

        return dset

    def __iter__(self):
        return self

    def __contains__(self, dset):
        return dset in self.datasets

    @property
    def attrs(self):
        """
        Global .h5 file attributes sourced from first .h5 file

        Returns
        -------
        attrs : dict
            .h5 file attributes sourced from first .h5 file
        """
        path = self.h5_files[0]
        attrs = dict(self._h5_map[path].attrs)

        return attrs

    @property
    def datasets(self):
        """
        Available datasets

        Returns
        -------
        list
            List of dataset present in .h5 files
        """
        return sorted(self._dset_map)

    @property
    def h5_files(self):
        """
        .h5 files data is being sourced from

        Returns
        -------
        list
            List of .h5 files data is being sourced from
        """
        return sorted(self._h5_map)

    @staticmethod
    def _get_dsets(h5_path):
        """
        Get datasets in given .h5 file

        Parameters
        ----------
        h5_path : str
            Path to .h5 file to get variables for

        Returns
        -------
        unique_dsets : list
            List of unique datasets in .h5 file
        shared_dsets : list
            List of shared datasets in .h5 file
        """
        unique_dsets = []
        shared_dsets = []
        try:
            with h5py.File(h5_path, mode='r') as f:
                for dset in f:
                    if dset not in ['meta', 'time_index', 'coordinates']:
                        unique_dsets.append(dset)
                    else:
                        shared_dsets.append(dset)
        except Exception as e:
            msg = ('Could not read file: "{}"'.format(h5_path))
            raise IOError(msg) from e

        return unique_dsets, shared_dsets

    @classmethod
    def _map_file_dsets(cls, h5_files):
        """
        Map 5min variables to their .h5 files in given directory

        Parameters
        ----------
        h5_files : list
            List of h5_files to source data from

        Returns
        -------
        dset_map : dict
            Dictionary mapping datasets to file paths
        """
        dset_map = {}
        for file in h5_files:
            unique_dsets, shared_dsets = cls._get_dsets(file)
            for dset in shared_dsets:
                if dset not in dset_map:
                    dset_map[dset] = file

            for dset in unique_dsets:
                dset_map[dset] = file

        return dset_map

    @staticmethod
    def _map_file_instances(h5_files):
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
            h5_map[f_path] = h5py.File(f_path, mode='r')

        return h5_map

    def _preflight_check(self):
        """
        Check time_index and coordinates accross files
        """
        time_index = None
        lat_lon = None

        bad_files = []
        for file in self.h5_files:
            with Resource(file) as f:
                if 'time_index' in f:
                    ti = f.time_index
                    if time_index is None:
                        time_index = ti.copy()
                    else:
                        check = time_index.equals(ti)
                        if not check:
                            bad_files.append(file)

                ll = f.lat_lon
                if lat_lon is None:
                    lat_lon = ll.copy()
                else:
                    check = np.allclose(lat_lon, ll)
                    if not check:
                        bad_files.append(file)

        bad_files = list(set(bad_files))
        if bad_files:
            msg = ("The following files' coordinates and time-index do not "
                   "match:\n{}".format(bad_files))
            raise ResourceRuntimeError(msg)

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map.values():
            f.close()


class MultiH5Path(MultiH5):
    """
    Class to handle multiple h5 file Resources derived from a path
    """

    def __init__(self, h5_path, check_files=False):
        """
        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets.
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        """
        self.h5_path, h5_files = self._get_h5_files(h5_path)
        super().__init__(h5_files, check_files=check_files)

    def __repr__(self):
        msg = ("{} for {}:\n Contains {} files and {} datasets"
               .format(self.__class__.__name__, self.h5_path,
                       len(self), len(self._dset_map)))

        return msg

    @staticmethod
    def _get_h5_files(h5_path):
        """
        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets.

        Returns
        -------
        h5_path : str
            Just like the input except unstupified
        file_paths : list
            List of full file paths found by matching the h5_path input.
        """
        h5_path = unstupify_path(h5_path)

        if os.path.isdir(h5_path):
            msg = ('h5_path must be a unix shell style pattern with '
                   'wildcard * in order to find files, but received '
                   'directory specification: {}'.format(h5_path))
            raise FileInputError(msg)

        file_paths = glob(h5_path)

        if not any(file_paths):
            msg = ('Could not find any file paths with pattern: {}'
                   .format(h5_path))
            raise FileInputError(msg)

        return h5_path, file_paths


class MultiFileResource(AbstractInterpolatedResource):
    """
    Class to handle fine spatial resolution resource data stored in
    multiple .h5 files

    See Also
    --------
    resource.Resource : Parent class

    Examples
    --------
    Due to the size of the 2018 NSRDB and 5min WTK, datasets are stored in
    multiple files. MultiFileResource and it's sub-classes allow for
    interaction with all datasets as if they are in a single file.
    MultiFileResource can take a directory containing all files to source
    data from, or a filepath with a wildcard (*) indicating the filename
    format.

    >>> file = '$TESTDATADIR/wtk/wtk_2010_*m.h5'
    >>> with MultiFileResource(file) as res:
    >>>     print(self._h5_files)
    ['$TESTDATADIR/wtk_2010_200m.h5',
     '$TESTDATADIR/wtk_2010_100m.h5']

    >>> file_100m = '$TESTDATADIR/wtk_2010_100m.h5'
    >>> with Resource(file_100m) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_100m', 'temperature_100m', 'time_index',
     'winddirection_100m', 'windspeed_100m']

    >>> file_200m = '$TESTDATADIR/wtk_2010_200m.h5'
    >>> with Resource(file_200m) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_200m', 'temperature_200m', 'time_index',
     'winddirection_200m', 'windspeed_200m']

    >>> with MultiFileResource(file) as res:
    >>>     print(res.datasets)
    ['coordinates', 'meta', 'pressure_100m', 'pressure_200m',
     'temperature_100m', 'temperature_200m', 'time_index',
     'winddirection_100m', 'winddirection_200m', 'windspeed_100m',
     'windspeed_200m']

    >>> with MultiFileResource(file) as res:
    >>>     wspd = res['windspeed_100m']
    >>>
    >>> wspd
    [[15.13 15.17 15.21 ... 15.3  15.32 15.31]
     [15.09 15.13 15.16 ... 15.26 15.29 15.31]
     [15.09 15.12 15.15 ... 15.24 15.23 15.26]
     ...
     [10.29 11.08 11.51 ... 14.43 14.41 14.19]
     [11.   11.19 11.79 ... 13.27 11.93 11.8 ]
     [12.16 12.44 13.09 ... 11.94 10.88 11.12]]
    """

    INTERPOLABLE_DSETS = ["temperature", "pressure", "windspeed",
                          "winddirection"]
    VARIABLE_NAME = "height"
    VARIABLE_UNIT = "m"

    def __init__(self, h5_source, unscale=True, str_decode=True,
                 check_files=False, use_lapse_rate=True):
        """
        Parameters
        ----------
        h5_source : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets. Can also be an
            explicit list of complete filepaths.
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        use_lapse_rate : bool
            If a dataset is only available at a single hub-height and this flag
            value is set to `True`, pressure / temperature values will be
            calculated using linear lapse rate adjustment from the available
            hub height to the requested one. If the flag value is set to
            `False`, the value of these variables at the single available
            hub-height will be returned for *all* requested heights. This
            option has no effect if data is available at multiple hub-heights.
        """
        self._unscale = unscale
        self._meta = None
        self._time_index = None
        self._lat_lon = None
        self._str_decode = str_decode
        self._group = None
        # Map variables to their .h5 files
        self._h5 = self._init_multi_h5(h5_source, check_files=check_files)
        self._h5_files = self._h5.h5_files
        self.h5_file = self._h5_files[0]
        self._attrs = None
        self._shapes = None
        self._chunks = None
        self._dtypes = None
        self._i = 0

        self._interp_var = None
        self._use_lapse = use_lapse_rate

        # this is where self.heights or self.depths gets set
        self._interpolation_variable = self._parse_interp_var(self.datasets)
        prop_name = "{}s".format(self.VARIABLE_NAME)
        setattr(self, prop_name, self._interpolation_variable)

    def __repr__(self):
        msg = "{}".format(self.__class__.__name__)
        return msg

    @staticmethod
    def _init_multi_h5(h5_source, check_files=False):
        """
        Initialize MultiH5 handler class based on input type

        Parameters
        ----------
        h5_source : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets. Can also be an
            explicit list of complete filepaths.
        check_files : bool
            Check to ensure files have the same coordinates and time_index

        Returns
        -------
        multi_h5 : MultiH5 | MultiH5Path
            Initialized multi h5 handler
        """
        if isinstance(h5_source, str):
            multi_h5 = MultiH5Path(h5_source, check_files=check_files)
        elif isinstance(h5_source, (list, tuple)):
            multi_h5 = MultiH5(h5_source, check_files=check_files)
        else:
            msg = ('Cannot initialize MultiH5 from {}, expecting a path or a '
                   'list of .h5 file paths'.format(type(h5_source)))
            raise ResourceRuntimeError(msg)

        return multi_h5


class MultiFileNSRDB(MultiFileResource, NSRDB):
    """
    Class to handle 2018 and beyond NSRDB data that is at 2km and
    sub 30 min resolution

    See Also
    --------
    resource.MultiFileResource : Parent class
    resource.NSRDB : Parent class
    """

    @classmethod
    def preload_SAM(cls, h5_source, sites, unscale=True, str_decode=True,
                    tech='pvwattsv7', time_index_step=None, means=False,
                    clearsky=False, bifacial=False, downscale=None,
                    check_files=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_source : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets. Can also be an
            explicit list of complete filepaths.
        sites : list
            List of sites to be provided to SAM
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        tech : str, optional
            SAM technology string, by default 'pvwattsv7'
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False
        clearsky : bool
            Boolean flag to pull clearsky instead of real irradiance
        bifacial : bool
            Boolean flag to pull surface albedo for bifacial modeling.
        downscale : NoneType | str
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a string in the Pandas frequency format,
            e.g. '5min'.
        check_files : bool
            Check to ensure files have the same coordinates and time_index

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        with cls(h5_source, unscale=unscale, str_decode=str_decode,
                 check_files=check_files) as res:
            # pylint: disable=assignment-from-no-return
            SAM_res = res._preload_SAM(res, sites, tech=tech,
                                       time_index_step=time_index_step,
                                       means=means, clearsky=clearsky,
                                       bifacial=bifacial, downscale=downscale)

        return SAM_res


class MultiFileWTK(MultiFileResource, WindResource):
    """
    Class to handle 5min WIND Toolkit data

    See Also
    --------
    resource.MultiFileResource : Parent class
    resource.WindResource : Parent class

    Examples
    --------
    MultiFileWTK automatically searches for files of the form *m.h5

    >>> file = '$TESTDATADIR/wtk'
    >>> with MultiFileWTK(file) as res:
    >>>     print(list(res._h5_files)
    >>>     print(res.datasets)
    ['$TESTDATADIR/wtk_2010_200m.h5',
     '$TESTDATADIR/wtk_2010_100m.h5']
    ['coordinates', 'meta', 'pressure_100m', 'pressure_200m',
     'temperature_100m', 'temperature_200m', 'time_index',
     'winddirection_100m', 'winddirection_200m', 'windspeed_100m',
     'windspeed_200m']

    MultiFileWTK, like WindResource can interpolate / extrapolate hub-heights

    >>> with MultiFileWTK(file) as res:
    >>>     wspd = res['windspeed_150m']
    >>>
    >>> wspd
    [[16.19     16.25     16.305    ... 16.375    16.39     16.39    ]
     [16.15     16.205    16.255001 ... 16.35     16.365    16.39    ]
     [16.154999 16.195    16.23     ... 16.335    16.32     16.34    ]
     ...
     [10.965    11.675    12.08     ... 15.18     14.805    14.42    ]
     [11.66     11.91     12.535    ... 13.31     12.23     12.335   ]
     [12.785    13.295    14.014999 ... 12.205    11.360001 11.64    ]]
    """

    @classmethod
    def preload_SAM(cls, h5_source, sites, hub_heights, unscale=True,
                    str_decode=True, time_index_step=None, means=False,
                    require_wind_dir=False, precip_rate=False, icing=False,
                    check_files=False):
        """
        Placeholder for classmethod that will pre-load project_points for SAM

        Parameters
        ----------
        h5_source : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same time index and
            coordinates but can have different datasets. Can also be an
            explicit list of complete filepaths.
        sites : list
            List of sites to be provided to SAM
        hub_heights : int | float | list
            Hub heights to extract for SAM
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False
        require_wind_dir : bool
            Boolean flag as to whether wind direction will be loaded.
        precip_rate : bool
            Boolean flag as to whether precipitationrate_0m will be preloaded
        icing : bool
            Boolean flag as to whether icing is analyzed.
            This will preload relative humidity.
        check_files : bool
            Check to ensure files have the same coordinates and time_index

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        with cls(h5_source, unscale=unscale, str_decode=str_decode,
                 check_files=check_files) as res:
            # pylint: disable=assignment-from-no-return
            SAM_res = res._preload_SAM(res, sites, hub_heights,
                                       time_index_step=time_index_step,
                                       means=means,
                                       require_wind_dir=require_wind_dir,
                                       precip_rate=precip_rate, icing=icing)

        return SAM_res
