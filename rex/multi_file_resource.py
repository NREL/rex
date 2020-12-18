# -*- coding: utf-8 -*-
"""
Classes to handle resource data
"""
import h5py
import numpy as np
import os

from rex.renewable_resource import NSRDB, WindResource
from rex.resource import Resource
from rex.utilities.exceptions import ResourceRuntimeError


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
        with h5py.File(h5_path, mode='r') as f:
            for dset in f:
                if dset not in ['meta', 'time_index', 'coordinates']:
                    unique_dsets.append(dset)
                else:
                    shared_dsets.append(dset)

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
    def __init__(self, h5_path, prefix='', suffix='.h5', check_files=False):
        """
        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        """
        self.h5_dir, pre, suf = self.multi_file_args(h5_path)
        if pre is None:
            pre = prefix

        if suf is None:
            suf = suffix

        h5_files = self._get_h5_files(self.h5_dir, prefix=pre, suffix=suf)
        super().__init__(h5_files, check_files=check_files)

    def __repr__(self):
        msg = ("{} for {}:\n Contains {} files and {} datasets"
               .format(self.__class__.__name__, self.h5_dir,
                       len(self), len(self._dset_map)))

        return msg

    @staticmethod
    def multi_file_args(h5_path):
        """
        Get multi-h5 directory arguments for multi file resource paths.

        Parameters
        ----------
        h5_path : str
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
        Returns
        -------
        h5_dir : str
            Directory containing multi-file resource files.
        prefix : str
            File prefix for files in h5_dir.
        suffix : str
            File suffix for files in h5_dir.
        """
        if '*' in h5_path:
            h5_dir, fn = os.path.split(h5_path)
            prefix, suffix = fn.split('*')
        elif os.path.isfile(h5_path):
            raise RuntimeError("MultiFileResource cannot handle a single file"
                               " use Resource instead.")
        else:
            h5_dir = h5_path
            prefix = None
            suffix = None

        return h5_dir, prefix, suffix

    @staticmethod
    def _get_h5_files(h5_dir, prefix='', suffix='.h5'):
        """
        Map 5min variables to their .h5 files in given directory

        Parameters
        ----------
        h5_dir : str
            Path to directory containing 5min .h5 files
        prefix : str
            Prefix for resource .h5 files
        suffix : str
            Suffix for resource .h5 files

        Returns
        -------
        h5_files : list
            List of .h5 files to source data from
        """
        h5_files = []
        for file in sorted(os.listdir(h5_dir)):
            if file.startswith(prefix) and file.endswith(suffix):
                h5_files.append(os.path.join(h5_dir, file))

        return h5_files


class MultiFileResource(Resource):
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
    PREFIX = ''
    SUFFIX = '.h5'

    def __init__(self, h5_source, unscale=True, str_decode=True,
                 check_files=False):
        """
        Parameters
        ----------
        h5_source : str | list
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source .h5 files
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        check_files : bool
            Check to ensure files have the same coordinates and time_index
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
        self._i = 0

    def __repr__(self):
        msg = "{}".format(self.__class__.__name__)
        return msg

    def _init_multi_h5(self, h5_source, check_files=False):
        """
        Initialize MultiH5 handler class based on input type

        Parameters
        ----------
        h5_source : str | list
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source .h5 files
        check_files : bool
            Check to ensure files have the same coordinates and time_index

        Returns
        -------
        multi_h5 : MultiH5 | MultiH5Path
            Initialized multi h5 handler
        """
        if isinstance(h5_source, str):
            multi_h5 = MultiH5Path(h5_source, prefix=self.PREFIX,
                                   suffix=self.SUFFIX, check_files=check_files)
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
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source .h5 files
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
            SAM_res = res._preload_SAM(sites, tech=tech,
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
    SUFFIX = 'm.h5'

    def __init__(self, h5_source, unscale=True, str_decode=True,
                 check_files=False):
        """
        Parameters
        ----------
        h5_source : str | list
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source .h5 files
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        check_files : bool
            Check to ensure files have the same coordinates and time_index
        """
        super().__init__(h5_source, unscale=unscale, str_decode=str_decode,
                         check_files=check_files)
        self._heights = None

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
            Path to directory containing multi-file resource file sets.
            Available formats:
                /h5_dir/
                /h5_dir/prefix*suffix
            Or list of source .h5 files
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
            SAM_res = res._preload_SAM(sites, hub_heights,
                                       time_index_step=time_index_step,
                                       means=means,
                                       require_wind_dir=require_wind_dir,
                                       precip_rate=precip_rate, icing=icing)

        return SAM_res
