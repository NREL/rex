# -*- coding: utf-8 -*-
"""
Classes to handle multiple years of resource data

Data split by time in chunks of less than a year can be opened by the
MultiTimeResource class, but not by these classes.
"""
import pandas as pd
import numpy as np
import os

from rex.multi_file_resource import MULTI_FILE_CLASS_MAP, MultiFileResource
from rex.multi_time_resource import MultiTimeH5, MultiTimeResource
from rex.renewable_resource import (NSRDB, SolarResource, WindResource,
                                    WaveResource)
from rex.resource import Resource
from rex.utilities.parse_keys import parse_slice
from rex.utilities.utilities import parse_year


class MultiYearH5(MultiTimeH5):
    """
    Class to handle multiple years of h5 Resources
    """

    def __init__(self, h5_path, years=None, res_cls=Resource, hsds=False,
                 hsds_kwargs=None, **res_cls_kwargs):
        """
        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be a path
            on HSDS starting with /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of integer years to access, by default None
        res_cls : obj
            Resource class to use to open and access resource data
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        self.h5_path = h5_path
        self._file_paths = self._get_file_paths(h5_path, hsds=hsds,
                                                hsds_kwargs=hsds_kwargs)
        self._file_paths, self._years = self._get_years(self._file_paths,
                                                        years)
        res_cls_kwargs.update({'hsds': hsds})
        self._h5_map = self._map_file_instances(self._file_paths,
                                                self._years,
                                                res_cls=res_cls,
                                                **res_cls_kwargs)
        self._years = self._h5_map['year'].values.tolist()

        self._datasets = None
        self._shape = None
        self._time_index = None

    def __repr__(self):
        msg = ("{} for {}:\n Contains data for {} years"
               .format(self.__class__.__name__, self.h5_path, len(self)))
        return msg

    def __len__(self):
        return len(set(self.years))

    def __getitem__(self, year):
        if isinstance(year, str):
            year = int(year)

        if year not in self._h5_map['year'].values:
            raise ValueError('{} is invalid, must be one of: {}'
                             .format(year, self.years))

        idx = np.where(self._h5_map['year'] == year)[0][0]
        h5 = self._h5_map.at[idx, 'h5']

        return h5

    def __iter__(self):
        return iter(self.years)

    def __contains__(self, year):
        return year in self.years

    @property
    def years(self):
        """
        Available years ordered the same way as self.files

        Returns
        -------
        list
            List of dataset present in .h5 files
        """
        return self._years

    @property
    def files(self):
        """
        Available file paths ordered the same way as self.years

        Returns
        -------
        list
        """
        return self._file_paths

    @property
    def time_index(self):
        """
        Multi-year datetime index

        Returns
        -------
        pandas.DatatimeIndex
        """
        if self._time_index is None:
            for h5 in self._h5_map['h5'].unique():
                if self._time_index is None:
                    self._time_index = h5.time_index
                else:
                    ti = self._time_index.append(h5.time_index)
                    self._time_index = ti

        return self._time_index

    @staticmethod
    def _map_file_instances(file_paths, years, res_cls=Resource,
                            **res_cls_kwargs):
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

        h5_map = pd.DataFrame({'fp': file_paths, 'year': years, 'h5': None})
        h5_map = h5_map.sort_values('year').reset_index(drop=True)

        if len(h5_map['year'].unique()) < len(h5_map):
            del res_cls_kwargs['hsds']  # no multi file res on hsds
            for _, subdf in h5_map.groupby('year'):
                fps = subdf['fp'].values.tolist()
                handle = MULTI_FILE_CLASS_MAP.get(res_cls, MultiFileResource)
                h5 = handle(fps, **res_cls_kwargs)
                for i in subdf.index:
                    h5_map.at[i, 'h5'] = h5

        else:
            for i, f_path in enumerate(h5_map['fp']):
                h5_map.at[i, 'h5'] = res_cls(f_path, **res_cls_kwargs)

        h5_map['dsets'] = [h5.dsets for h5 in h5_map['h5'].values]

        return h5_map

    @staticmethod
    def _get_years(file_paths, years):
        """Reduce file path list to requested years and/or return list of
        years corresponding to file paths

        Parameters
        ----------
        file_paths : list
            List of filepaths for this handler to handle.
        years : list | None
            List of years of interest. Should be a subset of years in file_map.
            Can also be None for all years found by the h5_path input.

        Returns
        -------
        file_paths : list
            List of filepaths for this handler to handle.
        years : list
            List of integer years corresponding to the file_paths list
        """

        fp_years = [int(parse_year(os.path.basename(fp), option='raise'))
                    for fp in file_paths]

        if years is None:
            years = fp_years
        elif any(int(y) not in fp_years for y in years):
            years = sorted([int(y) for y in years])
            raise RuntimeError('Requested years "{}" not all found in '
                               'file years "{}"'.format(years, fp_years))
        else:
            filtered_fps = []
            years = sorted([int(y) for y in years])
            for target_year in years:
                for fp_year, fp in zip(fp_years, file_paths):
                    if int(target_year) == int(fp_year):
                        filtered_fps.append(fp)
                        break
            file_paths = filtered_fps

        return file_paths, years

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
        if self._check_for_years(time_slice):
            years = time_slice
            year_slice = (slice(None), ) + ds_slice[1:]
            if isinstance(years, str):
                years = [years]

            for year in years:
                year = int(year)
                out.append(self[year]._get_ds(ds_name, year_slice))

            out = np.concatenate(out, axis=0)

        elif isinstance(time_slice, (int, np.integer)):
            time_step = self.time_index[time_slice]
            year = time_step.year
            year_index = self.year_index(year)
            year_slice = np.where(time_step == year_index)[0][0]
            year_slice = (year_slice, ) + ds_slice[1:]
            out = self[year]._get_ds(ds_name, year_slice)

        else:
            time_index = self.time_index[time_slice]
            year_map = time_index.year
            for year in year_map.unique():
                year_index = self.year_index(year)
                year_slice = year_index.isin(time_index[year_map == year])
                year_slice = \
                    self._check_time_slice(np.where(year_slice)[0])
                year_slice = (year_slice, ) + ds_slice[1:]
                out.append(self[year]._get_ds(ds_name, year_slice))

            out = np.concatenate(out, axis=0)

        return out

    def close(self):
        """
        Close all h5py.File instances
        """
        for f in self._h5_map['h5']:
            f.close()


class MultiYearResource(MultiTimeResource):
    """
    Class to handle multiple years of resource data stored accross multiple
    .h5 files. This also works if each year is split into multiple files each
    containing different datasets (e.g. for Sup3rCC and hi-res WTK+NSRDB). Data
    split by time in chunks of less than a year can be opened by the
    MultiTimeResource class, but not by this class.

    Note that files across years must have the same meta data, and files within
    the same year must have the same meta and time_index.

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

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 res_cls=Resource, hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards. Can also be a path on HSDS starting with
            /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        res_cls : obj
            Resource handler to us to open individual .h5 files
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        self.h5_path = h5_path
        self._time_index = None
        # Map variables to their .h5 files
        cls_kwargs = {'unscale': unscale, 'str_decode': str_decode,
                      'hsds': hsds, 'hsds_kwargs': hsds_kwargs}
        self._h5 = MultiYearH5(self.h5_path, years=years, res_cls=res_cls,
                               **cls_kwargs)
        self.h5_files = self._h5.h5_files
        self.h5_file = self.h5_files[0]

    @property
    def years(self):
        """
        Available years

        Returns
        -------
        list
            List of dataset present in .h5 files
        """
        return self.h5.years


class MultiYearSolarResource:
    """
    Class to handle multiple years of solar resource data stored accross
    multiple .h5 files
    """
    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be a path
            on HSDS starting with /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=SolarResource)


class MultiYearNSRDB(MultiYearResource):
    """
    Class to handle multiple years of NSRDB data stored accross
    multiple .h5 files
    """

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards. Can also be a path on HSDS starting with
            /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=NSRDB)


class MultiYearWindResource(MultiYearResource):
    """
    Class to handle multiple years of wind resource data stored accross
    multiple .h5 files
    """

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards. Can also be a path on HSDS starting with
            /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=WindResource)


class MultiYearWaveResource(MultiYearResource):
    """
    Class to handle multiple years of wave resource data stored accross
    multiple .h5 files
    """

    def __init__(self, h5_path, years=None, unscale=True, str_decode=True,
                 hsds=False, hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_path : str | list
            Unix shell style pattern path with * wildcards to multi-file
            resource file sets. Files must have the same coordinates
            but can have different datasets or time indexes. Can also be
            an explicit list of multi time files, which themselves can
            contain * wildcards. Can also be a path on HSDS starting with
            /nrel/ or a path on s3 starting with s3://
        years : list, optional
            List of years to access, by default None
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        super().__init__(h5_path, years=years, unscale=unscale, hsds=hsds,
                         hsds_kwargs=hsds_kwargs, str_decode=str_decode,
                         res_cls=WaveResource)
