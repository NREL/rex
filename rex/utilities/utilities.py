# -*- coding: utf-8 -*-
"""
Collection of helpful functions
"""
import datetime
import inspect
import json
import yaml
import os
from fnmatch import fnmatch
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import re
from scipy.spatial import cKDTree
import time
from warnings import warn

from rex.utilities.exceptions import (FileInputError, JSONError, RetryError,
                                      RetryWarning)


def safe_json_load(fpath):
    """Perform a json file load with better exception handling.

    Parameters
    ----------
    fpath : str
        Filepath to .json file.

    Returns
    -------
    dict
        Loaded json dictionary.

    Examples
    --------
    >>> json_path = "./path_to_json.json"
    >>> safe_json_load(json_path)
    {key1: value1,
     key2: value2}
    """

    validate_filepath(fpath, file_extension='.json', exception_type=JSONError)
    return _read_data_file(fpath, json.load, exception_type=JSONError)


def safe_yaml_load(fpath):
    """Perform a yaml file load with better exception handling.

    Parameters
    ----------
    fpath : str
        Filepath to .yaml (or .yml) file.

    Returns
    -------
    dict
        Loaded yaml dictionary.

    Examples
    --------
    >>> yaml_path = "./path_to_yaml.yaml"
    >>> safe_yaml_load(yaml_path)
    {key1: value1,
     key2: value2}
    """

    validate_filepath(fpath, file_extension=('.yml', '.yaml'),
                      exception_type=yaml.YAMLError)
    return _read_data_file(fpath, yaml.safe_load,
                           exception_type=yaml.YAMLError)


def validate_filepath(fpath, file_extension, exception_type):
    """Validate an input filepath.

    The input is verified to be a string with a valid ending, and
    it is validated that the file exists on disk. If any of these conditions
    are not met, an exception is raised.


    Parameters
    ----------
    fpath : str
        Filepath to validate.
    file_extension : str or iterable of str
        A single file extension or an iterable of acceptable
        file extensions for the input path.
    exception_type : `Exception`
        A class indicating the type of exception to raise if
        file extension is incorrect.

    Raises
    ------
    TypeError
        If the input `fpath` is not a string.
    exception_type
        If the input `fpath` does not end in a valid extension.
    FileNotFoundError
        If the input `fpath` does not exist on disk.
    """

    if not isinstance(fpath, str):
        raise TypeError('Filepath must be str to load: {}'.format(fpath))

    if not fpath.endswith(file_extension):
        raise exception_type('Filepath must end in {!r} to load: {}'
                             .format(file_extension, fpath))

    if not os.path.isfile(fpath):
        raise FileNotFoundError('Could not find file to load: {}'
                                .format(fpath))


def _read_data_file(fpath, load_method, exception_type):
    """Load the data in the file using a given load method.

    This function performs additional exception handling during the
    data loading process.

    Parameters
    ----------
    fpath : str
        Filepath containing data to load.
    load_method : callable
        Function that can be called on a stream to load the
        data it contains.
    exception_type : `Exception`
        A class indicating the type of exception to raise if
        data cannot be read.

    Returns
    -------
    data : dict
        Dictionary representation of the data in the file.

    Raises
    ------
    exception_type
        If there was an error loading the data.
    """

    try:
        with open(fpath, 'r') as f:
            data = load_method(f)
    except exception_type as e:
        msg = 'Error:\n{}\nCannot read file: "{}"'.format(e, fpath)
        raise exception_type(msg) from e

    return data


def jsonify_dict(di):
    """Jsonify a dictionary into a string with handling for int/float keys.

    Parameters
    ----------
    di : dict
        Dictionary to be jsonified.

    Returns
    -------
    sdi : str
        Jsonified dictionary. Int/float keys will be represented as strings
        because json objects outside of python cannot have int/float keys.
    """

    for k in list(di.keys()):
        try:
            float(k)
        except ValueError:
            pass
        else:
            di[str(k)] = di.pop(k)

    try:
        sdi = json.dumps(di)
    except TypeError as e:
        msg = ('Could not json serialize {}, received error: {}'
               .format(di, e))
        raise TypeError(msg) from e

    return sdi


def dict_str_load(dict_str):
    """
    Load jsonified string entries into dictionaries using JSON

    Parameters
    ----------
    dict_str : str
        JSON style string provided to CLI or in config

    Returns
    -------
    out_dict : dict
        Dictionary loaded by JSON

    Examples
    --------
    >>> json_str = "{bool_key: 'True', value_key: 'None'}"
    >>> dict_str_load(json_str)
    {bool_key: True,
     value_key: None}
    """
    dict_str = dict_str.replace('\'', '\"')
    dict_str = dict_str.replace('None', 'null')
    dict_str = dict_str.replace('True', 'true')
    dict_str = dict_str.replace('False', 'false')
    out_dict = json.loads(dict_str)

    return out_dict


def parse_year(inp, option='raise'):
    """
    Attempt to parse a year out of a string.

    Parameters
    ----------
    inp : str
        String from which year is to be parsed
    option : str
        Return option:
         - "bool" will return True if year is found, else False.
         - Return year int / raise a RuntimeError otherwise

    Returns
    -------
    out : int | bool
        Year int parsed from inp,
        or boolean T/F (if found and option is bool).

    Examples
    --------
    >>> year_str = "NSRDB_2018.h5"
    >>> parse_year(year_str)
    2018

    >>> year_str = "NSRDB_2018.h5"
    >>> parse_year(year_str, option='bool')
    True

    >>> year_str = "NSRDB_TMY.h5"
    >>> parse_year(year_str)
    RuntimeError: Cannot parse year from NSRDB_TMY.h5

    >>> year_str = "NSRDB_TMY.h5"
    >>> parse_year(year_str, option='bool')
    False
    """
    # char leading year cannot be 0-9
    # char trailing year can be end of str or not 0-9
    regex = r".*[^0-9]([1-2][0-9]{3})($|[^0-9])"

    match = re.match(regex, inp)

    if match:
        out = int(match.group(1))

        if 'bool' in option:
            out = True

    else:
        if 'bool' in option:
            out = False
        else:
            raise RuntimeError('Cannot parse year from {}'.format(inp))

    return out


def mean_irrad(arr):
    """Calc the annual irradiance at a site given an irradiance timeseries.

    Parameters
    ----------
    arr : np.ndarray | pd.Series
        Annual irradiance array in W/m2. Row dimension is time.

    Returns
    -------
    mean : float | np.ndarray
        Mean irradiance values in kWh/m2/day. Float if the input array is
        1D, 1darray if the input array is 2D (multi-site).
    """

    mean = arr.mean(axis=0) / 1000 * 24
    return mean


def check_res_file(res_file):
    """
    Check resource to see if the given path
    - It belongs to a multi-file handler
    - Is on local disk
    - Is a hsds path

    Parameters
    ----------
    res_file : str
        Filepath to single resource file, unix style multi-file path like
        /h5_dir/prefix*suffix.h5, or an hsds filepath (filename of hsds
        path can also contain wildcards *)

    Returns
    -------
    multi_h5_res : bool
        Boolean flag to use a MultiFileResource handler
    hsds : bool
        Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
        behind HSDS
    """
    multi_h5_res = False
    hsds = False

    if os.path.isfile(res_file):
        pass

    elif '*' in res_file:
        multi_h5_res = True

    elif os.path.isdir(res_file):
        msg = ('Cannot parse directory, need to add wildcard * suffix: {}'
               .format(res_file))
        raise FileInputError(msg)

    else:
        try:
            import h5pyd
            hsds_dir = os.path.dirname(res_file)
            with h5pyd.Folder(hsds_dir + '/') as f:
                hsds = True
                fps = [f'{hsds_dir}/{fn}' for fn in f
                       if fnmatch(f'{hsds_dir}/{fn}', res_file)]
                if not any(fps):
                    msg = ('{} is not a valid HSDS file path!'
                           .format(res_file))
                    raise FileNotFoundError(msg)
                elif len(fps) > 1:
                    multi_h5_res = True

        except Exception as ex:
            msg = ("{} is not a valid file path, and HSDS "
                   "cannot be check for a file at this path:{}!"
                   .format(res_file, ex))
            raise FileNotFoundError(msg) from ex

    return multi_h5_res, hsds


def parse_date_int(s):
    """Parse data parameters from an integer or string of format YYYYMMDD

    Parmeters
    ---------
    s : str | int
        Date string or integer of format YYYYMMDD

    Returns
    -------
    y : int
        Year integer parsed from input.
    m : int
        Month integer parsed from input.
    d : int
        Day integer parsed from input.
    """

    try:
        s = str(int(s))
    except ValueError as ex:
        e = ('Could not convert date string to int: "{}"'
             .format(s))
        raise ValueError(e) from ex

    assert len(s) == 8, 'Bad date string, should be YYYYMMDD: {}'.format(s)

    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])

    assert y > 1970, 'Bad date string, year < 1970: {}'.format(s)
    assert m < 13, 'Bad date string, month > 12: {}'.format(s)
    assert d < 32, 'Bad date string, day > 31: {}'.format(s)

    return y, m, d


def str_to_date(s):
    """Convert a date string of format YYYYMMDD to date object.

    Parameters
    ----------
    s : str
        Date string of format YYYYMMDD

    Returns
    -------
    d : datetime.date
        Date object.
    """
    d = datetime.date(*parse_date_int(s))
    return d


def str_to_datetime(s):
    """Convert a date string of format YYYYMMDD to datetime object.

    Parameters
    ----------
    s : str
        Date string of format YYYYMMDD

    Returns
    -------
    d : datetime.datetime
        Datetime object.
    """
    d = datetime.datetime(*parse_date_int(s))
    return d


def parse_table(table):
    """
    Load pandas DataFrame from .csv or .json file or dictionary

    Parameters
    ----------
    trans_table : str | pandas.DataFrame | dict
        Path to .csv or .json or dictionary containing table to parse

    Returns
    -------
    table : pandas.DataFrame
        DataFrame table
    """
    if isinstance(table, str):
        if table.endswith('.csv'):
            table = pd.read_csv(table)
            if 'Unnamed: 0' in table:
                table = table.drop(columns='Unnamed: 0')

        elif table.endswith('.json'):
            table = pd.read_json(table)
        else:
            raise ValueError('Cannot parse {}, expecting a .csv or .json file'
                             .format(table))
    elif isinstance(table, dict):
        table = pd.DataFrame(table)
    elif not isinstance(table, pd.DataFrame):
        raise ValueError('Cannot parse table from type {}, expecting a .csv, '
                         '.json, dictionary, or pandas.DataFrame'
                         .format(type(table)))

    return table


def get_class_properties(cls):
    """
    Get all class properties
    Used to check against config keys

    Returns
    -------
    properties : list
        List of class properties, each of which should represent a valid
        config key/entry
    """
    properties = [attr for attr, attr_obj
                  in inspect.getmembers(cls)
                  if isinstance(attr_obj, property)]

    return properties


def timestamp_format_to_redex(time_format):
    """
    convert time stamp format to redex

    Parameters
    ----------
    time_format : str
        datetime timestamp format

    Returns
    -------
    redex : str
        redex format for timestamp
    """

    time_keys = {'%Y': r'\d{4}',
                 '%m': r'\d{2}',
                 '%d': r'\d{2}',
                 '%H': r'\d{2}',
                 '%M': r'\d{2}',
                 '%S': r'\d{2}'}

    redex = time_format
    for key, item in time_keys.items():
        if key in redex:
            redex = redex.replace(key, item)

    return redex


def parse_timestamp(path, time_format='%Y-%m-%d_%H:%M:%S'):
    """
    extract timestamp with given format from given path

    Parameters
    ----------
    path : str
        file path
    time_format : str, optional
        datetime timestamp format, by default '%Y-%m-%d_%H:%M:%S'

    Returns
    -------
    str
        Portion of path that matches given format
    """
    pattern = timestamp_format_to_redex(time_format)
    pattern = re.compile(pattern)
    matcher = pattern.search(path)

    if matcher is None:
        raise RuntimeError("Could not find timestamp with format {} in {}!"
                           .format(time_format, path))

    return matcher.group()


def filename_timestamp(file_name, time_format='%Y-%m-%d_%H:%M:%S'):
    """
    extract timestamp from file name

    Parameters
    ----------
    file_name : str
        file name or file path
    time_format : str, optional
        datetime timestamp format, by default '%Y-%m-%d_%H:%M:%S'

    Returns
    -------
    str
        Portion of file_name that matches given format
    """
    timestamp = parse_timestamp(os.path.basename(file_name),
                                time_format=time_format)

    return timestamp


class Retry:
    """
    Retry Decorator to run a function multiple times
    """
    def __init__(self, tries=3, n_sec=1):
        """
        Parameters
        ----------
        tries : int, optional
            Number if times to retry function, by default 2
        n_sec : int, optional
            Number of seconds to wait between tries, by default 1
        """
        self._tries = tries
        self._wait = n_sec

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call

        Parameters
        ----------
        func : obj
            Function to retry on Exception
        args : tuple
            Function arguments
        kwargs : dict
            Function kwargs
        """
        def new_func(*args, **kwargs):
            i = 0
            error = None
            while i <= self._tries:
                try:
                    new_func = func(*args, **kwargs)
                    break
                except RetryError as ex:
                    msg = ('{} failed to run {} times:\n{}'
                           .format(func.__name__, i, ex))
                    raise RuntimeError(msg) from ex
                except Exception as ex:
                    error = ex
                    warn('Attempt {} failed:\n{}'.format(i, error),
                         RetryWarning)
                    time.sleep(self._wait)
                finally:
                    i += 1

            if i > self._tries:
                raise RetryError('Failed to run {}:\n{}'
                                 .format(func.__name__, error))

            return new_func

        return new_func


def check_eval_str(s):
    """Check an eval() string for questionable code.

    Parameters
    ----------
    s : str
        String to be sent to eval(). This is most likely a math equation to be
        evaluated. It will be checked for questionable code like imports and
        dunder statements.
    """
    bad_strings = ('import', 'os.', 'sys.', '.__', '__.')
    for bad_s in bad_strings:
        if bad_s in s:
            raise ValueError('Will not eval() string which contains "{}": {}'
                             .format(bad_s, s))


def check_tz(time_index):
    """
    Check datetime index for timezone, if None set to UTC

    Parameters
    ----------
    time_index : pandas.DatatimeIndex
        DatetimeIndex to check timezone for

    Returns
    -------
    time_index : pandas.DatatimeIndex
        Updated DatetimeIndex with timezone set
    """
    if not time_index.tz:
        time_index = time_index.tz_localize('utc')

    return time_index


def get_lat_lon_cols(df):
    """
    Get columns that contain (latitude, longitude) coordinates

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to extract coordinates (lat, lon) from

    Returns
    -------
    lat_lon_cols : list
        Column names in df that correspond to the latitude and longitude
        coordinates. There must be a single unique set of latitude and
        longitude columns.
    """
    lat_lon_cols = ['latitude', 'longitude']
    lat = False
    lon = False
    for c in df.columns:
        if c.lower() in ['lat', 'latitude']:
            if lat:
                msg = ("Multiple possible latitude columns were found: "
                       "({}, {})!".format(lat_lon_cols[0], c))
                raise RuntimeError(msg)

            lat_lon_cols[0] = c
            lat = True
        elif c.lower() in ['lon', 'long', 'longitude']:
            if lon:
                msg = ("Multiple possible longitude columns were found: "
                       "({}, {})!".format(lat_lon_cols[1], c))
                raise RuntimeError(msg)

            lat_lon_cols[1] = c
            lon = True

    if not lat or not lon:
        msg = ("A valid pair of latitude and longitude columns could not be "
               "found in: {}!".format(df.columns))
        raise RuntimeError(msg)

    return lat_lon_cols


def roll_timeseries(arr, timezones):
    """
    Roll timeseries from UTC to local time. Automatically compute time-shift
    from UTC offset (timezone) and time-series length.

    Parameters
    ----------
    arr : ndarray
        Input timeseries array of form (time, sites)
    timezones : ndarray | list
        Vector of timezone shifts from UTC to local time

    Returns
    -------
    local_arr : ndarray
        Array shifted to local time
    """
    if arr.shape[1] != len(timezones):
        msg = ('Number of timezone shifts ({}) does not match number of '
               'sites ({})'.format(len(timezones), arr.shape[1]))
        raise ValueError(msg)

    time_step = arr.shape[0] // 8760

    local_arr = np.zeros(arr.shape, dtype=arr.dtype)
    for tz in set(timezones):
        mask = timezones == tz
        local_arr[:, mask] = np.roll(arr[:, mask], int(tz * time_step), axis=0)

    return local_arr


def get_chunk_ranges(ds_dim, chunk_size):
    """
    Create list of chunk slices [(s_i, e_i), ...]

    Parameters
    ----------
    ds_len : int
        Length of dataset axis to chunk
    chunk_size : int
        Size of chunks

    Returns
    -------
    chunks : list
        List of chunk start and end positions
        [(s_i, e_i), (s_i+1, e_i+1), ...]
    """
    chunks = list(range(0, ds_dim, chunk_size))
    if chunks[-1] < ds_dim:
        chunks.append(ds_dim)
    else:
        chunks[-1] = ds_dim

    chunks = list(zip(chunks[:-1], chunks[1:]))

    return chunks


def split_sites_slice(sites_slice, n_sites, slice_size):
    """
    Break up sites_slice into slices of size slice_size

    Parameters
    ----------
    sites_slice : slice
        Sites to extract as a slice object to extract
    n_sites : int
        Total number of sites to extract
    slice_size : int
        Number of sites in each slice to extract either on each worker,
        or in series

    Returns
    -------
    slices : list
        List of slices to extract
    """
    stop = sites_slice.stop
    if stop is None:
        stop = n_sites

    if slice_size >= n_sites:
        msg = ('The slice_size {} is >= the number of sites to be '
               'extracted {}! A single slice will be extracted.'
               .format(slice_size, n_sites))
        warn(msg)

        slices = [slice(sites_slice.start, stop, sites_slice.step)]
    else:
        step = sites_slice.step
        if step is not None:
            slice_size *= step

        # Create slices of size slice_size
        slices = [slice(s, e, step) for s, e
                  in get_chunk_ranges(stop, slice_size)]

    return slices


def split_sites_list(sites, slice_size):
    """
    Split sites into sub-lists of ~ size slice_size

    Parameters
    ----------
    sites : list
        Sites to extract as a list or numpy object to extract
    slice_size : int
        Number of sites in each slice to extract either on each worker,
        or in series

    Returns
    -------
    slices : list
        List of slices to extract
    """
    if slice_size >= len(sites):
        msg = ('The slice_size {} is >= the number of sites to be '
               'extracted {}! A single slice will be extracted.'
               .format(slice_size, len(sites)))
        warn(msg)
        slices = [sites]
    else:
        slices = np.array_split(sites, len(sites) // slice_size)

    return slices


def slice_sites(shape, chunks, sites=None, chunks_per_slice=5):
    """
    Slice sites into given number of sub-sets with given number of chunks per
    sub-set

    Parameters
    ----------
    shape : tuple
        Shape of dataset array that data is being extracted from
    chunks : tuple
        Chunk size of dataset array in .h5 file from which dataset is being
        extracted
    sites : list | slice, optional
        Subset of sites to extract, by default None or all sites
    chunks_per_slice : int, optional
        Number of chunks to extract in each slice, by default 5

    Returns
    -------
    slices : list
        List of slices to extract
    """
    if chunks is not None:
        slice_size = chunks[1] * chunks_per_slice
    else:
        slice_size = chunks_per_slice * 100

    if sites is None:
        sites = slice(None)

    if isinstance(sites, slice):
        slices = split_sites_slice(sites, shape[1], slice_size)
    elif isinstance(sites, (list, tuple, np.ndarray)):
        slices = split_sites_list(sites, slice_size)
    else:
        msg = ('sites must be of type "None", "slice", "list", "tuple", '
               'or "np.ndarray", but {} was provided'.format(type(sites)))
        raise TypeError(msg)

    return slices


def res_dist_threshold(lat_lons, tree=None, margin=1.05):
    """
    Distance threshold for nearest neighbor searches performed on resource
    points. Calculated as half of the diagonal between closest resource points,
    with desired extra margin

    Parameters
    ----------
    lat_lons : ndarray
        n x 2 array of resource points coordinates (lat, lon)
    tree : cKDTree, optional
        Pre-build cKDTree of resource lat, lon coordintes. If None, build the
        cKDTree from scratch, by default None
    margin : float, optional
        Extra margin to multiply times the computed max distance between
        neighboring resource points, by default 1.05

    Returns
    -------
    float
        Distance threshold for nearest neighbor searches performed on resource
        points. Calculated as half of the diagonal between closest resource
        points, with desired extra margin
    """
    if tree is None:
        # pylint: disable=not-callable
        tree = cKDTree(lat_lons)

    dists = tree.query(lat_lons, k=2)[0][:, 1]
    dists = dists[(dists != 0)]

    return margin * (2 ** 0.5) * (dists.max() / 2)


def get_dtype(col):
    """
    Get column dtype for converstion to records array

    Parameters
    ----------
    col : pandas.Series
        Column from pandas DataFrame

    Returns
    -------
    out : str
        String representation of converted dtype for column:
        -  float = float32
        -  int = int16 or int32 depending on data range
        -  object/str = U* max length of strings in col
    """
    dtype = col.dtype

    if isinstance(dtype, CategoricalDtype):
        col = col.astype(type(col.values[0]))
        out = get_dtype(col)
    elif np.issubdtype(dtype, np.floating):
        out = 'float32'
    elif np.issubdtype(dtype, np.integer):
        if col.max() < 32767:
            out = 'int16'
        else:
            out = 'int32'
    elif np.issubdtype(dtype, np.object_):
        size = int(col.astype(str).str.len().max())
        out = 'S{:}'.format(size)
    else:
        out = dtype

    return out


def to_records_array(df):
    """
    Convert pandas DataFrame to numpy Records Array

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame to be converted

    Returns
    -------
    numpy.rec.array
        Records array of input df
    """
    meta_arrays = []
    dtypes = []
    for c_name, c_data in df.items():
        dtype = get_dtype(c_data)

        if np.issubdtype(dtype, np.bytes_):
            data = c_data.astype(str).str.encode('utf-8').values
        else:
            data = c_data.values

        arr = np.array(data, dtype=dtype)
        meta_arrays.append(arr)
        dtypes.append((c_name, dtype))

    return np.core.records.fromarrays(meta_arrays, dtype=dtypes)


def row_col_indices(sc_point_gids, row_length):
    """
    Convert supply curve point gids to row and col indices given row length

    Parameters
    ----------
    sc_point_gids : int | list | ndarray
        Supply curve point gid or list/array of gids
    row_length : int
        row length (shape[1])

    Returns
    -------
    row : int | list | ndarray
        row indices
    col : int | list | ndarray
        row indices
    """
    rows = sc_point_gids // row_length
    cols = sc_point_gids % row_length

    return rows, cols


def unstupify_path(path):
    """
    Utility to create sensical os agnostic paths from relative or local path
    such as:
    - ~/file
    - file
    - /.
    - ./file

    Parameters
    ----------
    path : str
        Path or relative path

    Returns
    -------
    path: str
        Absolute/real path
    """
    path = os.path.expanduser(path)
    if not os.path.isabs(path) and not path.startswith('/'):
        path = os.path.realpath(path)

    return path


def write_json(path, data):
    """
    Write data to given json file

    Parameters
    ----------
    path : str
        Path to .json file to save data too
    data : dict
        Data to save to json file at path
    """
    assert path.endswith('.json'), "path should be to a .json file"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
