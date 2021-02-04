# -*- coding: utf-8 -*-
"""
Collection of helpful functions
"""
import datetime
import inspect
import json
import os
import pandas as pd
import re
import time
from warnings import warn

from rex.utilities.exceptions import JSONError, RetryError, RetryWarning


def safe_json_load(fpath):
    """Perform a json file load with better exception handling.

    Parameters
    ----------
    fpath : str
        Filepath to .json file.

    Returns
    -------
    j : dict
        Loaded json dictionary.

    Examples
    --------
    >>> json_path = "./path_to_json.json"
    >>> safe_json_load(json_path)
    {key1: value1,
     key2: value2}
    """

    if not isinstance(fpath, str):
        raise TypeError('Filepath must be str to load json: {}'.format(fpath))

    if not fpath.endswith('.json'):
        raise JSONError('Filepath must end in .json to load json: {}'
                        .format(fpath))

    if not os.path.isfile(fpath):
        raise JSONError('Could not find json file to load: {}'.format(fpath))

    try:
        with open(fpath, 'r') as f:
            j = json.load(f)
    except json.decoder.JSONDecodeError as e:
        emsg = ('JSON Error:\n{}\nCannot read json file: '
                '"{}"'.format(e, fpath))
        raise JSONError(emsg) from e

    return j


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
        except ValueError as e:
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
        Filepath to single resource file, multi-h5 directory,
        or /h5_dir/prefix*suffix

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
    if os.path.isdir(res_file) or ('*' in res_file):
        multi_h5_res = True
    else:
        if not os.path.isfile(res_file):
            try:
                import h5pyd
                hsds_dir, hsds_file = os.path.split(res_file)
                with h5pyd.Folder(hsds_dir + '/') as f:
                    hsds = True
                    if hsds_file not in f:
                        msg = ('{} is not a valid HSDS file path!'
                               .format(res_file))
                        raise FileNotFoundError(msg)
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
        table = pd.DataFrame(dict)
    elif not isinstance(table, pd.DataFrame):
        raise ValueError('Cannot parse table from type {}, expecting a .csv, '
                         '.json, or pandas.DataFrame'.format(type(table)))

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
    datetime.datetime
        datetime timestamp
    """
    pattern = timestamp_format_to_redex(time_format)
    pattern = re.compile(pattern)
    matcher = pattern.search(os.path.basename(file_name))

    time = matcher.group()

    return time


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
