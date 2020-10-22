# -*- coding: utf-8 -*-
"""
Module to handle SAM Resource iterator to create site by site resource
DataFrames
"""
import numpy as np
import pandas as pd
from warnings import warn
import logging

from rex.utilities.exceptions import (ResourceKeyError, ResourceRuntimeError,
                                      ResourceValueError, SAMInputWarning)
from rex.utilities.parse_keys import parse_keys
from rex.utilities.solar_position import SolarPosition

logger = logging.getLogger(__name__)


class SAMResource:
    """
    Resource container for SAM. Resource handlers preload the datasets needed
    by SAM for the sites of interest. SAMResource handles all ETL needed before
    resource data is passed into SAM.
    """

    # Resource variables to load for each SAM technology
    RES_VARS = {'pv': ('dni', 'dhi', 'ghi', 'wind_speed', 'air_temperature'),
                'pvwattsv5': ('dni', 'dhi', 'ghi', 'wind_speed',
                              'air_temperature'),
                'pvwattsv7': ('dni', 'dhi', 'ghi', 'wind_speed',
                              'air_temperature'),
                'csp': ('dni', 'dhi', 'wind_speed', 'air_temperature',
                        'dew_point', 'surface_pressure'),
                'tcsmoltensalt': ('dni', 'dhi', 'wind_speed',
                                  'air_temperature', 'dew_point',
                                  'surface_pressure'),
                'solarwaterheat': ('dni', 'dhi', 'wind_speed',
                                   'air_temperature', 'dew_point',
                                   'surface_pressure'),
                'troughphysicalheat': ('dni', 'dhi', 'wind_speed',
                                       'air_temperature', 'dew_point',
                                       'surface_pressure'),
                'lineardirectsteam': ('dni', 'dhi', 'wind_speed',
                                      'air_temperature', 'dew_point',
                                      'surface_pressure'),
                'wind': ('pressure', 'temperature', 'winddirection',
                         'windspeed'),
                'windpower': ('pressure', 'temperature', 'winddirection',
                              'windspeed'),
                'wave': ('significant_wave_height', 'peak_period')}

    # valid data ranges for PV solar resource:
    PV_DATA_RANGES = {'dni': (0.0, 1360.0),
                      'dhi': (0.0, 1360.0),
                      'ghi': (0.0, 1360.0),
                      'wind_speed': (0, 120),
                      'air_temperature': (-200, 100)}

    # valid data ranges for CSP solar resource:
    CSP_DATA_RANGES = {'dni': (0.0, 1360.0),
                       'dhi': (0.0, 1360.0),
                       'ghi': (0.0, 1360.0),
                       'wind_speed': (0, 120),
                       'air_temperature': (-200, 100),
                       'dew_point': (-200, 100),
                       'surface_pressure': (300, 1100)}

    # valid data ranges for wind resource in SAM based on the cpp file:
    # https://github.com/NREL/ssc/blob/develop/shared/lib_windfile.cpp
    WIND_DATA_RANGES = {'windspeed': (0, 120),
                        'pressure': (0.5, 1.099),
                        'temperature': (-200, 100),
                        'rh': (0.1, 99.9)}

    WAVE_DATA_RANGES = {}

    # valid data ranges for trough physical process heat
    TPPH_DATA_RANGES = CSP_DATA_RANGES

    # valid data ranges for linear Fresnel
    LF_DATA_RANGES = CSP_DATA_RANGES

    # valid data ranges for solar water heater
    SWH_DATA_RANGES = CSP_DATA_RANGES

    # Data range mapping by SAM tech string
    DATA_RANGES = {'windpower': WIND_DATA_RANGES,
                   'wind': WIND_DATA_RANGES,
                   'pv': PV_DATA_RANGES,
                   'pvwattsv5': PV_DATA_RANGES,
                   'pvwattsv7': PV_DATA_RANGES,
                   'csp': CSP_DATA_RANGES,
                   'tcsmoltensalt': CSP_DATA_RANGES,
                   'troughphysicalheat': TPPH_DATA_RANGES,
                   'lineardirectsteam': LF_DATA_RANGES,
                   'solarwaterheat': SWH_DATA_RANGES,
                   'wave': WAVE_DATA_RANGES}

    def __init__(self, sites, tech, time_index, hub_heights=None,
                 require_wind_dir=False, means=False):
        """
        Parameters
        ----------
        sites : int | list | tuple | slice
            int, list, tuple, or slice indicating sites to send to SAM
        tech : str
            SAM technology string. See class attributes for options.
        time_index : pandas.DatetimeIndex
            Time-series datetime index
        hub_heights : int | float | list, optional
            Hub height(s) to extract wind data at, by default None
        require_wind_dir : bool, optional
            Boolean flag indicating that wind direction is required,
            by default False
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False
        """
        self._i = 0
        self._sites = self._parse_sites(sites)
        self._time_index = time_index
        self._shape = (len(time_index), len(self._sites))
        self._n = self._shape[1]
        self._var_list = None
        self._meta = None
        self._runnable = False
        self._res_arrays = {}
        self._h = hub_heights
        self._sza = None

        self._mean_arrays = None
        if means:
            self._mean_arrays = {}

        if tech.lower() in self.DATA_RANGES.keys():
            self._tech = tech.lower()
        else:
            msg = ('Selected tech {} is not valid. The following technology '
                   'strings are available: {}'
                   .format(tech, list(self.DATA_RANGES.keys())))
            logger.error(msg)
            raise ResourceValueError(msg)

        if self._tech == 'windpower':
            # hub height specified, get WTK wind data.
            if isinstance(self._h, (list, np.ndarray)):
                if len(self._h) != self._n:
                    msg = 'Must have a unique height for each site'
                    logger.error(msg)
                    raise ResourceValueError(msg)

            if not require_wind_dir:
                self._res_arrays['winddirection'] = np.zeros(self._shape,
                                                             dtype='float32')

    def __repr__(self):
        msg = "{} with {} {} sites".format(self.__class__.__name__,
                                           self._n, self._tech)
        return msg

    def __len__(self):
        return self._n

    def __getitem__(self, keys):
        var, var_slice = parse_keys(keys)

        if var == 'time_index':
            out = self.time_index
            out = out[var_slice[0]]
        elif var == 'meta':
            out = self.meta
            out = out.loc[var_slice[0]]
        elif isinstance(var, str):
            if var.startswith('mean_'):
                var = var.lstrip('mean_')
                out = self._get_var_mean(var, *var_slice)
            else:
                out = self._get_var_ts(var, *var_slice)
        elif isinstance(var, int):
            site = var
            out, _ = self._get_res_df(site)
        else:
            msg = 'Cannot interpret {}'.format(var)
            logger.error(msg)
            raise ResourceKeyError(msg)

        return out

    def __setitem__(self, keys, arr):
        var, var_slice = parse_keys(keys)

        if var == 'meta':
            self.meta = arr
        else:
            self._set_var_array(var, arr, *var_slice)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._n:
            site = self.sites[self._i]
            res_df, site_meta = self._get_res_df(site)
            self._i += 1
            return res_df, site_meta
        else:
            raise StopIteration

    @property
    def sites(self):
        """
        Sites being pre-loaded for SAM

        Returns
        -------
        sites : list
            List of sites to be provided to SAM
        """
        sites = self._sites

        return list(sites)

    @property
    def sites_slice(self):
        """Get the sites in slice format if possible

        Returns
        -------
        sites : list | slice
            Sites slice belonging to this instance of ProjectPoints.
            The type is slice if possible. Will be a list only if sites are
            non-sequential.
        """
        # try_slice is what the sites list would be if it is sequential
        if len(self.sites) > 1:
            try_step = self.sites[1] - self.sites[0]
        else:
            try_step = 1

        try_slice = slice(self.sites[0], self.sites[-1] + 1, try_step)
        try_list = list(range(*try_slice.indices(try_slice.stop)))

        if self.sites == try_list:
            # try_slice is equivelant to the site list
            sites = try_slice
        else:
            # cannot be converted to a sequential slice, return list
            sites = self.sites

        return sites

    @property
    def shape(self):
        """
        Shape of variable arrays

        Returns
        -------
        self._shape : tuple
            Shape (time_index, sites) of variable arrays
        """
        return self._shape

    @property
    def var_list(self):
        """
        Return variable list associated with SAMResource type

        Returns
        -------
        _var_list : list
            List of resource variables associated with resource type
            ('solar' or 'wind')
        """

        if self._var_list is None:
            if self._tech in self.RES_VARS:
                self._var_list = list(self.RES_VARS[self._tech])
            else:
                msg = ("SAM technology string {} is invalid! The following "
                       "technology strings are available: {}"
                       .format(self._tech, list(self.RES_VARS.keys())))
                logging.error(msg)
                raise ResourceValueError(msg)

        return self._var_list

    @property
    def time_index(self):
        """
        Return time_index

        Returns
        -------
        self._time_index : pandas.DatetimeIndex
            Time-series datetime index
        """
        return self._time_index

    @property
    def meta(self):
        """
        Return sites meta

        Returns
        -------
        self._meta : pandas.DataFrame
            DataFrame of sites meta data
        """
        return self._meta

    @meta.setter
    def meta(self, meta):
        """
        Set sites meta

        Parameters
        ----------
        meta : array | pandas.DataFrame
            Sites meta as records array or DataFrame
        """
        if len(meta) != self._n:
            msg = 'Meta does not contain {} sites'.format(self._n)
            logger.error(msg)
            raise ResourceValueError(msg)

        if not isinstance(meta, pd.DataFrame):
            meta = pd.DataFrame(meta, index=self.sites)
        else:
            if not np.array_equal(meta.index, self.sites):
                msg = 'Meta does not match sites!'
                logger.error(msg)
                raise ResourceValueError(msg)

        self._meta = meta

    @property
    def h(self):
        """
        Get heights for wind sites

        Returns
        -------
        self._h : int | float | list
            Hub height or height(s) for wind resource, None for solar resource
        """
        return self._h

    @property
    def lat_lon(self):
        """
        site latitudes and longitudes

        Returns
        -------
        ndarray
        """
        return self.meta[['latitude', 'longitude']].values

    @property
    def sza(self):
        """
        Solar zenith angle for sites of interest

        Returns
        -------
        ndarray
        """
        if self._sza is None:
            self._sza = \
                np.radians(SolarPosition(self.time_index, self.lat_lon).zenith)

        return self._sza

    @staticmethod
    def _parse_sites(sites):
        """
        Sites to extract resource for and send to SAM

        Parameters
        ----------
        sites : int | list | tuple | slice
            int, list, tuple, or slice indicating sites to send to SAM

        Returns
        -------
        sites : list
            list of sites to send to SAM
        """
        if isinstance(sites, int):
            sites = [sites]
        elif isinstance(sites, slice):
            stop = sites.stop
            if stop is None:
                msg = "sites as a slice must have an explicit stop value!"
                logger.error(msg)
                raise ResourceValueError(msg)

            sites = list(range(*sites.indices(stop)))
        elif not isinstance(sites, (list, tuple)):
            msg = ("sites must a list, tuple or slice, not a {}!"
                   .format(type(slice)))
            logger.error(msg)
            raise ResourceValueError(msg)

        return sites

    @staticmethod
    def check_units(var_name, var_array, tech):
        """
        Check units of variable array and convert to SAM units if needed

        Parameters
        ----------
        var_name : str
            Variable name
        var_array : ndarray
            Variable data
        tech : str
            SAM technology string (windpower, pvwattsv5, solarwaterheat, etc..)

        Returns
        -------
        var_array : ndarray
            Variable data with updated units if needed
        """
        pressure_change = ['csp', 'troughphysicalheat', 'lineardirectsteam',
                           'solarwaterheat']

        if 'pressure' in var_name and tech.lower() == 'windpower':
            # Check if pressure is in Pa, if so convert to atm
            if np.median(var_array) > 1e3:
                # convert pressure from Pa to ATM
                var_array *= 9.86923e-6

        elif 'pressure' in var_name and tech.lower() in pressure_change:
            if np.min(var_array) < 200:
                # convert pressure from 100 to 1000 hPa
                var_array *= 10
            if np.median(var_array) > 70000:
                # convert pressure from Pa to hPa
                var_array /= 100

        elif 'temperature' in var_name:
            # Check if tempearture is in K, if so convert to C
            if np.median(var_array) > 200.00:
                var_array -= 273.15

        return var_array

    @staticmethod
    def enforce_arr_range(var, arr, valid_range, sites):
        """Check an array for valid data range, warn, patch, and return.

        Parameters
        ----------
        var : str
            variable name
        arr : np.ndarray
            Array to be checked and patched
        valid_range : np.ndarray | tuple | list
            arr data will be ensured within the min/max values of valid_range
        sites : list
            Resource gid site list for warning printout.

        Returns
        -------
        arr : np.ndarray
            Patched array with valid range.
        """
        min_val = np.min(valid_range)
        max_val = np.max(valid_range)
        check_low = (arr < min_val)
        check_high = (arr > max_val)
        check = (check_low | check_high)
        if check.any():
            warn('Resource dataset "{}" out of viable SAM range ({}, {}) for '
                 'sites {}. Data min/max: {}/{}. Patching data...'
                 .format(var, min_val, max_val,
                         list(np.array(sites)[check.any(axis=0)]),
                         np.min(arr), np.max(arr)),
                 SAMInputWarning)

            arr[check_low] = min_val
            arr[check_high] = max_val

        return arr

    @staticmethod
    def roll_timeseries(time_series, timezone, time_interval):
        """
        Roll timeseries array to given timezone from UTC

        Parameters
        ----------
        time_series : ndarray
            time_series array to roll
        timezone : int
            Time zone as UTC offset
        time_interval : int
            Number of step-steps in an hour, needed to compute time shift

        Returns
        -------
        time_series : ndarray
            Time series in local time
        """
        shift = int(-1 * timezone * time_interval)
        time_series = np.roll(time_series, shift, axis=0)

        return time_series

    def check_irradiance_datasets(self, datasets, clearsky=False):
        """
        Check available irradiance datasets

        Parameters
        ----------
        datasets : list
            List of available datasets in resource .h5 file
        clearsky : bool, optional
            Flag to check for clearsky irradiance datasets, by default False
        """
        available = 0
        irradiance_vars = ['dni', 'dhi', 'ghi']
        if clearsky:
            irradiance_vars = ['clearsky_{}'.format(var)
                               for var in irradiance_vars]
        for var in irradiance_vars:
            if var in datasets and var in self.var_list:
                available += 1

        if available < 2:
            msg = ("At least 2 irradiance variables (dni, dhi, or ghi) are "
                   "needed to run SAM!")
            logger.error(msg)
            raise ResourceRuntimeError(msg)

    def compute_irradiance(self, clearsky=False):
        """
        Fillin missing irradiance dataset from available values and SZA

        Parameters
        ----------
        clearsky : bool, optional
            Flag to check for clearsky irradiance datasets, by default False
        """
        irradiance_vars = ['dni', 'dhi', 'ghi']
        if clearsky:
            irradiance_vars = ['clearsky_{}'.format(var)
                               for var in irradiance_vars]

        missing = None
        for var in irradiance_vars:
            if var in self.var_list and var not in self._res_arrays:
                missing = var
                break

        if missing is not None:
            dni_var, dhi_var, ghi_var = irradiance_vars
            logger.info('{} is missing and will be computed from {}'
                        .format(missing, irradiance_vars.remove(missing)))
            if missing == ghi_var:
                ghi = (self._res_arrays[dni_var] * np.cos(self.sza)
                       + self._res_arrays[dhi_var])
                ghi[ghi < 0] = 0
                self[ghi_var] = ghi
            elif missing == dni_var:
                dni = ((self._res_arrays[ghi_var] - self._res_arrays[dhi_var])
                       / np.cos(self.sza))
                dni = np.nan_to_num(dni)
                dni[dni < 0] = 0
                self[dni_var] = dni
            elif missing == dhi_var:
                dhi = (self._res_arrays[ghi_var]
                       - self._res_arrays[dni_var] * np.cos(self.sza))
                dhi[dhi < 0] = 0
                self[dhi_var] = dhi

    def set_clearsky(self):
        """Make the NSRDB var list for solar based on clearsky irradiance."""
        for i, var in enumerate(self.var_list):
            if var in ['dni', 'dhi', 'ghi']:
                self._var_list[i] = 'clearsky_{}'.format(var)

    def append_var_list(self, var):
        """
        Append a new variable to the SAM resource protected var_list.

        Parameters
        ----------
        var : str
            New resource variable to be added to the protected var_list
            property.
        """

        self.var_list.append(var)

    def _check_physical_ranges(self, var, arr, var_slice):
        """Check physical range of array and enforce usable SAM data.

        Parameters
        ----------
        var : str
            variable name
        arr : np.ndarray
            Array to be checked and patched
        var_slice : tuple of int | list | slice
            Slice of variable array to extract

        Returns
        -------
        arr : np.ndarray
            Patched array with valid range.
        """

        # Get site list corresponding to the var_slice. Only reduce the sites
        # list if the var_slice has a second entry (column slice of sites)
        arr_sites = self.sites
        if not isinstance(var_slice, slice):
            if (len(var_slice) > 1
                    and not isinstance(var_slice[1], slice)):
                arr_sites = list(np.array(self.sites)[np.array(var_slice[1])])

        if var in self.DATA_RANGES[self._tech]:
            valid_range = self.DATA_RANGES[self._tech][var]
            arr = self.enforce_arr_range(var, arr, valid_range, arr_sites)

        return arr

    def runnable(self):
        """
        Check to see if SAMResource iterator is runnable:
        - Meta must be loaded
        - Variables in var_list must be loaded

        Returns
        ------
        bool
            Returns True if runnable check passes
        """
        if self._meta is None:
            msg = 'meta has not been set!'
            logger.error(msg)
            raise ResourceRuntimeError(msg)
        else:
            for var in self.var_list:
                if var not in self._res_arrays.keys():
                    msg = '{} has not been set!'.format(var)
                    logger.error(msg)
                    raise ResourceRuntimeError(msg)

        return True

    def _set_var_array(self, var, arr, *var_slice):
        """
        Set variable array (units and physical ranges are checked while set).

        Parameters
        ----------
        var : str
            Resource variable name
        arr : ndarray
            Time series data of given variable for sites
        var_slice : tuple of int | list | slice
            Slice of variable array that corresponds to arr
        """
        if var in self.var_list:
            var_arr = self._res_arrays.get(var, np.zeros(self._shape,
                                                         dtype='float32'))
            if var_arr[var_slice].shape == arr.shape:
                arr = self.check_units(var, arr, self._tech)
                arr = self._check_physical_ranges(var, arr, var_slice)
                var_arr[var_slice] = arr
                self._res_arrays[var] = var_arr
                if self._mean_arrays is not None:
                    self._mean_arrays[var] = var_arr.mean(axis=0)
            else:
                msg = ('{} has shape {}, '
                       'needs proper shape: {}'.format(var,
                                                       arr.shape, self._shape))
                logger.error(msg)
                raise ResourceValueError(msg)
        else:
            msg = '{} not in {}'.format(var, self.var_list)
            logger.error(msg)
            raise ResourceKeyError(msg)

    def _get_var_mean(self, var, *var_slice):
        """
        Get variable means

        Parameters
        ----------
        var : str
            Resource variable name
        var_slice : int | list | slice
            Slice of variable array to extract

        Returns
        -------
        means : ndarray
            Vector of variable means
        """
        if self._mean_arrays is None:
            msg = ("Variable means were not computed, ensure ws_mean for "
                   "windpower, or dni_mean/ghi_mean for pvwatts is in "
                   "'output_request'")
            logger.error(msg)
            raise ResourceRuntimeError(msg)

        if var in self.var_list:
            try:
                var_array = self._mean_arrays[var]
            except KeyError:
                msg = '{} has yet to be set!'.format(var)
                logger.error(msg)
                raise ResourceKeyError(msg)

            means = var_array[var_slice]
        else:
            msg = '{} not in {}'.format(var, self.var_list)
            logger.error(msg)
            raise ResourceKeyError(msg)

        return means

    def _get_var_ts(self, var, *var_slice):
        """
        Get variable time-series

        Parameters
        ----------
        var : str
            Resource variable name
        var_slice : tuple of int | list | slice
            Slice of variable array to extract

        Returns
        -------
        ts : pandas.DataFrame
            Time-series for desired sites of variable var
        """
        if var in self.var_list:
            try:
                var_array = self._res_arrays[var]
            except KeyError:
                msg = '{} has yet to be set!'.format(var)
                logger.error(msg)
                raise ResourceKeyError(msg)

            sites = np.array(self.sites)
            if len(var_slice) == 2:
                sites = sites[var_slice[1]]

            ts = pd.DataFrame(var_array[var_slice],
                              index=self.time_index[var_slice[0]],
                              columns=sites)
        else:
            msg = '{} not in {}'.format(var, self.var_list)
            logger.error(msg)
            raise ResourceKeyError(msg)

        return ts

    def _get_res_df(self, site):
        """
        Get resource time-series

        Parameters
        ----------
        site : int
            Site to extract

        Returns
        -------
        res_df : pandas.DataFrame
            Time-series of SAM resource variables for given site
        site_meta : pandas.Series
            Meta data for the input site
        """

        self.runnable()
        try:
            idx = self.sites.index(site)
        except ValueError:
            msg = '{} is not in available sites'.format(site)
            logger.error(msg)
            raise ResourceValueError(msg)
        site_meta = self.meta.loc[site].copy()
        if self._h is not None:
            try:
                h = self._h[idx]
            except TypeError:
                h = self._h

            site_meta['height'] = h

        res_df = pd.DataFrame(index=self.time_index)
        res_df.name = site
        for var_name, var_array in self._res_arrays.items():
            res_df[var_name] = var_array[:, idx]

        return res_df, site_meta

    def curtail_windspeed(self, gids, curtailment):
        """
        Apply temporal curtailment mask to windspeed resource at given sites

        Parameters
        ----------
        gids : int | list
            gids for site or list of sites to curtail
        curtailment : ndarray
            Temporal multiplier for curtailment
        """
        shape = (self.shape[0],)
        if isinstance(gids, int):
            site_pos = self.sites.index(gids)
        else:
            shape += (len(gids),)
            site_pos = [self.sites.index(id) for id in gids]

        if curtailment.shape != shape:
            msg = "curtailment must be of shape: {}".format(shape)
            logger.error(msg)
            raise ResourceValueError(msg)

        if 'windspeed' in self._res_arrays:
            self._res_arrays['windspeed'][:, site_pos] *= curtailment
        else:
            msg = 'windspeed has not be loaded!'
            logger.error(msg)
            raise ResourceRuntimeError(msg)
