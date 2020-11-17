# -*- coding: utf-8 -*-
"""
Classes to handle renewable resource data
"""
import numpy as np
import pandas as pd
import warnings

from rex.resource import Resource
from rex.sam_resource import SAMResource
from rex.utilities.exceptions import (ResourceValueError, ExtrapolationWarning,
                                      ResourceWarning,
                                      MoninObukhovExtrapolationError)


class SolarResource(Resource):
    """
    Class to handle Solar Resource .h5 files

    See Also
    --------
    resource.Resource : Parent class
    """
    def _get_SAM_df(self, ds_name, site):
        """
        Get SAM solar resource DataFrame for given site

        Parameters
        ----------
        ds_name : str
            'Dataset' name == SAM
        site : int
            Site to extract SAM DataFrame for

        Returns
        -------
        res_df : pandas.DataFrame
            time-series DataFrame of resource variables needed to run SAM
        """
        if not self._unscale:
            raise ResourceValueError("SAM requires unscaled values")

        res_df = pd.DataFrame({'Year': self.time_index.year,
                               'Month': self.time_index.month,
                               'Day': self.time_index.day,
                               'Hour': self.time_index.hour})
        if len(self) > 8784:
            res_df['Minute'] = self.time_index.minute

        time_zone = self.meta.loc[site, 'timezone']
        time_interval = len(self.time_index) // 8760

        for var in ['dni', 'dhi', 'wind_speed', 'air_temperature']:
            ds_slice = (slice(None), site)
            var_array = self._get_ds(var, ds_slice)
            var_array = SAMResource.roll_timeseries(var_array, time_zone,
                                                    time_interval)
            res_df[var] = SAMResource.check_units(var, var_array,
                                                  tech='pvwattsv7')

        col_map = {'dni': 'DNI', 'dhi': 'DHI', 'wind_speed': 'Wind Speed',
                   'air_temperature': 'Temperature'}
        res_df = res_df.rename(columns=col_map)
        res_df.name = "{}-{}".format(ds_name, site)

        return res_df

    def _preload_SAM(self, sites, tech='pvwattsv7', time_index_step=None,
                     means=False, clearsky=False, bifacial=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        sites : list
            List of sites to be provided to SAM
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
        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, tech, self['time_index', time_slice],
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = self['meta', sites]

        if clearsky:
            SAM_res.set_clearsky()

        if bifacial and 'surface_albedo' not in SAM_res.var_list:
            SAM_res._var_list.append('surface_albedo')

        SAM_res.check_irradiance_datasets(self.datasets, clearsky=clearsky)
        for var in SAM_res.var_list:
            if var in self.datasets:
                SAM_res[var] = self[var, time_slice, sites]

        SAM_res.compute_irradiance(clearsky=clearsky)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, hsds=False,
                    str_decode=True, group=None, tech='pvwattsv7',
                    time_index_step=None, means=False, clearsky=False,
                    bifacial=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        sites : list
            List of sites to be provided to SAM
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

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(sites, tech=tech,
                                       time_index_step=time_index_step,
                                       means=means, clearsky=clearsky,
                                       bifacial=bifacial)

        return SAM_res


class NSRDB(SolarResource):
    """
    Class to handle NSRDB .h5 files

    See Also
    --------
    resource.Resource : Parent class
    """
    ADD_ATTR = 'psm_add_offset'
    SCALE_ATTR = 'psm_scale_factor'
    UNIT_ATTR = 'psm_units'

    def _preload_SAM(self, sites, tech='pvwattsv7', time_index_step=None,
                     means=False, clearsky=False, bifacial=False,
                     downscale=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        sites : list
            List of sites to be provided to SAM
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

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, tech, self['time_index', time_slice],
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = self['meta', sites]

        if clearsky:
            SAM_res.set_clearsky()

        if bifacial and 'surface_albedo' not in SAM_res.var_list:
            SAM_res._var_list.append('surface_albedo')

        SAM_res.check_irradiance_datasets(self.datasets, clearsky=clearsky)
        if not downscale:
            for var in SAM_res.var_list:
                if var in self.datasets:
                    SAM_res[var] = self[var, time_slice, sites]

            SAM_res.compute_irradiance(clearsky=clearsky)
        else:
            # contingent import to avoid dependencies
            from rex.utilities.downscale import downscale_nsrdb
            SAM_res = downscale_nsrdb(SAM_res, self, downscale,
                                      sam_vars=SAM_res.var_list)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, hsds=False,
                    str_decode=True, group=None, tech='pvwattsv7',
                    time_index_step=None, means=False, clearsky=False,
                    bifacial=False, downscale=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        sites : list
            List of sites to be provided to SAM
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

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(sites, tech=tech,
                                       time_index_step=time_index_step,
                                       means=means, clearsky=clearsky,
                                       bifacial=bifacial, downscale=downscale)

        return SAM_res


class WindResource(Resource):
    """
    Class to handle Wind Resource .h5 files

    See Also
    --------
    resource.Resource : Parent class

    Examples
    --------
    >>> file = '$TESTDATADIR/wtk/ri_100_wtk_2012.h5'
    >>> with WindResource(file) as res:
    >>>     print(res.datasets)
    ['meta', 'pressure_0m', 'pressure_100m', 'pressure_200m',
    'temperature_100m', 'temperature_80m', 'time_index', 'winddirection_100m',
    'winddirection_80m', 'windspeed_100m', 'windspeed_80m']

    WindResource can interpolate between available hub-heights (80 & 100)

    >>> with WindResource(file) as res:
    >>>     wspd_90m = res['windspeed_90m']
    >>>
    >>> wspd_90m
    [[ 6.865      6.77       6.565     ...  8.65       8.62       8.415    ]
     [ 7.56       7.245      7.685     ...  5.9649997  5.8        6.2      ]
     [ 9.775      9.21       9.225     ...  7.12       7.495      7.675    ]
      ...
     [ 8.38       8.440001   8.85      ... 11.934999  12.139999  12.4      ]
     [ 9.900001   9.895      9.93      ... 12.825     12.86      12.965    ]
     [ 9.895     10.01      10.305     ... 14.71      14.79      14.764999 ]]

    WindResource can also extrapolate beyond available hub-heights

    >>> with WindResource(file) as res:
    >>>     wspd_150m = res['windspeed_150m']
    >>>
    >>> wspd_150m
    ExtrapolationWarning: 150 is outside the height range (80, 100).
    Extrapolation to be used.
    [[ 7.336291   7.2570405  7.0532546 ...  9.736436   9.713792   9.487364 ]
     [ 8.038219   7.687255   8.208041  ...  6.6909685  6.362647   6.668326 ]
     [10.5515785  9.804363   9.770399  ...  8.026898   8.468434   8.67222  ]
     ...
     [ 9.079792   9.170363   9.634542  ... 13.472508  13.7102585 14.004617 ]
     [10.710078  10.710078  10.698757  ... 14.468795  14.514081  14.6386175]
     [10.698757  10.857258  11.174257  ... 16.585903  16.676476  16.653833 ]]
    """

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
        self._heights = None
        super().__init__(h5_file, unscale=unscale, hsds=hsds,
                         str_decode=str_decode, group=group)

    @staticmethod
    def _parse_hub_height(name):
        """
        Extract hub height from given string

        Parameters
        ----------
        name : str
            String to parse hub height from

        Returns
        -------
        h : int | float
            Hub Height as a numeric value
        """
        h = name.strip('m')
        try:
            h = int(h)
        except ValueError:
            h = float(h)

        return h

    @classmethod
    def _parse_name(cls, ds_name):
        """
        Extract dataset name and height from dataset name

        Parameters
        ----------
        ds_name : str
            Dataset name

        Returns
        -------
        name : str
            Variable name
        h : int | float
            Height of variable
        """
        try:
            if ds_name.endswith('m'):
                name = '_'.join(ds_name.split('_')[:-1])
                h = ds_name.split('_')[-1]
                h = cls._parse_hub_height(h)
            else:
                raise ValueError('{} does not end with "_m"'
                                 .format(ds_name))
        except ValueError:
            name = ds_name
            h = None

        return name, h

    @property
    def heights(self):
        """
        Extract available heights for pressure, temperature, windspeed, precip,
        and winddirection variables. Used for interpolation/extrapolation.

        Returns
        -------
        self._heights : list
            List of available heights for:
            windspeed, winddirection, temperature, and pressure
        """
        if self._heights is None:
            heights = {'pressure': [],
                       'temperature': [],
                       'windspeed': [],
                       'winddirection': []}

            ignore = ['meta', 'time_index', 'coordinates']
            for ds in self.datasets:
                if ds not in ignore:
                    ds_name, h = self._parse_name(ds)
                    if ds_name in heights.keys():
                        heights[ds_name].append(h)

            self._heights = heights

        return self._heights

    @staticmethod
    def get_nearest_h(h, heights):
        """
        Get two nearest h values in heights.
        Determine if h is inside or outside the range of heights
        (requiring extrapolation instead of interpolation)

        Parameters
        ----------
        h : int | float
            Height value of interest
        heights : list
            List of available heights

        Returns
        -------
        nearest_h : list
            list of 1st and 2nd nearest height in heights
        extrapolate : bool
            Flag as to whether h is inside or outside heights range
        """

        heights_arr = np.array(heights, dtype='float32')
        dist = np.abs(heights_arr - h)
        pos = dist.argsort()[:2]
        nearest_h = sorted([heights[p] for p in pos])
        extrapolate = np.all(h < heights_arr) or np.all(h > heights_arr)

        if extrapolate:
            h_min, h_max = np.sort(heights)[[0, -1]]
            msg = ('{} is outside the height range'.format(h),
                   '({}, {}).'.format(h_min, h_max),
                   'Extrapolation to be used.')
            warnings.warn(' '.join(msg), ExtrapolationWarning)

        return nearest_h, extrapolate

    @classmethod
    def monin_obukhov_extrapolation(cls, ts_1, h_1, z0, L, h):
        """
        Monin-Obukhov extrapolation

        Parameters
        ----------
         ts_1 : ndarray
            Time-series array at height h_1
        h_1 : int | float
            Height corresponding to time-seris ts_1
        z0: int | float | ndarray
            Roughness length
        L : ndarray
            time-series of Obukhov length (m; measure of stability)
        h : int | float
            Desired height

        Returns
        -------
        ndarray
            new wind speed from MO extrapolation.
        """
        # Non dimensional stability parameter at h
        zeta = cls.stability_function(h / L)
        # Non dimensional stability parameter at z0
        zeta_0 = cls.stability_function(z0 / L)
        # Non dimensional stability parameter at h_1
        zeta_1 = cls.stability_function(h_1 / L)

        # Logarithmic extrapolation equation
        out = (ts_1 * (np.log(h / z0) - zeta + zeta_0)
               / (np.log(h_1 / z0) - zeta_1 + zeta_0))

        return out

    @staticmethod
    def stability_function(zeta):
        """
        Calculate stability function depending on sign of L
        (negative is unstable, positive is stable)

        Parameters
        ----------
        zeta : ndarray
            Normalized length

        Returns
        -------
        numpy.ndarray
            stability measurements.
        """
        stab_fun = np.zeros(len(zeta))
        zeta = zeta.astype(float)

        # Unstable conditions
        x = (np.power(1 - 16 * zeta[zeta < 0], 0.25))
        paulson_func = (np.pi / 2 - 2 * np.arctan(x)
                        + np.log(np.power(1 + x, 2)
                        * (1 + np.power(x, 2)) / 8))

        y = np.power(1 - 10 * zeta[zeta < 0], 1. / 3)
        conv_func = (3 / 2 * np.log(np.power(y, 2) + y + 1. / 3) - np.sqrt(3)
                     * np.arctan(2 * y + 1 / np.sqrt(3)) + np.pi / np.sqrt(3))

        o = ((paulson_func + np.power(zeta[zeta < 0], 2) * conv_func)
             / (1 + np.power(zeta[zeta < 0], 2)))

        stab_fun[np.where(zeta < 0)] = o

        # Stable conditions
        a = 6.1
        b = 2.5

        o = np.log(zeta[zeta >= 0]
                   + (1 + np.power(zeta[zeta >= 0], b))**(1 / b))
        o *= -a
        stab_fun[np.where(zeta >= 0)] = o

        return stab_fun

    @staticmethod
    def power_law_interp(ts_1, h_1, ts_2, h_2, h, mean=True):
        """
        Power-law interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : ndarray
            Time-series array at height h_1
        h_1 : int | float
            Height corresponding to time-seris ts_1
        ts_2 : ndarray
            Time-series array at height h_2
        h_2 : int | float
            Height corresponding to time-seris ts_2
        h : int | float
            Height of desired time-series
        mean : bool
            Calculate average alpha versus point by point alpha

        Returns
        -------
        out : ndarray
            Time-series array at height h
        """
        if h_1 > h_2:
            h_1, h_2 = h_2, h_1
            ts_1, ts_2 = ts_2, ts_1

        if mean:
            alpha = (np.log(ts_2.mean() / ts_1.mean())
                     / np.log(h_2 / h_1))

            if alpha < 0.06:
                warnings.warn('Alpha is < 0.06', RuntimeWarning)
            elif alpha > 0.6:
                warnings.warn('Alpha is > 0.6', RuntimeWarning)
        else:
            # Replace zero values for alpha calculation
            ts_1[ts_1 == 0] = 0.001
            ts_2[ts_2 == 0] = 0.001

            alpha = np.log(ts_2 / ts_1) / np.log(h_2 / h_1)
            # The Hellmann exponent varies from 0.06 to 0.6
            alpha[alpha < 0.06] = 0.06
            alpha[alpha > 0.6] = 0.6

        out = ts_1 * (h / h_1)**alpha

        return out

    @staticmethod
    def linear_interp(ts_1, h_1, ts_2, h_2, h):
        """
        Linear interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : ndarray
            Time-series array at height h_1
        h_1 : int | float
            Height corresponding to time-seris ts_1
        ts_2 : ndarray
            Time-series array at height h_2
        h_2 : int | float
            Height corresponding to time-seris ts_2
        h : int | float
            Height of desired time-series

        Returns
        -------
        out : ndarray
            Time-series array at height h
        """
        if h_1 > h_2:
            h_1, h_2 = h_2, h_1
            ts_1, ts_2 = ts_2, ts_1

        # Calculate slope for every posiiton in variable arrays
        m = (ts_2 - ts_1) / (h_2 - h_1)
        # Calculate intercept for every position in variable arrays
        b = ts_2 - m * h_2

        out = m * h + b

        return out

    @staticmethod
    def shortest_angle(a0, a1):
        """
        Calculate the shortest angle distance between a0 and a1

        Parameters
        ----------
        a0 : int | float
            angle 0 in degrees
        a1 : int | float
            angle 1 in degrees

        Returns
        -------
        da : int | float
            shortest angle distance between a0 and a1
        """
        da = (a1 - a0) % 360
        return 2 * da % 360 - da

    @classmethod
    def circular_interp(cls, ts_1, h_1, ts_2, h_2, h):
        """
        Circular interpolate/extrapolate time-series data to height h

        Parameters
        ----------
        ts_1 : ndarray
            Time-series array at height h_1
        h_1 : int | float
            Height corresponding to time-seris ts_1
        ts_2 : ndarray
            Time-series array at height h_2
        h_2 : int | float
            Height corresponding to time-seris ts_2
        h : int | float
            Height of desired time-series

        Returns
        -------
        out : ndarray
            Time-series array at height h
        """
        h_f = (h - h_1) / (h_2 - h_1)

        da = cls.shortest_angle(ts_1, ts_2) * h_f
        da = np.sign(da) * (np.abs(da) % 360)

        out = (ts_2 + da) % 360

        return out

    def _check_hub_height(self, h):
        """
        Check requested hub-height against available windspeed hub-heights
        If only one hub-height is available change request to match available
        hub-height

        Parameters
        ----------
        h : int | float
            Requested hub-height

        Returns
        -------
        h : int | float
            Hub-height to extract
        """
        heights = self.heights['windspeed']
        if len(heights) == 1:
            h = heights[0]
            warnings.warn('Wind speed is only available at {h}m, '
                          'all variables will be extracted at {h}m'
                          .format(h=h), ResourceWarning)

        return h

    def _try_monin_obukhov_extrapolation(self, ts_1, ds_slice, h_1, h):
        rmol = 'inversemoninobukhovlength_2m'
        if rmol not in self:
            msg = ("{} is needed to run monin obukhov extrapolation"
                   .format(rmol))
            raise MoninObukhovExtrapolationError(msg)

        if 'roughness_length' in self:
            z0 = self._get_ds('roughness_length', ds_slice)
        elif 'z0' in self.meta:
            z0 = self.meta['z0']
        else:
            msg = ("roughness length ('z0') is needed to run monin obukhov"
                   "extrapolation")
            raise MoninObukhovExtrapolationError(msg)

        L = 1 / self._get_ds(rmol, ds_slice)

        out = self.monin_obukhov_extrapolation(ts_1, h_1, z0, L, h)

        return out

    def _get_ds_height(self, ds_name, ds_slice):
        """
        Extract data from given dataset at desired height, interpolate or
        extrapolate if neede

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
        var_name, h = self._parse_name(ds_name)
        heights = self.heights[var_name]
        if len(heights) == 1:
            h = heights[0]
            ds_name = '{}_{}m'.format(var_name, h)
            warnings.warn('Only one hub-height available, returning {}'
                          .format(ds_name), ResourceWarning)

        if h in heights:
            ds_name = '{}_{}m'.format(var_name, int(h))
            out = super()._get_ds(ds_name, ds_slice)
        else:
            (h1, h2), extrapolate = self.get_nearest_h(h, heights)
            if extrapolate:
                msg = 'Extrapolating {}'.format(ds_name)

            ts1 = super()._get_ds('{}_{}m'.format(var_name, h1), ds_slice)
            ts2 = super()._get_ds('{}_{}m'.format(var_name, h2), ds_slice)

            if (var_name == 'windspeed') and extrapolate:
                if h < h1:
                    try:
                        out = self._try_monin_obukhov_extrapolation(ts1,
                                                                    ds_slice,
                                                                    h1, h)
                        msg += ' using Monin Obukhov Extrapolation'
                        warnings.warn(msg, ExtrapolationWarning)
                    except MoninObukhovExtrapolationError:
                        out = self.power_law_interp(ts1, h1, ts2, h2, h)
                        msg += ' using Power Law Extrapolation'
                        warnings.warn(msg, ExtrapolationWarning)
                else:
                    out = self.power_law_interp(ts1, h1, ts2, h2, h)
                    msg += ' using Power Law Extrapolation'
                    warnings.warn(msg, ExtrapolationWarning)
            elif var_name == 'winddirection':
                out = self.circular_interp(ts1, h1, ts2, h2, h)
            else:
                out = self.linear_interp(ts1, h1, ts2, h2, h)

        return out

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
        var_name, h = self._parse_name(ds_name)
        if h is not None and var_name in self.heights:
            out = self._get_ds_height(ds_name, ds_slice)
        else:
            out = super()._get_ds(ds_name, ds_slice)

        return out

    def _get_SAM_df(self, ds_name, site, require_wind_dir=False,
                    icing=False):
        """
        Get SAM wind resource DataFrame for given site

        Parameters
        ----------
        ds_name : str
            'Dataset' name == SAM
        site : int
            Site to extract SAM DataFrame for
        require_wind_dir : bool
            Boolean flag as to whether wind direction will be loaded.
        icing : bool
            Boolean flag to include relativehumitidy for icing calculation

        Returns
        -------
        res_df : pandas.DataFrame
            time-series DataFrame of resource variables needed to run SAM
        """
        if not self._unscale:
            raise ResourceValueError("SAM requires unscaled values")

        _, h = self._parse_name(ds_name)
        h = self._check_hub_height(h)
        res_df = pd.DataFrame({'Year': self.time_index.year,
                               'Month': self.time_index.month,
                               'Day': self.time_index.day,
                               'Hour': self.time_index.hour})
        if len(self) > 8784:
            res_df['Minute'] = self.time_index.minute

        time_zone = self.meta.loc[site, 'timezone']
        time_interval = len(self.time_index) // 8760

        variables = ['pressure', 'temperature', 'winddirection', 'windspeed']
        if not require_wind_dir:
            variables.remove('winddirection')

        if icing:
            variables.append('relativehumidity')

        for var in variables:
            var_name = "{}_{}m".format(var, h)
            ds_slice = (slice(None), site)
            var_array = self._get_ds(var_name, ds_slice)
            var_array = SAMResource.roll_timeseries(var_array, time_zone,
                                                    time_interval)
            res_df[var] = SAMResource.check_units(var, var_array,
                                                  tech='windpower')
            res_df[var] = SAMResource.enforce_arr_range(
                var, res_df[var],
                SAMResource.WIND_DATA_RANGES[var], [site])

        col_map = {'pressure': 'Pressure', 'temperature': 'Temperature',
                   'windspeed': 'Speed', 'winddirection': 'Direction',
                   'relativehumidity': 'Relative Humidity'}
        res_df = res_df.rename(columns=col_map)
        res_df.name = "{}-{}".format(ds_name, site)

        return res_df

    def _preload_SAM(self, sites, hub_heights, time_index_step=None,
                     means=False, require_wind_dir=False,
                     precip_rate=False, icing=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        sites : list
            List of sites to be provided to SAM
        hub_heights : int | float | list
            Hub heights to extract for SAM
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None
        means : bool, optional
            Boolean flag to compute mean resource when res_array is set,
            by default False
        require_wind_dir : bool, optional
            Boolean flag as to whether wind direction will be loaded,
            by default False
        precip_rate : bool, optional
            Boolean flag as to whether precipitationrate_0m will be preloaded,
            by default False
        icing : bool, optional
            Boolean flag as to whether icing is analyzed.
            This will preload relative humidity, by default False

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, 'windpower',
                              self['time_index', time_slice],
                              hub_heights=hub_heights,
                              require_wind_dir=require_wind_dir,
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = self['meta', sites]
        var_list = SAM_res.var_list
        if not require_wind_dir:
            var_list.remove('winddirection')

        h = self._check_hub_height(SAM_res.h)
        if isinstance(h, (int, float)):
            for var in var_list:
                ds_name = "{}_{}m".format(var, h)
                SAM_res[var] = self[ds_name, time_slice, sites]
        else:
            _, unq_idx = np.unique(h, return_inverse=True)
            unq_h = sorted(list(set(h)))

            site_list = np.array(SAM_res.sites)
            height_slices = {}
            for i, h_i in enumerate(unq_h):
                pos = np.where(unq_idx == i)[0]
                height_slices[h_i] = (site_list[pos], pos)

            for var in var_list:
                for h_i, (h_pos, sam_pos) in height_slices.items():
                    ds_name = '{}_{}m'.format(var, h_i)
                    SAM_res[var, :, sam_pos] = self[ds_name, time_slice, h_pos]

        if precip_rate:
            var = 'precipitationrate'
            ds_name = '{}_0m'.format(var)
            SAM_res.append_var_list(var)
            SAM_res[var] = self[ds_name, time_slice, sites]

        if icing:
            var = 'rh'
            ds_name = 'relativehumidity_2m'
            SAM_res.append_var_list(var)
            SAM_res[var] = self[ds_name, time_slice, sites]

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, hub_heights, unscale=True,
                    hsds=False, str_decode=True, group=None,
                    time_index_step=None, means=False,
                    require_wind_dir=False, precip_rate=False, icing=False):
        """
        Placeholder for classmethod that will pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        sites : list
            List of sites to be provided to SAM
        hub_heights : int | float | list
            Hub heights to extract for SAM
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
        require_wind_dir : bool, optional
            Boolean flag as to whether wind direction will be loaded,
            by default False
        precip_rate : bool, optional
            Boolean flag as to whether precipitationrate_0m will be preloaded,
            by default False
        icing : bool, optional
            Boolean flag as to whether icing is analyzed.
            This will preload relative humidity, by default False

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(sites, hub_heights,
                                       require_wind_dir=require_wind_dir,
                                       precip_rate=precip_rate, icing=icing,
                                       means=means,
                                       time_index_step=time_index_step)

        return SAM_res


class WaveResource(Resource):
    """
    Class to handle Wave Resource .h5 files

    See Also
    --------
    resource.Resource : Parent class
    """

    def _get_SAM_df(self, ds_name, site):
        """
        Get SAM wave resource DataFrame for given site

        Parameters
        ----------
        ds_name : str
            'Dataset' name == SAM
        site : int
            Site to extract SAM DataFrame for

        Returns
        -------
        res_df : pandas.DataFrame
            time-series DataFrame of resource variables needed to run SAM
        """
        if not self._unscale:
            raise ResourceValueError("SAM requires unscaled values")

        res_df = pd.DataFrame({'Year': self.time_index.year,
                               'Month': self.time_index.month,
                               'Day': self.time_index.day,
                               'Hour': self.time_index.hour})
        if len(self) > 8784:
            res_df['Minute'] = self.time_index.minute

        time_zone = self.meta.loc[site, 'timezone']
        time_interval = len(self.time_index) // 8760

        for var in ['significant_wave_height', 'peak_period']:
            ds_slice = (slice(None), site)
            var_array = self._get_ds(var, ds_slice)
            var_array = SAMResource.roll_timeseries(var_array, time_zone,
                                                    time_interval)
            res_df[var] = SAMResource.check_units(var, var_array,
                                                  tech='pvwattsv7')

        col_map = {'significant_wave_height': 'Hs', 'peak_period': 'Tp'}
        res_df = res_df.rename(columns=col_map)
        res_df.name = "{}-{}".format(ds_name, site)

        return res_df

    def _preload_SAM(self, sites, means=False, time_index_step=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        sites : list
            List of sites to be provided to SAM
        means : bool
            Boolean flag to compute mean resource when res_array is set
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Wave resource for sites
            in project_points
        """
        SAM_res = super()._preload_SAM(sites, 'wave', means=means,
                                       time_index_step=time_index_step)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, hsds=False,
                    str_decode=True, group=None, means=False,
                    time_index_step=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            h5_file to extract resource from
        sites : list
            List of sites to be provided to SAM
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
        means : bool
            Boolean flag to compute mean resource when res_array is set
        time_index_step: int, optional
            Step size for time_index, used to reduce temporal resolution,
            by default None

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Wave resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(sites, means=means,
                                       time_index_step=time_index_step)

        return SAM_res
