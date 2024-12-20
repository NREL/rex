# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Classes to handle renewable resource data
"""
import re
from abc import abstractmethod
import numpy as np
import os
import pandas as pd
import warnings
import logging

from rex.resource import BaseResource
from rex.sam_resource import SAMResource
from rex.utilities.exceptions import (ResourceValueError, ExtrapolationWarning,
                                      ResourceWarning, ResourceRuntimeError,
                                      ResourceKeyError,
                                      MoninObukhovExtrapolationError)
from rex.utilities.parse_keys import parse_keys


logger = logging.getLogger(__name__)


class WaveResource(BaseResource):
    """
    Class to handle Wave BaseResource .h5 files

    See Also
    --------
    resource.BaseResource : Parent class
    """

    def get_SAM_df(self, site):
        """
        Get SAM wave resource DataFrame for given site

        Parameters
        ----------
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

        for var in ['significant_wave_height', 'energy_period']:
            ds_slice = (slice(None), site)
            var_array = self._get_ds(var, ds_slice)
            var_array = SAMResource.roll_timeseries(var_array, time_zone,
                                                    time_interval)
            res_df[var] = var_array

        col_map = {'significant_wave_height': 'wave_height',
                   'energy_period': 'wave_period'}
        res_df = res_df.rename(columns=col_map)
        res_df.name = "SAM_-{}".format(site)

        return res_df

    @staticmethod
    def _preload_SAM(res, sites, means=False, time_index_step=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
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
        SAM_res = super()._preload_SAM(res, sites, 'wave', means=means,
                                       time_index_step=time_index_step)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, str_decode=True,
                    group=None, hsds=False, hsds_kwargs=None, means=False,
                    time_index_step=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
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
        kwargs = {"unscale": unscale, "hsds": hsds, 'hsds_kwargs': hsds_kwargs,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(res, sites, means=means,
                                       time_index_step=time_index_step)

        return SAM_res


class AbstractInterpolatedResource(BaseResource):
    """Class to handle resource dataset interpolation.

    Default type of interpolation is linear.

    Pressure and Temperature lapse rates are used if p and t are only given at
    a single hub height. The lapse rates are from the International Standard
    Atmosphere (ISA) or ICAO Standard Atmosphere:
    (https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/
     phak/media/06_phak_ch4.pdf)
    """

    LAPSE_RATES = {'temperature': 6.56, 'pressure': 11_109}
    """Air Temperature and Pressure lapse rate in C/km and Pa/km"""

    def __init__(self, h5_file, mode='r', unscale=True, str_decode=True,
                 group=None, use_lapse_rate=True, hsds=False,
                 hsds_kwargs=None):
        """
        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        mode : str, optional
            Mode to instantiate h5py.File instance, by default 'r'
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        use_lapse_rate : bool
            If a dataset is only available at a single hub-height and this flag
            value is set to `True`, pressure / temperature values will be
            calculated using linear lapse rate adjustment from the available
            hub height to the requested one. If the flag value is set to
            `False`, the value of these variables at the single available
            hub-height will be returned for *all* requested heights. This
            option has no effect if data is available at multiple hub-heights.
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        """
        self._interp_var = None
        self._use_lapse = use_lapse_rate
        super().__init__(h5_file, unscale=unscale, str_decode=str_decode,
                         group=group, hsds=hsds, mode=mode,
                         hsds_kwargs=hsds_kwargs)

        # this is where self.heights or self.depths gets set
        self._interpolation_variable = self._parse_interp_var(self.datasets)
        prop_name = "{}s".format(self.VARIABLE_NAME)
        setattr(self, prop_name, self._interpolation_variable)

    @classmethod
    def _parse_interp_var(cls, datasets):
        """Extract available interpolation variable values for the
        interpolable datasets. Used for interpolation/extrapolation.

        Parameters
        ----------
        datasets : list
            List of dataset names that will be parsed for interpolation value
            suffixes like "windspeed_100m" -> windspeed at 100 meters

        Returns
        -------
        dict
            Dictionary of available interpolation variable values for
            the interpolable datasets. For example, this could be:
            {'windspeed': [10, 100, 200]}
        """
        interp_var = {dset: [] for dset in cls.INTERPOLABLE_DSETS}
        ignore = ['meta', 'time_index', 'coordinates']
        for ds in datasets:
            if ds not in ignore:
                ds_name, val = cls._parse_name(ds)
                if ds_name in interp_var and val is not None:
                    interp_var[ds_name].append(val)

        return interp_var

    @classmethod
    def _parse_name(cls, ds_name):
        """Extract dataset name and interpolation variable value from
        dataset name.

        Parameters
        ----------
        ds_name : str
            Dataset name

        Returns
        -------
        name : str
            Dataset name without interpolation value.
        val : int | float
            Variable value.
        """

        regex = "_-?[0-9]+(.[0-9]+)?{}$".format(cls.VARIABLE_UNIT)
        regex = re.search(regex, ds_name)
        if regex:
            val = regex.group()
            name = ds_name.rstrip(val)
            val = val.lstrip('_').rstrip(cls.VARIABLE_UNIT)

            try:
                val = int(val)
            except ValueError:
                val = float(val)

            return name, val

        return ds_name, None

    @classmethod
    def _get_nearest_val(cls, val, vals):
        """
        Get two nearest two values in `vals`.
        Determine if val is inside or outside the range of vals
        (requiring extrapolation instead of interpolation)

        Parameters
        ----------
        val : int | float
            Value of interest.
        vals : list
            List of available values.

        Returns
        -------
        nearest_val : list
            List of 1st and 2nd nearest val in vals.
        extrapolate : bool
            Flag as to whether val is inside or outside vals range
        """

        vals_arr = np.array(vals, dtype='float32')
        dist = np.abs(vals_arr - val)
        pos = dist.argsort()[:2]
        nearest_d = sorted([vals[p] for p in pos])
        extrapolate = np.all(val < vals_arr) or np.all(val > vals_arr)

        if extrapolate:
            v_min, v_max = np.sort(vals)[[0, -1]]
            msg = ('{} is outside the {} range'.format(val, cls.VARIABLE_NAME),
                   '({}, {}).'.format(v_min, v_max),
                   'Extrapolation to be used.')
            warnings.warn(' '.join(msg), ExtrapolationWarning)

        return nearest_d, extrapolate

    def _get_closest_existing_dset_name(self, dset):
        """Get the name of an existing dataset closest to interp value. """

        var_name, val = self._parse_name(dset)
        if val is not None and var_name in self._interpolation_variable:
            available_vals = self._interpolation_variable[var_name]
            (val, _), _ = self._get_nearest_val(val, available_vals)
            dset = '{}_{}{}'.format(var_name, val, self.VARIABLE_UNIT)

        return dset

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
        dset = self._get_closest_existing_dset_name(dset)
        return super().get_dset_properties(dset)

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
            dset = self._get_closest_existing_dset_name(dset)
            attrs = super().get_attrs(dset=dset)

        return attrs

    def _get_ds(self, ds_name, ds_slice):
        """
        Extract data from given dataset

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract from
            ds, each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        var_name, val = self._parse_name(ds_name)
        if val is not None and var_name in self._interpolation_variable:
            out = self._get_ds_interpolated(ds_name, ds_slice)
        else:
            out = super()._get_ds(ds_name, ds_slice)

        return out

    def _get_ds_interpolated(self, ds_name, ds_slice):
        """Extract data from given dataset at desired interpolation value.

        Data is interpolated or extrapolated as needed.

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract
            from ds, each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        var_name, val = self._parse_name(ds_name)
        interpolation_values = self._interpolation_variable[var_name]

        if not interpolation_values:
            warnings.warn('No {0} info available for {1!r}, returning '
                          'single {2!r} value for requested {0} {3}{4}'
                          .format(self.VARIABLE_NAME, ds_name, var_name, val,
                                  self.VARIABLE_UNIT), ResourceWarning)
            out = super()._get_ds(var_name, ds_slice)

        elif val in interpolation_values:
            ds_name = '{}_{}{}'.format(var_name, int(val), self.VARIABLE_UNIT)
            out = super()._get_ds(ds_name, ds_slice)

        elif (len(interpolation_values) == 1
              and self._use_lapse
              and var_name in self.LAPSE_RATES):
            out = self._get_ds_lapse(ds_name, ds_slice)

        elif len(interpolation_values) == 1:
            val = interpolation_values[0]
            ds_name = '{}_{}{}'.format(var_name, int(val), self.VARIABLE_UNIT)
            warnings.warn('Only one {} available, returning {!r}'
                          .format(self.VARIABLE_NAME, ds_name),
                          ResourceWarning)
            out = super()._get_ds(ds_name, ds_slice)

        else:
            out = self._get_calculated_ds(val, ds_name, var_name, ds_slice)

        return out

    def _get_ds_lapse(self, ds_name, ds_slice,
                      valid_units=('pa', 'pascals',
                                   'c', 'celsius',
                                   'k', 'kelvin')):
        """Extract data from given dataset where there is only temperature or
        pressure data at a single elevation and a lapse rate must be used to
        adjust to a new elevation.

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract
            from ds, each arg is for a sequential axis
        valid_units : tuple
            Tuple of valid lower-case units that can be lapse-rate adjusted. If
            the dataset doesnt have units in this list, a warning will be
            raised.

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """

        var_name, val = self._parse_name(ds_name)
        interpolation_values = self._interpolation_variable[var_name]
        ds_name = '{}_{}{}'.format(var_name, int(interpolation_values[0]),
                                   self.VARIABLE_UNIT)
        lapse_rate = self.LAPSE_RATES[var_name]

        if self.VARIABLE_UNIT != 'm':
            msg = ('Cannot use temperature/pressure lapse rate when vertical '
                   'interpolation unit is not "m": {}'
                   .format(self.VARIABLE_UNIT))
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info('Only one {} available for {} at {}, using '
                    'lapse rate of {} to get to {}'
                    .format(self.VARIABLE_NAME, var_name,
                            interpolation_values[0], lapse_rate, val))

        ds_units = str(self.units.get(ds_name, None))
        if ds_units.lower() not in valid_units:
            msg = ('Dataset "{}" is being adjusted to elevation {} via lapse '
                   'rate but valid units not found: {} (must be Pa or C). '
                   'Proceeding, but if this is not the desired behavior, set '
                   'use_lapse_rate=False'.format(ds_name, val, ds_units))
            logger.warning(msg)
            warnings.warn(msg, ResourceWarning)

        out = super()._get_ds(ds_name, ds_slice)
        delta_h = (interpolation_values[0] - val) / 1000  # to kilometers
        lapse = delta_h * lapse_rate
        return out + lapse

    def _get_calculated_ds(self, val, ds_name, var_name, ds_slice):
        """Get interpolated/extrapolated values for the dataset. """
        vals = self._interpolation_variable[var_name]
        (v1, v2), extrapolate = self._get_nearest_val(val, vals)

        if extrapolate:
            msg = ('Extrapolating {} using linear extrapolation'
                   .format(ds_name))
            warnings.warn(msg, ExtrapolationWarning)

        dset_name_1 = '{}_{}{}'.format(var_name, v1, self.VARIABLE_UNIT)
        ts1 = super()._get_ds(dset_name_1, ds_slice)
        dset_name_2 = '{}_{}{}'.format(var_name, v2, self.VARIABLE_UNIT)
        ts2 = super()._get_ds(dset_name_2, ds_slice)

        out = linear_interp(ts1, v1, ts2, v2, val)
        return out

    @staticmethod
    def _set_sam_res(res, values, dsets, SAM_res, time_slice, sites):
        """
        Set the resource for individual sites at various values
        (i.e. hub-heights, depths, etc).

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        values : list | int
            List of interpolation values e.g. hub heights or geothermal depths
        dsets : list
            List of dataset names to set
        SAM_res : SAMResource
            SAMResource object to load resource data into
        time_slice : slice
            Slice object representing any temporal subsampling
        sites : list | slice
            Spatial indices to load.
            (sites is synonymous with gids aka spatial indices)
        """

        if isinstance(values, (int, float)):
            SAM_res.load_rex_resource(res, dsets, time_slice, sites,
                                      hh=values, hh_unit=res.VARIABLE_UNIT)

        else:
            _, unique_index = np.unique(values, return_inverse=True)
            unique_values = sorted(list(set(values)))
            for dset in dsets:
                for index, value in enumerate(unique_values):
                    pos = np.where(unique_index == index)[0]
                    sites = np.array(SAM_res.sites)[pos]
                    ds_name = '{}_{}{}'.format(dset, value, res.VARIABLE_UNIT)
                    SAM_res[dset, :, pos] = res[ds_name, time_slice, sites]

    @property
    @abstractmethod
    def INTERPOLABLE_DSETS(self):
        """
        list: Names of the datasets allowed to be interpolated/extrapolated.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def VARIABLE_NAME(self):
        """
        str: Name of the variable to interpolate over (e.g. "height").
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def VARIABLE_UNIT(self):
        """
        str: Abbreviation used in dataset names for the interpolation
        variable unit  (e.g. "m").
        """
        raise NotImplementedError


class SolarResource(AbstractInterpolatedResource):
    """
    Class to handle Solar BaseResource .h5 files

    See Also
    --------
    resource.BaseResource : Parent class
    """

    INTERPOLABLE_DSETS = ["temperature", "windspeed"]
    VARIABLE_NAME = "height"
    VARIABLE_UNIT = "m"

    def get_SAM_df(self, site, extra_cols=None):
        """
        Get SAM solar resource DataFrame for given site

        Parameters
        ----------
        site : int
            Site to extract SAM DataFrame for.
        extra_cols : dict, optional
            A dictionary where they keys are extra columns
            to extract from the SAM solar resource DataFrame
            and the values are the names the new columns should
            have (e.g. extra_cols={'surface_albedo': 'Surface
            Albedo'} will extract the 'surface_albedo' from the
            resource file and call it 'Surface Albedo' in the output).

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

        if len(self) > 8784 or (self.time_index.minute != 0).any():
            res_df['Minute'] = self.time_index.minute

        time_zone = self.meta.loc[site, 'timezone']
        time_interval = len(self.time_index) // 8760

        main_cols = ['dni', 'dhi', 'wind_speed', 'air_temperature']
        extra_cols = extra_cols or {}
        for var in main_cols + list(extra_cols):
            ds_slice = (slice(None), site)
            var_array = self._get_ds(var, ds_slice)
            var_array = SAMResource.roll_timeseries(var_array, time_zone,
                                                    time_interval)
            res_df[var] = SAMResource.check_units(var, var_array,
                                                  tech='pvwattsv8')

        col_map = {'dni': 'DNI', 'dhi': 'DHI', 'wind_speed': 'Wind Speed',
                   'air_temperature': 'Temperature'}
        col_map.update(extra_cols)
        res_df = res_df.rename(columns=col_map)
        res_df.name = "SAM_-{}".format(site)

        return res_df

    @staticmethod
    def _preload_SAM(res, sites, tech='pvwattsv8', time_index_step=None,
                     means=False, clearsky=False, bifacial=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        tech : str, optional
            SAM technology string, by default 'pvwattsv8'
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
        SAM_res = SAMResource(sites, tech, res['time_index', time_slice],
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = res['meta', sites]

        if clearsky:
            SAM_res.set_clearsky()

        if bifacial and 'surface_albedo' not in SAM_res.var_list:
            SAM_res._var_list.append('surface_albedo')

        SAM_res.check_irradiance_datasets(res.datasets, clearsky=clearsky)
        SAM_res.load_rex_resource(res, SAM_res.var_list, time_slice, sites,
                                  hh=2)
        SAM_res.compute_irradiance(clearsky=clearsky)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, str_decode=True,
                    group=None, hsds=False, hsds_kwargs=None,
                    tech='pvwattsv8', time_index_step=None, means=False,
                    clearsky=False, bifacial=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        tech : str, optional
            SAM technology string, by default 'pvwattsv8'
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
        kwargs = {"unscale": unscale, "hsds": hsds, 'hsds_kwargs': hsds_kwargs,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(res, sites, tech=tech,
                                       time_index_step=time_index_step,
                                       means=means, clearsky=clearsky,
                                       bifacial=bifacial)

        return SAM_res


class NSRDB(SolarResource):
    """
    Class to handle NSRDB .h5 files

    See Also
    --------
    resource.BaseResource : Parent class
    """
    ADD_ATTR = ['add_offset', 'psm_add_offset']
    SCALE_ATTR = ['scale_factor', 'psm_scale_factor']
    UNIT_ATTR = ['units', 'psm_units']

    @staticmethod
    def _preload_SAM(res, sites, tech='pvwattsv8', time_index_step=None,
                     means=False, clearsky=False, bifacial=False,
                     downscale=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        tech : str, optional
            SAM technology string, by default 'pvwattsv8'
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
        downscale : NoneType | dict
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a dict of downscaling kwargs with a minimum
            requirement of the desired frequency e.g. 'frequency': '5min'
            and an option to add "variability_kwargs".

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, tech, res['time_index', time_slice],
                              means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = res['meta', sites]

        if clearsky:
            SAM_res.set_clearsky()

        if bifacial and 'surface_albedo' not in SAM_res.var_list:
            SAM_res._var_list.append('surface_albedo')

        SAM_res.check_irradiance_datasets(res.datasets, clearsky=clearsky)
        if not downscale:
            SAM_res.load_rex_resource(res, SAM_res.var_list, time_slice,
                                      sites, hh=2)
            SAM_res.compute_irradiance(clearsky=clearsky)
        else:
            # contingent import to avoid dependencies
            from rex.utilities.downscale import downscale_nsrdb
            frequency = downscale.pop('frequency')
            SAM_res = downscale_nsrdb(SAM_res, res, sam_vars=SAM_res.var_list,
                                      frequency=frequency,
                                      variability_kwargs=downscale)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, unscale=True, str_decode=True,
                    group=None, hsds=False, hsds_kwargs=None,
                    tech='pvwattsv8', time_index_step=None, means=False,
                    clearsky=False, bifacial=False, downscale=None):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        tech : str, optional
            SAM technology string, by default 'pvwattsv8'
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
        downscale : NoneType | dict
            Option for NSRDB resource downscaling to higher temporal
            resolution. Expects a dict of downscaling kwargs with a minimum
            requirement of the desired frequency e.g. 'frequency': '5min'
            and an option to add "variability_kwargs".

        Returns
        -------
        SAM_res : SAMResource
            Instance of SAMResource pre-loaded with Solar resource for sites
            in project_points
        """
        kwargs = {"unscale": unscale, "hsds": hsds, 'hsds_kwargs': hsds_kwargs,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(res, sites, tech=tech,
                                       time_index_step=time_index_step,
                                       means=means, clearsky=clearsky,
                                       bifacial=bifacial, downscale=downscale)

        return SAM_res


class WindResource(AbstractInterpolatedResource):
    """
    Class to handle Wind BaseResource .h5 files

    Notes
    -----
    Interpolation between hub-heights is performed using linear
    interpolation. While wind follows a log profile macroscopically,
    using power law interpolation doesn't allow for negative wind shear,
    which we see often at the taller hub heights we care about. In other
    words, power law interpolation is a bad assumption for near surface
    wind, and linear interpolation is a much better approach.
    Extrapolation beyond the resource data hub heights is still
    performed using power law interpolation.

    See Also
    --------
    resource.AbstractInterpolatedResource : Parent class

    Examples
    --------
    >>> import os
    >>> from rex import TESTDATADIR, WindResource
    >>> file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
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

    INTERPOLABLE_DSETS = ["temperature", "pressure", "windspeed",
                          "winddirection"]
    VARIABLE_NAME = "height"
    VARIABLE_UNIT = "m"

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)
        _, ds_name = os.path.split(ds)
        if 'SAM' in ds_name:
            site = ds_slice[0]
            if isinstance(site, (int, np.integer)):
                _, height = self._parse_name(ds_name)
                out = self.get_SAM_df(site, height)
            else:
                msg = "Can only extract SAM DataFrame for a single site"
                raise ResourceRuntimeError(msg)
        else:
            out = super().__getitem__(keys)

        return out

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

        Notes
        -----
        While wind follows a log profile macroscopically, using power
        law interpolation doesn't allow for negative wind shear, which
        we see often at the taller hub heights we care about. This means
        yous should prefer linear interpolation over power law
        interpolation when possible for near surface wind.
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

    @staticmethod
    def _check_hub_height(heights, h):
        """
        Check requested hub-height against available windspeed hub-heights
        If only one hub-height is available change request to match available
        hub-height

        Parameters
        ----------
        heights : dict
            Dictionary of available interpolation variable values for
            the interpolable datasets. For example, this could be:
            {'windspeed': [10, 100, 200]}
        h : int | float
            Requested hub-height

        Returns
        -------
        h : int | float
            Hub-height to extract
        """
        heights = heights['windspeed']
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

    def _get_ds_interpolated(self, ds_name, ds_slice):
        """Extract data from given dataset at desired interpolation value.

        Data is interpolated or extrapolated as needed.

        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple
            Tuple of (int, slice, list, ndarray) of what to extract
            from ds, each arg is for a sequential axis

        Returns
        -------
        out : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        var_name, __ = self._parse_name(ds_name)
        heights = self.heights[var_name]

        if not heights:
            msg = ("Missing height info for dataset '{}' in {}"
                   .format(var_name, self.h5_file))
            logger.error(msg)
            raise ResourceKeyError(msg)

        return super()._get_ds_interpolated(ds_name, ds_slice)

    def _get_calculated_ds(self, val, ds_name, var_name, ds_slice):
        """Get interpolated/extrapolated values for the dataset.

        Note that interpolation between hub-heights is performed using
        linear interpolation. While wind follows a log profile
        macroscopically, using power law interpolation doesn't allow for
        negative wind shear, which we see often at the taller hub
        heights we care about. In other words, power law interpolation
        is a bad assumption for near surface wind, and linear
        interpolation is a much better approach. Extrapolation beyond
        the resource data hub heights is still performed using power law
        interpolation.
        """
        heights = self._interpolation_variable[var_name]
        (h1, h2), extrapolate = self._get_nearest_val(val, heights)

        if extrapolate:
            msg = 'Extrapolating {}'.format(ds_name)

        dset_name_1 = '{}_{}{}'.format(var_name, h1, self.VARIABLE_UNIT)
        ts1 = super()._get_ds(dset_name_1, ds_slice)
        dset_name_2 = '{}_{}{}'.format(var_name, h2, self.VARIABLE_UNIT)
        ts2 = super()._get_ds(dset_name_2, ds_slice)

        if (var_name == 'windspeed') and extrapolate:
            if val < h1:
                try:
                    out = self._try_monin_obukhov_extrapolation(ts1,
                                                                ds_slice,
                                                                h1, val)
                    msg += ' using Monin Obukhov Extrapolation'
                    warnings.warn(msg, ExtrapolationWarning)
                except MoninObukhovExtrapolationError:
                    out = self.power_law_interp(ts1, h1, ts2, h2, val)
                    msg += ' using Power Law Extrapolation'
                    warnings.warn(msg, ExtrapolationWarning)
            else:
                out = self.power_law_interp(ts1, h1, ts2, h2, val)
                msg += ' using Power Law Extrapolation'
                warnings.warn(msg, ExtrapolationWarning)
        elif var_name == 'winddirection':
            out = self.circular_interp(ts1, h1, ts2, h2, val)
        else:
            out = linear_interp(ts1, h1, ts2, h2, val)
        return out

    def get_SAM_df(self, site, height, require_wind_dir=False, icing=False,
                   add_header=False):
        """
        Get SAM wind resource DataFrame for given site

        Parameters
        ----------
        site : int
            Site to extract SAM DataFrame for
        height : int
            Hub height to extract SAM variables at
        require_wind_dir : bool, optional
            Boolean flag as to whether wind direction will be loaded,
            by default False
        icing : bool, optional
            Boolean flag to include relativehumitidy for icing calculation,
            by default False
        add_header : bool, optional
            Add units and hub_height below variable names, needed for SAM .csv,
            by default False

        Returns
        -------
        res_df : pandas.DataFrame
            time-series DataFrame of resource variables needed to run SAM
        """
        if not self._unscale:
            raise ResourceValueError("SAM requires unscaled values")

        height = self._check_hub_height(self.heights, height)
        units = ['year', 'month', 'day', 'hour']
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
            variables.append('relativehumidity_2m')

        for var in variables:
            var_name = "{}_{}m".format(var, height)
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
                   'relativehumidity_2m': 'Relative Humidity'}
        res_df = res_df.rename(columns=col_map)
        res_df.name = "SAM_{}m-{}".format(height, site)

        if add_header:
            header = pd.DataFrame(columns=res_df.columns)
            header_units = units + [self.get_units(v) for v in variables]
            header_heights = [height] * len(header_units)
            header = pd.DataFrame([header_units, header_heights],
                                  columns=res_df.columns)
            res_df = pd.concat((header, res_df)).reset_index(drop=True)

        return res_df

    @staticmethod
    def _preload_SAM(res, sites, hub_heights, time_index_step=None,
                     means=False, require_wind_dir=False,
                     precip_rate=False, icing=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
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
                              res['time_index', time_slice],
                              hub_heights=hub_heights,
                              require_wind_dir=require_wind_dir,
                              means=means)

        sites = SAM_res.sites_slice
        SAM_res['meta'] = res['meta', sites]
        var_list = SAM_res.var_list
        if not require_wind_dir:
            var_list.remove('winddirection')

        h = res._check_hub_height(res.heights, SAM_res.h)
        res._set_sam_res(res, h, var_list, SAM_res, time_slice, sites)

        if precip_rate:
            var = 'precipitationrate'
            ds_name = '{}_0m'.format(var)
            SAM_res.append_var_list(var)
            SAM_res[var] = res[ds_name, time_slice, sites]

        if icing:
            var = 'rh'
            ds_name = 'relativehumidity_2m'
            SAM_res.append_var_list(var)
            SAM_res[var] = res[ds_name, time_slice, sites]

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, hub_heights, unscale=True,
                    str_decode=True, group=None, hsds=False, hsds_kwargs=None,
                    time_index_step=None, means=False,
                    require_wind_dir=False, precip_rate=False, icing=False):
        """
        Placeholder for classmethod that will pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        hub_heights : int | float | list
            Hub heights to extract for SAM
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
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
        kwargs = {"unscale": unscale, "hsds": hsds, 'hsds_kwargs': hsds_kwargs,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(res, sites, hub_heights,
                                       require_wind_dir=require_wind_dir,
                                       precip_rate=precip_rate, icing=icing,
                                       means=means,
                                       time_index_step=time_index_step)

        return SAM_res


class GeothermalResource(AbstractInterpolatedResource):
    """
    Class to handle Geothermal Resource .h5 files

    See Also
    --------
    resource.AbstractInterpolatedResource : Parent class

    Examples
    --------
    >>> file = '$TESTDATADIR/geo/template_geo_data.h5'
    >>> with GeothermalResource(file) as res:
    >>>     print(res.datasets)
    ['meta', 'temperature_3500m', 'temperature_4500m']

    GeothermalResource can linearly interpolate between available depths
    (3.5km & 4.5km).

    >>> with GeothermalResource(file) as res:
    >>>     temp_4km = res['temperature_4000m']
    >>>
    >>> temp_4km
    [450.5, 434. , 383.5, 422. , 387. , 316.5, 438. , 419. , 424. ,
     438.5]

    GeothermalResource can also linearly extrapolate beyond available depths

    >>> with GeothermalResource(file) as res:
    >>>     temp_5km = res['temperature_5000m']
    >>>
    >>> temp_5km
    ExtrapolationWarning: 5000 is outside the depth range (3500, 4500).
    Extrapolation to be used.
    [501.5, 338. , 428.5, 548. , 405. , 301.5, 440. , 565. , 446. ,
     341.5]
    """

    LAPSE_RATES = {}
    INTERPOLABLE_DSETS = ["temperature", "potential_MW"]
    VARIABLE_NAME = "depth"
    VARIABLE_UNIT = "m"

    @staticmethod
    def _preload_SAM(res, sites, depths, time_index_step=None,
                     means=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler or similar (NSRDB, WindResource,
            MultiFileResource, etc...)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        depths :  int | float | list
            Depths to extract for SAM
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
        time_slice = slice(None, None, time_index_step)
        SAM_res = SAMResource(sites, "geothermal",
                              res['time_index', time_slice],
                              depths=depths, means=means)
        sites = SAM_res.sites_slice
        SAM_res['meta'] = res['meta', sites]
        res._set_sam_res(res, SAM_res.d, SAM_res.var_list, SAM_res, time_slice,
                         sites)

        return SAM_res

    @classmethod
    def preload_SAM(cls, h5_file, sites, depths, unscale=True, str_decode=True,
                    group=None, hsds=False, hsds_kwargs=None,
                    time_index_step=None, means=False):
        """
        Pre-load project_points for SAM

        Parameters
        ----------
        h5_file : str
            String filepath to .h5 file to extract resource from. Can also
            be a path to an HSDS file (starts with /nrel/) or S3 file
            (starts with s3://)
        sites : list
            List of sites to be provided to SAM
            (sites is synonymous with gids aka spatial indices)
        depths :  int | float | list
            Depths to extract for SAM
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        str_decode : bool
            Boolean flag to decode the bytestring meta data into normal
            strings. Setting this to False will speed up the meta data read.
        group : str
            Group within .h5 resource file to open
        hsds : bool, optional
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS, by default False. This is now redundant; file paths
            starting with /nrel/ will be treated as hsds=True by default
        hsds_kwargs : dict, optional
            Dictionary of optional kwargs for h5pyd, e.g., bucket, username,
            password, by default None
        tech : str, optional
            SAM technology string, by default 'geothermal'
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
        kwargs = {"unscale": unscale, "hsds": hsds, 'hsds_kwargs': hsds_kwargs,
                  "str_decode": str_decode, "group": group}
        with cls(h5_file, **kwargs) as res:
            SAM_res = res._preload_SAM(res, sites, depths,
                                       time_index_step=time_index_step,
                                       means=means)

        return SAM_res


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

    # Calculate slope for every position in variable arrays
    m = (ts_2 - ts_1) / (h_2 - h_1)
    # Calculate intercept for every position in variable arrays
    b = ts_2 - m * h_2

    out = m * h + b

    return out
