# -*- coding: utf-8 -*-
"""Solar resource downscaling utility methods.

Created on April 8 2019

@author: gbuster
"""
import numpy as np
import pandas as pd
import logging

from rex.utilities.solar_position import SolarPosition
from rex.utilities.utilities import get_lat_lon_cols, pd_date_range

from nsrdb.all_sky import CLEAR_TYPES
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.utilities.interpolation import temporal_lin, temporal_step


logger = logging.getLogger(__name__)


def make_time_index(year, frequency, set_timezone=True):
    """Make the NSRDB target time index.

    Parameters
    ----------
    year : int
        Year for time index.
    frequency : str
        String in the Pandas frequency format, e.g. '5min'.
    set_timezone : bool
        Flag to set a timezone-aware time index. will be set to UTC with
        zero offset.

    Returns
    -------
    ti : pd.DatetimeIndex
        Pandas datetime index for a full year at the requested resolution.
    """
    ti = pd_date_range('1-1-{y}'.format(y=year), '1-1-{y}'.format(y=year + 1),
                       freq=frequency)[:-1]

    if set_timezone:
        ti = ti.tz_localize('UTC')

    return ti


def interp_cld_props(data, ti_native, ti_new,
                     var_list=('cld_reff_dcomp', 'cld_opd_dcomp')):
    """Interpolate missing cloud properties (NOT CLOUD TYPE).

    Parameters
    ----------
    data : dict
        Namespace of variables for input to all_sky. Must include the cloud
        variables in var_list and "cloud_type".
    ti_native : pd.DateTimeIndex
        Native time index of the original NSRDB data.
    ti_new : pd.DateTimeIndex
        Intended downscaled time index.
    var_list : list | tuple
        Cloud variables to downscale.

    Returns
    -------
    data : dict
        Namespace of variables with the cloud variables in var_list downscaled
        to the requested ti_new.
    """

    for var in var_list:

        # make sparse dataframe with new time_index
        data[var] = pd.DataFrame(data[var], index=ti_native).reindex(ti_new)

        # find location of bad data
        cld_fill_flag = ((data[var] < 0) | data[var].isnull())

        # replace to-fill values with nan
        data[var].values[cld_fill_flag] = np.nan

        # set clear timesteps cloud props to zero for better transitions
        data[var].values[np.isin(data['cloud_type'], CLEAR_TYPES)] = 0.0

        # interpolate empty values
        data[var] = data[var].interpolate(method='linear', axis=0).values

        logger.debug('Downscaled array for "{}" has shape {} and {} NaN values'
                     .format(var, data[var].shape, np.isnan(data[var]).sum()))

    return data


def downscale_nsrdb(SAM_res, res, frequency='5min',
                    sam_vars=('dhi', 'dni', 'wind_speed', 'air_temperature'),
                    variability_kwargs=None):
    """Downscale the NSRDB resource and return the preloaded SAM_res.

    Parameters
    ----------
    SAM_res : SAMResource
        SAM resource object.
    res : NSRDB
        NSRDB resource handler.
    frequency : str
        String in the Pandas frequency format, e.g. '5min'.
    sam_vars : tuple | list
        Variables to save to SAM resource handler before returning.
    variability_kwargs : Nonetype | dict
        Downscaling kwargs to the NSRDB all sky method call. Should include
        maximum GHI synthetic variability fraction ("var_frac") which
        will be set to 0.05 (5%) if variability_kwargs is None.

    Returns
    -------
    SAM_res : SAMResource
        SAM resource object with downscaled solar resource data loaded.
        Time index and shape are also updated.
    """

    logger.debug('Downscaling NSRDB resource data to "{}".'.format(frequency))

    # variables required for all-sky not including clouds, ti, sza
    var_list = ('aod',
                'surface_pressure',
                'surface_albedo',
                'ssa',
                'asymmetry',
                'alpha',
                'ozone',
                'total_precipitable_water',
                )

    # Indexing variable
    sites_slice = SAM_res.sites_slice

    # get downscaled time_index
    time_index = make_time_index(res.time_index.year[0], frequency)
    SAM_res._time_index = time_index
    SAM_res._shape = (len(time_index), len(SAM_res.sites))

    logger.debug('Native resource time index has length {}: \n{}'
                 .format(len(res.time_index), res.time_index))
    logger.debug('Target resource time index has length {}: \n{}'
                 .format(len(time_index), time_index))

    # downscale variables into an all-sky input variable namespace
    all_sky_ins = {'time_index': time_index}
    for var in var_list:
        arr = res[var, :, sites_slice]
        arr = temporal_lin(res[var, :, sites_slice], res.time_index,
                           time_index)
        all_sky_ins[var] = arr
        logger.debug('Downscaled array for "{}" has shape {} and {} NaN values'
                     .format(var, arr.shape, np.isnan(arr).sum()))

    # calculate downscaled solar zenith angle
    lat_lon_cols = get_lat_lon_cols(res.meta)
    lat_lon = res.meta.loc[SAM_res.sites, lat_lon_cols]\
        .values.astype(np.float32)
    sza = SolarPosition(time_index, lat_lon).zenith
    all_sky_ins['solar_zenith_angle'] = sza
    logger.debug('Downscaled array for "solar_zenith_angle" '
                 'has shape {} and {} NaN values'
                 .format(sza.shape, np.isnan(sza).sum()))

    # get downscaled cloud properties
    all_sky_ins['cloud_type'] = temporal_step(
        res['cloud_type', :, sites_slice], res.time_index, time_index)
    all_sky_ins['cld_opd_dcomp'] = res['cld_opd_dcomp', :, sites_slice]
    all_sky_ins['cld_reff_dcomp'] = res['cld_reff_dcomp', :, sites_slice]
    all_sky_ins = interp_cld_props(all_sky_ins, res.time_index, time_index)

    # add all sky kwargs such as variability
    if variability_kwargs is None:
        variability_kwargs = {'var_frac': 0.05}

    all_sky_ins['variability_kwargs'] = variability_kwargs

    # run all-sky
    logger.debug('Running all-sky for "{}".'.format(SAM_res))
    all_sky_outs = all_sky(**all_sky_ins)

    # set downscaled data to sam resource handler
    for k, v in all_sky_outs.items():
        if k in sam_vars:
            SAM_res[k] = v

    # downscale extra vars needed for SAM but not for all-sky
    for var in sam_vars:
        if var not in SAM_res._res_arrays:
            SAM_res[var] = temporal_lin(res[var, :, sites_slice],
                                        res.time_index, time_index)

    return SAM_res
