# -*- coding: utf-8 -*-
"""
Module to perform bias correction of renewable energy resource data
"""
import numpy as np
import logging

from rex.utilities.bc_utils import QuantileDeltaMapping

logger = logging.getLogger(__name__)


def _irrad_pre_proc(ghi, dni, dhi):
    """Irradiance data pre processing to get ancillary variables
    (run before bias correction).

    Parameters
    ----------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)

    Returns
    -------
    ghi_zeros : np.ndarray
        2D boolean array, True where ghi==0, same shape as ghi input
    dni_zeros : np.ndarray
        2D boolean array, True where dni==0, same shape as dni input
    dhi_zeros : np.ndarray
        2D boolean array, True where dhi==0, same shape as dhi input
    cos_sza : np.ndarray
        2D array for cos(solar_zenith_angle) calculated from the basic
        relationship ``cos_sza = (ghi - dhi) / dni``
    """

    ghi_zeros = ghi == 0
    dni_zeros = dni == 0
    dhi_zeros = dhi == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sza = (ghi - dhi) / dni

    return ghi_zeros, dni_zeros, dhi_zeros, cos_sza


def _irrad_post_proc(ghi, dni, ghi_zeros, dni_zeros, dhi_zeros, cos_sza):
    """Irradiance data post processing to calculate DHI and set limits on
    irradiance variables (run after bias correction).

    Parameters
    ----------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    ghi_zeros : np.ndarray
        2D boolean array, True where ghi==0, same shape as ghi input
    dni_zeros : np.ndarray
        2D boolean array, True where dni==0, same shape as dni input
    dhi_zeros : np.ndarray
        2D boolean array, True where dhi==0, same shape as dhi input
    cos_sza : np.ndarray
        2D array for cos(solar_zenith_angle) calculated from the basic
        relationship ``cos_sza = (ghi - dhi) / dni``

    Returns
    -------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    """

    ghi = np.maximum(0, ghi)
    dni = np.maximum(0, dni)

    ghi[ghi_zeros] = 0
    dni[dni_zeros] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        dhi = ghi - (dni * cos_sza)

    dhi[dni_zeros] = ghi[dni_zeros]
    dhi[dhi_zeros] = 0
    dhi = np.maximum(0, dhi)

    assert not np.isnan(dhi).any()

    return ghi, dni, dhi


def lin_irrad(ghi, dni, dhi, scalar=1, adder=0):
    """Correct GHI and DNI using linear correction factors. Both irradiance
    variables are corrected as ``irradiance * scalar + adder``. DHI is
    preserved based on the relationship ``dhi = ghi - (dni * cos(sza))``. Times
    when GHI and DNI are zero are preserved and negative values are protected
    against.

    Parameters
    ----------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    scalar : np.ndarray
        1D array of linear scalar values in the shape (space,)
    adder : np.ndarray
        1D array of linear adder values in the shape (space,)

    Returns
    -------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    """

    ghi_zeros, dni_zeros, dhi_zeros, cos_sza = _irrad_pre_proc(ghi, dni, dhi)

    ghi = ghi * scalar + adder
    dni = dni * scalar + adder

    ghi, dni, dhi = _irrad_post_proc(ghi, dni, ghi_zeros, dni_zeros, dhi_zeros,
                                     cos_sza)

    return ghi, dni, dhi


def lin_ws(ws, scalar=1, adder=0):
    """Correct windspeed using linear correction factors. Windspeed is
    corrected as ``windspeed * scalar + adder`` with a minimum of zero.

    Parameters
    ----------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    scalar : np.ndarray
        1D array of linear scalar values in the shape (space,)
    adder : np.ndarray
        1D array of linear adder values in the shape (space,)

    Returns
    -------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    """
    ws = ws * scalar + adder
    ws = np.maximum(ws, 0)
    return ws


def qdm_irrad(ghi, dni, dhi,
              ghi_params_oh, dni_params_oh,
              ghi_params_mh, dni_params_mh,
              ghi_params_mf=None, dni_params_mf=None,
              dist='empirical', relative=True,
              sampling='linear', log_base=10,
              delta_denom_min=0.01, delta_denom_zero=None):
    """Correct irradiance using the quantile delta mapping based on the method
    from Cannon et al., 2015

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).

    Parameters
    ----------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    ghi_params_oh : np.ndarray | list
        2D array of **observed historical** distribution parameters created
        from a multi-year set of data where the shape is (space, N). This
        can be the output of a parametric distribution fit like
        ``scipy.stats.weibull_min.fit()`` where N is the number of
        parameters for that distribution, or this can define the x-values
        of N points from an empirical CDF that will be linearly
        interpolated between. If this is an empirical CDF, this must
        include the 0th and 100th percentile values and have even
        percentile spacing between values.
    dni_params_oh : np.ndarray | list
        Same requirements as ghi_params_oh. This input arg is for the
        **observed historical distribution** for DNI.
    ghi_params_mh : np.ndarray | list
        Same requirements as ghi_params_oh. This input arg is for the **modeled
        historical distribution** for GHI.
    dni_params_mh : np.ndarray | list
        Same requirements as ghi_params_oh. This input arg is for the **modeled
        historical distribution** for DNI.
    ghi_params_mf : np.ndarray | list | None
        Same requirements as ghi_params_oh. This input arg is for the **modeled
        future distribution** for GHI. If this is None, this defaults to
        ghi_params_mh (no future data, just corrected to modeled historical
        distribution)
    dni_params_mf : np.ndarray | list | None
        Same requirements as ghi_params_oh. This input arg is for the **modeled
        future distribution** for DNI. If this is None, this defaults to
        dni_params_mh. (no future data, just corrected to modeled historical
        distribution)
    dist : str | np.ndarray
        Probability distribution name to use to model the data which
        determines how the param args are used. This can "empirical" or any
        continuous distribution name from ``scipy.stats``. Can also be a 1D
        array of dist inputs if being used from reV, but they must all be
        the same option.
    relative : bool | np.ndarray
        Flag to preserve relative rather than absolute changes in
        quantiles. relative=False (default) will multiply by the change in
        quantiles while relative=True will add. See Equations 4-6 from
        Cannon et al., 2015 for more details. Can also be a 1D array of
        dist inputs if being used from reV, but they must all be the same
        option.
    sampling : str | np.ndarray
        If dist="empirical", this is an option for how the quantiles were
        sampled to produce the params inputs, e.g., how to sample the
        y-axis of the distribution (see sampling functions in
        ``rex.utilities.bc_utils``). "linear" will do even spacing, "log"
        will concentrate samples near quantile=0, and "invlog" will
        concentrate samples near quantile=1. Can also be a 1D array of dist
        inputs if being used from reV, but they must all be the same option.
    log_base : int | float | np.ndarray
        Log base value if sampling is "log" or "invlog". A higher value
        will concentrate more samples at the extreme sides of the
        distribution. Can also be a 1D array of dist inputs if being used from
        reV, but they must all be the same option.
    delta_denom_min : float | None
        Option to specify a minimum value for the denominator term in the
        calculation of a relative delta value. This prevents division by a
        very small number making delta blow up and resulting in very large
        output bias corrected values. See equation 4 of Cannon et al., 2015
        for the delta term.
    delta_denom_zero : float | None
        Option to specify a value to replace zeros in the denominator term
        in the calculation of a relative delta value. This prevents
        division by a very small number making delta blow up and resulting
        in very large output bias corrected values. See equation 4 of
        Cannon et al., 2015 for the delta term.

    Returns
    -------
    ghi : np.ndarray
        2D array of global horizontal irradiance values in shape (time, space)
    dni : np.ndarray
        2D array of direct normal irradiance values in shape (time, space)
    dhi : np.ndarray
        2D array of diffuse horizontal irradiance values in shape (time, space)
    """

    ghi_zeros, dni_zeros, dhi_zeros, cos_sza = _irrad_pre_proc(ghi, dni, dhi)

    ghi_qdm = QuantileDeltaMapping(ghi_params_oh, ghi_params_mh,
                                   ghi_params_mf, dist=dist,
                                   relative=relative, sampling=sampling,
                                   log_base=log_base,
                                   delta_denom_min=delta_denom_min,
                                   delta_denom_zero=delta_denom_zero)
    dni_qdm = QuantileDeltaMapping(dni_params_oh, dni_params_mh,
                                   dni_params_mf, dist=dist,
                                   relative=relative, sampling=sampling,
                                   log_base=log_base,
                                   delta_denom_min=delta_denom_min,
                                   delta_denom_zero=delta_denom_zero)

    # This will prevent inverse CDF functions from returning zero resulting in
    # a divide by zero error in the calculation of the QDM delta. These zeros
    # get fixed later in _irrad_post_proc
    ghi[ghi_zeros] = 1
    dni[dni_zeros] = 1

    ghi = ghi_qdm(ghi)
    dni = dni_qdm(dni)

    ghi, dni, dhi = _irrad_post_proc(ghi, dni, ghi_zeros, dni_zeros, dhi_zeros,
                                     cos_sza)

    return ghi, dni, dhi


def qdm_ws(ws, params_oh, params_mh, params_mf=None, dist='empirical',
           relative=True, sampling='linear', log_base=10,
           delta_denom_min=0.01, delta_denom_zero=None):
    """Correct windspeed using quantile delta mapping based on the method from
    Cannon et al., 2015

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).

    Parameters
    ----------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    params_oh : np.ndarray | list
        2D array of **observed historical** distribution parameters created
        from a multi-year set of data where the shape is (space, N). This
        can be the output of a parametric distribution fit like
        ``scipy.stats.weibull_min.fit()`` where N is the number of
        parameters for that distribution, or this can define the x-values
        of N points from an empirical CDF that will be linearly
        interpolated between. If this is an empirical CDF, this must
        include the 0th and 100th percentile values and have even
        percentile spacing between values.
    params_mh : np.ndarray | list
        Same requirements as params_oh. This input arg is for the **modeled
        historical distribution**.
    params_mf : np.ndarray | list | None
        Same requirements as params_oh. This input arg is for the **modeled
        future distribution**. If this is None, this defaults to params_mh
        (no future data, just corrected to modeled historical distribution)
    dist : str | np.ndarray
        Probability distribution name to use to model the data which
        determines how the param args are used. This can "empirical" or any
        continuous distribution name from ``scipy.stats``. Can also be a 1D
        array of dist inputs if being used from reV, but they must all be
        the same option.
    relative : bool | np.ndarray
        Flag to preserve relative rather than absolute changes in
        quantiles. relative=False (default) will multiply by the change in
        quantiles while relative=True will add. See Equations 4-6 from
        Cannon et al., 2015 for more details. Can also be a 1D array of
        dist inputs if being used from reV, but they must all be the same
        option.
    sampling : str | np.ndarray
        If dist="empirical", this is an option for how the quantiles were
        sampled to produce the params inputs, e.g., how to sample the
        y-axis of the distribution (see sampling functions in
        ``rex.utilities.bc_utils``). "linear" will do even spacing, "log"
        will concentrate samples near quantile=0, and "invlog" will
        concentrate samples near quantile=1. Can also be a 1D array of dist
        inputs if being used from reV, but they must all be the same option.
    log_base : int | float | np.ndarray
        Log base value if sampling is "log" or "invlog". A higher value
        will concentrate more samples at the extreme sides of the
        distribution. Can also be a 1D array of dist inputs if being used from
        reV, but they must all be the same option.
    delta_denom_min : float | None
        Option to specify a minimum value for the denominator term in the
        calculation of a relative delta value. This prevents division by a
        very small number making delta blow up and resulting in very large
        output bias corrected values. See equation 4 of Cannon et al., 2015
        for the delta term.
    delta_denom_zero : float | None
        Option to specify a value to replace zeros in the denominator term
        in the calculation of a relative delta value. This prevents
        division by a very small number making delta blow up and resulting
        in very large output bias corrected values. See equation 4 of
        Cannon et al., 2015 for the delta term.

    Returns
    -------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    """

    qdm = QuantileDeltaMapping(params_oh, params_mh, params_mf, dist=dist,
                               relative=relative, sampling=sampling,
                               log_base=log_base,
                               delta_denom_min=delta_denom_min,
                               delta_denom_zero=delta_denom_zero)

    # This will prevent inverse CDF functions from returning zero resulting in
    # a divide by zero error in the calculation of the QDM delta. These zeros
    # get fixed later
    ws_zeros = ws == 0
    ws[ws_zeros] = 0.01

    ws = qdm(ws)

    ws = np.maximum(ws, 0)
    ws[ws_zeros] = 0

    return ws
