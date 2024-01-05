# -*- coding: utf-8 -*-
"""
Module to perform bias correction of renewable energy resource data
"""
import scipy
import numpy as np
import logging


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


class _QuantileDeltaMapping:
    """Class for quantile delta mapping based on the method from
    Cannon et al., 2015

    Note that this is a utility class for implementing QDM and should not be
    requested directly as a method in the reV/rex bias correction table input

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).
    """

    def __init__(self, params_oh, params_mh, params_mf, dist, relative):
        """
        Parameters
        ----------
        params_oh : np.ndarray
            2D array of **observed historical** distribution parameters created
            from a multi-year set of data where the shape is (space, N). This
            can be the output of a parametric distribution fit like
            ``scipy.stats.weibull_min.fit()`` where N is the number of
            parameters for that distribution, or this can define the x-values
            of N points from an empirical CDF that will be linearly
            interpolated between. If this is an empirical CDF, this must
            include the 0th and 100th percentile values and have even
            percentile spacing between values.
        params_mh : np.ndarray
            Same requirements as params_oh. This input arg is for the **modeled
            historical distribution**.
        params_mf : np.ndarray | None
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
        """

        self.params_oh = params_oh
        self.params_mh = params_mh
        self.params_mf = params_mf if params_mf is not None else params_mh
        self.relative = self._clean_kwarg(relative)
        self.dist_name = self._clean_kwarg(dist)
        self.scipy_dist = None

        if self.dist_name != 'empirical':
            self.scipy_dist = getattr(scipy.stats, self.dist_name, None)
            if self.scipy_dist is None:
                msg = ('Could not get requested distribution "{}" from '
                       '``scipy.stats``. Please double check your spelling '
                       'and select "empirical" or one of the continuous '
                       'distribution options from here: '
                       'https://docs.scipy.org/doc/scipy/reference/stats.html'
                       .format(self.dist_name))
                logger.error(msg)
                raise KeyError(msg)

    @staticmethod
    def _clean_kwarg(inp):
        """Clean any kwargs inputs (e.g., dist, relative) that might be
        provided as an array and must be collapsed into a single string or
        boolean value"""
        unique = np.unique(inp)
        msg = ('_QuantileDeltaMapping kwargs must have only one unique input '
               'even if being called with arrays as part of reV but found: {}'
               .format(unique))
        assert len(unique) == 1, msg

        while isinstance(inp, np.ndarray):
            inp = inp[0]

        return inp

    def _clean_params(self, params, arr_shape):
        """Verify and clean 2D parameter arrays for passing into empirical
        distribution or scipy continuous distribution functions.

        Parameters
        ----------
        params : np.ndarray
            Input params shape should be (space, N) where N is the number of
            parameters for the distribution.
        arr_shape : tuple
            Array shape should be (time, space).

        Returns
        -------
        params : np.ndarray | list
            If a scipy continuous dist is set, this output will be params
            unpacked along axis=1 into a list so that the list entries
            represent the scipy distribution parameters
            (e.g., shape, scale, loc) and each list entry is of shape (space,)
        """

        msg = f'params must be 2D array but received {type(params)}'
        assert isinstance(params, np.ndarray), msg
        msg = (f'params must be 2D array of shape ({arr_shape[1]}, N) '
               f'but received shape {params.shape}')
        assert len(params.shape) == 2, msg
        assert params.shape[0] == arr_shape[1], msg

        if self.scipy_dist is not None:
            params = [params[:, i] for i in range(params.shape[1])]

        return params

    def cdf(self, x, params):
        """Run the CDF function e.g., convert physical variable to quantile"""

        if self.scipy_dist is None:
            p = np.zeros_like(x)
            for idx in range(x.shape[1]):
                xp = params[idx, :]
                fp = np.linspace(0, 1, len(xp))
                p[:, idx] = np.interp(x[:, idx], xp, fp)

        else:
            p = self.scipy_dist.cdf(x, *params)

        return p

    def ppf(self, p, params):
        """Run the inverse CDF function (percent point function) e.g., convert
        quantile to physical variable"""

        if self.scipy_dist is None:
            x = np.zeros_like(p)
            for idx in range(p.shape[1]):
                fp = params[idx, :]
                xp = np.linspace(0, 1, len(fp))
                x[:, idx] = np.interp(p[:, idx], xp, fp)
        else:
            x = self.scipy_dist.ppf(p, *params)

        return x

    def __call__(self, arr):
        """Run the QDM function to bias correct an array

        Parameters
        ----------
        arr : np.ndarray
            2D array of values in shape (time, space)

        Returns
        -------
        arr : np.ndarray
            Bias corrected copy of the input array with same shape.
        """

        params_oh = self._clean_params(self.params_oh, arr.shape)
        params_mh = self._clean_params(self.params_mh, arr.shape)
        params_mf = self._clean_params(self.params_mf, arr.shape)

        p_mf = self.cdf(arr, params_mf)
        x_oh = self.ppf(p_mf, params_oh)

        if self.relative:
            delta = arr / self.ppf(p_mf, params_mh)
            arr_bc = x_oh * delta
        else:
            delta = arr - self.ppf(p_mf, params_mh)
            arr_bc = x_oh + delta

        msg = ('Input shape {} does not match QDM bias corrected output '
               'shape {}!'.format(arr.shape, arr_bc.shape))
        assert arr.shape == arr_bc.shape, msg

        return arr_bc


def qdm_irrad(ghi, dni, dhi,
              ghi_params_oh, dni_params_oh,
              ghi_params_mh, dni_params_mh,
              ghi_params_mf=None, dni_params_mf=None,
              dist='empirical', relative=True):
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

    ghi_qdm = _QuantileDeltaMapping(ghi_params_oh, ghi_params_mh,
                                    ghi_params_mf, dist, relative)
    dni_qdm = _QuantileDeltaMapping(dni_params_oh, dni_params_mh,
                                    dni_params_mf, dist, relative)

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
           relative=True):
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

    Returns
    -------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    """

    qdm = _QuantileDeltaMapping(params_oh, params_mh, params_mf, dist,
                                relative)
    ws = qdm(ws)
    ws = np.maximum(ws, 0)
    return ws
