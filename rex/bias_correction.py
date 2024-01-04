# -*- coding: utf-8 -*-
"""
Module to perform bias correction of renewable energy resource data
"""
import scipy
import numpy as np
import logging


logger = logging.getLogger(__name__)


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

    ghi_zeros = ghi == 0
    dni_zeros = dni == 0
    dhi_zeros = dhi == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sza = (ghi - dhi) / dni

    ghi = ghi * scalar + adder
    dni = dni * scalar + adder

    ghi = np.maximum(0, ghi)
    dni = np.maximum(0, dni)

    ghi[ghi_zeros] = 0
    dni[dni_zeros] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        dhi = ghi - (dni * cos_sza)

    dhi[dni_zeros] = ghi[dni_zeros]
    dhi = np.maximum(0, dhi)
    dhi[dhi_zeros] = 0

    assert not np.isnan(dhi).any()

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


class PQDM:
    """Class for parametric quantile delta mapping based on a parametric
    implementation of the method from Cannon et al., 2015

    Note that this is a utility class for implementing PQDM and should not be
    requested directly as a method in the reV/rex bias correction table input

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).
    """

    def __init__(self, params_oh, params_mh, params_mf, dist, relative):
        """
        Parameters
        ----------
        ws : np.ndarray
            2D array of windspeed values in shape (time, space)
        params_oh : np.ndarray
            2D array of probability distribution parameters created using a
            function like ``scipy.stats.weibull_min.fit()`` where the shape is
            (space, N) with N being the number of parameters required by the
            specified distribution e.g., (shape, loc, scale) for weibull_min.
            This input arg is for the **observed historical distribution**.
        params_mh : np.ndarray
            Same requirements as params_oh. This input arg is for the **modeled
            historical distribution**.
        params_mf : np.ndarray | None
            Same requirements as params_oh. This input arg is for the **modeled
            future distribution**. If this is None, this defaults to params_mh
            (no future data).
        dist : str | np.ndarray
            Parametric probability distribution name to use to model the
            windspeed. This can be any distribution name from ``scipy.stats``,
            but "weibull_min" is a common choice for windspeed distributions.
            Can also be an array of dist inputs if being used from reV, but
            they must all be the same option.
        relative : bool | np.ndarray
            Flag to preserve relative rather than absolute changes in
            quantiles. relative=False (default) will multiply by the change in
            quantiles while relative=True will add. See Equations 4-6 from
            Cannon et al., 2015 for more details. Can also be an array of dist
            inputs if being used from reV, but they must all be the same
            option.
        """
        self.params_oh = params_oh
        self.params_mh = params_mh
        self.params_mf = params_mf if params_mf is not None else params_mh
        self.dist_name = self._clean_kwarg(dist)
        self.relative = self._clean_kwarg(relative)

        self.scipy_dist = getattr(scipy.stats, self.dist_name, None)
        if self.scipy_dist is None:
            msg = ('Could not get requested distribution "{}" from '
                   '``scipy.stats``. Please double check your spelling and '
                   'select one of the options from here: '
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
        msg = ('PQDM kwargs must have only one unique input even if being '
               'called with arrays as part of reV but found: {}'
               .format(unique))
        assert len(unique) == 1, msg

        while isinstance(inp, np.ndarray):
            inp = inp[0]

        return inp

    @staticmethod
    def _clean_params(params, arr_shape):
        """Re-organize 2D parameter arrays for passing into scipy distribution
        functions.

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
            If the two inputs are of the expected shape, this output will be
            params unpacked along axis=1 into a list so that the list entries
            represent the scipy distribution parameters (e.g., shape, scale,
            loc) and each list entry is of shape (space,)
        """
        if arr_shape[1] == params.shape[0]:
            params = [params[:, i] for i in range(params.shape[1])]
        return params

    def __call__(self, arr):
        """Run the PQDM function to bias correct an array

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

        p_mf = self.scipy_dist.cdf(arr, *params_mf)
        x_oh = self.scipy_dist.ppf(p_mf, *params_oh)

        if self.relative:
            delta = arr / self.scipy_dist.ppf(p_mf, *params_mh)
            arr_bc = x_oh * delta
        else:
            delta = arr - self.scipy_dist.ppf(p_mf, *params_mh)
            arr_bc = x_oh + delta

        msg = ('Input shape {} does not match PQDM bias corrected output '
               'shape {}!'.format(arr.shape, arr_bc.shape))
        assert arr.shape == arr_bc.shape, msg

        return arr_bc


def pqdm_ws(ws, params_oh, params_mh, params_mf=None, dist='weibull_min',
            relative=True):
    """Correct windspeed using parametric quantile delta mapping based on a
    parametric implementation of the method from Cannon et al., 2015

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).

    Parameters
    ----------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    params_oh : np.ndarray | list
        2D array of probability distribution parameters created using a
        function like ``scipy.stats.weibull_min.fit()`` where the shape is
        (space, N) with N being the number of parameters required by the
        specified distribution e.g., (shape, loc, scale) for weibull_min. This
        input arg is for the **observed historical distribution**.
    params_mh : np.ndarray | list
        Same requirements as params_oh. This input arg is for the **modeled
        historical distribution**.
    params_mf : np.ndarray | list | None
        Same requirements as params_oh. This input arg is for the **modeled
        future distribution**. If this is None, this defaults to params_mh
        (no future data).
    dist : str | np.ndarray
        Parametric probability distribution name to use to model the windspeed.
        This can be any distribution name from ``scipy.stats``, but
        "weibull_min" is a common choice for windspeed distributions. Can also
        be an array of dist inputs if being used from reV, but they must all be
        the same option.
    relative : bool | np.ndarray
        Flag to preserve relative rather than absolute changes in quantiles.
        relative=False (default) will multiply by the change in quantiles while
        relative=True will add. See Equations 4-6 from Cannon et al., 2015 for
        more details. Can also be an array of dist inputs if being used from
        reV, but they must all be the same option.

    Returns
    -------
    ws : np.ndarray
        2D array of windspeed values in shape (time, space)
    """

    pqdm = PQDM(params_oh, params_mh, params_mf, dist, relative)
    ws = pqdm(ws)
    ws = np.maximum(ws, 0)
    return ws
