# -*- coding: utf-8 -*-
"""
rex bias correction utilities.
"""
import scipy
import numpy as np
import logging


logger = logging.getLogger(__name__)


def sample_q_linear(n_samples):
    """Sample quantiles from 0 to 1 inclusive linearly with even spacing

    Parameters
    ----------
    n_samples : int
        Number of points to sample between 0 and 1

    Returns
    -------
    quantiles : np.ndarray
        1D array of evenly spaced samples from 0 to 1
    """
    quantiles = np.linspace(0, 1, n_samples)
    return quantiles


def sample_q_log(n_samples, log_base):
    """Sample quantiles from 0 to 1 while concentrating samples near quantile=0

    Parameters
    ----------
    n_samples : int
        Number of points to sample between 0 and 1
    log_base : int | float
        Log base value. A higher value will concentrate more samples at the
        extreme sides of the distribution.

    Returns
    -------
    quantiles : np.ndarray
        1D array of log-spaced samples from 0 to 1
    """
    quantiles = np.logspace(0, 1, n_samples, base=log_base)
    quantiles = (quantiles - 1) / (log_base - 1)
    return quantiles


def sample_q_invlog(n_samples, log_base):
    """Sample quantiles from 0 to 1 while concentrating samples near quantile=1

    Parameters
    ----------
    n_samples : int
        Number of points to sample between 0 and 1
    log_base : int | float
        Log base value. A higher value will concentrate more samples at the
        extreme sides of the distribution.

    Returns
    -------
    quantiles : np.ndarray
        1D array of log-spaced samples from 0 to 1
    """
    quantiles = np.logspace(0, 1, n_samples, base=log_base)
    quantiles = (quantiles - 1) / (log_base - 1)
    quantiles = np.array(sorted(1 - quantiles))
    return quantiles


class QuantileDeltaMapping:
    """Class for quantile delta mapping based on the method from
    Cannon et al., 2015

    Note that this is a utility class for implementing QDM and should not be
    requested directly as a method in the reV/rex bias correction table input

    Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
    Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in
    Quantiles and Extremes? Journal of Climate 28, 6938–6959 (2015).
    """

    def __init__(self, params_oh, params_mh, params_mf, dist='empirical',
                 relative=True, sampling='linear', log_base=10):
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
        sampling : str | np.ndarray
            If dist="empirical", this is an option for how the quantiles were
            sampled to produce the params inputs, e.g., how to sample the
            y-axis of the distribution (see sampling functions in
            ``rex.utilities.bc_utils``). "linear" will do even spacing, "log"
            will concentrate samples near quantile=0, and "invlog" will
            concentrate samples near quantile=1. Can also be a 1D array of dist
            inputs if being used from reV, but they must all be the same
            option.
        log_base : int | float | np.ndarray
            Log base value if sampling is "log" or "invlog". A higher value
            will concentrate more samples at the extreme sides of the
            distribution. Can also be a 1D array of dist inputs if being used
            from reV, but they must all be the same option.
        """

        self.params_oh = params_oh
        self.params_mh = params_mh
        self.params_mf = params_mf if params_mf is not None else params_mh
        self.relative = bool(self._clean_kwarg(relative))
        self.dist_name = str(self._clean_kwarg(dist)).casefold()
        self.sampling = str(self._clean_kwarg(sampling)).casefold()
        self.log_base = float(self._clean_kwarg(log_base))
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

        if len(params.shape) == 1:
            params = np.expand_dims(params, 0)

        msg = (f'params must be 2D array of shape ({arr_shape[1]}, N) '
               f'but received shape {params.shape}')
        assert len(params.shape) == 2, msg
        assert params.shape[0] == arr_shape[1], msg

        if self.scipy_dist is not None:
            params = [params[:, i] for i in range(params.shape[1])]

        return params

    def _get_quantiles(self, n_samples):
        """If dist='empirical', this will get the quantile values for the CDF
        x-values specified in the input params"""

        if self.sampling == 'linear':
            quantiles = sample_q_linear(n_samples)
        elif self.sampling == 'log':
            quantiles = sample_q_log(n_samples, self.log_base)
        elif self.sampling == 'invlog':
            quantiles = sample_q_invlog(n_samples, self.log_base)
        else:
            msg = ('sampling option must be linear, log, or invlog, but '
                   'received: {}'.format(self.sampling))
            logger.error(msg)
            raise KeyError(msg)

        return quantiles

    def cdf(self, x, params):
        """Run the CDF function e.g., convert physical variable to quantile"""

        if self.scipy_dist is None:
            p = np.zeros_like(x)
            for idx in range(x.shape[1]):
                xp = params[idx, :]
                fp = self._get_quantiles(len(xp))
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
                xp = self._get_quantiles(len(fp))
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

        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, 1)

        params_oh = self._clean_params(self.params_oh, arr.shape)
        params_mh = self._clean_params(self.params_mh, arr.shape)
        params_mf = self._clean_params(self.params_mf, arr.shape)

        # Equation references are from Section 3 of Cannon et al 2015:
        # Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
        # Precipitation by Quantile Mapping: How Well Do Methods Preserve
        # Changes in Quantiles and Extremes? Journal of Climate 28, 6938–6959
        # (2015).

        q_mf = self.cdf(arr, params_mf)  # Eq.3: Tau_m_p = F_m_p(x_m_p)
        x_oh = self.ppf(q_mf, params_oh)  # Eq.5: x^_o:m_h:p = F-1_o_h(Tau_m_p)
        x_mh_mf = self.ppf(q_mf, params_mh)  # Eq.4 denom: F-1_m_h(Tau_m_p)

        if self.relative:
            x_mh_mf[x_mh_mf == 0] = 0.001  # arbitrary limit to prevent div 0
            delta = arr / x_mh_mf  # Eq.4: x_m_p / F-1_m_h(Tau_m_p)
            arr_bc = x_oh * delta  # Eq.6: x^_m_p = x^_o:m_h:p * delta
        else:
            delta = arr - x_mh_mf  # Eq.4: x_m_p - F-1_m_h(Tau_m_p)
            arr_bc = x_oh + delta  # Eq.6: x^_m_p = x^_o:m_h:p + delta

        msg = ('Input shape {} does not match QDM bias corrected output '
               'shape {}!'.format(arr.shape, arr_bc.shape))
        assert arr.shape == arr_bc.shape, msg

        return arr_bc
