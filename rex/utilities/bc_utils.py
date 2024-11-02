# -*- coding: utf-8 -*-
"""
rex bias correction utilities.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np
import scipy

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


def sample_cdf(quantiles, x_values, n_samples):
    """Randomly draw a number of real values from a CDF.

    quantiles : np.ndarray
        1D array of quantile values from 0 to 1. Must be monotonic.
    x_values : np.ndarray
        Values on the x-axis of a CDF corresponding to quantiles. Must be
        monotonic.
    n_samples : int
        Number of sample to draw

    Returns
    -------
    samples : np.ndarray
        1D array of real values sampled from the CDF made up by quantiles and
        x_values
    """
    samples = np.random.uniform(0, 1, n_samples)
    samples = np.interp(samples, quantiles, x_values)
    return samples


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
                 relative=True, sampling='linear', log_base=10,
                 delta_denom_min=None, delta_denom_zero=None,
                 delta_range=None):
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
        delta_range : tuple | None
            Option to set a (min, max) on the delta term in QDM. This can help
            prevent QDM from making non-realistic increases/decreases in
            otherwise physical values. See equation 4 of Cannon et al., 2015
            for the delta term.
        """

        self.params_oh = params_oh
        self.params_mh = params_mh
        self.params_mf = params_mf if params_mf is not None else params_mh
        self.relative = bool(self._clean_kwarg(relative))
        self.dist_name = str(self._clean_kwarg(dist)).casefold()
        self.sampling = str(self._clean_kwarg(sampling)).casefold()
        self.log_base = float(self._clean_kwarg(log_base))
        self.scipy_dist = None
        self.delta_denom_min = delta_denom_min
        self.delta_denom_zero = delta_denom_zero
        self.delta_range = delta_range

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

    @staticmethod
    def _clean_params(params, arr_shape, scipy_dist):
        """Verify and clean 2D parameter arrays for passing into empirical
        distribution or scipy continuous distribution functions.

        Parameters
        ----------
        params : np.ndarray
            Input params shape should be (space, N) where N is the number of
            parameters for the distribution.
        arr_shape : tuple
            Array shape should be (time, space).
        scipy_dist : scipy.stats.rv_continuous | None
            Any continuous distribution class from ``scipy.stats`` or None if
            using an empirical distribution (taken from attribute
            ``QuantileDeltaMapping.scipy_dist``)

        Returns
        -------
        params : np.ndarray | list
            If a scipy continuous dist is set, this output will be params
            unpacked along axis=1 into a list so that the list entries
            represent the scipy distribution parameters
            (e.g., shape, scale, loc) and each list entry is of shape (space,)
        """

        msg = f'params must be 2D array but received {type(params)}'
        assert hasattr(params, 'shape'), msg

        if len(params.shape) == 1:
            params = np.expand_dims(params, 0)

        msg = (f'params must be 2D array of shape ({arr_shape[1]}, N) '
               f'but received shape {params.shape}')
        assert len(params.shape) == 2, msg
        assert params.shape[0] == arr_shape[1], msg

        if scipy_dist is not None:
            params = [params[:, i] for i in range(params.shape[1])]

        return params

    @staticmethod
    def _get_quantiles(n_samples, sampling, log_base):
        """If dist='empirical', this will get the quantile values for the CDF
        x-values specified in the input params"""

        if sampling == 'linear':
            quantiles = sample_q_linear(n_samples)
        elif sampling == 'log':
            quantiles = sample_q_log(n_samples, log_base)
        elif sampling == 'invlog':
            quantiles = sample_q_invlog(n_samples, log_base)
        else:
            msg = ('sampling option must be linear, log, or invlog, but '
                   'received: {}'.format(sampling))
            logger.error(msg)
            raise KeyError(msg)

        return quantiles

    @classmethod
    def cdf(cls, x, params, scipy_dist, sampling, log_base):
        """Run the CDF function e.g., convert physical variable to quantile"""

        if scipy_dist is None:
            p = np.zeros_like(x)
            for idx in range(x.shape[1]):
                xp = params[idx, :]
                fp = cls._get_quantiles(len(xp), sampling, log_base)
                p[:, idx] = np.interp(x[:, idx], xp, fp)
        else:
            p = scipy_dist.cdf(x, *params)

        return p

    @classmethod
    def ppf(cls, p, params, scipy_dist, sampling, log_base):
        """Run the inverse CDF function (percent point function) e.g., convert
        quantile to physical variable"""

        if scipy_dist is None:
            x = np.zeros_like(p)
            for idx in range(p.shape[1]):
                fp = params[idx, :]
                xp = cls._get_quantiles(len(fp), sampling, log_base)
                x[:, idx] = np.interp(p[:, idx], xp, fp)
        else:
            x = scipy_dist.ppf(p, *params)

        return x

    @classmethod
    def run_qdm(cls, arr, params_oh, params_mh, params_mf,
                scipy_dist, relative, sampling, log_base, delta_denom_min,
                delta_denom_zero, delta_range):
        """Run the actual QDM operation from args without initializing the
        ``QuantileDeltaMapping`` object

        Parameters
        ----------
        arr : np.ndarray
            2D array of values in shape (time, space)
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
        params_mf : np.ndarray
            Same requirements as params_oh. This input arg is for the **modeled
            future distribution**.
        scipy_dist : scipy.stats.rv_continuous | None
            Any continuous distribution class from ``scipy.stats`` or None if
            using an empirical distribution (taken from attribute
            ``QuantileDeltaMapping.scipy_dist``)
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
        delta_range : tuple | None
            Option to set a (min, max) on the delta term in QDM. This can help
            prevent QDM from making non-realistic increases/decreases in
            otherwise physical values. See equation 4 of Cannon et al., 2015
            for the delta term.

        Returns
        -------
        arr : np.ndarray
            Bias corrected copy of the input array with same shape.
        """

        params_oh = cls._clean_params(params_oh, arr.shape, scipy_dist)
        params_mh = cls._clean_params(params_mh, arr.shape, scipy_dist)
        params_mf = cls._clean_params(params_mf, arr.shape, scipy_dist)

        # Equation references are from Section 3 of Cannon et al 2015:
        # Cannon, A. J., Sobie, S. R. & Murdock, T. Q. Bias Correction of GCM
        # Precipitation by Quantile Mapping: How Well Do Methods Preserve
        # Changes in Quantiles and Extremes? Journal of Climate 28, 6938–6959
        # (2015).

        logger.debug('Computing CDF on modeled future data')
        # Eq.3: Tau_m_p = F_m_p(x_m_p)
        q_mf = cls.cdf(arr, params_mf, scipy_dist, sampling, log_base)

        logger.debug('Computing PPF on observed historical data')
        # Eq.5: x^_o:m_h:p = F-1_o_h(Tau_m_p)
        x_oh = cls.ppf(q_mf, params_oh, scipy_dist, sampling, log_base)

        logger.debug('Computing PPF on modeled historical data')
        # Eq.4 denom: F-1_m_h(Tau_m_p)
        x_mh_mf = cls.ppf(q_mf, params_mh, scipy_dist, sampling, log_base)

        logger.debug('Finished computing distributions.')

        if relative:
            if delta_denom_zero is not None:
                x_mh_mf[x_mh_mf == 0] = delta_denom_zero
            if delta_denom_min is not None:
                x_mh_mf = np.maximum(x_mh_mf, delta_denom_min)
            delta = arr / x_mh_mf  # Eq.4: x_m_p / F-1_m_h(Tau_m_p)
            if delta_range is not None:
                delta = np.maximum(delta, np.min(delta_range))
                delta = np.minimum(delta, np.max(delta_range))
            arr_bc = x_oh * delta  # Eq.6: x^_m_p = x^_o:m_h:p * delta

        else:
            delta = arr - x_mh_mf  # Eq.4: x_m_p - F-1_m_h(Tau_m_p)
            if delta_range is not None:
                delta = np.maximum(delta, np.min(delta_range))
                delta = np.minimum(delta, np.max(delta_range))
            arr_bc = x_oh + delta  # Eq.6: x^_m_p = x^_o:m_h:p + delta

        return arr_bc

    def __call__(self, arr, max_workers=1):
        """Run the QDM function to bias correct an array

        Parameters
        ----------
        arr : np.ndarray
            2D array of values in shape (time, space)
        max_workers : int, None
            Number of parallel workers to use in QDM bias correction. 1 will
            run in serial (default), None will use all available cores.

        Returns
        -------
        arr : np.ndarray
            Bias corrected copy of the input array with same shape.
        """

        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, 1)

        if max_workers == 1:
            arr_bc = self.run_qdm(arr, self.params_oh, self.params_mh,
                                  self.params_mf, self.scipy_dist,
                                  self.relative, self.sampling, self.log_base,
                                  self.delta_denom_min, self.delta_denom_zero,
                                  self.delta_range)
        else:
            max_workers = max_workers or os.cpu_count()
            sslices = np.array_split(np.arange(arr.shape[1]), arr.shape[1])
            sslices = [slice(idx[0], idx[-1] + 1) for idx in sslices]
            arr_bc = arr.copy()
            futures = {}
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                for idx in range(arr.shape[1]):
                    idx = slice(idx, idx + 1)
                    fut = exe.submit(self.run_qdm, arr[:, idx],
                                     self.params_oh[idx],
                                     self.params_mh[idx],
                                     self.params_mf[idx], self.scipy_dist,
                                     self.relative, self.sampling,
                                     self.log_base, self.delta_denom_min,
                                     self.delta_denom_zero, self.delta_range)
                    futures[fut] = idx
                for future in as_completed(futures):
                    idx = futures[future]
                    arr_bc[:, idx] = future.result()

        msg = ('Input shape {} does not match QDM bias corrected output '
               'shape {}!'.format(arr.shape, arr_bc.shape))
        assert arr.shape == arr_bc.shape, msg

        return arr_bc
