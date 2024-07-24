# -*- coding: utf-8 -*-
"""
pytests for bias correction utilities
"""

import numpy as np

from rex.temporal_stats.temporal_stats import cdf
from rex.utilities.bc_utils import QuantileDeltaMapping


def test_qdm():
    """Test basic QuantileDeltaMapping functionality with dummy distributions

    Note that because we use a sampled empirical distribution the bias
    corrected data will not be perfectly precise, hence the rtol values
    """

    ndata = int(1e4)

    arr_oh = np.random.gamma(0.5, scale=1.0, size=ndata)
    arr_mh = np.random.gamma(0.75, scale=1.0, size=ndata)
    arr_mf = np.random.gamma(1, scale=1.0, size=ndata)

    params_oh = cdf(arr_oh, n_samples=100, sampling='invlog', log_base=10)
    params_mh = cdf(arr_mh, n_samples=100, sampling='invlog', log_base=10)
    params_mf = cdf(arr_mf, n_samples=100, sampling='invlog', log_base=10)

    qdm_abs_hist = QuantileDeltaMapping(params_oh, params_mh, params_mf=None,
                                        dist='empirical', relative=False,
                                        sampling='invlog', log_base=10)
    qdm_rel_hist = QuantileDeltaMapping(params_oh, params_mh, params_mf=None,
                                        dist='empirical', relative=True,
                                        sampling='invlog', log_base=10)
    qdm_abs_fut = QuantileDeltaMapping(params_oh, params_mh, params_mf,
                                       dist='empirical', relative=False,
                                       sampling='invlog', log_base=10)
    qdm_rel_fut = QuantileDeltaMapping(params_oh, params_mh, params_mf,
                                       dist='empirical', relative=True,
                                       sampling='invlog', log_base=10)

    # absolute changes
    arr_mh_bc = qdm_abs_hist(arr_mh)
    arr_mf_bc = qdm_abs_fut(arr_mf)

    # check mean/max values for historical only (simple quantile mapping)
    assert not np.allclose(arr_mh.mean(), arr_oh.mean(), rtol=1e-2)
    assert not np.allclose(arr_mh.max(), arr_oh.max(), rtol=1e-2)
    assert np.allclose(arr_mh_bc.mean(), arr_oh.mean(), rtol=1e-2)
    assert np.allclose(arr_mh_bc.max(), arr_oh.max(), rtol=1e-2)

    # check trend
    diff_raw = arr_mf.mean() - arr_mh.mean()
    diff_bc = arr_mf_bc.mean() - arr_mh_bc.mean()
    assert np.allclose(diff_raw, diff_bc, rtol=1e-2)

    # relative changes
    arr_mh_bc = qdm_rel_hist(arr_mh)
    arr_mf_bc = qdm_rel_fut(arr_mf)

    # check trend
    diff_raw = arr_mf.mean() / arr_mh.mean()
    diff_bc = arr_mf_bc.mean() / arr_mh_bc.mean()
    assert np.allclose(diff_raw, diff_bc, rtol=5e-2)
