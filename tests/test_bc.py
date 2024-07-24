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

    ntime = int(1e4)

    arr_oh = np.random.gamma(0.5, scale=1.0, size=ntime)
    arr_mh = np.random.gamma(0.75, scale=1.0, size=ntime)
    arr_mf = np.random.gamma(1, scale=1.0, size=ntime)

    params_oh = cdf(arr_oh, n_samples=100, sampling='invlog', log_base=10)
    params_mh = cdf(arr_mh, n_samples=100, sampling='invlog', log_base=10)
    params_mf = cdf(arr_mf, n_samples=100, sampling='invlog', log_base=10)
    params_oh = params_oh[np.newaxis]  # qdm expects (space, time) for params
    params_mh = params_mh[np.newaxis]  # qdm expects (space, time) for params
    params_mf = params_mf[np.newaxis]  # qdm expects (space, time) for params

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
    arr_mh_bc = qdm_abs_hist(arr_mh[:, np.newaxis])
    arr_mf_bc = qdm_abs_fut(arr_mf[:, np.newaxis])

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
    arr_mh_bc = qdm_rel_hist(arr_mh[:, np.newaxis])
    arr_mf_bc = qdm_rel_fut(arr_mf[:, np.newaxis])

    # check trend
    diff_raw = arr_mf.mean() / arr_mh.mean()
    diff_bc = arr_mf_bc.mean() / arr_mh_bc.mean()
    assert np.allclose(diff_raw, diff_bc, rtol=5e-2)


def test_qdm_parallel():
    """Test parallelization of QuantileDeltaMapping"""

    ntime = int(1e4)
    nspace = 1000
    size = (ntime, nspace)

    arr_oh = np.random.gamma(0.5, scale=1.0, size=size)
    arr_mh = np.random.gamma(0.75, scale=1.0, size=size)
    arr_mf = np.random.gamma(1, scale=1.0, size=size)

    params_oh = []
    params_mh = []
    params_mf = []
    for idx in range(nspace):
        params_oh.append(cdf(arr_oh[:, idx], n_samples=100)[np.newaxis])
        params_mh.append(cdf(arr_mh[:, idx], n_samples=100)[np.newaxis])
        params_mf.append(cdf(arr_mf[:, idx], n_samples=100)[np.newaxis])

    params_oh = np.concatenate(params_oh, axis=0)
    params_mh = np.concatenate(params_mh, axis=0)
    params_mf = np.concatenate(params_mf, axis=0)

    qdm_rel_fut = QuantileDeltaMapping(params_oh, params_mh, params_mf,
                                       dist='empirical', relative=True)

    arr_mf_bc_ser = qdm_rel_fut(arr_mf, max_workers=1)
    arr_mf_bc_par = qdm_rel_fut(arr_mf, max_workers=2)
    assert np.allclose(arr_mf_bc_ser, arr_mf_bc_par)
