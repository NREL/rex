# -*- coding: utf-8 -*-
"""
pytests for bias correction utilities
"""

import numpy as np
from flaky import flaky

from rex.temporal_stats.temporal_stats import cdf
from rex.utilities.bc_utils import (QuantileDeltaMapping, sample_q_invlog,
                                    sample_cdf)


@flaky(max_runs=3, min_passes=1)
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


def test_difficult_qdm():
    """Relative QDM can blow up with extremely large deltas resulting from very
    small values in the denominator. This test verifies the protection against
    this using the ``delta_denom_min`` feature. The distributions are real
    distributions from Sup3rCC EC-Earth3 SSP585 wind data"""

    params_mf = np.array([0.04, 1.51, 2.14, 2.64, 3.06, 3.44, 3.78,
                          4.11, 4.43, 4.74, 5.03, 5.32, 5.6, 5.88, 6.15,
                          6.42, 6.69, 6.95, 7.22, 7.48, 7.73, 7.99, 8.23,
                          8.48, 8.72, 8.95, 9.2, 9.44, 9.67, 9.91, 10.15,
                          10.39, 10.63, 10.88, 11.11, 11.35, 11.59, 11.84,
                          12.1, 12.36, 12.63, 12.92, 13.23, 13.58, 13.96,
                          14.42, 14.99, 15.75, 17.03, 25.48])

    params_mh = np.array([0., 1.53, 2.19, 2.71, 3.15, 3.53, 3.89, 4.22, 4.53,
                          4.83, 5.13, 5.42, 5.71, 5.98, 6.25, 6.53, 6.8, 7.06,
                          7.3, 7.54, 7.79, 8.03, 8.27, 8.5, 8.74, 8.98, 9.21,
                          9.46, 9.7, 9.94, 10.18, 10.41, 10.61, 10.87, 11.11,
                          11.34, 11.56, 11.81, 12.06, 12.33, 12.61, 12.88,
                          13.16, 13.49, 13.83, 14.23, 14.72, 15.41, 16.71,
                          23.42])

    params_oh = np.array([2.000e-02, 1.860e+00, 2.550e+00, 3.080e+00,
                          3.530e+00, 3.950e+00, 4.330e+00, 4.690e+00,
                          5.040e+00, 5.360e+00, 5.660e+00, 5.950e+00,
                          6.220e+00, 6.490e+00, 6.740e+00, 6.990e+00,
                          7.230e+00, 7.470e+00, 7.690e+00, 7.900e+00,
                          8.120e+00, 8.340e+00, 8.550e+00, 8.770e+00,
                          8.970e+00, 9.180e+00, 9.380e+00, 9.570e+00,
                          9.770e+00, 9.960e+00, 1.016e+01, 1.035e+01,
                          1.055e+01, 1.076e+01, 1.097e+01, 1.118e+01,
                          1.140e+01, 1.163e+01, 1.188e+01, 1.211e+01,
                          1.236e+01, 1.267e+01, 1.297e+01, 1.334e+01,
                          1.377e+01, 1.427e+01, 1.479e+01, 1.547e+01,
                          1.669e+01, 2.191e+01])

    params_oh = params_oh[np.newaxis]  # qdm expects (space, time) for params
    params_mh = params_mh[np.newaxis]  # qdm expects (space, time) for params
    params_mf = params_mf[np.newaxis]  # qdm expects (space, time) for params

    quantiles = sample_q_invlog(params_mf.shape[1], log_base=10)
    arr = sample_cdf(quantiles, params_mf[0], int(1e4))
    arr[0] = 0.04  # trouble!

    # test bad result with large delta
    qdm_rel_fut = QuantileDeltaMapping(params_oh, params_mh, params_mf,
                                       dist='empirical', relative=True,
                                       sampling='invlog', log_base=10)
    arr_bc = qdm_rel_fut(arr[:, np.newaxis])
    assert arr_bc.max() > 1000  # bad result

    # test that delta_denom_min fixes this
    qdm_rel_fut = QuantileDeltaMapping(params_oh, params_mh, params_mf,
                                       dist='empirical', relative=True,
                                       sampling='invlog', log_base=10,
                                       delta_denom_min=0.01)
    arr_bc = qdm_rel_fut(arr[:, np.newaxis])
    assert arr_bc.max() < 40  # fixed result
