# -*- coding: utf-8 -*-
"""
pytests for regridder
"""
import os
import tempfile

import pytest
import numpy as np
import pandas as pd
import dask.array as da

from rex import TESTDATADIR, Resource
from rex.utilities.regridder import Regridder, CachedRegridder


RANDOM_GENERATOR = np.random.default_rng(seed=42)
WTK_H5 = os.path.join(TESTDATADIR, "wtk", "ri_100_wtk_2012.h5")


@pytest.mark.parametrize(
    "params",
    [{"target": (0.5, 0.5), "k": 4, "expected": (0 + 1 + 4 + 5) / 4},
     {"target": (2.5, 0.5), "k": 4, "expected": (2 + 3 + 6 + 7) / 4},
     {"target": (0.5, 2.5), "k": 4, "expected": (8 + 9 + 12 + 13) / 4},
     {"target": (2.5, 2.5), "k": 4, "expected": (10 + 11 + 14 + 15) / 4},
     {"target": (1.5, 1.5), "k": 4, "expected": (5 + 6 + 9 + 10) / 4},
     {"target": (0.8, 0), "k": 1, "expected": 1},
     {"target": (0, 0.9), "k": 1, "expected": 4},
     {"target": (1, 0.9), "k": 2, "expected": (1 / 9 * 1 + 5) / (1 / 9 + 1)}])
def test_regridding_basic(params):
    """Run basic tests through regridder"""
    X, Y  = np.meshgrid(np.arange(4), np.arange(4))
    source = pd.DataFrame({"latitude": X.flatten(), "longitude": Y.flatten()})

    vals = da.from_array(np.arange(16).reshape(4, 4))
    # vals:
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]

    x, y = params["target"]
    target = pd.DataFrame({"latitude": [x], "longitude": [y]})
    regridder = Regridder(source, target, k_neighbors=params["k"])
    out = regridder(vals.flatten()[:, None]).compute()
    assert np.allclose(out, params["expected"], atol=0.0001)


@pytest.mark.parametrize(
    "params",
    [{"target": (0.5, 0.5), "k": 4, "expected": (0 + 1 + 4 + 5) / 4},
     {"target": (2.5, 0.5), "k": 4, "expected": (2 + 3 + 6 + 7) / 4},
     {"target": (0.5, 2.5), "k": 4, "expected": (8 + 9 + 12 + 13) / 4},
     {"target": (2.5, 2.5), "k": 4, "expected": (10 + 11 + 14 + 15) / 4},
     {"target": (1.5, 1.5), "k": 4, "expected": (5 + 6 + 9 + 10) / 4},
     {"target": (0.8, 0), "k": 1, "expected": 1},
     {"target": (0, 0.9), "k": 1, "expected": 4},
     {"target": (1, 0.9), "k": 2, "expected": (1 / 9 * 1 + 5) / (1 / 9 + 1)}])
def test_regridding_cached(params):
    """Run basic tests through cached regridder"""
    X, Y  = np.meshgrid(np.arange(4), np.arange(4))
    source = pd.DataFrame({"latitude": X.flatten(), "longitude": Y.flatten()})

    vals = np.arange(16).reshape(4, 4)
    # vals:
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]

    x, y = params["target"]
    target = pd.DataFrame({"latitude": [x], "longitude": [y]})

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, "{array_name}.pkl")
        CachedRegridder.build_cache(cache_pattern, source, target,
                                    k_neighbors=params["k"])

        regridder = CachedRegridder(cache_pattern)
        out = regridder(vals.flatten()[:, None])
        assert np.allclose(out, params["expected"], atol=0.0001)


def test_regridding_with_dask():
    """Make sure regridding reproduces original data when coordinates in the
    meta is the same"""

    with Resource(WTK_H5) as res:
        source_meta = res.meta.copy()
        source_meta['gid'] = np.arange(len(source_meta))
        shuffled_meta = source_meta.sample(frac=1, random_state=0)
        ws = res['windspeed_100m']

        regridder = Regridder(source_meta=source_meta,
                              target_meta=shuffled_meta, max_workers=1)
        out = regridder(ws.T).T
        assert np.array_equal(ws[:, shuffled_meta['gid'].values], out)

        new_shuffled_meta = shuffled_meta.copy()
        rand = RANDOM_GENERATOR.uniform(0, 1e-12,
                                        size=(2 * len(shuffled_meta)))
        rand = rand.reshape((len(shuffled_meta), 2))
        new_shuffled_meta['latitude'] += rand[:, 0]
        new_shuffled_meta['longitude'] += rand[:, 1]

        out = Regridder.run(source_meta=source_meta,
                            target_meta=new_shuffled_meta,
                            source_data=ws.T,
                            max_workers=1, min_distance=0)

        assert np.allclose(ws[:, new_shuffled_meta['gid'].values], out.T,
                           atol=0.1)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
