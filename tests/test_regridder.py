# -*- coding: utf-8 -*-
"""
pytests for regridder
"""
import os

import pytest
import numpy as np

from rex import TESTDATADIR, Resource
from rex.utilities.regridder import Regridder


RANDOM_GENERATOR = np.random.default_rng(seed=42)
WTK_H5 = os.path.join(TESTDATADIR, "wtk", "ri_100_wtk_2012.h5")


def test_regridding_with_dask():
    """Make sure regridding reproduces original data when coordinates in the
    meta is the same"""

    with Resource(WTK_H5) as res:
        source_meta = res.meta.copy()
        source_meta['gid'] = np.arange(len(source_meta))
        shuffled_meta = source_meta.sample(frac=1, random_state=0)
        ws = res['windspeed_100m', ...]

        regridder = Regridder(source_meta=source_meta,
                              target_meta=shuffled_meta, max_workers=1)

        out = regridder(ws.T).T.compute()

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

        assert np.allclose(ws[:, new_shuffled_meta['gid'].values],
                           out.T.compute(), atol=0.1)


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
