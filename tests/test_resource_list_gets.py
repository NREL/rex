# -*- coding: utf-8 -*-
"""
pytests for resource handlers
"""
import h5py
import numpy as np
import os
import pytest

from rex import TESTDATADIR
from rex.renewable_resource import NSRDB


def get_baseline(path, dset, ds_slice):
    """
    Extract baseline data
    """
    with h5py.File(path, mode='r') as f:
        arr = f[dset][...]

    print('numpy get')
    print(arr.shape)
    print(ds_slice)
    print(arr[ds_slice].shape)
    return arr[ds_slice]


@pytest.mark.parametrize('ds_slice',
                         [(slice(None), list(range(5, 100, 20))),
                          (slice(None), [23, 24, 27, 21, 1, 2, 7, 5]),
                          (list(range(8)), [23, 24, 27, 21, 1, 2, 7, 5]),
                          ([230, 240, 270, 210, 1, 2, 7, 5],
                           [23, 24, 27, 21, 1, 2, 7, 5]),
                          (10, [23, 24, 27, 21, 1, 2, 7, 5]),
                          ([230, 240, 270, 210, 1, 2, 7, 5], slice(None)),
                          ([230, 240, 270, 210, 1, 2, 7, 5], list(range(8))),
                          ([230, 240, 270, 210, 1, 2, 7, 5], 10),
                          (10, 10),
                          (list(range(5)), list(range(5)))])
def test_2d_list_gets(ds_slice):
    """
    Test advanced list gets
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/nsrdb_wspd_chunked_2012.h5')
    dset = 'wind_speed'
    baseline = get_baseline(path, dset, ds_slice)

    with NSRDB(path, unscale=False) as f:
        dset_slice = (dset, ) + ds_slice
        test = f[dset_slice]

    assert np.allclose(baseline, test)


@pytest.mark.parametrize('ds_slice',
                         [(slice(None), list(range(5))),
                          (slice(None), [2, 3, 1, 4]),
                          (list(range(4)), [2, 3, 1, 4]),
                          (9, [2, 3, 1, 4]),
                          ([2, 3, 1, 4], 9),
                          (slice(None), [2, 3, 1, 4], 9),
                          (9, [2, 3, 1, 4], 9),
                          (9, [2, 3, 1, 4], 9, [2, 3, 1, 4]),
                          (9, [2, 3, 1, 4], slice(None), [2, 3, 1, 4]),
                          (9, [2, 3, 1, 4], slice(None), 7),
                          ([2, 3, 1, 4], [2, 3, 1, 4], [2, 3, 1, 4], 8),
                          ([2, 3, 1, 4], [2, 3, 1, 4], [2, 3, 1, 4],
                           [2, 3, 1, 4]),
                          ([2, 3, 1, 4], 8, [2, 3, 1, 4], 8),
                          (slice(None), slice(None), slice(None),
                           [2, 3, 1, 4]),
                          # These fail due to a numpy bug
                          # (8, slice(None), slice(None), [2, 3, 1, 4]),
                          # (slice(None), [2, 3, 1, 4], slice(None), 8),
                          ])
def test_4d_list_gets(ds_slice):
    """
    Test advanced list gets
    """
    path = os.path.join(TESTDATADIR, 'wave/test_virutal_buoy.h5')
    dset = 'directional_wave_spectrum'
    baseline = get_baseline(path, dset, ds_slice)
    print('ds_slice = ', ds_slice)
    with NSRDB(path, unscale=False) as f:
        dset_slice = (dset, ) + ds_slice
        test = f[dset_slice]

    print('baseline shape = ', baseline.shape)
    print('rex shape = ', test.shape)
    assert np.allclose(baseline, test)


def test_index_error():
    """
    test incompatible list IndexError
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/nsrdb_wspd_chunked_2012.h5')
    dset = 'wind_speed'
    with pytest.raises(IndexError):
        bad_slice = (list(range(5)), list(range(10)))
        with NSRDB(path, unscale=False) as f:
            dset_slice = (dset, ) + bad_slice
            f[dset_slice]  # pylint: disable=W0104


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
