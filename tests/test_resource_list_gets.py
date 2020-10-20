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

PATH = os.path.join(TESTDATADIR, 'nsrdb/nsrdb_wspd_chunked_2012.h5')
DSET = 'wind_speed'


def get_baseline(ds_slice):
    """
    Extract baseline data
    """
    with h5py.File(PATH, mode='r') as f:
        arr = f[DSET][...]

    return arr[ds_slice]


@pytest.mark.parametrize('ds_slice',
                         [(slice(None), list(range(5, 100, 20))),
                          (slice(None), [23, 24, 27, 21, 1, 2, 7, 5]),
                          ([23, 24, 27, 21, 1, 2, 7, 5], slice(None)),
                          (list(range(5)), list(range(5)))])
def test_list_gets(ds_slice):
    """
    Test advanced list gets
    """
    baseline = get_baseline(ds_slice)

    with NSRDB(PATH, unscale=False) as f:
        dset_slice = (DSET, ) + ds_slice
        test = f[dset_slice]

    assert np.allclose(baseline, test)


def test_index_error():
    """
    test incompatible list IndexError
    """
    with pytest.raises(IndexError):
        bad_slice = (list(range(5)), list(range(10)))
        with NSRDB(PATH, unscale=False) as f:
            dset_slice = (DSET, ) + bad_slice
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
