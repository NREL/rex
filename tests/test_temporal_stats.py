# -*- coding: utf-8 -*-
"""
pytests for  Rechunk h5
"""
import numpy as np
import os
import pytest

from rex.renewable_resource import WindResource
from rex.resource_extraction.temporal_stats import TemporalStats
from rex import TESTDATADIR

PURGE_OUT = True

RES_H5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
DATASET = 'windspeed_100m'
with WindResource(RES_H5) as f:
    TIME_INDEX = f.time_index
    RES_DATA = f['windspeed_100m']


@pytest.mark.parametrize("max_workers", [1, None])
def test_means(max_workers):
    """
    Test TemporalStats means
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, statistics=('mean'),
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    truth = np.mean(RES_DATA, axis=0)
    msg = 'Annual means do not match!'
    assert np.allclose(truth, test_stats['mean'].values), msg

    mask = TIME_INDEX.month == 1
    truth = np.mean(RES_DATA[mask], axis=0)
    msg = 'January means do not match!'
    assert np.allclose(truth, test_stats['Jan_mean'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.mean(RES_DATA[mask], axis=0)
    msg = 'Midnight means do not match!'
    assert np.allclose(truth, test_stats['00_mean'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.mean(RES_DATA[mask], axis=0)
    msg = 'January-midnight means do not match!'
    assert np.allclose(truth, test_stats['Jan-00_mean'].values), msg


@pytest.mark.parametrize("max_workers", [1, None])
def test_medians(max_workers):
    """
    Test TemporalStats medians
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, statistics=('median'),
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    truth = np.median(RES_DATA, axis=0)
    msg = 'Annual medians do not match!'
    assert np.allclose(truth, test_stats['median'].values), msg

    mask = TIME_INDEX.month == 1
    truth = np.median(RES_DATA[mask], axis=0)
    msg = 'January medians do not match!'
    assert np.allclose(truth, test_stats['Jan_median'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.median(RES_DATA[mask], axis=0)
    msg = 'Midnight medians do not match!'
    assert np.allclose(truth, test_stats['00_median'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.median(RES_DATA[mask], axis=0)
    msg = 'January-midnight medians do not match!'
    assert np.allclose(truth, test_stats['Jan-00_median'].values), msg


@pytest.mark.parametrize("max_workers", [1, None])
def test_stdevs(max_workers):
    """
    Test TemporalStats stdevs
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, statistics=('std'),
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    truth = np.std(RES_DATA, axis=0, ddof=1)
    msg = 'Annual stdevs do not match!'
    assert np.allclose(truth, test_stats['std'].values), msg

    mask = TIME_INDEX.month == 1
    truth = np.std(RES_DATA[mask], axis=0, ddof=1)
    msg = 'January stdevs do not match!'
    assert np.allclose(truth, test_stats['Jan_std'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.std(RES_DATA[mask], axis=0, ddof=1)
    msg = 'Midnight stdevs do not match!'
    assert np.allclose(truth, test_stats['00_std'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.std(RES_DATA[mask], axis=0, ddof=1)
    msg = 'January-midnight stdevs do not match!'
    assert np.allclose(truth, test_stats['Jan-00_std'].values), msg


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
