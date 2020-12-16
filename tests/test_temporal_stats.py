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


@pytest.mark.parametrize(("max_workers", "sites"),
                         [(1, slice(None)),
                          (1, slice(None, None, 10)),
                          (1, list(range(10))),
                          (1, [7, 3, 5, 9]),
                          (None, slice(None)),
                          (None, slice(None, None, 10)),
                          (None, list(range(10))),
                          (None, [7, 3, 5, 9])])
def test_means(max_workers, sites):
    """
    Test TemporalStats means
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, sites=sites,
                                   statistics='mean',
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    if isinstance(sites, list):
        sites = sorted(sites)

    res_data = RES_DATA[:, sites]
    gids = np.arange(RES_DATA.shape[1], dtype=int)[sites]

    msg = ('gids do not match!')
    assert np.allclose(gids, test_stats.index.values), msg

    truth = np.nanmean(res_data, axis=0)
    msg = 'Annual means do not match!'
    assert np.allclose(truth, test_stats['mean'].values), msg

    mask = TIME_INDEX.month == 1
    truth = np.nanmean(res_data[mask], axis=0)
    msg = 'January means do not match!'
    assert np.allclose(truth, test_stats['Jan_mean'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.nanmean(res_data[mask], axis=0)
    msg = 'Midnight means do not match!'
    assert np.allclose(truth, test_stats['00_mean'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.nanmean(res_data[mask], axis=0)
    msg = 'January-midnight means do not match!'
    assert np.allclose(truth, test_stats['Jan-00_mean'].values), msg


@pytest.mark.parametrize(("max_workers", "sites"),
                         [(1, slice(None)),
                          (1, slice(None, None, 10)),
                          (1, list(range(10))),
                          (1, [7, 3, 5, 9]),
                          (None, slice(None)),
                          (None, slice(None, None, 10)),
                          (None, list(range(10))),
                          (None, [7, 3, 5, 9])])
def test_medians(max_workers, sites):
    """
    Test TemporalStats medians
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, sites=sites,
                                   statistics='median',
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    if isinstance(sites, list):
        sites = sorted(sites)

    res_data = RES_DATA[:, sites]
    gids = np.arange(RES_DATA.shape[1], dtype=int)[sites]

    msg = ('gids do not match!')
    assert np.allclose(gids, test_stats.index.values), msg

    truth = np.nanmedian(res_data, axis=0)
    msg = 'Annual medians do not match!'
    assert np.allclose(truth, test_stats['median'].values), msg

    mask = TIME_INDEX.month == 1
    truth = np.nanmedian(res_data[mask], axis=0)
    msg = 'January medians do not match!'
    assert np.allclose(truth, test_stats['Jan_median'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.nanmedian(res_data[mask], axis=0)
    msg = 'Midnight medians do not match!'
    assert np.allclose(truth, test_stats['00_median'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.nanmedian(res_data[mask], axis=0)
    msg = 'January-midnight medians do not match!'
    assert np.allclose(truth, test_stats['Jan-00_median'].values), msg


@pytest.mark.parametrize(("max_workers", "sites"),
                         [(1, slice(None)),
                          (1, slice(None, None, 10)),
                          (1, list(range(10))),
                          (1, [7, 3, 5, 9]),
                          (None, slice(None)),
                          (None, slice(None, None, 10)),
                          (None, list(range(10))),
                          (None, [7, 3, 5, 9])])
def test_stdevs(max_workers, sites):
    """
    Test TemporalStats stdevs
    """
    test_stats = TemporalStats.all(RES_H5, DATASET, sites=sites,
                                   statistics='std',
                                   res_cls=WindResource,
                                   max_workers=max_workers)
    if isinstance(sites, list):
        sites = sorted(sites)

    res_data = RES_DATA[:, sites]
    gids = np.arange(RES_DATA.shape[1], dtype=int)[sites]

    msg = ('gids do not match!')
    assert np.allclose(gids, test_stats.index.values), msg

    truth = np.nanstd(res_data, axis=0)
    msg = 'Annual stdevs do not match!'
    assert np.allclose(truth, test_stats['std'].values, rtol=0.0001), msg

    mask = TIME_INDEX.month == 1
    truth = np.nanstd(res_data[mask], axis=0)
    msg = 'January stdevs do not match!'
    assert np.allclose(truth, test_stats['Jan_std'].values), msg

    mask = TIME_INDEX.hour == 0
    truth = np.nanstd(res_data[mask], axis=0)
    msg = 'Midnight stdevs do not match!'
    assert np.allclose(truth, test_stats['00_std'].values), msg

    mask = (TIME_INDEX.month == 1) & (TIME_INDEX.hour == 0)
    truth = np.nanstd(res_data[mask], axis=0)
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
