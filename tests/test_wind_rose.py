# -*- coding: utf-8 -*-
"""
pytests for WindRose
"""
from click.testing import CliRunner
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import traceback

from rex.multi_year_resource import MultiYearWindResource
from rex.renewable_resource import WindResource
from rex.joint_pd.wind_rose import WindRose
from rex.joint_pd.wind_rose_cli import main
from rex.utilities.loggers import LOGGERS
from rex import TESTDATADIR

PURGE_OUT = True

WIND_H5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
HUB_HEIGHT = 100
with WindResource(WIND_H5) as f:
    WSPD = f[f'windspeed_{HUB_HEIGHT}m']
    WDIR = f[f'winddirection_{HUB_HEIGHT}m']


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def wind_rose(wspd, wdir, site, wspd_bins, wdir_bins):
    """
    Compute wind rose for a single site
    """
    wspd_bins = WindRose._make_bins(*wspd_bins)
    wdir_bins = WindRose._make_bins(*wdir_bins)
    out = np.histogram2d(wspd[:, site], wdir[:, site],
                         bins=(wspd_bins, wdir_bins),
                         density=True)

    columns = pd.Index(wdir_bins[:-1], name='wdir')
    index = pd.Index(wspd_bins[:-1], name='wspd')
    out = pd.DataFrame(out[0], columns=columns, index=index)
    out = out.melt(ignore_index=False).reset_index()

    out = out.set_index(['wspd', 'wdir'])
    out.columns = [site]

    return out


@pytest.mark.parametrize("hub_height", [80, 90])
def test_hub_height(hub_height):
    """
    Test WindRose serial vs parallel
    """
    wspd_bins = (0, 30, 1)
    wdir_bins = (0, 360, 5)
    with WindResource(WIND_H5) as f:
        wspd = f[f'windspeed_{hub_height}m']
        wdir = f[f'winddirection_{hub_height}m']

    test = WindRose.run(WIND_H5, hub_height,
                        wspd_bins=wspd_bins,
                        wdir_bins=wdir_bins)
    site = np.random.choice(test.columns.values, 1)[0]
    truth = wind_rose(wspd, wdir, site, wspd_bins, wdir_bins)

    assert_frame_equal(test[[site]], truth, check_dtype=False)


@pytest.mark.parametrize("max_workers", [1, None])
def test_workers(max_workers):
    """
    Test WindRose serial vs parallel
    """
    wspd_bins = (0, 30, 1)
    wdir_bins = (0, 360, 5)
    test = WindRose.run(WIND_H5, HUB_HEIGHT,
                        wspd_bins=wspd_bins,
                        wdir_bins=wdir_bins,
                        max_workers=max_workers)
    site = np.random.choice(test.columns.values, 1)[0]
    truth = wind_rose(WSPD, WDIR, site, wspd_bins, wdir_bins)

    assert_frame_equal(test[[site]], truth, check_dtype=False)


@pytest.mark.parametrize("sites",
                         [slice(None), slice(None, None, 10), list(range(20)),
                          np.random.choice(range(100), 20, replace=False)])
def test_sites(sites):
    """
    Test WindRose with different sites
    """
    wspd_bins = (0, 30, 1)
    wdir_bins = (0, 360, 5)
    test = WindRose.run(WIND_H5, HUB_HEIGHT,
                        wspd_bins=wspd_bins,
                        wdir_bins=wdir_bins,
                        sites=sites)
    site = np.random.choice(test.columns.values, 1)[0]
    truth = wind_rose(WSPD, WDIR, site, wspd_bins, wdir_bins)

    assert_frame_equal(test[[site]], truth, check_dtype=False)


@pytest.mark.parametrize(("wspd", "wdir"),
                         [((0, 30, 5), (0, 360, 5)),
                          ((0, 25, 1), (0, 360, 10)),
                          ((0, 30, 2), (0, 360, 20))])
def test_bins(wspd, wdir):
    """
    Test WindRose with different sites
    """
    test = WindRose.run(WIND_H5, HUB_HEIGHT,
                        wspd_bins=wspd,
                        wdir_bins=wdir,)
    site = np.random.choice(test.columns.values, 1)[0]
    truth = wind_rose(WSPD, WDIR, site, wspd, wdir)

    assert_frame_equal(test[[site]], truth, check_dtype=False)


def test_cli(runner):
    """
    Test CLI
    """
    wind_h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')
    with MultiYearWindResource(wind_h5) as f:
        wspd = f[f'windspeed_{HUB_HEIGHT}m']
        wdir = f[f'winddirection_{HUB_HEIGHT}m']

    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(main, ['-h5', wind_h5,
                                      '-height', HUB_HEIGHT,
                                      '-res', 'MultiYear',
                                      '-o', td])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        name = os.path.splitext(os.path.basename(wind_h5))[0]
        name = name.replace('*', '')
        out_fpath = '{}_wind_rose-{}m.csv'.format(name, HUB_HEIGHT)
        test = pd.read_csv(os.path.join(td, out_fpath))
        test = test.set_index(['wspd', 'wdir'])
        test.columns = test.columns.astype(int)

        site = np.random.choice(test.columns.values, 1)[0]
        wspd_bins = (0, 30, 1)
        wdir_bins = (0, 360, 5)
        truth = wind_rose(wspd, wdir, site, wspd_bins, wdir_bins)

        assert_frame_equal(test[[site]], truth, check_dtype=False)

    LOGGERS.clear()


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
