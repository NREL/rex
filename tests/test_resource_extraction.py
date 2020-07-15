# -*- coding: utf-8 -*-
"""
pytests for resource extractors
"""
import numpy as np
import os
import pandas as pd
import pytest

from rex.resource_extraction import (MultiFileWindX, MultiFileNSRDBX,
                                     NSRDBX, WindX)
from rex.resource_extraction.resource_extraction import TREE_DIR
from rex import TESTDATADIR


@pytest.fixture
def NSRDBX_cls():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
    return NSRDBX(path)


@pytest.fixture
def MultiFileNSRDBX_cls():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb*2018.h5')
    return MultiFileNSRDBX(path)


@pytest.fixture
def WindX_cls():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    return WindX(path)


@pytest.fixture
def MultiFileWindX_cls():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk', 'wtk*m.h5')
    return MultiFileWindX(path)


def check_props(res_cls):
    """
    Test extraction class properties
    """
    meta = res_cls['meta']

    assert np.all(np.in1d(res_cls.countries, meta['country'].unique()))
    assert np.all(np.in1d(res_cls.states, meta['state'].unique()))
    assert np.all(np.in1d(res_cls.counties, meta['county'].unique()))


def extract_site(res_cls, ds_name):
    """
    Run tests extracting a single site
    """
    time_index = res_cls['time_index']
    meta = res_cls['meta']
    print(len(meta))
    site = np.random.choice(len(meta), 1)[0]
    lat_lon = meta.loc[site, ['latitude', 'longitude']].values
    truth_ts = res_cls[ds_name, :, site]
    truth_df = pd.DataFrame({ds_name: truth_ts}, index=time_index)

    site_ts = res_cls.get_lat_lon_ts(ds_name, lat_lon)
    assert np.allclose(truth_ts, site_ts)

    site_df = res_cls.get_lat_lon_df(ds_name, lat_lon)
    assert site_df.equals(truth_df)

    tree_file = res_cls._get_tree_file(res_cls.h5_file)
    assert tree_file in os.listdir(TREE_DIR.name)


def extract_region(res_cls, ds_name, region, region_col='county'):
    """
    Run tests extracting all gids in a region
    """
    time_index = res_cls['time_index']
    meta = res_cls['meta']
    sites = meta.index[(meta[region_col] == region)].values
    truth_ts = res_cls[ds_name, :, sites]
    truth_df = pd.DataFrame(truth_ts, columns=sites, index=time_index)

    lat_lon = meta.loc[sites, ['latitude', 'longitude']].values
    region_ts = res_cls.get_lat_lon_ts(ds_name, lat_lon)
    assert np.allclose(truth_ts, region_ts)

    region_df = res_cls.get_lat_lon_df(ds_name, lat_lon)
    assert region_df.equals(truth_df)

    region_ts = res_cls.get_region_ts(ds_name, region, region_col=region_col)
    assert np.allclose(truth_ts, region_ts)

    region_df = res_cls.get_region_df(ds_name, region, region_col=region_col)
    assert region_df.equals(truth_df)


def extract_map(res_cls, ds_name, timestep, region=None, region_col='county'):
    """
    Run tests extracting a single timestep
    """
    time_index = res_cls['time_index']
    meta = res_cls['meta']
    lat_lon = meta[['latitude', 'longitude']].values
    idx = np.where(time_index == pd.to_datetime(timestep))[0][0]
    gids = slice(None)
    if region is not None:
        gids = meta.index[(meta[region_col] == region)].values
        lat_lon = lat_lon[gids]

    truth = res_cls[ds_name, idx, gids]
    truth = pd.DataFrame({'longitude': lat_lon[:, 1],
                          'latitude': lat_lon[:, 0],
                          ds_name: truth})

    ts_map = res_cls.get_timestep_map(ds_name, timestep, region=region,
                                      region_col=region_col)
    assert ts_map.equals(truth)


class TestNSRDBX:
    """
    NSRDBX Resource Extractor
    """
    @staticmethod
    def test_props(NSRDBX_cls):
        """
        test NSRDBX properties
        """
        check_props(NSRDBX_cls)
        NSRDBX_cls.close()

    @staticmethod
    def test_site(NSRDBX_cls, ds_name='dni'):
        """
        test site data extraction
        """
        extract_site(NSRDBX_cls, ds_name)
        NSRDBX_cls.close()

    @staticmethod
    def test_region(NSRDBX_cls, ds_name='ghi', region='Washington',
                    region_col='county'):
        """
        test region data extraction
        """
        extract_region(NSRDBX_cls, ds_name, region, region_col=region_col)
        NSRDBX_cls.close()

    @staticmethod
    def test_full_map(NSRDBX_cls, ds_name='ghi',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(NSRDBX_cls, ds_name, timestep)
        NSRDBX_cls.close()

    @staticmethod
    def test_region_map(NSRDBX_cls, ds_name='dhi',
                        timestep='2012-12-25 12:00:00',
                        region='Washington', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(NSRDBX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        NSRDBX_cls.close()


class TestMultiFileNSRDBX:
    """
    MultiFileNSRDBX Resource Extractor
    """
    @staticmethod
    def test_props(MultiFileNSRDBX_cls):
        """
        test MultiFileNSRDBX properties
        """
        check_props(MultiFileNSRDBX_cls)
        MultiFileNSRDBX_cls.close()

    @staticmethod
    def test_site(MultiFileNSRDBX_cls, ds_name='dni'):
        """
        test site data extraction
        """
        extract_site(MultiFileNSRDBX_cls, ds_name)
        MultiFileNSRDBX_cls.close()

    @staticmethod
    def test_region(MultiFileNSRDBX_cls, ds_name='ghi', region='Clallam',
                    region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiFileNSRDBX_cls, ds_name, region,
                       region_col=region_col)
        MultiFileNSRDBX_cls.close()

    @staticmethod
    def test_full_map(MultiFileNSRDBX_cls, ds_name='ghi',
                      timestep='2018-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiFileNSRDBX_cls, ds_name, timestep)
        MultiFileNSRDBX_cls.close()

    @staticmethod
    def test_region_map(MultiFileNSRDBX_cls, ds_name='dhi',
                        timestep='2018-12-25 12:00:00',
                        region='Clallam', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiFileNSRDBX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiFileNSRDBX_cls.close()


class TestWindX:
    """
    WindX Resource Extractor
    """
    @staticmethod
    def test_props(WindX_cls):
        """
        test WindX properties
        """
        check_props(WindX_cls)
        WindX_cls.close()

    @staticmethod
    def test_site(WindX_cls, ds_name='windspeed_100m'):
        """
        test site data extraction
        """
        extract_site(WindX_cls, ds_name)
        WindX_cls.close()

    @staticmethod
    def test_region(WindX_cls, ds_name='windspeed_50m', region='Providence',
                    region_col='county'):
        """
        test region data extraction
        """
        extract_region(WindX_cls, ds_name, region, region_col=region_col)
        WindX_cls.close()

    @staticmethod
    def test_full_map(WindX_cls, ds_name='windspeed_100m',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(WindX_cls, ds_name, timestep)
        WindX_cls.close()

    @staticmethod
    def test_region_map(WindX_cls, ds_name='windspeed_50m',
                        timestep='2012-12-25 12:00:00',
                        region='Providence', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(WindX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        WindX_cls.close()


class TestMultiFileWindX:
    """
    MultiFileWindX Resource Extractor
    """
    @staticmethod
    def test_props(MultiFileWindX_cls):
        """
        test MultiFileWindX properties
        """
        check_props(MultiFileWindX_cls)
        MultiFileWindX_cls.close()

    @staticmethod
    def test_site(MultiFileWindX_cls, ds_name='windspeed_100m'):
        """
        test site data extraction
        """
        extract_site(MultiFileWindX_cls, ds_name)
        MultiFileWindX_cls.close()

    @staticmethod
    def test_region(MultiFileWindX_cls, ds_name='windspeed_50m',
                    region='Klamath', region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiFileWindX_cls, ds_name, region,
                       region_col=region_col)
        MultiFileWindX_cls.close()

    @staticmethod
    def test_full_map(MultiFileWindX_cls, ds_name='windspeed_100m',
                      timestep='2010-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiFileWindX_cls, ds_name, timestep)
        MultiFileWindX_cls.close()

    @staticmethod
    def test_region_map(MultiFileWindX_cls, ds_name='windspeed_50m',
                        timestep='2010-12-25 12:00:00',
                        region='Klamath', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiFileWindX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiFileWindX_cls.close()


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
