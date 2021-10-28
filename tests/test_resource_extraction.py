# -*- coding: utf-8 -*-
"""
pytests for resource extractors
"""
from click.testing import CliRunner
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import tempfile
import traceback

from rex.resource_extraction.wind_cli import main
from rex.resource_extraction.resource_extraction import (MultiFileWindX,
                                                         MultiFileNSRDBX,
                                                         MultiTimeWindX,
                                                         MultiTimeNSRDBX,
                                                         MultiYearWindX,
                                                         MultiYearNSRDBX,
                                                         NSRDBX, WindX, WaveX)
from rex.resource_extraction.resource_extraction import TREE_DIR
from rex.utilities.exceptions import ResourceValueError
from rex.utilities.loggers import LOGGERS
from rex import TESTDATADIR


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


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
    Init MultiFileNSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb*2018.h5')
    return MultiFileNSRDBX(path)


@pytest.fixture
def MultiYearNSRDBX_cls():
    """
    Init MultiYearNSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_*.h5')
    return MultiYearNSRDBX(path)


@pytest.fixture
def MultiTimeNSRDBX_cls():
    """
    Init MulitTimeNSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_*.h5')
    return MultiTimeNSRDBX(path)


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


@pytest.fixture
def MultiYearWindX_cls():
    """
    Init MultiYearWindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')
    return MultiYearWindX(path)


@pytest.fixture
def MultiTimeWindX_cls():
    """
    Init MultiTimeWindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')
    return MultiTimeWindX(path)


def check_props(res_cls):
    """
    Test extraction class properties
    """
    time_index = res_cls.time_index
    meta = res_cls.meta
    res_shape = (len(time_index), len(meta))

    assert len(res_cls) == len(time_index)
    assert res_cls.shape == res_shape

    assert np.all(np.isin(['meta', 'time_index'],
                          res_cls.datasets))
    assert np.all(~np.isin(['meta', 'time_index', 'coordinates'],
                           res_cls.resource_datasets))

    assert np.all(np.in1d(res_cls.countries, meta['country'].unique()))
    assert np.all(np.in1d(res_cls.states, meta['state'].unique()))
    assert np.all(np.in1d(res_cls.counties, meta['county'].unique()))


def extract_site(res_cls, ds_name):
    """
    Run tests extracting a single site
    """
    time_index = res_cls.time_index
    meta = res_cls.meta
    site = np.random.choice(len(meta), 1)[0]
    lat_lon = meta.loc[site, ['latitude', 'longitude']].values

    truth_ts = res_cls[ds_name, :, site]
    truth_df = pd.DataFrame(truth_ts, columns=[site],
                            index=pd.Index(time_index, name='time_index'))

    site_ts = res_cls.get_lat_lon_ts(ds_name, lat_lon)
    assert np.allclose(truth_ts, site_ts)

    site_df = res_cls.get_lat_lon_df(ds_name, lat_lon)
    assert_frame_equal(site_df, truth_df, check_dtype=False)

    tree_file = res_cls._get_tree_file(res_cls.resource.h5_file)
    assert tree_file in os.listdir(TREE_DIR.name)


def extract_region(res_cls, ds_name, region, region_col='county'):
    """
    Run tests extracting all gids in a region
    """
    time_index = res_cls.time_index
    meta = res_cls.meta
    sites = meta.index[(meta[region_col] == region)].values
    truth_ts = res_cls[ds_name, :, sites]
    truth_df = pd.DataFrame(truth_ts, columns=sites,
                            index=pd.Index(time_index, name='time_index'))

    region_ts = res_cls.get_region_ts(ds_name, region, region_col=region_col)
    assert np.allclose(truth_ts, region_ts)

    region_df = res_cls.get_region_df(ds_name, region, region_col=region_col)
    assert_frame_equal(region_df, truth_df, check_dtype=False)


def extract_box(res_cls, ds_name):
    """
    Run tests extracting all gids in a bounding box
    """
    time_index = res_cls.time_index
    lat_lon = res_cls.lat_lon

    lat_lon_1 = (lat_lon[:, 0].min(), lat_lon[:, 1].min())
    lat_lon_2 = (lat_lon[:, 0].max(), lat_lon[:, 1].max())

    truth_ts = res_cls[ds_name]
    truth_df = pd.DataFrame(truth_ts, columns=list(range(len(lat_lon))),
                            index=pd.Index(time_index, name='time_index'))

    box_ts = res_cls.get_box_ts(ds_name, lat_lon_1, lat_lon_2)
    assert np.allclose(truth_ts, box_ts)

    box_df = res_cls.get_box_df(ds_name, lat_lon_1, lat_lon_2)
    assert_frame_equal(box_df, truth_df, check_dtype=False)


def extract_map(res_cls, ds_name, timestep, region=None, region_col='county'):
    """
    Run tests extracting a single timestep
    """
    time_index = res_cls.time_index
    meta = res_cls.meta
    lat_lon = res_cls.lat_lon
    idx = np.where(time_index == pd.to_datetime(timestep, utc=True))[0][0]
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

    @staticmethod
    def test_box(NSRDBX_cls, ds_name='dhi'):
        """
        test bounding box extraction
        """
        extract_box(NSRDBX_cls, ds_name)
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

    @staticmethod
    def test_box(MultiFileNSRDBX_cls, ds_name='dhi'):
        """
        test bounding box extraction
        """
        extract_box(MultiFileNSRDBX_cls, ds_name)
        MultiFileNSRDBX_cls.close()


class TestMultiYearNSRDBX:
    """
    MultiYearNSRDBX Resource Extractor
    """
    @staticmethod
    def test_props(MultiYearNSRDBX_cls):
        """
        test MultiYearNSRDBX properties
        """
        check_props(MultiYearNSRDBX_cls)
        MultiYearNSRDBX_cls.close()

    @staticmethod
    def test_site(MultiYearNSRDBX_cls, ds_name='dni'):
        """
        test site data extraction
        """
        extract_site(MultiYearNSRDBX_cls, ds_name)
        MultiYearNSRDBX_cls.close()

    @staticmethod
    def test_region(MultiYearNSRDBX_cls, ds_name='ghi', region='Washington',
                    region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiYearNSRDBX_cls, ds_name, region,
                       region_col=region_col)
        MultiYearNSRDBX_cls.close()

    @staticmethod
    def test_full_map(MultiYearNSRDBX_cls, ds_name='ghi',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiYearNSRDBX_cls, ds_name, timestep)
        MultiYearNSRDBX_cls.close()

    @staticmethod
    def test_region_map(MultiYearNSRDBX_cls, ds_name='dhi',
                        timestep='2012-12-25 12:00:00',
                        region='Washington', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiYearNSRDBX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiYearNSRDBX_cls.close()

    @staticmethod
    def test_box(MultiYearNSRDBX_cls, ds_name='dhi'):
        """
        test bounding box extraction
        """
        extract_box(MultiYearNSRDBX_cls, ds_name)
        MultiYearNSRDBX_cls.close()


class TestMultiTimeNSRDBX:
    """
    MultiTimeNSRDBX Resource Extractor
    """
    @staticmethod
    def test_props(MultiTimeNSRDBX_cls):
        """
        test MultiTimeNSRDBX properties
        """
        check_props(MultiTimeNSRDBX_cls)
        MultiTimeNSRDBX_cls.close()

    @staticmethod
    def test_site(MultiTimeNSRDBX_cls, ds_name='dni'):
        """
        test site data extraction
        """
        extract_site(MultiTimeNSRDBX_cls, ds_name)
        MultiTimeNSRDBX_cls.close()

    @staticmethod
    def test_region(MultiTimeNSRDBX_cls, ds_name='ghi', region='Washington',
                    region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiTimeNSRDBX_cls, ds_name,
                       region, region_col=region_col)
        MultiTimeNSRDBX_cls.close()

    @staticmethod
    def test_full_map(MultiTimeNSRDBX_cls, ds_name='ghi',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiTimeNSRDBX_cls, ds_name, timestep)
        MultiTimeNSRDBX_cls.close()

    @staticmethod
    def test_region_map(MultiTimeNSRDBX_cls, ds_name='dhi',
                        timestep='2012-12-25 12:00:00',
                        region='Washington', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiTimeNSRDBX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiTimeNSRDBX_cls.close()

    @staticmethod
    def test_box(MultiTimeNSRDBX_cls, ds_name='dhi'):
        """
        test bounding box extraction
        """
        extract_box(MultiTimeNSRDBX_cls, ds_name)
        MultiTimeNSRDBX_cls.close()


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

    @staticmethod
    def test_box(WindX_cls, ds_name='windspeed_100m'):
        """
        test bounding box extraction
        """
        extract_box(WindX_cls, ds_name)
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

    @staticmethod
    def test_box(MultiFileWindX_cls, ds_name='windspeed_100m'):
        """
        test bounding box extraction
        """
        extract_box(MultiFileWindX_cls, ds_name)
        MultiFileWindX_cls.close()


class TestMultiYearWindX:
    """
    MultiYearWindX Resource Extractor
    """
    @staticmethod
    def test_props(MultiYearWindX_cls):
        """
        test MultiYearWindX properties
        """
        check_props(MultiYearWindX_cls)
        MultiYearWindX_cls.close()

    @staticmethod
    def test_site(MultiYearWindX_cls, ds_name='windspeed_100m'):
        """
        test site data extraction
        """
        extract_site(MultiYearWindX_cls, ds_name)
        MultiYearWindX_cls.close()

    @staticmethod
    def test_region(MultiYearWindX_cls, ds_name='windspeed_50m',
                    region='Providence', region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiYearWindX_cls, ds_name, region,
                       region_col=region_col)
        MultiYearWindX_cls.close()

    @staticmethod
    def test_full_map(MultiYearWindX_cls, ds_name='windspeed_100m',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiYearWindX_cls, ds_name, timestep)
        MultiYearWindX_cls.close()

    @staticmethod
    def test_region_map(MultiYearWindX_cls, ds_name='windspeed_50m',
                        timestep='2012-12-25 12:00:00',
                        region='Providence', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiYearWindX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiYearWindX_cls.close()

    @staticmethod
    def test_box(MultiYearWindX_cls, ds_name='windspeed_100m'):
        """
        test bounding box extraction
        """
        extract_box(MultiYearWindX_cls, ds_name)
        MultiYearWindX_cls.close()


class TestMultiTimeWindX:
    """
    MultiTimeWindX Resource Extractor
    """
    @staticmethod
    def test_props(MultiTimeWindX_cls):
        """
        test MultiTimeWindX properties
        """
        check_props(MultiTimeWindX_cls)
        MultiTimeWindX_cls.close()

    @staticmethod
    def test_site(MultiTimeWindX_cls, ds_name='windspeed_100m'):
        """
        test site data extraction
        """
        extract_site(MultiTimeWindX_cls, ds_name)
        MultiTimeWindX_cls.close()

    @staticmethod
    def test_region(MultiTimeWindX_cls, ds_name='windspeed_50m',
                    region='Providence', region_col='county'):
        """
        test region data extraction
        """
        extract_region(MultiTimeWindX_cls, ds_name, region,
                       region_col=region_col)
        MultiTimeWindX_cls.close()

    @staticmethod
    def test_full_map(MultiTimeWindX_cls, ds_name='windspeed_100m',
                      timestep='2012-07-04 12:00:00'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiTimeWindX_cls, ds_name, timestep)
        MultiTimeWindX_cls.close()

    @staticmethod
    def test_region_map(MultiTimeWindX_cls, ds_name='windspeed_50m',
                        timestep='2012-12-25 12:00:00',
                        region='Providence', region_col='county'):
        """
        test map data extraction for all gids
        """
        extract_map(MultiTimeWindX_cls, ds_name, timestep, region=region,
                    region_col=region_col)
        MultiTimeWindX_cls.close()

    @staticmethod
    def test_box(MultiTimeWindX_cls, ds_name='windspeed_100m'):
        """
        test bounding box extraction
        """
        extract_box(MultiTimeWindX_cls, ds_name)
        MultiTimeWindX_cls.close()


@pytest.mark.parametrize('gid', [1, [1, 2, 4, 8]])
def test_WaveX(gid):
    """
    Test custom WaveX get methods for 4d 'directional_wave_spectrum' dataset
    """
    path = os.path.join(TESTDATADIR, 'wave/test_virutal_buoy.h5')
    ds_name = 'directional_wave_spectrum'
    with WaveX(path) as f:
        truth = f[ds_name, :, :, :, gid]
        index = pd.MultiIndex.from_product(
            [f.time_index, f['frequency'], f['direction']],
            names=['time_index', 'frequency', 'direction'])

        # pylint: disable=no-member
        test_ts = f.get_gid_ts(ds_name, gid)
        test_df = f.get_gid_df(ds_name, gid)

    assert np.allclose(truth, test_ts)
    index_len = np.product(truth.shape[:3])
    gids = len(gid) if isinstance(gid, list) else 1
    assert np.allclose(truth.reshape((index_len, gids)), test_df.values)
    assert all(test_df.index == index)


def test_check_lat_lon():
    """
    Test lat_lon check
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    with WindX(path) as f:
        lat_lon = f.lat_lon
        bad_lat_lon = [lat_lon[:, 0].max() + 1, lat_lon[:, 1].max() + 1]
        with pytest.raises(ResourceValueError):
            # pylint: disable=no-member
            f.lat_lon_gid(bad_lat_lon)


def test_check_dist():
    """
    Test lat lon distance check
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    with WindX(path) as f:
        lat_lon = f.lat_lon
        bad_lat_lon = [lat_lon[:, 0].min(), lat_lon[:, 1].min()]
        with pytest.raises(ResourceValueError):
            # pylint: disable=no-member
            f.lat_lon_gid(bad_lat_lon)


@pytest.mark.parametrize('datasets', [None, 'windspeed_100m',
                                      ['pressure_100m',
                                       'temperature_100m',
                                       'windspeed_100m']])
def test_save_region(WindX_cls, datasets):
    """
    test save_region to .h5
    """
    region = 'Providence'
    region_col = 'county'

    gids = WindX_cls.region_gids(region, region_col=region_col)
    meta = WindX_cls.meta.loc[gids].reset_index(drop=True)
    meta.index.name = 'gid'
    truth = {'meta': meta,
             'coordinates': WindX_cls.lat_lon[gids],
             'time_index': WindX_cls.time_index}

    if datasets is None:
        dsets = WindX_cls.resource_datasets
    else:
        dsets = datasets
        if isinstance(dsets, str):
            dsets = [dsets]

    for dset in dsets:
        truth[dset] = WindX_cls[dset, :, gids]

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, 'test.h5')
        WindX_cls.save_region(out_path, region, datasets=datasets,
                              region_col=region_col)

        if datasets is not None:
            assert 'meta' not in datasets

        with WindX(out_path) as f:
            for dset in f:
                test = f[dset]
                if dset == 'meta':
                    assert_frame_equal(truth[dset], test)
                elif dset == 'time_index':
                    truth[dset].equals(test)
                else:
                    assert np.allclose(truth[dset], test)

    WindX_cls.close()


def test_cli_site(runner, WindX_cls):
    """
    Test rex CLI site get
    """
    lat_lon = WindX_cls.lat_lon
    gid = np.random.choice(len(lat_lon), 1)[0]
    ds_name = 'windspeed_100m'

    truth = WindX_cls.get_gid_df(ds_name, gid)

    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(main, ['-h5', path,
                                      '-o', td,
                                      'dataset', '-d', ds_name,
                                      'site', '-gid', gid])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_path = "{}-{}.csv".format(ds_name, gid)
        site_df = pd.read_csv(os.path.join(td, out_path), index_col=0)
        assert np.allclose(site_df.values, truth.values)

    WindX_cls.close()
    LOGGERS.clear()


def test_cli_SAM(runner, WindX_cls):
    """
    Test rex CLI SAM get
    """
    lat_lon = WindX_cls.lat_lon
    gid = np.random.choice(len(lat_lon), 1)[0]

    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with tempfile.TemporaryDirectory() as td:
        truth_path = os.path.join(td, 'truth.csv')
        WindX_cls.get_SAM_gid(100, gid, out_path=truth_path)
        truth = pd.read_csv(truth_path, skiprows=1)

        result = runner.invoke(main, ['-h5', path,
                                      '-o', td,
                                      'sam-datasets', '-h', 100,
                                      '-gid', gid])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg
        print(os.listdir(td))

        test = pd.read_csv(os.path.join(td, 'SAM_{}.csv'.format(gid)),
                           skiprows=1)
        assert_frame_equal(truth, test)

    WindX_cls.close()
    LOGGERS.clear()


def test_cli_region(runner, WindX_cls):
    """
    Test rex CLI region get
    """
    ds_name = 'windspeed_50m'
    region = 'Providence'
    region_col = 'county'

    truth = WindX_cls.get_region_df(ds_name, region, region_col=region_col)

    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(main, ['-h5', path,
                                      '-o', td,
                                      'dataset', '-d', ds_name,
                                      'region', '-r', region,
                                      '-col', region_col])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_path = "{}-{}.csv".format(ds_name, region)
        site_df = pd.read_csv(os.path.join(td, out_path), index_col=0)
        assert np.allclose(site_df.values, truth.values)

    WindX_cls.close()
    LOGGERS.clear()


def test_cli_box(runner, WindX_cls):
    """
    Test rex CLI bounding box get
    """
    lat_lon = WindX_cls.lat_lon
    ds_name = 'windspeed_100m'
    lat_lon_1 = (lat_lon[:, 0].min(), lat_lon[:, 1].min())
    lat_lon_2 = (lat_lon[:, 0].max(), lat_lon[:, 1].max())

    truth = WindX_cls.get_box_df(ds_name, lat_lon_1, lat_lon_2)

    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with tempfile.TemporaryDirectory() as td:
        result = runner.invoke(main, ['-h5', path,
                                      '-o', td,
                                      'dataset', '-d', ds_name,
                                      'box', '-ll1', *lat_lon_1,
                                      '-ll2', *lat_lon_2])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        out_path = "{}-box.csv".format(ds_name)
        site_df = pd.read_csv(os.path.join(td, out_path), index_col=0)
        assert np.allclose(site_df.values, truth.values)

    WindX_cls.close()
    LOGGERS.clear()


def _check_raster_lat_lons(meta, raster_index, shape):
    """Check that the raster index results in a right-side-up 2D spatial
    raster"""
    lats = meta.loc[raster_index.flatten(), 'latitude'].values.reshape(shape)
    lons = meta.loc[raster_index.flatten(), 'longitude'].values.reshape(shape)

    for i in range(1, shape[0]):
        assert all(lats[i - 1, :] > lats[i, :])

    for j in range(1, shape[1]):
        assert all(lons[:, j] > lons[:, j - 1])


def _plot_raster(meta, raster_index, shape, gid_target, close,
                 vector_dx, vector_dy,
                 start_xy, point_x, point_y, end_xy):
    """Plot raster test outputs"""
    import matplotlib.pyplot as plt
    lats = meta.loc[raster_index.flatten(), 'latitude']
    lons = meta.loc[raster_index.flatten(), 'longitude']
    lats = lats.values.reshape(shape)
    lons = lons.values.reshape(shape)

    a = plt.imshow(lats)
    plt.colorbar(a, label='lats')
    plt.savefig('./lats.png')
    plt.close()

    a = plt.imshow(lons)
    plt.colorbar(a, label='lons')
    plt.savefig('./lons.png')
    plt.close()

    plt.plot([gid_target[1], gid_target[1] + vector_dx[1]],
             [gid_target[0], gid_target[0] + vector_dx[0]])
    plt.plot([gid_target[1], gid_target[1] + vector_dy[1]],
             [gid_target[0], gid_target[0] + vector_dy[0]])

    plt.scatter(meta.loc[close, 'longitude'],
                meta.loc[close, 'latitude'], marker='s')
    plt.scatter(meta.loc[raster_index.flatten(), 'longitude'],
                meta.loc[raster_index.flatten(), 'latitude'],
                s=50, marker='x')

    plt.scatter(meta.longitude, meta.latitude, s=10)
    plt.scatter(gid_target[1], gid_target[0], marker='x', s=20)
    plt.scatter(start_xy[1], start_xy[0], marker='x', s=20)
    plt.scatter(point_x[1], point_x[0], marker='x', s=20)
    plt.scatter(point_y[1], point_y[0], marker='x', s=20)
    plt.scatter(end_xy[1], end_xy[0], marker='x', s=20)

    plt.axis('equal')
    plt.legend(['X Vector', 'Y Vector', 'Closest', 'Raster Meta',
                'Full Meta', 'GID Target', 'Start XY', 'Point X',
                'Point Y', 'End XY'])
    plt.savefig('./test.png', dpi=300)


def test_get_raster_index(plot=False):
    """Test the get raster meta index functionality. Plotting is the best way
    to sanity check this."""
    res_fp = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')

    # use a custom meta df because NSRDB/WTK resource test files are too small
    fp = os.path.join(TESTDATADIR, 'wtk/hawaii_grid.csv')
    meta = pd.read_csv(fp)

    target = (16, -162)
    shape = (10, 5)

    with NSRDBX(res_fp) as ext:
        gid_target, vector_dx, vector_dy, close = \
            ext.get_grid_vectors(target, meta=meta)
        _, start_xy, point_x, point_y, end_xy = ext._get_raster_index(
            meta, gid_target, vector_dx, vector_dy, (4, 4))
        raster_index = ext.get_raster_index(target, shape, meta=meta)

    if plot:
        xrange = (-162.05, -161.85)
        yrange = (15.95, 16.2)
        mask = ((xrange[0] < meta['longitude'])
                & (xrange[1] > meta['longitude'])
                & (yrange[0] < meta['latitude'])
                & (yrange[1] > meta['latitude']))
        meta = meta[mask]
        _plot_raster(meta, raster_index, shape, gid_target, close,
                     vector_dx, vector_dy,
                     start_xy, point_x, point_y, end_xy)

    assert raster_index.shape == shape

    _check_raster_lat_lons(meta, raster_index, shape)


def test_get_raster_index_big(plot=False):
    """Test the retrieval of a very large raster that needs to be chunked"""
    res_fp = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')

    # use a custom meta df because NSRDB/WTK resource test files are too small
    fp = os.path.join(TESTDATADIR, 'wtk/hawaii_grid.csv')
    meta = pd.read_csv(fp)

    target = (16, -163)
    shape = (50, 500)

    with NSRDBX(res_fp) as ext:
        gid_target, vector_dx, vector_dy, close = \
            ext.get_grid_vectors(target, meta=meta)
        _, start_xy, point_x, point_y, end_xy = ext._get_raster_index(
            meta, gid_target, vector_dx, vector_dy, (4, 4))
        raster_index = ext.get_raster_index(target, shape, meta=meta,
                                            max_delta=10)
    if plot:
        _plot_raster(meta, raster_index, shape, gid_target, close,
                     vector_dx, vector_dy,
                     start_xy, point_x, point_y, end_xy)

    assert not (raster_index == 0).any()
    _check_raster_lat_lons(meta, raster_index, shape)


def test_get_raster_index_skewed(plot=False):
    """Test retrieval of raster index on skewed data in RI"""
    res_fp = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    target = (41.25, -71.8)
    shape = (5, 5)

    with WindX(res_fp) as ext:
        meta = ext.meta
        gid_target, vector_dx, vector_dy, close = \
            ext.get_grid_vectors(target)
        _, start_xy, point_x, point_y, end_xy = ext._get_raster_index(
            meta, gid_target, vector_dx, vector_dy, shape)
        raster_index = ext.get_raster_index(target, shape)

    if plot:
        xrange = (-71.85, -71.6)
        yrange = (41.1, 41.4)
        mask = ((xrange[0] < meta['longitude'])
                & (xrange[1] > meta['longitude'])
                & (yrange[0] < meta['latitude'])
                & (yrange[1] > meta['latitude']))
        meta = meta[mask]
        _plot_raster(meta, raster_index, shape, gid_target, close,
                     vector_dx, vector_dy,
                     start_xy, point_x, point_y, end_xy)

    _check_raster_lat_lons(meta, raster_index, shape)


def test_get_raster_nsrdb(plot=False):
    """Test the raster retrieval on the NSRDB meta data which is sorted
    differently than wtk."""
    res_fp = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')

    # use a custom meta df because NSRDB/WTK resource test files are too small
    fp = os.path.join(TESTDATADIR, 'nsrdb/ri_full_meta.csv')
    meta = pd.read_csv(fp)

    target = (41.45, -71.74)
    shape = (13, 7)
    shape = (9, 5)

    with NSRDBX(res_fp) as ext:
        gid_target, vector_dx, vector_dy, close = \
            ext.get_grid_vectors(target, meta=meta)
        _, start_xy, point_x, point_y, end_xy = ext._get_raster_index(
            meta, gid_target, vector_dx, vector_dy, (4, 4))
        raster_index = ext.get_raster_index(target, shape, meta=meta,
                                            max_delta=4)

    if plot:
        _plot_raster(meta, raster_index, shape, gid_target, close,
                     vector_dx, vector_dy,
                     start_xy, point_x, point_y, end_xy)

    assert not (raster_index == 0).any()
    _check_raster_lat_lons(meta, raster_index, shape)


def test_get_bad_raster_index():
    """Test the get raster meta index functionality with a bad target input"""
    res_fp = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')

    # use a custom meta df because NSRDB/WTK resource test files are too small
    fp = os.path.join(TESTDATADIR, 'wtk/hawaii_grid.csv')
    meta = pd.read_csv(fp)

    target = (-90, -162)
    shape = (10, 5)
    with pytest.raises(RuntimeError):
        with NSRDBX(res_fp) as ext:
            ext.get_raster_index(target, shape, meta=meta)

    target = (16, 0)
    shape = (10, 5)
    with pytest.raises(RuntimeError):
        with NSRDBX(res_fp) as ext:
            ext.get_raster_index(target, shape, meta=meta)


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
