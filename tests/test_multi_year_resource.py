# -*- coding: utf-8 -*-
# pylint: disable=all
"""
pytests for multi year resource handlers
"""
import numpy as np
import os
from pandas.testing import assert_frame_equal
import pytest

from rex import TESTDATADIR
from rex.multi_file_resource import MultiH5Path
from rex.multi_year_resource import (MultiYearH5, MultiYearNSRDB,
                                     MultiYearWindResource)
from rex.resource import Resource


@pytest.fixture
def MultiYearNSRDB_res():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_*.h5')

    return MultiYearNSRDB(path)


@pytest.fixture
def MultiYearWind_res():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')

    return MultiYearWindResource(path)


def check_res(res_cls):
    """
    Run test on len and shape methods
    """
    time_index = None
    for file in res_cls.h5_files:
        with Resource(file) as f:
            if time_index is None:
                time_index = f.time_index
            else:
                time_index = time_index.append(f.time_index)

    with Resource(res_cls.h5_files[0]) as f:
        meta = f.meta

    res_shape = (len(time_index), len(meta))

    assert len(res_cls) == len(time_index)
    assert res_cls.shape == res_shape

    assert np.all(np.isin(['meta', 'time_index'],
                          res_cls.datasets))
    assert np.all(~np.isin(['meta', 'time_index', 'coordinates'],
                           res_cls.resource_datasets))


def check_attrs(res_cls, dset):
    """
    Check dataset attributes extraction
    """
    truth = res_cls.get_attrs(dset=dset)
    test = res_cls.attrs[dset]

    msg = "{} attributes do not match!".format(dset)
    assert truth == test, msg

    truth = res_cls.get_scale_factor(dset)
    test = res_cls.scale_factors[dset]

    msg = "{} scale factors do not match!".format(dset)
    assert truth == test, msg

    truth = res_cls.get_units(dset)
    test = res_cls.units[dset]

    msg = "{} units do not match!".format(dset)
    assert truth == test, msg


def check_properties(res_cls, dset):
    """
    Check dataset properties extraction
    """
    shape, dtype, chunks = res_cls.get_dset_properties(dset)

    test = res_cls.shapes[dset]
    msg = "{} shape does not match!".format(dset)
    assert shape == test, msg

    test = res_cls.dtypes[dset]
    msg = "{} dtype does not match!".format(dset)
    assert dtype == test, msg

    test = res_cls.chunks[dset]
    msg = "{} chunks do not match!".format(dset)
    assert chunks == test, msg


def check_meta(res_cls):
    """
    Run tests on meta data
    """
    with Resource(res_cls.h5_files[0]) as f:
        truth = f.meta

    test = res_cls['meta']
    assert_frame_equal(truth, test, check_dtype=False)

    test = res_cls.lat_lon
    assert np.allclose(truth[['latitude', 'longitude']].values, test)


def check_time_index(res_cls):
    """
    Run tests on time_index
    """
    truth = None
    for file in res_cls.h5_files:
        with Resource(file) as f:
            if truth is None:
                truth = f.time_index
            else:
                truth = truth.append(f.time_index)

    test = res_cls.time_index

    assert np.all(test == truth)


def check_dset(res_cls, ds_name):
    """
    Run tests on dataset ds_name
    """
    truth = []
    for file in res_cls.h5_files:
        truth.append(res_cls.h5._h5_map[file][ds_name])

    truth = np.concatenate(truth, axis=0)

    test = res_cls[ds_name]
    assert np.allclose(truth, test)

    test = res_cls[ds_name, :, 10]
    assert np.allclose(truth[:, 10], test)

    test = res_cls[ds_name, :, 10:20]
    assert np.allclose(truth[:, 10:20], test)

    test = res_cls[ds_name, :, [1, 3, 5, 7]]
    assert np.allclose(truth[:, [1, 3, 5, 7]], test)

    test = res_cls[ds_name, :, [2, 6, 3, 20]]
    assert np.allclose(truth[:, [2, 6, 3, 20]], test)


def check_years(res_cls, ds_name):
    """
    Run tests on dataset ds_name
    """
    for years in ['2012', ['2012', '2013'], ['2013', '2012']]:
        test = res_cls[ds_name, years]
        if not isinstance(years, list):
            years = [years]

        truth = []
        for year in years:
            truth.append(res_cls.h5[year][ds_name])

        truth = np.concatenate(truth, axis=0)
        assert np.allclose(truth, test)


@pytest.mark.parametrize('years',
                         [['2012'], [2013], [2012, 2013],
                          ['2013', '2012']])
def test_years_kwarg(years):
    """
    Test years kwarg
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')

    with MultiYearWindResource(path, years=years) as res_cls:
        check_res(res_cls)
        check_meta(res_cls)
        check_time_index(res_cls)
        check_dset(res_cls, 'windspeed_90m')


def test_years_error():
    """
    Test years RuntimeError when years don't exist
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')
    years = [2014, 2015]
    with pytest.raises(RuntimeError):
        MultiYearWindResource(path, years=years)


class TestMultiYearNSRDB:
    """
    Multi Year NSRDB Resource handler tests
    """
    @staticmethod
    def test_res(MultiYearNSRDB_res):
        """
        test NSRDB class calls
        """
        check_res(MultiYearNSRDB_res)
        MultiYearNSRDB_res.close()

    @staticmethod
    def test_meta(MultiYearNSRDB_res):
        """
        test extraction of NSRDB meta data
        """
        check_meta(MultiYearNSRDB_res)
        MultiYearNSRDB_res.close()

    @staticmethod
    def test_time_index(MultiYearNSRDB_res):
        """
        test extraction of NSRDB time_index
        """
        check_time_index(MultiYearNSRDB_res)
        MultiYearNSRDB_res.close()

    @staticmethod
    def test_ds(MultiYearNSRDB_res, ds_name='dni'):
        """
        test extraction of a variable array, attributes, and properties
        """
        check_dset(MultiYearNSRDB_res, ds_name)
        check_attrs(MultiYearNSRDB_res, ds_name)
        check_properties(MultiYearNSRDB_res, ds_name)
        check_years(MultiYearNSRDB_res, ds_name)
        MultiYearNSRDB_res.close()


class TestMultiYearWindResource:
    """
    Multi Year WindResource Resource handler tests
    """
    @staticmethod
    def test_res(MultiYearWind_res):
        """
        test WindResource class calls
        """
        check_res(MultiYearWind_res)
        MultiYearWind_res.close()

    @staticmethod
    def test_meta(MultiYearWind_res):
        """
        test extraction of WindResource meta data
        """
        check_meta(MultiYearWind_res)
        MultiYearWind_res.close()

    @staticmethod
    def test_time_index(MultiYearWind_res):
        """
        test extraction of WindResource time_index
        """
        check_time_index(MultiYearWind_res)
        MultiYearWind_res.close()

    @staticmethod
    def test_ds(MultiYearWind_res, ds_name='windspeed_100m'):
        """
        test extraction of a variable array, attributes, and properties
        """
        check_dset(MultiYearWind_res, ds_name)
        check_attrs(MultiYearWind_res, ds_name)
        check_properties(MultiYearWind_res, ds_name)
        check_years(MultiYearWind_res, ds_name)
        MultiYearWind_res.close()

    @staticmethod
    def test_new_hubheight(MultiYearWind_res, ds_name='windspeed_90m'):
        """
        test extraction of interpolated hub-height
        """
        check_dset(MultiYearWind_res, ds_name)
        check_years(MultiYearWind_res, ds_name)
        MultiYearWind_res.close()


def test_map_hsds_files():
    """
    Test map hsds files method
    """
    files = [f'/nrel/US_wave/West_Coast/West_Coast_wave_{year}.h5'
             for year in range(1979, 2011)]
    hsds_kwargs = {'endpoint': 'https://developer.nrel.gov/api/hsds',
                   'api_key': 'oHP7dGu4VZeg4rVo8PZyb5SVmYigedRHxi3OfiqI'}
    path = '/nrel/US_wave/West_Coast/West_Coast_wave_*.h5'

    h5_dir, prefix, suffix = MultiH5Path.multi_file_args(path)
    file_map = MultiYearH5._map_files(h5_dir, prefix=prefix, suffix=suffix,
                                      hsds=True, hsds_kwargs=hsds_kwargs)

    assert all(f in files for f in file_map.values())


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
