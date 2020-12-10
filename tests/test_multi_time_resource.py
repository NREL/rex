# -*- coding: utf-8 -*-
"""
pytests for multi time resource handlers
"""
import numpy as np
import os
from pandas.testing import assert_frame_equal
import pytest

from rex import TESTDATADIR
from rex.multi_time_resource import MultiTimeResource
from rex.resource import Resource
from rex.renewable_resource import WindResource, NSRDB


@pytest.fixture
def MultiYearNSRDB_res():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_*.h5')

    return MultiTimeResource(path, res_cls=NSRDB)


@pytest.fixture
def MultiYearWind_res():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_*.h5')

    return MultiTimeResource(path, res_cls=WindResource)


def check_res(res_cls):
    """
    Run test on len and shape methods
    """
    time_index = None
    for file in res_cls.h5_files:
        print(file)
        with Resource(file) as f:
            print(f.datasets)
            if time_index is None:
                time_index = f.time_index
            else:
                time_index = time_index.append(f.time_index)

    with Resource(res_cls.h5_files[0]) as f:
        meta = f.meta

    res_shape = (len(time_index), len(meta))

    assert len(res_cls) == len(time_index)
    assert res_cls.shape == res_shape


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
        test extraction of a variable array
        """
        check_dset(MultiYearNSRDB_res, ds_name)
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
        test extraction of a variable array
        """
        check_dset(MultiYearWind_res, ds_name)
        MultiYearWind_res.close()

    @staticmethod
    def test_new_hubheight(MultiYearWind_res, ds_name='windspeed_90m'):
        """
        test extraction of a variable array
        """
        check_dset(MultiYearWind_res, ds_name)
        MultiYearWind_res.close()


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
