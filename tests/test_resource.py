# -*- coding: utf-8 -*-
"""
pytests for resource handlers
"""
from datetime import datetime
import h5py
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile

from rex import TESTDATADIR, Resource, Outputs
from rex.multi_file_resource import (MultiH5, MultiH5Path, MultiFileNSRDB,
                                     MultiFileWTK)
from rex.renewable_resource import (NSRDB, WindResource)
from rex.utilities.exceptions import ResourceKeyError, ResourceRuntimeError


def NSRDB_res():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/ri_100_nsrdb_2012.h5')
    return NSRDB(path)


def NSRDB_2018():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb*2018.h5')
    return MultiFileNSRDB(path)


def NSRDB_2018_list():
    """
    Init NSRDB resource handler
    """
    path = os.path.join(TESTDATADIR, 'nsrdb/nsrdb*2018.h5')
    path, h5_files = MultiH5Path._get_h5_files(path)

    return MultiFileNSRDB(h5_files)


def WindResource_res():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    return WindResource(path)


def FiveMinWind_res():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk', 'wtk*m.h5')
    return MultiFileWTK(path)


def FiveMinWind_list():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/wtk*m.h5')
    path, h5_files = MultiH5Path._get_h5_files(path)
    return MultiFileWTK(h5_files)


def wind_group():
    """
    Init WindResource resource handler
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_group.h5')
    return WindResource(path, group='group')


def check_res(res_cls):
    """
    Run test on len and shape methods
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
    with h5py.File(res_cls.h5_file, 'r') as f:
        ds_name = 'meta'
        if res_cls._group:
            ds_name = '{}/{}'.format(res_cls._group, ds_name)

        baseline = pd.DataFrame(f[ds_name][...])

    sites = slice(0, len(baseline))
    meta = res_cls['meta', sites]
    cols = ['latitude', 'longitude', 'elevation', 'timezone']
    assert np.allclose(baseline[cols].values[sites], meta[cols].values)
    sites = len(baseline)
    sites = slice(int(sites / 3), int(sites / 2))
    meta = res_cls['meta', sites]
    cols = ['latitude', 'longitude', 'elevation', 'timezone']
    assert np.allclose(baseline[cols].values[sites], meta[cols].values)

    sites = 5
    meta = res_cls['meta', sites]
    cols = ['latitude', 'longitude', 'elevation', 'timezone']
    assert np.allclose(baseline[cols].values[sites], meta[cols].values)

    sites = sorted(np.random.choice(len(baseline), 5, replace=False))
    meta = res_cls['meta', sites]
    cols = ['latitude', 'longitude', 'elevation', 'timezone']
    assert np.allclose(baseline[cols].values[sites], meta[cols].values)

    meta = res_cls['meta']
    cols = ['latitude', 'longitude', 'elevation', 'timezone']
    assert np.allclose(baseline[cols].values, meta[cols].values)
    assert isinstance(meta, pd.DataFrame)

    meta_shape = meta.shape
    max_sites = int(meta_shape[0] * 0.8)
    # single site
    meta = res_cls['meta', max_sites]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (1, meta_shape[1])
    # site slice

    meta = res_cls['meta', :max_sites]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (max_sites, meta_shape[1])
    # site list
    sites = sorted(np.random.choice(meta_shape[0], max_sites, replace=False))
    meta = res_cls['meta', sites]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (len(sites), meta_shape[1])
    # select columns
    meta = res_cls['meta', :, ['latitude', 'longitude']]
    assert isinstance(meta, pd.DataFrame)
    assert meta.shape == (meta_shape[0], 2)

    lat_lon = res_cls.lat_lon
    assert np.allclose(baseline[['latitude', 'longitude']].values, lat_lon)


def check_time_index(res_cls):
    """
    Run tests on time_index
    """
    time_index = res_cls['time_index']
    time_shape = time_index.shape
    assert isinstance(time_index, pd.DatetimeIndex)
    assert str(time_index.tz) == 'UTC'
    # single timestep
    time_index = res_cls['time_index', 50]
    assert isinstance(time_index, datetime)
    # time slice
    time_index = res_cls['time_index', 100:200]
    assert isinstance(time_index, pd.DatetimeIndex)
    assert time_index.shape == (100,)
    # list of timesteps
    steps = sorted(np.random.choice(time_shape[0], 50, replace=False))
    time_index = res_cls['time_index', steps]
    assert isinstance(time_index, pd.DatetimeIndex)
    assert time_index.shape == (50,)


def check_dset(res_cls, ds_name):
    """
    Run tests on dataset ds_name
    """
    ds_shape = res_cls.shape
    max_sites = int(ds_shape[1] * 0.8)
    arr = res_cls[ds_name]
    ds = res_cls[ds_name]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == ds_shape
    assert np.allclose(arr, ds)
    # single site all time
    ds = res_cls[ds_name, :, 1]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0],)
    # single time all sites
    ds = res_cls[ds_name, 10]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[1],)
    assert np.allclose(arr[10], ds)
    # single value
    ds = res_cls[ds_name, 10, max_sites]
    assert isinstance(ds, (np.integer, np.floating))
    assert np.allclose(arr[10, max_sites], ds)
    # site slice
    sites = slice(int(max_sites / 2), max_sites)
    ds = res_cls[ds_name, :, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], sites.stop - sites.start)
    assert np.allclose(arr[:, sites], ds)
    # time slice
    ds = res_cls[ds_name, 10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (10, ds_shape[1])
    assert np.allclose(arr[10:20], ds)
    # slice in time and space
    ds = res_cls[ds_name, 100:200, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, sites.stop - sites.start)
    assert np.allclose(arr[100:200, sites], ds)
    # site list
    sites = sorted(np.random.choice(ds_shape[1], max_sites, replace=False))
    ds = res_cls[ds_name, :, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], len(sites))
    assert np.allclose(arr[:, sites], ds)
    # site list single time
    sites = sorted(np.random.choice(ds_shape[1], max_sites, replace=False))
    ds = res_cls[ds_name, 0, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (len(sites),)
    assert np.allclose(arr[0, sites], ds)
    # time list
    times = sorted(np.random.choice(ds_shape[0], 100, replace=False))
    ds = res_cls[ds_name, times]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, ds_shape[1])
    assert np.allclose(arr[times], ds)
    # time list single site
    ds = res_cls[ds_name, times, 0]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100,)
    assert np.allclose(arr[times, 0], ds)
    # boolean mask
    mask = res_cls.time_index.month == 7
    ds = res_cls[ds_name, mask]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (mask.sum(), ds_shape[1])
    assert np.allclose(arr[mask], ds)
    # time and site lists
    with pytest.raises(IndexError):
        assert res_cls[ds_name, times, sites]


def check_dset_handler(res_cls, ds_name):
    """
    Run tests on dataset ds_name
    """
    ds_shape = res_cls.shape
    max_sites = int(ds_shape[1] * 0.8)
    dset = res_cls.open_dataset(ds_name)
    arr = dset[...]
    ds = res_cls[ds_name]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == ds_shape
    assert np.allclose(arr, ds)
    # single site all time
    ds = dset[:, 1]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0],)
    # single time all sites
    ds = dset[10]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[1],)
    assert np.allclose(arr[10], ds)
    # single value
    ds = dset[10, max_sites]
    assert isinstance(ds, (np.integer, np.floating))
    assert np.allclose(arr[10, max_sites], ds)
    # site slice
    sites = slice(int(max_sites / 2), max_sites)
    ds = dset[:, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], sites.stop - sites.start)
    assert np.allclose(arr[:, sites], ds)
    # time slice
    ds = dset[10:20]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (10, ds_shape[1])
    assert np.allclose(arr[10:20], ds)
    # slice in time and space
    ds = dset[100:200, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, sites.stop - sites.start)
    assert np.allclose(arr[100:200, sites], ds)
    # site list
    sites = sorted(np.random.choice(ds_shape[1], max_sites, replace=False))
    ds = dset[:, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (ds_shape[0], len(sites))
    assert np.allclose(arr[:, sites], ds)
    # site list single time
    sites = sorted(np.random.choice(ds_shape[1], max_sites, replace=False))
    ds = dset[0, sites]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (len(sites),)
    assert np.allclose(arr[0, sites], ds)
    # time list
    times = sorted(np.random.choice(ds_shape[0], 100, replace=False))
    ds = dset[times]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100, ds_shape[1])
    assert np.allclose(arr[times], ds)
    # time list single site
    ds = dset[times, 0]
    assert isinstance(ds, np.ndarray)
    assert ds.shape == (100,)
    assert np.allclose(arr[times, 0], ds)
    # time and site lists
    with pytest.raises(IndexError):
        assert dset[times, sites]


def check_dset_map(res_cls, ds_name):
    """
    Test MultiH5 dset map
    """
    res_values = res_cls.h5[ds_name][:, 0]

    file = res_cls.h5._dset_map[ds_name]
    with h5py.File(file, 'r') as f:
        native_values = f[ds_name][:, 0]

    assert np.allclose(res_values, native_values)


def check_scale(res_cls, ds_name):
    """
    Test unscaling of variable
    """
    native_value = res_cls[ds_name, 0, 0]
    scaled_value = res_cls.h5[ds_name][0, 0]
    scale_factor = res_cls.get_scale_factor(ds_name)
    if scale_factor != 1:
        assert native_value != scaled_value

    assert native_value == (scaled_value / scale_factor)


def check_interp(res_cls, var, h):
    """
    Test linear interpolation of Wind variables
    """
    ds_name = '{}_{}m'.format(var, h)
    ds_value = res_cls[ds_name, 0, 0]

    (h1, h2), _ = res_cls.get_nearest_h(h, res_cls.heights[var])

    ds_name = '{}_{}m'.format(var, h1)
    h1_value = res_cls[ds_name, 0, 0]
    ds_name = '{}_{}m'.format(var, h2)
    h2_value = res_cls[ds_name, 0, 0]
    interp_value = (h2_value - h1_value) / (h2 - h1) * (h - h1) + h1_value

    assert ds_value == interp_value


class TestNSRDB:
    """
    NSRDB Resource handler tests
    """
    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_res(res_cls):
        """
        test NSRDB class calls
        """
        check_res(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_meta(res_cls):
        """
        test extraction of NSRDB meta data
        """
        check_meta(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_time_index(res_cls):
        """
        test extraction of NSRDB time_index
        """
        check_time_index(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_ds(res_cls, ds_name='dni'):
        """
        test extraction of a variable array, attributes, and properties
        """
        check_dset(res_cls, ds_name)
        check_dset_handler(res_cls, ds_name)
        check_attrs(res_cls, ds_name)
        check_properties(res_cls, ds_name)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_unscale_dni(res_cls):
        """
        test unscaling of dni values
        """
        check_scale(res_cls, 'dni')
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_res(),
                              NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_unscale_pressure(res_cls):
        """
        test unscaling of pressure values
        """
        check_scale(res_cls, 'surface_pressure')
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [NSRDB_2018(),
                              NSRDB_2018_list()])
    def test_dset_map(res_cls, ds_name='dni'):
        """
        Test MultiH5 dset map
        """
        check_dset_map(res_cls, ds_name)
        res_cls.close()


class TestWindResource:
    """
    WindResource Resource handler tests
    """

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_res(res_cls):
        """
        test WindResource class calls
        """
        check_res(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_meta(res_cls):
        """
        test extraction of WindResource meta data
        """
        check_meta(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_time_index(res_cls):
        """
        test extraction of WindResource time_index
        """
        check_time_index(res_cls)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_ds(res_cls, ds_name='windspeed_100m'):
        """
        test extraction of a variable array, attributes, and properties
        """
        check_dset(res_cls, ds_name)
        check_dset_handler(res_cls, ds_name)
        check_attrs(res_cls, ds_name)
        check_properties(res_cls, ds_name)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize(('res_cls', 'ds_name'),
                             [(WindResource_res(), 'windspeed_90m'),
                              (FiveMinWind_res(), 'windspeed_110m'),
                              (FiveMinWind_list(), 'windspeed_110m'),
                              (wind_group(), 'windspeed_90m')])
    def test_new_hubheight(res_cls, ds_name):
        """
        test extraction of a variable array
        """
        check_dset(res_cls, ds_name)
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_unscale_windspeed(res_cls):
        """
        test unscaling of windspeed values
        """
        check_scale(res_cls, 'windspeed_100m')
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [WindResource_res(),
                              FiveMinWind_res(),
                              FiveMinWind_list(),
                              wind_group()])
    def test_unscale_pressure(res_cls):
        """
        test unscaling of pressure values
        """
        check_scale(res_cls, 'pressure_100m')
        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize(('res_cls', 'h'),
                             [(WindResource_res(), 90),
                              (FiveMinWind_res(), 110),
                              (FiveMinWind_list(), 110),
                              (wind_group(), 90)])
    def test_interpolation(res_cls, h):
        """
        test variable interpolation
        """
        ignore = ['winddirection', 'precipitationrate', 'relativehumidity']
        for var in res_cls.heights.keys():
            if var not in ignore:
                check_interp(res_cls, var, h)

        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize(('res_cls', 'h'),
                             [(WindResource_res(), 110),
                              (FiveMinWind_res(), 90),
                              (FiveMinWind_list(), 90),
                              (wind_group(), 100)])
    def test_extrapolation(res_cls, h):
        """
        test variable interpolation
        """
        for var in ['temperature', 'pressure']:
            check_interp(res_cls, var, h)

        res_cls.close()

    @staticmethod
    @pytest.mark.parametrize('res_cls',
                             [FiveMinWind_res(),
                              FiveMinWind_list(), ])
    def test_dset_map(res_cls, ds_name='windspeed_100m'):
        """
        Test MultiH5 dset map
        """
        check_dset_map(res_cls, ds_name)
        res_cls.close()


def test_group_raise():
    """
    test WindResource class group check
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_group.h5')
    with pytest.raises(ResourceKeyError):
        with WindResource(path) as res:
            check_res(res)


def test_missing_dset():
    """
    test WindResource missing data set
    """
    with pytest.raises(ResourceKeyError) as excinfo:
        with WindResource_res() as res:
            __ = res['dne_dset']

    assert 'dne_dset not in' in str(excinfo.value)


def test_missing_dset_for_heights():
    """
    test WindResource missing data set for `self.heights`
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    with tempfile.TemporaryDirectory() as td:
        res_fp = os.path.join(td, 'ri_100_wtk_2012.h5')
        shutil.copy(path, res_fp)
        with h5py.File(res_fp, 'a') as fh:
            del fh['temperature_80m']
            del fh['temperature_100m']

        with pytest.raises(ResourceKeyError) as excinfo:
            with WindResource(res_fp) as res:
                __ = res._get_ds_height('temperature', (0, 0))

    expected_str = "Missing height info for dataset 'temperature'"
    assert expected_str in str(excinfo.value)


def test_check_files():
    """
    test MultiH5 check_files
    """
    h5_files = [os.path.join(TESTDATADIR, 'nsrdb', 'nsrdb_irradiance_2018.h5'),
                os.path.join(TESTDATADIR, 'wtk', 'wtk_2010_100m.h5')]
    with pytest.raises(ResourceRuntimeError):
        with MultiH5(h5_files, check_files=True) as f:
            check_res(f)


def test_time_index_out_of_bounds():
    """
    Test that resource allows a time index that is "out of bounds" for pandas
    """
    with tempfile.TemporaryDirectory() as td:
        out_fp = os.path.join(td, "temp.h5")
        time = np.arange(1000001, 1001001)
        shapes = {"data": (len(time), 1)}
        attrs = {"data": None}
        chunks = {"data": None}
        dtypes = {"data": "float32"}

        Outputs.init_h5(out_fp, ["data"], shapes,attrs, chunks, dtypes,
            meta=pd.DataFrame(data={"lat": [0], "lon": [1]}),
            time_index=time,
        )
        with Outputs(out_fp, 'a') as out:
            out["data", :, 0] = time + 10

        with Resource(out_fp) as res:
            assert np.allclose(res.time_index, time)
            assert np.allclose(res["data"][:, 0], time + 10)


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
