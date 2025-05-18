# -*- coding: utf-8 -*-
"""
pytests for rex xarray backend
"""
import os
import sys
from tempfile import TemporaryDirectory

import h5py
import pytest
import pandas as pd
import numpy as np
import xarray as xr

from rex import (TESTDATADIR, Resource, Outputs, MultiYearResource,
                 MultiFileResource)


NSRDB_2012 = os.path.join(TESTDATADIR, 'nsrdb', 'ri_100_nsrdb_2012.h5')
NSRDB_2013 = os.path.join(TESTDATADIR, 'nsrdb', 'ri_100_nsrdb_2013.h5')
SZA_2012 = os.path.join(TESTDATADIR, 'sza', 'nsrdb_sza_2012.h5')
SZA_2013 = os.path.join(TESTDATADIR, 'sza', 'nsrdb_sza_2013.h5')
WAVE_2010 = os.path.join(TESTDATADIR, 'wave', 'ri_wave_2010.h5')
WTK_2012_FP = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_2012.h5')
WTK_2013_FP = os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_2013.h5')
WTK_2012_GRP_FP = os.path.join(TESTDATADIR, 'wtk', 'ri_wtk_2012_group.h5')
WTK_2010_100M = os.path.join(TESTDATADIR, 'wtk', 'wtk_2010_100m.h5')
WTK_2010_200M = os.path.join(TESTDATADIR, 'wtk', 'wtk_2010_200m.h5')


def check_ti(truth_ti, ds):
    """Check that the time index of the dataset matches expectations"""

    for t_var in ["time_index", "time"]:
        assert t_var in ds.coords
        assert len(ds[t_var].shape) == 1
        assert ds[t_var].dtype == np.dtype('datetime64[ns]')
        assert np.allclose(ds[t_var].isel(time=0).astype('int64'),
                           truth_ti[0].value)
        assert np.allclose(ds[t_var].isel(time=[0, 2]).astype('int64'),
                           truth_ti[[0, 2]].astype('int64'))
        assert np.allclose(ds[t_var].isel(time=slice(0, 2)).astype('int64'),
                           truth_ti[0:2].astype('int64'))


def check_meta(truth_meta, ds):
    """Check that the meta of the dataset matches expectations"""

    for col in truth_meta.columns:
        truth_vals = truth_meta[col].to_numpy()
        truth_dtype = truth_vals.dtype
        assert col in ds.coords
        assert len(ds[col].shape) == 1
        if isinstance(truth_vals[0], str):
            continue
        assert (ds[col].isel(gid=0).values.astype(truth_dtype)
                == truth_vals[0])
        assert np.allclose(ds[col].isel(gid=[0, 2]).astype(truth_dtype),
                           truth_vals[[0, 2]])
        assert np.allclose(ds[col].isel(gid=slice(0, 2)).astype(truth_dtype),
                           truth_vals[0:2])


def check_shape(truth_shape, ds):
    """Check that the shape of the dataset matches expectations"""
    assert ds.sizes == {'time': truth_shape[0], 'gid': truth_shape[1]}


def check_data(truth_datasets, ds):
    """Check that the values of the dataset match expectations"""

    for name, values in truth_datasets.items():
        assert np.allclose(ds[name], values)


@pytest.mark.parametrize('fp', [WTK_2012_FP, WTK_2013_FP, WTK_2010_100M,
                                WTK_2010_200M, SZA_2012, SZA_2013, NSRDB_2012,
                                NSRDB_2013, WAVE_2010])
def test_open_with_xr(fp):
    """Test basic opening and read operations on various files"""
    with Resource(fp) as res:
        truth_meta = res.meta
        truth_ti = res.time_index
        truth_shape = res.shape
        truth_datasets = {name: res[name][:] for name in res.resource_datasets}

    with xr.open_dataset(fp, engine="rex") as ds:
        check_ti(truth_ti, ds)
        check_meta(truth_meta, ds)
        check_shape(truth_shape, ds)
        check_data(truth_datasets, ds)

        assert set(ds.indexes) == {"time", "gid"}


def test_encoding():
    """Test that the encoding is set properly"""
    with Resource(WTK_2012_FP) as res:
        truth_shape = res.shape

    with xr.open_dataset(WTK_2012_FP, engine="rex") as ds:
        expected = {'chunksizes': None,
                    'fletcher32': False,
                    'shuffle': False,
                    'dtype': 'u2',
                    'scale_factor': 10,
                    'source': WTK_2012_FP,
                    'original_shape': truth_shape}
        assert ds["pressure_0m"].encoding == expected


def test_var_attrs():
    """Test the attrs values for a dataset variable"""
    with xr.open_dataset(WTK_2012_FP, engine="rex") as ds:
        expected = {'standard_name': 'air_pressure', 'units': 'Pa'}
        assert ds["pressure_0m"].attrs == expected


def test_ds_attrs():
    """Test the attrs values for a dataset"""
    meta = pd.DataFrame(
        {"latitude": [41.29], "longitude": [-71.86], "timezone": [-5]}
    )
    meta.index.name = "gid"
    test_attrs = {"test": 1, "another": "Attr"}
    with TemporaryDirectory() as td:
        test_file = os.path.join(td, "test_geo.h5")

        with Outputs(test_file, "w") as f:
            f.meta = meta
            f.time_index = pd.date_range(start="1/1/2018", end="1/1/2019",
                                         freq="h")[:-1]
            f.run_attrs = test_attrs

        Outputs.add_dataset(test_file, "pressure_0m", np.array([1]),
                            np.float32, attrs={"units": "C"})

        with xr.open_dataset(test_file, engine="rex") as ds:
            for k, v in test_attrs.items():
                assert ds.attrs[k] == v
            assert ds.attrs["package"] == "rex"
            assert "version" in ds.attrs
            assert "full_version_record" in ds.attrs

            expected = {'standard_name': 'air_pressure', 'units': 'C'}
            assert ds["pressure_0m"].attrs == expected


def test_open_group():
    """Test opening a group within the file"""
    with Resource(WTK_2012_GRP_FP, group="group") as res:
        truth_meta = res.meta
        truth_ti = res.time_index
        truth_shape = res.shape
        truth_datasets = {name: res[name][:] for name in res.resource_datasets}

    with xr.open_dataset(WTK_2012_GRP_FP, group="group", engine="rex") as ds:
        check_ti(truth_ti, ds)
        check_meta(truth_meta, ds)
        check_shape(truth_shape, ds)
        check_data(truth_datasets, ds)


@pytest.mark.parametrize(
    'glob_fp',
    [os.path.join(TESTDATADIR, 'nsrdb', 'ri_100_nsrdb_201*.h5'),
     os.path.join(TESTDATADIR, 'sza', 'nsrdb_sza_201*.h5'),
     os.path.join(TESTDATADIR, 'wtk', 'ri_100_wtk_201*.h5')])
def test_open_mf_year(glob_fp):
    """Test opening a multi-file dataset across years"""
    with MultiYearResource(glob_fp) as res:
        truth_shape = res.shape

    with xr.open_mfdataset(glob_fp, engine="rex") as ds:
        assert ds.sizes == {'time': truth_shape[0], 'gid': truth_shape[1]}


def test_open_mf_ds():
    """Test opening multi-file dataset across variables"""
    glob_fp = os.path.join(TESTDATADIR, 'wtk', 'wtk_2010_*m.h5')
    with MultiFileResource(glob_fp) as res:
        truth_shape = res.shape
        datasets = res.resource_datasets

    with xr.open_mfdataset(glob_fp, engine="rex") as ds:
        assert ds.sizes == {'time': truth_shape[0], 'gid': truth_shape[1]}
        assert all(ds_name in ds for ds_name in datasets)


def test_open_drop_var():
    """Test dropping of variables when opening a file"""
    with xr.open_dataset(WTK_2012_FP, engine="rex") as ds:
        assert "pressure_0m" in ds

    with xr.open_dataset(WTK_2012_FP, drop_variables={"pressure_0m"},
                         engine="rex") as ds:
        assert "pressure_0m" not in ds


def test_detect_var_dims():
    """Test that var dimensions are detected properly"""
    meta = pd.DataFrame(
        {"latitude": [41.29], "longitude": [-71.86], "timezone": [-5]}
    )
    meta.index.name = "gid"
    with TemporaryDirectory() as td:
        test_file = os.path.join(td, "test_geo.h5")

        with Outputs(test_file, "w") as f:
            f.meta = meta
            f.time_index = pd.date_range(start="1/1/2018", end="1/1/2019",
                                         freq="h")[:-1]

        Outputs.add_dataset(test_file, "spatial_var", np.array([1]),
                            np.float32, attrs={"units": "C"})
        Outputs.add_dataset(test_file, "temporal_var", np.zeros((8760,)),
                            np.float32, attrs={"units": "MW"})
        Outputs.add_dataset(test_file, "spatiotemporal_var",
                            np.ones((8760, 1)), np.float32,
                            attrs={"units": "MW"})

        with Resource(test_file) as res:
            truth_ti = res.time_index
            truth_shape = res.shape
            truth_datasets = {name: res[name][:]
                              for name in res.resource_datasets}

        with xr.open_dataset(test_file, engine="rex") as ds:
            assert set(ds.indexes) == {"time", "gid"}

            assert ds["spatial_var"].dims == ("gid",)
            assert ds["temporal_var"].dims == ("time", )
            assert ds["spatiotemporal_var"].dims == ("time", "gid")
            assert ds["latitude"].dims == ("gid",)
            assert ds["longitude"].dims == ("gid",)
            assert ds["timezone"].dims == ("gid",)

            assert ds["spatial_var"].isel(gid=0) == 1
            assert ds["temporal_var"].isel(time=542) == 0
            assert ds["spatiotemporal_var"].isel(gid=0, time=542) == 1

            assert np.allclose(ds["latitude"], [41.29])
            assert np.allclose(ds["longitude"], [-71.86])

            check_ti(truth_ti, ds)
            check_shape(truth_shape, ds)
            check_data(truth_datasets, ds)


def test_coords_dset():
    """Test that the coordinates dataset is loaded properly"""
    meta = pd.DataFrame(
        {"latitude": [45.29], "longitude": [71.86], "timezone": [-5]}
    )
    meta.index.name = "gid"
    with TemporaryDirectory() as td:
        test_file = os.path.join(td, "test_geo.h5")
        with Outputs(test_file, "w") as f:
            f.meta = meta
            f.time_index = pd.date_range(start="1/1/2018", end="1/1/2019",
                                         freq="h")[:-1]

        with h5py.File(test_file, "a") as fh:
            fh.create_dataset("coordinates", data=np.array([[41.29, -71.86]]))

        with xr.open_dataset(test_file, engine="rex") as ds:
            assert set(ds.indexes) == {"time", "gid"}
            assert ds["latitude"].dims == ("gid",)
            assert ds["longitude"].dims == ("gid",)
            assert ds["timezone"].dims == ("gid",)

            assert np.allclose(ds["latitude"], [41.29])
            assert np.allclose(ds["longitude"], [-71.86])


@pytest.mark.skipif(sys.version_info[:2] <= (3, 9),
                    reason="DataTrees require Python 3.10+ to run")
@pytest.mark.parametrize('fp', [WTK_2012_FP, WTK_2013_FP, WTK_2010_100M,
                                WTK_2010_200M, SZA_2012, SZA_2013, NSRDB_2012,
                                NSRDB_2013, WAVE_2010])
def test_open_data_tree_no_groups(fp):
    """Test basic opening and read operations for a data tree"""
    with Resource(fp) as res:
        truth_meta = res.meta
        truth_ti = res.time_index
        truth_shape = res.shape
        truth_datasets = {name: res[name][:] for name in res.resource_datasets}

    with xr.open_datatree(fp, engine="rex") as ds:
        check_ti(truth_ti, ds)
        check_meta(truth_meta, ds)
        check_shape(truth_shape, ds)
        check_data(truth_datasets, ds)

        assert set(ds.indexes) == {"time", "gid"}


@pytest.mark.skipif(sys.version_info[:2] <= (3, 9),
                    reason="DataTrees require Python 3.10+ to run")
def test_open_data_tree_with_group():
    """Test opening a data tree for a file with a group"""
    with Resource(WTK_2012_GRP_FP, group="group") as res:
        truth_meta = res.meta
        truth_ti = res.time_index
        truth_shape = res.shape
        truth_datasets = {name: res[name][:] for name in res.resource_datasets}

    with xr.open_datatree(WTK_2012_GRP_FP, engine="rex") as ds:
        check_ti(truth_ti, ds["group"])
        check_meta(truth_meta, ds["group"])
        check_shape(truth_shape, ds["group"])
        check_data(truth_datasets, ds["group"])


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
