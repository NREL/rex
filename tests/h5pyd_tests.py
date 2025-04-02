# -*- coding: utf-8 -*-
"""
pytests for accessing NREL .h5 files on the cloud via hsds/h5pyd

Note that this file cannot be named "test_*.py" because it is run with a
separate github action that sets up a local hsds server before running the
test.
"""
import pytest
import h5pyd
import numpy as np
import xarray as xr

from rex import NSRDB, WindResource, open_mfdataset_hsds


def test_file_list():
    """Test listing the nrel hsds directory with h5pyd"""
    with h5pyd.Folder('/nrel/') as folder:
        dir_list = list(folder)
    assert len(dir_list) >= 9
    assert 'nsrdb' in dir_list
    assert 'wtk' in dir_list
    assert 'sup3rcc' in dir_list


def test_nsrdb():
    """Test retrieving NSRDB data"""
    fp = '/nrel/nsrdb/v3/nsrdb_2020.h5'
    with NSRDB(fp, hsds=True) as res:
        dsets = res.dsets
        ghi = res['ghi', :, int(1e5)]

    assert len(dsets) == 28
    assert not any(ghi < 0)
    assert all(ghi < 1300)
    assert any(ghi > 800)

    with xr.open_dataset(fp, engine="rex", hsds=True) as ds:
        assert np.allclose(ds["ghi"].isel(gid=int(1e5)), ghi)


def test_wtk():
    """Test retrieving WTK data"""
    fp = '/nrel/wtk/conus/wtk_conus_2007.h5'
    with WindResource(fp, hsds=True) as res:
        dsets = res.dsets
        ws = res['windspeed_80m', :, [100000, 100100]]

    assert len(dsets) == 37
    assert not (ws < 0).any()
    assert (ws < 50).all()
    assert (ws > 10).any()

    with xr.open_dataset(fp, engine="rex", hsds=True) as ds:
        assert np.allclose(ds["windspeed_80m"].isel(gid=[100000, 100100]), ws)


def test_sup3rcc():
    """Test retrieving sup3rcc data"""
    fp = ('/nrel/sup3rcc/conus_ecearth3_ssp585_r1i1p1f1/'
          'sup3rcc_conus_ecearth3_ssp585_r1i1p1f1_2059.h5')
    with WindResource(fp, hsds=True) as res:
        dsets = res.dsets
        ghi = res['ghi', :, 100000:100002]
        ws = res['windspeed_88m', :, 100000:100002]
        temp = res['temperature_2m', :, 100000:100002]

    assert len(dsets) == 14
    assert not (ghi < 0).any()
    assert (ghi < 1300).all()
    assert (ghi > 800).any()
    assert not (ws < 0).any()
    assert (ws < 50).all()
    assert (ws > 10).any()
    assert (temp > 20).any()
    assert (temp < -20).any()

    with xr.open_dataset(fp, engine="rex", hsds=True) as ds:
        assert np.allclose(ds["ghi"].isel(gid=slice(100000, 100002)), ghi)


@pytest.mark.parametrize('fps', ["/nrel/wtk/conus/wtk_conus_200[8,9].h5",
                                 ("/nrel/wtk/conus/wtk_conus_2008.h5",
                                  "/nrel/wtk/conus/wtk_conus_2009.h5")])
def test_mf_hsds_xr(fps):
    """Test opening multiple files via HSDS with xarray"""

    with open_mfdataset_hsds(fps, parallel=True, chunks="auto") as ds:
        assert ds.sizes == {'time': 17544, 'gid': 2488136}
        assert str(ds.time_index.isel(time=0).values).startswith("2008")
        assert str(ds.time_index.isel(time=-1).values).startswith("2009")
