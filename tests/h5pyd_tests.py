# -*- coding: utf-8 -*-
"""
pytests for accessing NREL .h5 files on the cloud via hsds/h5pyd

Note that this file cannot be named "test_*.py" because it is run with a
separate github action that sets up a local hsds server before running the
test.
"""
import h5pyd
from rex import Resource, NSRDB, WindResource

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
    with NSRDB('/nrel/nsrdb/v3/nsrdb_2020.h5', hsds=True) as res:
        dsets = res.dsets
        ghi = res['ghi', :, int(1e5)]

    assert len(dsets) == 28
    assert not any(ghi < 0)
    assert all(ghi < 1300)
    assert any(ghi > 800)


def test_wtk():
    """Test retrieving WTK data"""
    with WindResource('/nrel/wtk/conus/wtk_conus_2007.h5', hsds=True) as res:
        dsets = res.dsets
        ws = res['windspeed_88m', :, int(1e5)]

    assert len(dsets) == 37
    assert not any(ws < 0)
    assert all(ws < 50)
    assert any(ws > 10)


def test_sup3rcc():
    """Test retrieving sup3rcc data"""
    fp = ('/nrel/sup3rcc/conus_ecearth3_ssp585_r1i1p1f1/'
          'sup3rcc_conus_ecearth3_ssp585_r1i1p1f1_2059.h5')
    with WindResource(fp, hsds=True) as res:
        dsets = res.dsets
        ghi = res['ghi', :, int(1e5)]
        ws = res['windspeed_88m', :, int(1e5)]
        temp = res['temperature_2m', :, int(1e5)]

    assert len(dsets) == 14
    assert not any(ghi < 0)
    assert all(ghi < 1300)
    assert any(ghi > 800)
    assert not any(ws < 0)
    assert all(ws < 50)
    assert any(ws > 10)
    assert any(temp > 20)
    assert any(temp < -20)
