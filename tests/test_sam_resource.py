# -*- coding: utf-8 -*-
"""
pytests for sam_resource
"""
import numpy as np
import os
import pytest

from rex.renewable_resource import WindResource
from rex.sam_resource import SAMResource
from rex import TESTDATADIR


def test_sites_slice():
    """
    Test to ensure SAMResource.sites_slice returns slice when possible, else
    a list
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with WindResource(h5) as f:
        time_index = f.time_index

    sites = list(range(10))
    hub_heights = 80
    sam_res = SAMResource(sites, 'windpower', time_index,
                          hub_heights=hub_heights)
    msg = "sites were not returned as a slice"
    assert isinstance(sam_res.sites_slice, slice), msg

    sites = [0, 2, 5, 7, 9, 4, 3]
    sam_res = SAMResource(sites, 'windpower', time_index,
                          hub_heights=hub_heights)
    msg = "sites were not returned as the same input list"
    assert sites == sites


def test_check_units():
    """
    Test SAMResource unit convertion
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    var_name = 'pressure_100m'
    with WindResource(h5) as f:
        pa = f[var_name, :, :10]

    atm = SAMResource.check_units(var_name, pa.copy(), 'windpower')
    msg = "Pressure was not converted from pa to atm"
    assert np.allclose(atm, (pa * 9.86923e-6)), msg


def test_valid_range():
    """
    Test SAMResource valid range enforcement
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    var = 'pressure'
    var_name = '{}_100m'.format(var)
    tech = 'windpower'
    sites = list(range(10))
    with WindResource(h5) as f:
        pa = f[var_name, :, sites]

    atm = SAMResource.check_units(var_name, pa * 10, 'windpower')
    valid_range = SAMResource.DATA_RANGES[tech][var]
    valid = SAMResource.enforce_arr_range(var, atm, valid_range, sites)

    assert np.all(valid == valid_range[1])


def test_preload_sam_hh():
    """Test the preload_SAM method with invalid resource data ranges.
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_invalid.h5')
    sites = slice(0, 200)
    hub_heights = 80

    SAM_res = WindResource.preload_SAM(h5, sites, hub_heights)

    msg1 = 'Invalid pressure range was not corrected.'
    msg2 = 'Invalid temperature range was not corrected.'
    msg3 = 'Invalid windspeed range was not corrected.'

    assert np.min(SAM_res._res_arrays['pressure']) >= 0.5, msg1
    assert np.min(SAM_res._res_arrays['temperature']) >= -200, msg2
    assert np.max(SAM_res._res_arrays['windspeed']) <= 120, msg3


def test_preload_sam_hh():
    """Test the preload_SAM method with a single hub height windspeed in res.

    In this case, all variables should be loaded at the single windspeed hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    sites = slice(0, 200)
    hub_heights = 80

    SAM_res = WindResource.preload_SAM(h5, sites, hub_heights)

    with WindResource(h5) as wind:
        p = wind['pressure_100m'] * 9.86923e-6
        t = wind['temperature_100m']
        msg1 = ('Error: pressure should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')
        msg2 = ('Error: temperature should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')
        assert np.allclose(SAM_res['pressure', :, :].values, p), msg1
        assert np.allclose(SAM_res['temperature', :, :].values, t), msg2


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
