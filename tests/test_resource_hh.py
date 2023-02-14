# -*- coding: utf-8 -*-
"""
pytests for resource handlers with a single hub height
"""
import numpy as np
import os
import pytest

from rex.utilities.exceptions import ResourceWarning
from rex.renewable_resource import WindResource
from rex.sam_resource import SAMResource
from rex import TESTDATADIR


def test_single_hh():
    """Test that resource with data at a single hub height will always return
    the data at that hub height (and also return a warning)

    ***Without lapse rate feature enabled (new as of 1/2023)

    Only pressure_0m and temperature_80m are available from this file
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_1.h5')
    with pytest.warns(ResourceWarning):
        with WindResource(h5, use_lapse_rate=False) as wind:
            # Existing datasets are P0m and T80m
            assert np.array_equal(wind['pressure_80m'], wind['pressure_0m'])
            assert np.array_equal(wind['temperature_10m'],
                                  wind['temperature_80m'])


def test_p_lapse_rate():
    """Test pressure-based lapse rate when there's only a single pressure level
    available.

    Will throw warning because stupid test file doesn't have units.

    Only pressure_0m and temperature_80m are available from this file
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_1.h5')
    with pytest.warns(ResourceWarning):
        with WindResource(h5, use_lapse_rate=True) as wind:
            assert (wind['pressure_80m'] < wind['pressure_0m']).all()
            assert (wind['pressure_1m'] < wind['pressure_0m']).all()
            assert (wind['pressure_0.1m'] < wind['pressure_0m']).all()
            assert (wind['pressure_0m'] == wind['pressure_0m']).all()
            assert (wind['pressure_-10m'] > wind['pressure_0m']).all()


def test_t_lapse_rate():
    """Test temperature-based lapse rate when there's only a single
    temperature level available.

    Will throw warning because stupid test file doesn't have units.

    Only pressure_0m and temperature_80m are available from this file
    """
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_1.h5')
    with pytest.warns(ResourceWarning):
        with WindResource(h5, use_lapse_rate=True) as wind:
            assert (wind['temperature_90m'] < wind['temperature_80m']).all()
            assert (wind['temperature_80m'] == wind['temperature_80m']).all()
            assert (wind['temperature_70m'] > wind['temperature_80m']).all()
            assert (wind['temperature_0.1m'] > wind['temperature_80m']).all()
            assert (wind['temperature_-10m'] > wind['temperature_80m']).all()


def test_check_hh():
    """Test that check hub height method will return the hh at the single
    windspeed"""
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_2.h5')
    msg = ('Wind resource method _check_hub_height() failed! Should have '
           'returned 100 because theres only windspeed at 100m')
    with WindResource(h5) as wind:
        assert (wind._check_hub_height(wind.heights, 120) == 100), msg


def test_sam_df_hh():
    """Test that if there's only windspeed at one HH, all data is returned
    from that hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_2.h5')
    with WindResource(h5) as wind:
        sam_df = wind.get_SAM_df(0, 80)

        arr1 = wind['pressure_100m', :, 0] * 9.86923e-6
        arr1 = SAMResource.roll_timeseries(arr1, -5, 1)
        arr2 = sam_df['Pressure'].values

        msg1 = ('Error: pressure should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')

        assert np.array_equal(arr1, arr2), msg1


def test_preload_sam_hh():
    """Test the preload_SAM method with a single hub height windspeed in res.

    In this case, all variables should be loaded at the single windspeed hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_incomplete_2.h5')
    sites = slice(0, 200)
    hub_heights = 80

    SAM_res = WindResource.preload_SAM(h5, sites, hub_heights)

    with WindResource(h5) as wind:
        p = wind['pressure_100m'] * 9.86923e-6
        t = wind['temperature_100m']
        msg1 = ('Error: pressure should have been loaded at 100m b/c '
                'there is only windspeed at 100m.')
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
