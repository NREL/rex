# -*- coding: utf-8 -*-
"""
pytests for resource handlers with a single hub height
"""
import numpy as np
import os
import pytest

from rex.renewable_resource import WindResource
from rex import TESTDATADIR


def test_single_hh():
    """Test that resource with data at a single hub height will always return
    the data at that hub height (and also return a warning)"""
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_1.h5')
    with WindResource(h5) as wind:
        # Existing datasets are P0m and T80m
        assert np.array_equal(wind['pressure_80m'], wind['pressure_0m'])
        assert np.array_equal(wind['temperature_10m'], wind['temperature_80m'])


def test_check_hh():
    """Test that check hub height method will return the hh at the single
    windspeed"""
    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    msg = ('Wind resource method _check_hub_height() failed! Should have '
           'returned 100 because theres only windspeed at 100m')
    with WindResource(h5) as wind:
        assert (wind._check_hub_height(120) == 100), msg


def test_sam_df_hh():
    """Test that if there's only windspeed at one HH, all data is returned
    from that hh
    """

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012_incomplete_2.h5')
    with WindResource(h5) as wind:
        sam_df = wind._get_SAM_df('pressure_80m', 0)

        arr1 = wind['pressure_100m', :, 0] * 9.86923e-6
        arr2 = sam_df['pressure_100m'].values

        msg1 = ('Error: pressure should have been loaded at 100m '
                'b/c there is only windspeed at 100m.')

        assert np.array_equal(arr1, arr2), msg1


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
