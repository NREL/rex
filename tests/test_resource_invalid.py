# -*- coding: utf-8 -*-
"""
pytests for resource handlers with invalid ranges
"""
import numpy as np
import os
import pytest

from rex.renewable_resource import WindResource
from rex import TESTDATADIR


def test_min_pressure():
    """Test that minimum pressure range enforcement works."""

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_invalid.h5')

    # sites set with bad pressure.
    for site in [3, 7, 43, 79, 151, 179]:

        with WindResource(h5) as wind:
            og_min = np.min(wind['pressure_100m']) * 9.86923e-6
            sam_df = wind._get_SAM_df('pressure_100m', site)
            patched_min = np.min(sam_df['Pressure'].values)

            msg1 = 'Not a good test set. Min pressure is {}'.format(og_min)
            msg2 = ('Physical range enforcement failed. '
                    'Original pressure min was {}, '
                    'patched min was {}'.format(og_min, patched_min))

            assert og_min < 0.5, msg1
            assert patched_min == 0.5, msg2


def test_min_temp():
    """Test that minimum temperature range enforcement works."""

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_invalid.h5')

    # sites set with bad temperature.
    for site in [5, 12, 45, 54, 97, 103, 142, 166]:

        with WindResource(h5) as wind:
            og_min = np.min(wind['temperature_100m'])
            sam_df = wind._get_SAM_df('temperature_100m', site)
            patched_min = np.min(sam_df['Temperature'].values)

            msg1 = 'Not a good test set. Min temp is {}'.format(og_min)
            msg2 = ('Physical range enforcement failed. '
                    'Original temp min was {}, '
                    'patched min was {}'.format(og_min, patched_min))

            assert og_min < -200, msg1
            assert patched_min == -200, msg2


def test_max_ws():
    """Test that max windspeed range enforcement works."""

    h5 = os.path.join(TESTDATADIR, 'wtk/ri_wtk_2012_invalid.h5')

    # sites set with bad wind speeds.
    for site in [7, 54, 66, 89, 110, 149, 188]:

        with WindResource(h5) as wind:
            og_max = np.max(wind['windspeed_100m'])
            sam_df = wind._get_SAM_df('windspeed_100m', site)
            patched_max = np.max(sam_df['Speed'].values)

            msg1 = 'Not a good test set. Min wind speed is {}'.format(og_max)
            msg2 = ('Physical range enforcement failed. '
                    'Original wind speed min was {}, '
                    'patched min was {}'.format(og_max, patched_max))

            assert og_max > 120, msg1
            assert patched_max == 120, msg2


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
