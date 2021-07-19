# -*- coding: utf-8 -*-
"""
pytests for hsds conection
"""
from rex.resource_extraction.resource_extraction import MultiYearResourceX


def test_hsds():
    """
    Test HSDS reads
    """
    dsets = ['coordinates',
             'directionality_coefficient',
             'energy_period',
             'maximum_energy_direction',
             'mean_absolute_period',
             'mean_wave_direction',
             'mean_zero-crossing_period',
             'meta',
             'omni-directional_wave_power',
             'peak_period',
             'significant_wave_height',
             'spectral_width',
             'time_index',
             'water_depth']
    files = [f'/nrel/US_wave/West_Coast/West_Coast_wave_{year}.h5'
             for year in range(1979, 2011)]

    hsds_kwargs = {'endpoint': 'https://developer.nrel.gov/api/hsds',
                   'api_key': 'oHP7dGu4VZeg4rVo8PZyb5SVmYigedRHxi3OfiqI'}
    path = '/nrel/US_wave/West_Coast/West_Coast_wave_*.h5'
    with MultiYearResourceX(path, hsds=True, hsds_kwargs=hsds_kwargs) as f:
        assert all(dset in dsets for dset in f.datasets)
        assert all(h5_file in files for h5_file in f._res.h5_files)
