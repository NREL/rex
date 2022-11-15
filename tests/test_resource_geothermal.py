# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
pytests for Geothermal resource handlers
"""
import numpy as np
import os
import pytest
import tempfile

from rex import TESTDATADIR, Resource, Outputs
from rex.renewable_resource import GeothermalResource
from rex.utilities.exceptions import ExtrapolationWarning, ResourceWarning


SAMPLE_ATTRS = {'units': 'Celsius'}


@pytest.fixture
def sample_meta():
    """Sample 10-point meta file. """
    sample_res_file = os.path.join(TESTDATADIR, "wtk", "ri_100_wtk_2012.h5")
    with Resource(sample_res_file) as res:
        meta = res.meta

    meta = meta.iloc[0:10].copy()
    return meta[['latitude', 'longitude', 'country', 'state', 'county',
                 'timezone', 'elevation']]


def test_no_depth(sample_meta):
    """Test extracting data from file with no depth info."""

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'outputs.h5')

        Outputs.init_h5(fp, ["temperature", "potential_MW"],
                        shapes={"temperature": (10,), "potential_MW": (10,)},
                        attrs={"temperature": SAMPLE_ATTRS,
                               "potential_MW": SAMPLE_ATTRS},
                        chunks={"temperature": (10,), "potential_MW": (10,)},
                        dtypes={"temperature": np.float32,
                                "potential_MW": np.float32},
                        meta=sample_meta)

        with Outputs(fp, "a") as out:
            out["temperature"] = np.random.randint(300, 500, size=10)
            out["potential_MW"] = np.random.randint(100, 200, size=10)

        with GeothermalResource(fp) as res:
            assert "potential_MW" in res.dsets
            assert "temperature" in res.dsets
            temps = res['temperature']
            assert np.all(temps >= 300)
            assert np.all(temps <= 500)

            with pytest.warns(ResourceWarning) as record:
                temps_3km = res['temperature_3000m']
            warn_msg = record[0].message.args[0]
            expected_message = ("No depth info available for "
                                "'temperature_3000m', returning single "
                                "'temperature' value for requested depth "
                                "3000m")
            assert warn_msg == expected_message
            assert np.allclose(temps, temps_3km)

            potential = res['potential_MW']
            assert np.all(potential >= 100)
            assert np.all(potential <= 200)

            with pytest.warns(ResourceWarning) as record:
                potential_3km = res['potential_MW_3000m']
            warn_msg = record[0].message.args[0]
            expected_message = ("No depth info available for "
                                "'potential_MW_3000m', returning single "
                                "'potential_MW' value for requested depth "
                                "3000m")
            assert warn_msg == expected_message
            assert np.allclose(potential, potential_3km)

            assert res.depths == {'temperature': [], 'potential_MW': []}


def test_single_depth(sample_meta):
    """Test extracting data from file with a single depth."""

    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, 'outputs.h5')

        Outputs.init_h5(fp, ["temperature_3500m"],
                        shapes={"temperature_3500m": (10,)},
                        attrs={"temperature_3500m": SAMPLE_ATTRS},
                        chunks={"temperature_3500m": (10,)},
                        dtypes={"temperature_3500m": np.float32},
                        meta=sample_meta)

        with Outputs(fp, "a") as out:
            out["temperature_3500m"] = np.random.randint(300, 500, size=10)

        with GeothermalResource(fp) as res:
            assert "temperature_3500m" in res.dsets
            temps = res['temperature_3500m']
            assert np.all(temps >= 300)
            assert np.all(temps <= 500)

            with pytest.warns(ResourceWarning) as record:
                temps_3km = res['temperature_3000m']
            warn_msg = record[0].message.args[0]
            assert "Only one depth available" in warn_msg
            assert np.allclose(temps, temps_3km)

            assert res.depths == {'temperature': [3500], 'potential_MW': []}


def test_interpolation_and_extrapolation():
    """Test interpolation and extrapolation of data. """
    fp = os.path.join(TESTDATADIR, "geo", "template_geo_data.h5")
    with GeothermalResource(fp) as res:
        assert "temperature_3500m" in res.dsets
        assert "temperature_4500m" in res.dsets
        temps_3500m = res['temperature_3500m']
        temps_4500m = res['temperature_4500m']
        assert not np.allclose(temps_3500m, temps_4500m)

        temps_4000m = res['temperature_4000m']
        expected_temps = (temps_4500m - temps_3500m) / 2 + temps_3500m
        assert np.allclose(temps_4000m, expected_temps)

        with pytest.warns(ExtrapolationWarning) as record:
            temps_5000m = res['temperature_5000m']
        warn_msg = record[0].message.args[0]
        assert "5000 is outside the depth range (3500, 4500)" in warn_msg

        expected_temps = (temps_4500m - temps_3500m) / 2 + temps_4500m
        assert np.allclose(temps_5000m, expected_temps)

        assert not res.depths['potential_MW']
        assert all(d in res.depths['temperature'] for d in [3500, 4500])
        assert (res.get_attrs("temperature_3000m")
                == res.get_attrs("temperature_3100m"))
        assert (res.get_dset_properties("temperature_3000m")
                == res.get_dset_properties("temperature_3100m"))


def test_parse_name():
    """Test the `_parse_name` function. """
    func = GeothermalResource._parse_name

    name, val = func("temp")
    assert name == "temp"
    assert val is None

    name, val = func("temp_1m")
    assert name == "temp"
    assert val == 1

    name, val = func("temp_3.5m")
    assert name == "temp"
    assert np.isclose(val, 3.5)


def test_get_nearest_val():
    """Test the `_get_nearest_val` function. """
    sample_depths = [500, 1000, 3000, 100]
    func = GeothermalResource._get_nearest_val

    nearest, extrapolate = func(0, sample_depths)
    assert extrapolate
    assert nearest == [100, 500]

    nearest, extrapolate = func(700, sample_depths)
    assert not extrapolate
    assert nearest == [500, 1000]

    nearest, extrapolate = func(2900, sample_depths)
    assert not extrapolate
    assert nearest == [1000, 3000]

    nearest, extrapolate = func(5000, sample_depths)
    assert extrapolate
    assert nearest == [1000, 3000]


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
