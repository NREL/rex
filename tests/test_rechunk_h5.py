# -*- coding: utf-8 -*-
"""
pytests for Rechunk h5
"""
from click.testing import CliRunner
import h5py
import numpy as np
import os
import pytest
import tempfile
import traceback

from rex.resource import Resource
from rex.rechunk_h5.rechunk_h5 import (get_dataset_attributes,
                                       RechunkH5)
from rex.rechunk_h5.rechunk_cli import main
from rex.utilities.loggers import LOGGERS
from rex.utilities.utilities import to_records_array
from rex import TESTDATADIR

LOGGERS.clear()


@pytest.fixture(scope="module")
def runner():
    """
    cli runner
    """
    return CliRunner()


def create_var_attrs(h5_file, t_chunk=(8 * 7 * 24)):
    """
    Create DataFrame for rechunk attributes

    Parameters
    ----------
    h5_file : str
        Source .h5 file
    t_chunk : int
        Number of timesteps per chunk

    Returns
    -------
    var_attrs : pandas.DataFrame
        rechunk variable attributes
    """
    var_attrs = get_dataset_attributes(h5_file)
    for var, _ in var_attrs.iterrows():
        if var == 'time_index':
            var_attrs.loc[var, 'dtype'] = 'S20'
            var_attrs.at[var, 'attrs'] = {'freq': 'h', 'timezone': 'UTC'}
        elif var == 'meta':
            var_attrs.loc[var, 'chunks'] = None
            var_attrs.loc[var, 'dtype'] = None
        elif var == 'coordinates':
            var_attrs.loc[var, 'chunks'] = None
        else:
            var_attrs.loc[var, 'chunks'] = (t_chunk, 10)

    return var_attrs


def check_rechunk(src, dst, missing=None):
    """
    Compare src and dst .h5 files
    """
    with h5py.File(dst, mode='r') as f_dst:
        with h5py.File(src, mode='r') as f_src:
            if missing is not None:
                for dset in missing:
                    assert dset in f_src
                    assert dset not in f_dst
            else:
                missing = []

            for dset in f_src:
                if dset not in missing:
                    assert dset in f_dst
                    ds_dst = f_dst[dset]
                    ds_src = f_src[dset]
                    assert ds_dst.shape == ds_src.shape
                    if dset != 'time_index':
                        assert ds_dst.dtype == ds_src.dtype

                    chunks = ds_dst.chunks
                    if chunks is not None:
                        assert chunks != ds_src.chunks


def test_to_records_array():
    """
    Test converstion of pandas DataFrame to numpy records array for .h5
    ingestion
    """
    path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with Resource(path) as f:
        meta = f.meta
        truth = f.h5['meta'][...]

    test = to_records_array(meta)

    for c in truth.dtype.names:
        msg = "{} did not get converted propertly!".format(c)
        assert np.all(test[c] == truth[c]), msg


@pytest.mark.parametrize('t_chunk', [None, 8 * 7 * 24])
def test_chunks(t_chunk):
    """
    Test rechunk chunk size
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    var_attrs = create_var_attrs(src_path, t_chunk)

    with tempfile.TemporaryDirectory() as td:
        rechunk_path = os.path.join(td, 'rechunk.h5')
        RechunkH5.run(src_path, rechunk_path, var_attrs=var_attrs)

        check_rechunk(src_path, rechunk_path)


@pytest.mark.parametrize('drop', [None,
                                  ['pressure_0m', ],
                                  ['pressure_100m',
                                   'temperature_100m',
                                   'windspeed_100m']])
def test_rechunk_h5(runner, drop):
    """
    Test RechunkH5
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    var_attrs = create_var_attrs(src_path)
    if drop is not None:
        var_attrs = var_attrs.drop(drop)

    with tempfile.TemporaryDirectory() as td:
        rechunk_path = os.path.join(td, 'rechunk.h5')
        attrs_path = os.path.join(td, 'var_attrs.json')
        var_attrs.to_json(attrs_path)

        result = runner.invoke(main, ['-src', src_path,
                                      '-dst', rechunk_path,
                                      '-vap', attrs_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        check_rechunk(src_path, rechunk_path, missing=drop)

    LOGGERS.clear()


def test_chunk_size(runner):
    """
    Test chunk size
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')

    with tempfile.TemporaryDirectory() as td:
        rechunk_path = os.path.join(td, 'rechunk.h5')

        result = runner.invoke(main, ['-src', src_path,
                                      '-dst', rechunk_path])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        check_rechunk(src_path, rechunk_path)

    LOGGERS.clear()


def test_downscale(runner):
    """
    Test downscaling resolution during RechunkH5
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    truth_path = os.path.join(TESTDATADIR, 'wtk/rechunk_3hr.h5')
    var_attrs = create_var_attrs(src_path, t_chunk=(7 * 24))

    with tempfile.TemporaryDirectory() as td:
        rechunk_path = os.path.join(td, 'rechunk.h5')
        attrs_path = os.path.join(td, 'var_attrs.json')
        var_attrs.to_json(attrs_path)

        result = runner.invoke(main, ['-src', src_path,
                                      '-dst', rechunk_path,
                                      '-vap', attrs_path,
                                      '-res', '3h'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        check_rechunk(truth_path, rechunk_path)

    LOGGERS.clear()


def test_hub_height(runner):
    """
    Test hub_height RechunkH5 kwarg
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    var_attrs = create_var_attrs(src_path, t_chunk=(7 * 24))

    with tempfile.TemporaryDirectory() as td:
        rechunk_path = os.path.join(td, 'rechunk.h5')
        attrs_path = os.path.join(td, 'var_attrs.json')
        var_attrs.to_json(attrs_path)

        result = runner.invoke(main, ['-src', src_path,
                                      '-dst', rechunk_path,
                                      '-vap', attrs_path,
                                      '-hgt', '100'])
        msg = ('Failed with error {}'
               .format(traceback.print_exception(*result.exc_info)))
        assert result.exit_code == 0, msg

        missing = ['pressure_0m', 'pressure_200m', 'temperature_80m',
                   'winddirection_80m', 'windspeed_80m']
        check_rechunk(src_path, rechunk_path, missing=missing)


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
