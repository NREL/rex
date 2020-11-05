# -*- coding: utf-8 -*-
"""
pytests for  Rechunk h5
"""
import h5py
import numpy as np
import os
import pytest

from rex.rechunk_h5.combine_h5 import CombineH5
from rex import TESTDATADIR

PURGE_OUT = True


def check_combine(src, comb, axis=1):
    """
    Compare src and comb .h5 files
    """
    with h5py.File(comb, mode='r') as f_comb:
        with h5py.File(src, mode='r') as f_src:
            for dset in f_comb:
                assert dset in f_src
                ds_comb = f_comb[dset]
                ds_src = f_src[dset]
                comb_data = ds_comb[...]
                src_data = ds_src[...]
                shape = ds_src.shape
                assert ds_comb.dtype == ds_src.dtype
                assert ds_comb.chunks == ds_src.chunks
                if axis == 1:
                    if dset == 'meta':
                        assert ds_comb.shape == (shape[0] * 2, )
                        assert np.array_equal(comb_data[:shape[0]], src_data)
                        assert np.array_equal(comb_data[shape[0]:], src_data)
                    elif dset == 'coordinates':
                        assert ds_comb.shape == (shape[0] * 2, 2)
                        assert np.array_equal(comb_data[:shape[0]], src_data)
                        assert np.array_equal(comb_data[shape[0]:], src_data)
                    elif dset == 'time_index':
                        assert ds_comb.shape == shape
                        assert np.array_equal(comb_data, src_data)
                    else:
                        assert ds_comb.shape == (shape[0], shape[1] * 2)
                        assert np.array_equal(comb_data[:, :shape[1]],
                                              src_data)
                        assert np.array_equal(comb_data[:, shape[1]:],
                                              src_data)
                else:
                    if dset == 'meta':
                        assert ds_comb.shape == shape
                        assert np.array_equal(comb_data, src_data)
                    elif dset == 'coordinates':
                        assert ds_comb.shape == shape
                        assert np.array_equal(comb_data, src_data)
                    elif dset == 'time_index':
                        assert ds_comb.shape == (shape[0] * 2,)
                        assert np.array_equal(comb_data[:shape[0]], src_data)
                        assert np.array_equal(comb_data[shape[0]:], src_data)
                    else:
                        assert ds_comb.shape == (shape[0] * 2, shape[1])
                        assert np.array_equal(comb_data[:shape[0]], src_data)
                        assert np.array_equal(comb_data[shape[0]:], src_data)


@pytest.mark.parametrize('axis', [0, 1])
def test_combine_h5(axis):
    """
    Test CombineH5
    """
    src_path = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    combine_path = os.path.join(TESTDATADIR, 'wtk/combine.h5')

    CombineH5.run(combine_path, src_path, src_path, axis=axis)

    check_combine(src_path, combine_path, axis=axis)

    if PURGE_OUT:
        os.remove(combine_path)


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
