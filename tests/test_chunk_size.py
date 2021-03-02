# -*- coding: utf-8 -*-
"""
pytests for chunk size
"""
from math import ceil
import numpy as np
import os
import pytest

from rex.rechunk_h5.chunk_size import TimeseriesChunkSize, ArrayChunkSize


@pytest.mark.parametrize('freq', [0.5, 2, 5, 10, 12])
def test_time_chunks(freq):
    """
    Test time chunk size
    """
    truth = (ceil(ceil(8 / freq) * 24 * 7 * freq), 10)

    test = TimeseriesChunkSize.compute((8760 * freq, 10), 'float32')

    assert truth == test


@pytest.mark.parametrize(('dtype', 'chunk_size'),
                         [('int8', 2),
                          ('int8', 4),
                          ('int8', 8),
                          ('int16', 2),
                          ('int16', 4),
                          ('int16', 8),
                          ('float32', 2),
                          ('float32', 4),
                          ('float32', 8), ])
def test_timeseries_chunks(dtype, chunk_size):
    """
    Test timeseries chunk size
    """
    time_chunks = 8 * 24 * 7
    pixel_size = np.dtype(dtype).itemsize * 10**-6
    site_chunk = ceil(chunk_size / (time_chunks * pixel_size))
    truth = (time_chunks, site_chunk)

    test = TimeseriesChunkSize.compute((8760, 10000), dtype,
                                       chunk_size=chunk_size)

    assert truth == test


@pytest.mark.parametrize(('dtype', 'chunk_size', 'bytes'),
                         [('int8', 2, 1),
                          ('int8', 4, 1),
                          ('int8', 8, 1),
                          ('int16', 2, 2),
                          ('int16', 4, 2),
                          ('int16', 8, 2),
                          ('float32', 2, 4),
                          ('float32', 4, 4),
                          ('float32', 8, 4), ])
def test_array_chunks(chunk_size, dtype, bytes):
    """
    Test array chunk size
    """
    arr = np.zeros((10000, 10000), dtype=dtype)
    truth = (arr.size * bytes * 10**-6) / chunk_size

    truth = (ceil(arr.shape[0] / truth), 10000)

    test = ArrayChunkSize.compute(arr, chunk_size=chunk_size)

    assert truth == test


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
