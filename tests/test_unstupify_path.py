# -*- coding: utf-8 -*-
"""
pytests for unstupify_path utility
"""
import os
import pytest

from rex import TESTDATADIR
# from rex.utilities.utilities import unstupify_path

HERE = os.path.realpath(__file__)


def test_unstupify_path():
    """
    Test unstupify path logic
    """
    print(TESTDATADIR)
    print(HERE)
    raise Exception('Trigger test')


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
